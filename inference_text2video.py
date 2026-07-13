import argparse
import logging
import torch
from pathlib import Path
from datetime import datetime
from omegaconf import OmegaConf

# Custom modules from the MimicMotion project
from mimicmotion.utils.loader import create_pipeline      # builds the generation pipeline
from mimicmotion.utils.utils import save_to_mp4_custom    # saves tensor frames as MP4
from utils.text_processing import TextProcessor           # maps text to video file paths
from utils.video_processor import VideoProcessor          # preprocesses videos and runs pipeline
from constants import ASPECT_RATIO                        # defines target aspect ratio

# ----------------------------------------------------------------------
# Logging setup – a basic console logger; later a file handler may be added.
# ----------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s: [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def set_logger(log_file=None, log_level=logging.INFO):
    """
    Adds a file handler to the global logger if a log file path is provided.
    This allows logging to both console and a file.
    """
    if log_file:
        log_handler = logging.FileHandler(log_file, "w")
        log_handler.setFormatter(
            logging.Formatter("[%(asctime)s][%(name)s][%(levelname)s]: %(message)s")
        )
        log_handler.setLevel(log_level)
        logger.addHandler(log_handler)

# ----------------------------------------------------------------------
# Main pipeline execution – decorated with @torch.no_grad() to disable
# gradient computation and save memory during inference.
# ----------------------------------------------------------------------
@torch.no_grad()
def main(config):
    """
    config: an OmegaConf DictConfig object loaded from a YAML file.
    It contains all hyperparameters, paths, and task-specific settings.
    """

    # ------------------------------------------------------------------
    # 1. Device and precision setup
    # ------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # If the config does NOT disable float16, set the default tensor type to half precision.
    # This speeds up inference and reduces memory on supported GPUs.
    if not config.get('no_use_float16', False):
        torch.set_default_dtype(torch.float16)

    # ------------------------------------------------------------------
    # 2. Initialize helper classes
    # ------------------------------------------------------------------
    # TextProcessor: Given a text query (or a CSV mapping), it returns a list
    # of video file paths that match the text (e.g., using precomputed embeddings).
    text_processor = TextProcessor(
        config.csv_path,          # CSV mapping text -> video metadata
        config.embeddings_path,   # Precomputed text/video embeddings for retrieval
        config.gloss_words_path,  # Possibly a glossary/vocabulary file
        config.video_folder       # Root folder containing video files
    )

    # VideoProcessor: Handles video I/O, pose extraction, reference image loading,
    # and orchestrates the generation pipeline for each video.
    video_processor = VideoProcessor(ASPECT_RATIO)

    # ------------------------------------------------------------------
    # 3. Build the generative pipeline
    # ------------------------------------------------------------------
    # create_pipeline returns a callable object (or a pipeline object) that
    # takes pose and image inputs and generates video frames.
    pipeline = create_pipeline(
        config,
        device,
        unet_base_path=config.unet_path,      # pretrained UNet weights (for diffusion)
        pos_net_base_path=config.pos_net_path # pose encoder network weights
    )

    # ------------------------------------------------------------------
    # 4. Prepare output directory and logging
    # ------------------------------------------------------------------
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    if config.get('log_file', None):
        set_logger(config.log_file)

    # ------------------------------------------------------------------
    # 5. Determine input source: video list file OR free‑text query
    # ------------------------------------------------------------------
    input_text = config.get('input_text', "")

    if not input_text:
        logger.info("No input provided. Exiting.")
        return

    # Case A: input_text is given → retrieve matching videos via TextProcessor
    if input_text:
        video_paths = text_processor.get_videos_from_text(input_text)
        if not video_paths:
            logger.info("No videos found for the given text. Exiting.")
            return

    # ------------------------------------------------------------------
    # 6. Process each video, collect generated frame tensors
    # ------------------------------------------------------------------
    all_video_frames = []
    # The config.test_case is expected to be a list; we take the first entry
    # which holds resolution, sampling stride, fps, etc. for this run.
    task_config = config.test_case[0]


    for video_path in video_paths:
        print(video_path)
        logger.info(f"Processing video: {video_path}")

        # ------------------------------------------------------------------
        # 6a. Preprocess: extract pose sequence and load reference image
        # ------------------------------------------------------------------
        # pose_pixels: tensor of pose keypoints/heatmaps extracted from the video
        # image_pixels: tensor of the reference appearance image (resized to target resolution)
        pose_pixels, image_pixels = video_processor.preprocess(
            video_path,
            task_config.ref_image_path,          # reference image for appearance
            resolution=task_config.resolution,   # e.g., (256, 256)
            sample_stride=task_config.sample_stride  # frame sampling step for pose extraction
        )

        # ------------------------------------------------------------------
        # 6b. Run the generative pipeline to synthesize frames
        # ------------------------------------------------------------------
        # The pipeline uses the reference image as appearance and the pose sequence
        # as motion guidance to produce a video of the same length as the pose sequence.
        video_frames = video_processor.run_pipeline(
            pipeline,
            image_pixels,
            pose_pixels,
            device,
            task_config
        )

        # video_frames is a 4D tensor: [T, C, H, W] (frames, channels, height, width)
        all_video_frames.append(video_frames)

    # ------------------------------------------------------------------
    # 7. Concatenate all generated clips into one continuous video
    # ------------------------------------------------------------------
    merged_frames = torch.cat(all_video_frames, dim=0)   # concatenate along time dimension

    # Generate a timestamped filename
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    output_path = Path(config.output_dir) / f"merged_{timestamp}.mp4"

    # Save the merged tensor as an MP4 file with the specified fps
    save_to_mp4_custom(
        merged_frames,
        str(output_path),
        fps=task_config.fps
    )

    logger.info(f"Video saved to {output_path}")
    logger.info("--- Finished ---")


# ----------------------------------------------------------------------
# Entry point: parse command‑line argument for the config file,
# load it using OmegaConf, and invoke main().
# ----------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the MimicMotion pipeline.")
    parser.add_argument("--config_path", type=str, required=True,
                       help="Path to the YAML configuration file.")
    args = parser.parse_args()

    config = OmegaConf.load(args.config_path)
    main(config)
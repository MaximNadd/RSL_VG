import logging
from pathlib import Path
import numpy as np
import imageio
from moviepy import ImageSequenceClip

import cv2
import torch

logger = logging.getLogger(__name__)

def write_video(filename, video_tensor, fps=30, video_codec='mp4v', options=None):
    """
    Drop-in replacement for torchvision.io.write_video using OpenCV.
    video_tensor: torch.Tensor of shape [T, H, W, C] with values in [0, 1] or [0, 255]
    """
    if video_tensor.dtype != torch.uint8:
        video_tensor = (video_tensor * 255).clamp(0, 255).to(torch.uint8)
    frames = video_tensor.cpu().numpy()
    h, w = frames.shape[1], frames.shape[2]
    fourcc = cv2.VideoWriter_fourcc(*video_codec)
    out = cv2.VideoWriter(filename, fourcc, fps, (w, h))
    for frame in frames:
        # OpenCV uses BGR, so convert from RGB
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(bgr)
    out.release()


def save_to_mp4(frames, save_path, fps=7):
    frames = frames.permute((0, 2, 3, 1))  # (f, c, h, w) to (f, h, w, c)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    write_video(save_path, frames, fps=fps)

def save_to_mp4_custom(frames, save_path, fps=7):
    """
    Save a sequence of frames to an MP4 video file using imageio.

    Parameters:
    - frames (torch.Tensor): Tensor of shape (frames, channels, height, width).
    - save_path (str or Path): Path where the video will be saved.
    - fps (int): Frames per second for the output video.
    """
    # Ensure frames are on CPU and convert to numpy array
    frames = frames.permute(0, 2, 3, 1).cpu().numpy()  # (f, c, h, w) -> (f, h, w, c)

    # Create the directory if it doesn't exist
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    # Normalize frames if necessary (assuming frames are in [0, 1])
    if frames.dtype in [np.float32, np.float64]:
        frames = (frames * 255).astype(np.uint8)
    else:
        frames = frames.astype(np.uint8)

    # Save frames as a video using imageio
    with imageio.get_writer(save_path, fps=fps, codec='libx264', format='FFMPEG') as writer:
        for frame in frames:
            writer.append_data(frame)
    logger.info(f"Video saved to {save_path}")

def save_to_mp4_custom2(frames, save_path, fps=7):
    """
    Save a sequence of frames to an MP4 video file using moviepy.

    Parameters:
    - frames (torch.Tensor): Tensor of shape (frames, channels, height, width).
    - save_path (str or Path): Path where the video will be saved.
    - fps (int): Frames per second for the output video.
    """
    # Ensure frames are on CPU and convert to numpy array
    frames = frames.permute(0, 2, 3, 1).cpu().numpy()  # (f, c, h, w) -> (f, h, w, c)

    # Create the directory if it doesn't exist
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    # Normalize frames if necessary (assuming frames are in [0, 1])
    if frames.dtype in [np.float32, np.float64]:
        frames = (frames * 255).astype(np.uint8)
    else:
        frames = frames.astype(np.uint8)

    # Verify that frames are in RGB format
    if frames.shape[-1] != 3:
        raise ValueError(f"Expected frames with 3 channels (RGB), but got {frames.shape[-1]} channels.")

    # Create a video clip using moviepy
    clip = ImageSequenceClip(list(frames), fps=fps)

    # Write the video file
    try:
        clip.write_videofile(save_path, codec='libx264')
        logger.info(f"Video saved to {save_path}")
    except Exception as e:
        logger.error(f"Failed to save video to {save_path}: {e}")
        raise
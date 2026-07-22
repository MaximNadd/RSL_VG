import torch
from torchvision.datasets.folder import pil_loader
from torchvision.transforms.functional import pil_to_tensor, resize, center_crop, to_pil_image
import math
import numpy as np
# These imports are from the MimicMotion repository:
# - get_video_pose: extracts DWpose (DensePose) skeletons from every frame of a video.
# - get_image_pose: extracts a single DWpose skeleton from a reference image.
from mimicmotion.dwpose.preprocess import get_video_pose, get_image_pose


class VideoProcessor:
    """
    Preprocesses a reference image and a driving video for the MimicMotion model.
    Handles aspect-ratio-preserving resizing, pose extraction (using DWpose),
    and normalizes pixel values to the [-1, 1] range expected by the diffusion model.
    """

    def __init__(self, aspect_ratio):
        """
        Args:
            aspect_ratio: The target aspect ratio (width/height) for the output video.
                         For example, 9/16 for portrait, 16/9 for landscape.
                         This is used to determine the exact target dimensions.
        """
        self.aspect_ratio = aspect_ratio

    def preprocess(self, video_path, image_path, resolution=576, sample_stride=2):
        """
        Load and preprocess the reference image and driving video.

        Args:
            video_path: Path to the driving video file (e.g., .mp4).
            image_path: Path to the reference image file (e.g., .jpg/.png).
            resolution: The base resolution (shortest side) of the output.
                        Default is 576.
            sample_stride: Frame sampling rate. If stride=2, it takes every 2nd frame
                           from the driving video. This reduces the total number of
                           frames to fit memory limits or desired duration.

        Returns:
            pose_pixels: Tensor of shape (1, T+1, H, W, C)??? Wait, let's trace.
                         Actually, it returns (T+1, H, W, C) as numpy, then converted.
                         Let's be precise: 
                         - pose_pixels: torch.Tensor of shape (1, T+1, C, H, W)? No, look at the concat.
                         Let's trace carefully inside the function.
        """
        # 1. Load the reference image using torchvision's standard PIL loader.
        #    PIL loader returns a PIL Image object in RGB mode.
        image_pixels = pil_loader(image_path)

        # Convert PIL Image to a torch tensor of shape (C, H, W) with values in [0, 255].
        image_pixels = pil_to_tensor(image_pixels)  # (C, H, W)

        # Extract original height and width.
        h, w = image_pixels.shape[-2:]

        # 2. Compute the target height and width after resizing, while respecting aspect ratio.
        #    The goal: resize the image so that the *smaller* side becomes 'resolution',
        #    and the *larger* side is scaled proportionally, then rounded down to the
        #    nearest multiple of 64 (required by the VAE / UNet downsampling blocks).
        if h > w:
            # Portrait: height is larger than width -> target width is 'resolution',
            # target height is scaled by aspect_ratio, then rounded to multiple of 64.
            w_target = resolution
            # int(resolution / aspect_ratio // 64) * 64 ensures height is divisible by 64.
            h_target = int(resolution / self.aspect_ratio // 64) * 64
        else:
            # Landscape: width is larger than height -> target height is 'resolution',
            # target width is scaled, then rounded to multiple of 64.
            w_target = int(resolution / self.aspect_ratio // 64) * 64
            h_target = resolution

        # 3. Determine the actual resize dimensions (before center cropping).
        #    We want to cover the target crop area, so we scale the image so that
        #    the smaller dimension exactly fits, then center-crop the excess.
        h_w_ratio = float(h) / float(w)

        if h_w_ratio < h_target / w_target:
            # Image is too tall relative to target -> resize so height matches target,
            # and width becomes larger (to be cropped later).
            h_resize = h_target
            w_resize = math.ceil(h_target / h_w_ratio)
        else:
            # Image is too wide relative to target -> resize so width matches target,
            # and height becomes larger (to be cropped later).
            h_resize = math.ceil(w_target * h_w_ratio)
            w_resize = w_target

        # 4. Apply the resize and center crop transformations.
        #    'antialias=None' uses the default antialiasing (usually True in newer versions).
        image_pixels = resize(image_pixels, [h_resize, w_resize], antialias=None)
        image_pixels = center_crop(image_pixels, [h_target, w_target])

        # 5. Convert the preprocessed image tensor to a NumPy array in HWC format.
        #    The DWpose extraction functions expect HWC (height, width, channels) with
        #    values in [0, 255] and dtype uint8.
        image_pixels_np = image_pixels.permute((1, 2, 0)).numpy()  # (H, W, C)

        # 6. Extract poses using the DWpose detector.
        #    - get_image_pose: processes a single image and returns a pose map.
        #      Output shape: (H, W, C) – a heatmap-like representation of the skeleton.
        image_pose = get_image_pose(image_pixels_np)

        #    - get_video_pose: processes the driving video. It samples frames according to
        #      'sample_stride' and extracts poses for each sampled frame.
        #      Output shape: (T, H, W, C), where T = number of sampled frames.
        video_pose = get_video_pose(video_path, image_pixels_np, sample_stride=sample_stride)

        # 7. Combine the reference image pose and the video poses into a single array.
        #    The reference pose is inserted as the first frame (frame 0) of the sequence.
        #    shape of np.expand_dims(image_pose, 0) -> (1, H, W, C)
        #    shape of video_pose -> (T, H, W, C)
        #    concatenate on axis=0 -> (T+1, H, W, C)
        pose_pixels = np.concatenate([np.expand_dims(image_pose, 0), video_pose])

        # 8. Prepare the image pixels for the model.
        #    - np.expand_dims(image_pixels_np, 0) -> (1, H, W, C)
        #    - np.transpose(..., (0, 3, 1, 2)) -> (1, C, H, W) -> this is the standard
        #      batch, channels, height, width format expected by PyTorch.
        image_pixels = np.transpose(np.expand_dims(image_pixels_np, 0), (0, 3, 1, 2))

        # 9. Convert to PyTorch tensors and normalize to [-1, 1] range.
        #    The diffusion model was trained on normalized values.
        #    We copy the arrays to avoid negative stride issues with torch.from_numpy.
        return (
            torch.from_numpy(pose_pixels.copy()) / 127.5 - 1,   # pose_pixels: (T+1, H, W, C) in [-1, 1]
            torch.from_numpy(image_pixels) / 127.5 - 1          # image_pixels: (1, C, H, W) in [-1, 1]
        )

    def run_pipeline(self, pipeline, image_pixels, pose_pixels, device, task_config):
        """
        Execute the MimicMotion inference pipeline (diffusion-based video generation).

        Args:
            pipeline: The MimicMotion pipeline object (from diffusers or custom implementation).
            image_pixels: The preprocessed reference image tensor (1, C, H, W) in [-1, 1].
            pose_pixels: The combined pose tensor (T+1, H, W, C) in [-1, 1].
            device: torch device (e.g., 'cuda' or 'cpu').
            task_config: A configuration object containing parameters like:
                         seed, num_frames, frames_overlap, fps, noise_aug_strength,
                         num_inference_steps, guidance_scale.

        Returns:
            video_frames: torch.Tensor of shape (T, C, H, W) with pixel values in [0, 255]
                          (uint8 range). The first frame (reference image) is removed.
        """
        # 1. Denormalize the reference image from [-1, 1] back to [0, 255],
        #    then convert each tensor (batch dimension) to a PIL Image.
        #    The pipeline expects a list of PIL Images for the reference image.
        #    'to_pil_image' expects input in (C, H, W) format and values in [0, 255].
        #    We iterate over the batch dimension (size 1).
        image_pixels = [
            to_pil_image(img.to(torch.uint8))   # img is (C, H, W)
            for img in (image_pixels + 1.0) * 127.5   # denormalize: [-1,1] -> [0,255]
        ]

        # 2. Add a batch dimension to the pose sequence and move to the target device.
        #    pose_pixels is currently (T+1, H, W, C). Unsqueeze(0) -> (1, T+1, H, W, C).
        pose_pixels = pose_pixels.unsqueeze(0).to(device)

        # 3. Initialize a random generator for reproducibility.
        generator = torch.Generator(device=device)
        generator.manual_seed(task_config.seed)

        # 4. Call the MimicMotion pipeline.
        #    Key parameters explained:
        #    - image_pose: the pose sequence (including the reference pose as first frame).
        #    - num_frames: total frames to generate (matches the number of poses provided).
        #    - tile_size: number of frames processed in one tile (for memory efficiency).
        #    - tile_overlap: overlap between tiles to smooth transitions.
        #    - height/width: spatial dimensions of the generated video.
        #    - fps: frames per second (affects temporal motion strength).
        #    - noise_aug_strength: amount of noise added to the reference image (for diversity).
        #    - num_inference_steps: diffusion sampling steps (higher = better quality, slower).
        #    - min/max_guidance_scale: classifier-free guidance scale (control adherence to poses).
        #    - decode_chunk_size: number of latents to decode at once (memory optimization).
        #    - output_type: "pt" returns PyTorch tensors.
    
        frames = pipeline(
            image_pixels,
            image_pose=pose_pixels,
            num_frames=pose_pixels.size(1),                     # T+1 frames
            tile_size=task_config.num_frames,
            tile_overlap=task_config.frames_overlap,
            height=pose_pixels.shape[-2],                       # H
            width=pose_pixels.shape[-1],                        # W
            fps=task_config.fps,
            noise_aug_strength=task_config.noise_aug_strength,
            num_inference_steps=task_config.num_inference_steps,
            generator=generator,
            min_guidance_scale=task_config.guidance_scale,
            max_guidance_scale=task_config.guidance_scale,
            decode_chunk_size=4,
            output_type="pt",
            device=device
        ).frames.cpu()  # Move the generated frames from GPU to CPU.

        # 5. Convert the generated frames from range [-1, 1] to [0, 255] as uint8.
        #    frames shape: (1, T+1, C, H, W) - batch, time, channels, height, width.
        video_frames = (frames * 255.0).to(torch.uint8)

        # 6. Remove the first frame of the video.
        #    The first frame (index 0) is the reference image itself (conditioned input).
        #    The remaining frames (index 1 to end) are the generated animation.
        #    Resulting shape: (T, C, H, W) with uint8 values.
        return video_frames[0, 1:]  # Remove the batch dimension and the first frame
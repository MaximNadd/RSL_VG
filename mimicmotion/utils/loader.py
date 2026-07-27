import logging

import torch
import torch.utils.checkpoint
from diffusers.models import AutoencoderKLTemporalDecoder
from diffusers.schedulers import EulerDiscreteScheduler
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

from ..modules.unet import UNetSpatioTemporalConditionModel
from ..modules.pose_net import PoseNet
from ..pipelines.pipeline_mimicmotion import MimicMotionPipeline

logger = logging.getLogger(__name__)


class MimicMotionModel(torch.nn.Module):
    def __init__(self,
                 base_model_path,
                 unet_base_path="",
                 pos_net_base_path=""):
        """
        Construct base model components and load pretrained SVD model except pose-net.
        Args:
            base_model_path (str): path to pretrained Stable Video Diffusion (SVD) model
            unet_base_path (str): optional separate UNet checkpoint path
            pos_net_base_path (str): optional separate PoseNet checkpoint path
        """
        super().__init__()

        # --- UNet ---
        if unet_base_path:
            # Load the UNet from a specific checkpoint and cast to FP16 for efficiency.
            self.unet = UNetSpatioTemporalConditionModel.from_pretrained(unet_base_path).half()
        else:
            # Load only the UNet configuration from the base model's subfolder.
            # No pretrained weights are loaded here.
            self.unet = UNetSpatioTemporalConditionModel.from_config(
                UNetSpatioTemporalConditionModel.load_config(base_model_path, subfolder="unet"))

        # --- VAE (temporal decoder) ---
        # Load the pretrained AutoencoderKLTemporalDecoder from the base path and cast to FP16.
        self.vae = AutoencoderKLTemporalDecoder.from_pretrained(base_model_path, subfolder="vae").half()

        # --- CLIP image encoder ---
        # The vision encoder produces embeddings for the reference image.
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(base_model_path, subfolder="image_encoder")

        # --- Noise scheduler ---
        # Euler scheduler is used for the diffusion sampling process.
        self.noise_scheduler = EulerDiscreteScheduler.from_pretrained(base_model_path, subfolder="scheduler")

        # --- CLIP feature extractor ---
        # Preprocesses input images to the format expected by the CLIP encoder.
        self.feature_extractor = CLIPImageProcessor.from_pretrained(base_model_path, subfolder="feature_extractor")

        # --- PoseNet ---
        if pos_net_base_path:
            # Attempt to load PoseNet from a separate checkpoint.
            # NOTE: The following line is buggy because `from_pretrained` is a class method,
            # not an instance method. It will raise an AttributeError, triggering the except block.
            try:
                self.pose_net = PoseNet(noise_latent_channels=self.unet.config.block_out_channels[0]).from_pretrained(pos_net_base_path)
            except:
                # Fallback: manually load the state dict and handle DataParallel prefix.
                from collections import OrderedDict

                state_dict = torch.load(pos_net_base_path, map_location='cpu')

                # Remove the 'module.' prefix that appears when the model was saved with DataParallel.
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    new_key = k.replace('module.', '') if k.startswith('module.') else k
                    new_state_dict[new_key] = v

                # Create a fresh PoseNet with the correct number of input channels
                # (taken from the UNet's first block output channels).
                self.pose_net = PoseNet(noise_latent_channels=self.unet.config.block_out_channels[0])
                # Load the cleaned state dict with strict=True (all keys must match).
                self.pose_net.load_state_dict(new_state_dict, strict=True)
        else:
            # If no separate checkpoint is given, create an untrained PoseNet.
            # It will be initialised randomly and later loaded from a full checkpoint in create_pipeline.
            self.pose_net = PoseNet(noise_latent_channels=self.unet.config.block_out_channels[0])


def create_pipeline(infer_config, device,
                    unet_base_path='', pos_net_base_path=''):
    """
    Create the MimicMotion pipeline and load pretrained weights.

    Args:
        infer_config: an object that contains `base_model_path` and `ckpt_path`.
        device (str or torch.device): 'cpu' or 'cuda:{device_id}'.
        unet_base_path (str): optional separate UNet checkpoint.
        pos_net_base_path (str): optional separate PoseNet checkpoint.
    """
    if unet_base_path and pos_net_base_path:
        # Both UNet and PoseNet are loaded from separate checkpoints.
        # Instantiate the model, move to the target device, and set to eval mode.
        mimicmotion_models = MimicMotionModel(infer_config.base_model_path,
                                              unet_base_path=unet_base_path,
                                              pos_net_base_path=pos_net_base_path).to(device=device).eval()
    else:
        # Only the base model path is provided; the full model is built from that.
        # We then load the complete state dict from a single checkpoint file.
        mimicmotion_models = MimicMotionModel(infer_config.base_model_path).to(device=device).eval()
        # Load the weights with strict=False – this allows missing keys (e.g., pose_net
        # may be absent in the checkpoint if it was not trained yet) without crashing.
        mimicmotion_models.load_state_dict(torch.load(infer_config.ckpt_path, map_location=device, weights_only=False), strict=False)

    # Build the actual Diffusers-style pipeline by passing all sub‑models.
    pipeline = MimicMotionPipeline(
        vae=mimicmotion_models.vae,
        image_encoder=mimicmotion_models.image_encoder,
        unet=mimicmotion_models.unet,
        scheduler=mimicmotion_models.noise_scheduler,
        feature_extractor=mimicmotion_models.feature_extractor,
        pose_net=mimicmotion_models.pose_net
    )
    return pipeline


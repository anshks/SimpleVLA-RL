"""Utils for world model simulation integration."""

import numpy as np
import torch
from PIL import Image
from pathlib import Path

try:
    from world_model_eval.world_model import WorldModel
    from world_model_eval.utils import rescale_bridge_action, predict
except ImportError as e:
    print(f"Warning: can't import world_model_eval: {e}")


def load_png_to_tensor(png_path, target_size=256):
    """
    Load PNG file and convert to tensor format expected by world model.

    Args:
        png_path: Path to PNG file
        target_size: Target image size (default 256x256)

    Returns:
        Tensor of shape (H, W, C) with values in [0, 1] range
    """
    img = Image.open(png_path).convert("RGB")
    img = img.resize((target_size, target_size))
    img_array = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(img_array)


def worldmodel_frame_to_vla_input(frame):
    """
    Convert world model output frame to VLA input format.
    Follows Libero's observation format (RGB image only, no state used).

    Args:
        frame: Tensor of shape (1, 1, H, W, C) from world model, values in [0, 1]

    Returns:
        Dictionary with 'full_image' and 'state' keys
    """
    # Remove batch and time dimensions: (1, 1, H, W, C) -> (H, W, C)
    frame_np = frame[0, 0].cpu().numpy()

    # Convert to uint8 [0, 255] range
    frame_uint8 = np.clip(frame_np * 255, 0, 255).astype(np.uint8)

    # Return in same format as Libero (state is dummy, not used for Bridge)
    return {
        "full_image": frame_uint8,
        "state": np.zeros(7, dtype=np.float32)  # Dummy state, not used
    }


def pad_and_rescale_action(action, target_dim=10):
    """
    Pad 7D action to 10D and rescale for world model.

    Handles batched chunked actions: (batch_size, num_chunks, action_dim).

    Args:
        action: Action tensor of shape (batch_size, num_chunks, action_dim), typically 7D from VLA
        target_dim: Target action dimension (default 10)

    Returns:
        Rescaled action tensor of shape (batch_size, num_chunks, target_dim)
    """
    # Ensure action is a torch tensor
    if not isinstance(action, torch.Tensor):
        action = torch.from_numpy(action)

    batch_size, num_chunks, action_dim = action.shape

    # Pad from 7D to 10D with zeros along last dimension
    if action_dim < target_dim:
        pad_size = target_dim - action_dim
        padding = torch.zeros(
            (batch_size, num_chunks, pad_size),
            dtype=action.dtype,
            device=action.device,
        )
        action = torch.cat([action, padding], dim=-1)

    # Rescale using world model's rescaling function
    # Flatten to 2D, rescale each action, then reshape back
    action_flat = action.reshape(-1, target_dim)  # (batch_size * num_chunks, target_dim)
    action_rescaled = torch.stack([rescale_bridge_action(a) for a in action_flat])
    action = action_rescaled.reshape(batch_size, num_chunks, target_dim)

    return action


def get_worldmodel_checkpoint_config(checkpoint_name):
    """
    Get configuration for a specific world model checkpoint.

    Args:
        checkpoint_name: Name of the checkpoint file

    Returns:
        Dictionary with use_pixel_rope and default_cfg values
    """
    CHECKPOINTS_TO_KWARGS = {
        "bridge_v2_ckpt.pt": {
            "use_pixel_rope": True,
            "default_cfg": 1.0,
        },
        "mixed_openx_9robots_20frames_0p1actiondropout_580ksteps.pt": {
            "use_pixel_rope": False,
            "default_cfg": 3.0,
        },
        "ckpt_000480000.pt": {  # LIBERO world model
            "use_pixel_rope": False,
            "default_cfg": 3.0,
        },
    }

    # Extract just the filename if full path is provided
    checkpoint_file = Path(checkpoint_name).name

    # Return config if found, otherwise use defaults
    return CHECKPOINTS_TO_KWARGS.get(
        checkpoint_file,
        {
            "use_pixel_rope": False,
            "default_cfg": 1.0,
        },
    )


def load_world_model(checkpoint_path, rank):
    """
    Load and initialize world model from checkpoint.

    Args:
        checkpoint_path: Path to world model checkpoint file

    Returns:
        Initialized WorldModel instance
    """
    # Get checkpoint-specific config
    config = get_worldmodel_checkpoint_config(checkpoint_path)

    # Load world model with checkpoint path and configuration
    world_model = WorldModel(
        checkpoint_path=checkpoint_path,
        use_pixel_rope=config["use_pixel_rope"],
        default_cfg=config["default_cfg"],
        rank=rank,
    )

    return world_model

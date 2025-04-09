# utils.py
import os
import numpy as np
from PIL import Image
import torch

def set_random_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def calculate_positive_ratio(tensor: torch.Tensor) -> torch.Tensor:
    """Calculate the ratio of positive values in the tensor."""
    return (tensor > 0).float().mean()

def channel_mean_shift(z_T: torch.Tensor, target_shift: float = 0.11) -> torch.Tensor:
    """
    Apply channel mean shift for color guidance.
    For channels 1 and 2, gradually shift the values until the target positive ratio is reached.
    """
    z_T_star = z_T.clone()
    for c in [1, 2]:
        channel = z_T[:, c, :, :]
        initial_ratio = calculate_positive_ratio(channel)
        target_ratio = initial_ratio + target_shift

        delta = 0.0
        while True:
            shifted = channel + delta
            current_ratio = calculate_positive_ratio(shifted)
            if current_ratio >= target_ratio:
                break
            delta += 0.01

        z_T_star[:, c, :, :] = shifted
    return z_T_star

def create_2d_gaussian(height, width, std_dev, center_x=0, center_y=0):
    """
    Create a 2D Gaussian distribution where the center (center_x, center_y) is 1 and the periphery gradually decays towards 0.
    Coordinates are normalized in the range [-1, 1] using the specified standard deviation (std_dev).
    Returns a tensor with shape (1, 1, height//8, width//8).
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    y = torch.linspace(-1, 1, height // 8, device=device)
    x = torch.linspace(-1, 1, width // 8, device=device)
    y_grid, x_grid = torch.meshgrid(y, x, indexing="ij")

    x_grid = x_grid - center_x
    y_grid = y_grid - center_y

    gaussian = torch.exp(-((x_grid ** 2 + y_grid ** 2) / (2 * std_dev ** 2)))

    gaussian = gaussian.unsqueeze(0).unsqueeze(0)
    return gaussian

def tkg_noise(latents: torch.Tensor, device: str) -> torch.Tensor:
    """
    Apply noise processing to latent variables based on the tkg method.
    """
    from utils import channel_mean_shift, create_2d_gaussian
    z_T_star = channel_mean_shift(latents)
    mask = create_2d_gaussian(height=latents.shape[2], width=latents.shape[3], std_dev=0.5)
    mask = torch.nn.functional.interpolate(mask, size=(latents.shape[2], latents.shape[3]), mode='bilinear', align_corners=False)
    mask = mask.expand(-1, latents.shape[1], -1, -1)
    mask = mask.to(device).to(torch.float16)
    latents = mask * latents + (1 - mask) * z_T_star
    return latents

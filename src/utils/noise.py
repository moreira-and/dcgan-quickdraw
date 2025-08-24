import torch
from src.config import device


def noise(n, latent_dim):
    """
    Creates a noise vector to be the generator input
    """
    return torch.randn(n, latent_dim).to(device)

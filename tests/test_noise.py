import torch
from src.config import params, device
from src.utils.noise import noise
from src.modeling.models import Generator, Discriminator


def generate_image():
    generator = Generator(params.model.latent_dim).to(device)
    generator.eval()
    with torch.no_grad():
        img = generator(noise(1, params.model.latent_dim).to(device))
    return img


def test_generator_output_shape():
    img = generate_image()
    assert img.shape[1:] == (1, 28, 28) or img.shape[1:] == (28, 28)


def test_discriminator_output():
    img = generate_image()
    discriminator = Discriminator().to(device)
    discriminator.eval()
    with torch.no_grad():
        score = discriminator(img)
    assert score.shape == (1, 1)

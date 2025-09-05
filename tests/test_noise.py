import torch
from src.config import params, device
from src.utils.noise import noise
from src.modeling.models import Generator, Discriminator


def _generate_image():
    generator = Generator(params.model.generator.latent_dim).to(device)
    generator.eval()
    with torch.no_grad():
        img = generator(noise(1, params.model.generator.latent_dim).to(device))
    return img


def test_generator_output_shape():
    img = _generate_image()
    assert img.shape[1:] == (1, 28, 28) or img.shape[1:] == (28, 28)


def test_discriminator_output():
    img = _generate_image()
    discriminator = Discriminator().to(device)
    discriminator.eval()
    with torch.no_grad():
        score = discriminator(img)
    assert score.shape == (1, 1)

from loguru import logger
import time
import typer

from src.config import cfg, device
from src.utils.noise import noise
from src.modeling.models import Generator, Discriminator

import torch
from matplotlib import pyplot as plt

app = typer.Typer()


@app.command()
def main():
    # -----------------------------------------
    start_time = time.time()
    logger.info("Performing inference for model...")
    # -----------------------------------------

    generator = Generator(cfg.model.LATENT_DIM).to(device)

    ## testando o gerador, ele deve produzir uma imagem ruidosa
    generator.eval()
    with torch.no_grad():
        gen = generator(noise(1, cfg.model.LATENT_DIM).to(device))
    plt.imshow(gen.cpu().squeeze().numpy(), cmap="gray")

    discriminator = Discriminator().to(device)
    # testando o discriminador ele deve retornar um tensor da seguinte forma tensor([[valor]])
    discriminator.eval()
    with torch.no_grad():
        dis = discriminator(gen)
    print(dis)

    # -----------------------------------------
    elapsed_time = time.time() - start_time
    logger.success(f"Inference complete. Elapsed time: {elapsed_time:.2f} seconds")
    # -----------------------------------------


if __name__ == "__main__":
    app()

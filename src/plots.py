from loguru import logger
from tqdm import tqdm
import time

import typer

from src.config import PROCESSED_DATA_DIR

import torch
import matplotlib.pyplot as plt

import numpy as np

app = typer.Typer()


@app.command()
def main():
    # -----------------------------------------
    start_time = time.time()
    logger.info("Generating plot from data...")
    # -----------------------------------------

    for class_dir in tqdm(sorted(PROCESSED_DATA_DIR.iterdir()), desc="Plotting classes"):
        if not class_dir.is_dir():
            continue

        # Pega apenas o primeiro arquivo .pt da classe
        img_file = next(class_dir.glob("*.pt"), None)
        if img_file is None:
            continue

        img = torch.load(img_file)

        # Se vier esparso, densifica
        if getattr(img, "layout", torch.strided) != torch.strided:
            img = img.to_dense()

        # Se vier batelada [B,C,H,W], pega o primeiro
        if img.dim() == 4:
            img = img[0]

        # Garanta float
        img = img.to(torch.float32)

        # Desnormalização
        # img = img.mul_(0.5).add_(0.5).clamp_(0, 1)

        # Se for 2D (grayscale), recupere o canal
        if img.dim() == 2:
            img = img.unsqueeze(0)  # [1,H,W]

        # CHW -> HWC
        img = img.movedim(0, -1)

        # Numpy seguro
        img_np = img.detach().cpu().contiguous().numpy()

        # Opcional: forçar 3 canais para plot/salvar
        if img_np.shape[-1] == 1:
            img_np = np.repeat(img_np, 3, axis=-1)

        plt.figure()
        plt.imshow(img_np, cmap="gray")
        plt.title(class_dir.name)
        plt.axis("off")
        plt.show()

    # -----------------------------------------
    elapsed_time = time.time() - start_time
    logger.success(f"Plot generation complete. Elapsed time: {elapsed_time:.2f} seconds")
    # -----------------------------------------


if __name__ == "__main__":
    app()

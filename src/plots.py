from loguru import logger
from tqdm import tqdm
import time

import typer

from src.config import PROCESSED_DATA_DIR

import torch
import matplotlib.pyplot as plt

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

        # Carrega o tensor
        img_tensor = torch.load(img_file)  # [C, H, W]

        # Converte para [H, W] e desfaz normalize [-1,1] → [0,1]
        img = img_tensor.squeeze() * 0.5 + 0.5
        # [C,H,W] → [H,W,C]
        img_np = img.permute(1, 2, 0).numpy()

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

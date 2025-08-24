from pathlib import Path

from loguru import logger

# from tqdm import tqdm
import time

import typer

from src.config import MODELS_DIR, PROCESSED_DATA_DIR

import torch
from torch.utils.data import TensorDataset, DataLoader

app = typer.Typer()


@app.command()
def main(
    # -----------------------------------------
    model_path: Path = MODELS_DIR / "model.pkl",
    # -----------------------------------------
):
    # -----------------------------------------
    start_time = time.time()
    logger.info("Training some model...")
    # -----------------------------------------

    all_imgs, all_labels = [], []

    # Recarregar por classe
    for class_idx, class_name in enumerate(sorted(PROCESSED_DATA_DIR.iterdir())):
        if not class_name.is_dir():
            continue

        for img_file in class_name.glob("*.pt"):
            img_tensor = torch.load(img_file)
            all_imgs.append(img_tensor)
            all_labels.append(class_idx)  # usa o índice da classe

    # Concatenar em tensores contínuos
    all_imgs = torch.stack(all_imgs)  # [num_samples, C, H, W]
    all_labels = torch.tensor(all_labels)  # [num_samples]

    # Criar TensorDataset
    dataset = TensorDataset(all_imgs, all_labels)

    # DataLoader para treino
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Teste: imprimir forma do primeiro batch
    for batch_imgs, batch_labels in loader:
        print(batch_imgs.shape, batch_labels.shape)
        break

    # -----------------------------------------
    elapsed_time = time.time() - start_time
    logger.success(f"Modeling training complete. Elapsed time: {elapsed_time:.2f} seconds")
    # -----------------------------------------


if __name__ == "__main__":
    app()

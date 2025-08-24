from loguru import logger
import time
from tqdm import tqdm
import typer

from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR

import torch
from torchvision import datasets, transforms

app = typer.Typer()


@app.command()
def main():
    # -----------------------------------------
    start_time = time.time()
    logger.info("Generating features from dataset...")
    # -----------------------------------------

    # Transform: resize + tensor + normalize
    transform = transforms.Compose(
        [
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),  # [-1,1]
        ]
    )

    # Dataset ImageFolder
    dataset = datasets.ImageFolder(root=RAW_DATA_DIR, transform=transform)
    logger.info(f"Dataset loaded with {len(dataset)} images and {len(dataset.classes)} classes.")

    for class_name, class_idx in tqdm(
        dataset.class_to_idx.items(), desc="Saving tensors by class"
    ):
        class_dir = PROCESSED_DATA_DIR / class_name
        class_dir.mkdir(exist_ok=True)

        # Filtra imagens dessa classe
        for i, (img, label) in enumerate(dataset):
            if label == class_idx:
                torch.save(img, class_dir / f"{i}.pt")

        print(f"Processed {i + 1} images for class '{class_name}'")

    # -----------------------------------------
    elapsed_time = time.time() - start_time
    logger.success(f"Features generation complete. Elapsed time: {elapsed_time:.2f} seconds")
    # -----------------------------------------


if __name__ == "__main__":
    app()

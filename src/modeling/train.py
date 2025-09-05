from pathlib import Path

from loguru import logger

# from tqdm import tqdm
import time

import typer

from src.config import MODELS_DIR, PROCESSED_DATA_DIR, params
from src.modeling.models import Generator, Discriminator
from src.modeling.trainers import DCGANTrainer

import torch
from torch.utils.data import TensorDataset, DataLoader

app = typer.Typer()


@app.command()
def main(
    # -----------------------------------------
    artifact_name: str = "gan_artifact.pth",
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
    loader = DataLoader(dataset, batch_size=params.dataset.batch_size, shuffle=True)

    # -----------------------------------------

    generator_instance = Generator(params.model.generator.latent_dim)

    discriminator_instance = Discriminator()

    DCGANTrainer_instance = DCGANTrainer(generator_instance, discriminator_instance)

    # -----------------------------------------

    DCGANTrainer_instance.train(
        loader, epochs=params.train.epochs, num_classes=len(torch.unique(all_labels))
    )

    # Salvar os modelos treinados
    DCGANTrainer_instance.save_models(artifact_name)

    # -----------------------------------------
    elapsed_time = time.time() - start_time
    logger.success(f"Modeling training complete. Elapsed time: {elapsed_time:.2f} seconds")
    # -----------------------------------------


if __name__ == "__main__":
    app()

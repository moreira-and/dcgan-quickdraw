from src.config import device, params, MODELS_DIR
from src.utils import noise

import torch
import torch.optim as optim
from torch.nn import BCEWithLogitsLoss

from tqdm import tqdm

from matplotlib import pyplot as plt


class DCGANTrainer:
    def __init__(
        self, generator, discriminator, g_optimizer=None, d_optimizer=None, loss_function=None
    ):
        self.generator = generator.to(device)
        self.discriminator = discriminator.to(device)
        self.g_optimizer = g_optimizer or optim.Adam(
            generator.parameters(), lr=params.model.generator.lr
        )
        self.d_optimizer = d_optimizer or optim.Adam(
            discriminator.parameters(), lr=params.model.discriminator.lr
        )
        self.loss_function = loss_function or BCEWithLogitsLoss()
        self.batch_size = params.dataset.batch_size
        self.latent_dim = params.model.generator.latent_dim
        self.num_classes = None

    # training loop
    def _train_discriminator(self, real_imgs):

        self.d_optimizer.zero_grad()
        # x_train = x_train.unsqueeze(1)

        # Mover as imagens reais para o dispositivo correto (CPU ou GPU)
        # real_imgs = x_train.to(device)

        batch_size = real_imgs.size(0)

        # Passar imagens reais pelo discriminador
        real = self.discriminator(real_imgs)
        # Calcular a perda para as imagens reais, comparando com um tensor de uns
        real_loss = self.loss_function(real, torch.ones(batch_size, 1).to(device))

        # Gerar imagens falsas com o gerador
        fake = self.generator(noise(batch_size, self.latent_dim)).detach()
        # Mover as imagens falsas para o dispositivo correto (CPU ou GPU)
        fake = fake.to(device)
        # Passe as imagens falsas pelo discriminador
        fake = self.discriminator(fake)
        # Calcular a perda para as imagens falsas, comparando com um tensor de zeros
        fake_loss = self.loss_function(fake, torch.zeros(batch_size, 1).to(device))

        # backpropagation
        discriminator_loss = real_loss + fake_loss
        discriminator_loss.backward()
        self.d_optimizer.step()

        return discriminator_loss

    def _train_generator(self):

        self.g_optimizer.zero_grad()

        # Gerar imagens falsas com o gerador
        fake = self.generator(noise(self.batch_size, self.latent_dim))
        fake = fake.to(device)
        # Passar as imagens falsas pelo discriminador
        fake = self.discriminator(fake)

        # Calcular a perda do gerador comparando com um tensor de uns
        generator_loss = self.loss_function(fake, torch.ones(self.batch_size, 1).to(device))

        generator_loss.backward()
        self.g_optimizer.step()

        return generator_loss

    def train(self, data_loader, epochs, num_classes=1):

        self.discriminator.train()
        self.generator.train()
        self.num_classes = num_classes

        print("Training...")
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")
            d_loss = 0.0
            g_loss = 0.0

            for real_imgs, _ in tqdm(data_loader):  # ignora labels, só imagens importam
                d_loss = self._train_discriminator(real_imgs)
                g_loss = self._train_generator()
                print(f"Epoch {epoch+1}/{epochs} | d_loss: {d_loss:.4f} | g_loss: {g_loss:.4f}")

        self._test()

    def _test(self):
        # testando o gerador treinado
        self.generator.eval()
        with torch.no_grad():

            fig, axes = plt.subplots(3, 5, figsize=(12, 6))
            for i, ax in enumerate(axes.flat):
                gen = self.generator(noise(1, self.latent_dim).to(device))
                # como estamos utilizando um batch de 64, vamos pegar a primeira imagem de cada batch
                ax.imshow(gen.cpu().numpy()[0, 0, ...], cmap="gray")
                ax.axis("off")

            plt.show()

    def save_models(self, artifact_file: str = "gan_artifact.pth"):
        MODELS_DIR.mkdir(parents=True, exist_ok=True)  # garante que o diretório existe
        artifact_path = MODELS_DIR / artifact_file

        torch.save(
            {
                "generator_state": self.generator.state_dict(),
                "discriminator_state": self.discriminator.state_dict(),
                "latent_dim": self.latent_dim,
                "num_classes": self.num_classes,
                "img_shape": (
                    1,
                    params.dataset.image_size,
                    params.dataset.image_size,
                ),  # ajustar caso a imagem seja diferente
            },
            artifact_path,
        )

        print(f"Artifact saved at: {artifact_path}")

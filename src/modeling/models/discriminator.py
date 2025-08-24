import torch.nn as nn


class Discriminator(nn.Module):
    """
    The Generator is a neural network that receives a random noise vector as input and generates an image from that vector.
    The goal of the Generator is to fool the Discriminator by generating images that are indistinguishable from real images.
    """

    def __init__(self):
        super().__init__()
        self.model_pt1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=0),  # shape: 28x28 -> 13x13
            nn.ReLU(),
            nn.Dropout2d(0.4),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=0),  # shape: 13x13 -> 6x6
            nn.ReLU(),
            nn.Dropout2d(0.4),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0),  # shape: 6x6 -> 4x4
            nn.ReLU(),
            nn.Dropout2d(0.4),
        )

        self.model_pt2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding="same"),
            nn.ReLU(),
            nn.Dropout2d(0.4),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 1),
        )

    def forward(self, x):
        x = self.model_pt1(x)
        x = self.model_pt2(x)
        return x

import torch.nn as nn


class View(nn.Module):
    """
    Class for resizing the tensor, similar to numpy's reshape and tf.reshape
    """

    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)

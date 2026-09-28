"""Model definition for MNIST DP disparity experiments.

Matches the architecture from:
"Disparate Impact in Differential Privacy from Gradient Misalignment" (arXiv:2206.07737)
"""

import torch
from torch import nn


class PaperCNN(nn.Module):
    """Exact 2-layer CNN used in the paper:
    Conv2d(1, 32, 3, stride=1, padding=0) -> Tanh ->
    Conv2d(32, 16, 3, stride=1, padding=0) -> Tanh ->
    Flatten -> Linear(16 * 24 * 24, 10)
    Total parameters: 97,114 (no pooling layers).
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=0)
        self.act1 = nn.Tanh()
        self.conv2 = nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=0)
        self.act2 = nn.Tanh()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(16 * 24 * 24, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act1(self.conv1(x))
        x = self.act2(self.conv2(x))
        x = self.flatten(x)
        return self.fc(x)


def get_model(seed: int, device: str = "cpu") -> PaperCNN:
    """Instantiates the paper's CNN initialized deterministically with the given seed."""
    torch.manual_seed(seed)
    model = PaperCNN().to(device)
    return model

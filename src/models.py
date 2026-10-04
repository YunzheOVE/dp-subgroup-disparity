"""Model definition for MNIST DP disparity experiments.

Matches the architecture from:
"Disparate Impact in Differential Privacy from Gradient Misalignment" (arXiv:2206.07737)
"""

import torch
from torch import nn


class PaperCNN(nn.Module):
    """Exact 2-layer CNN used for Table 2 MNIST benchmarks in the paper:
    Conv2d(1, 32, 3, stride=1, padding=0) -> Tanh ->
    Conv2d(32, 16, 3, stride=1, padding=0) -> Tanh ->
    Flatten -> Linear(16 * 24 * 24, 10)
    Total parameters: 97,114 (no pooling layers).
    Matches the authors' official script: mnist_script.sh (--config net=cnn --config hidden_channels=32,16).
    (The paper's Appendix B.3 reports 80,522 MNIST parameters, which conflicts with the released CNN configuration.)
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


class PaperAdultMLP(nn.Module):
    """Paper's 2-hidden-layer MLP for Adult tabular benchmark:
    Linear(d, 256) -> Tanh -> Linear(256, 256) -> Tanh -> Linear(256, 2)
    With biases and raw output logits (no dropout, no batch normalization).
    Input dimension d is dynamically derived from one-hot encoded features.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 256, num_classes: int = 2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act1 = nn.Tanh()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.act2 = nn.Tanh()
        self.fc3 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act1(self.fc1(x))
        x = self.act2(self.fc2(x))
        return self.fc3(x)


def get_model(seed: int, device: str = "cpu", model_type: str = "cnn", input_dim: int = None) -> nn.Module:
    """Instantiates the paper's model initialized deterministically with the given seed.
    Defaults to 'cnn' (PaperCNN) for full backward compatibility with MNIST.
    """
    torch.manual_seed(seed)
    if model_type.lower() in ("cnn", "mnist"):
        model = PaperCNN().to(device)
    elif model_type.lower() in ("mlp", "adult"):
        if input_dim is None:
            raise ValueError("input_dim must be provided when instantiating PaperAdultMLP.")
        model = PaperAdultMLP(input_dim=input_dim).to(device)
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Expected 'cnn' or 'mlp'.")
    return model


def get_adult_model(seed: int, input_dim: int, device: str = "cpu") -> PaperAdultMLP:
    """Convenience factory for the Adult MLP."""
    return get_model(seed=seed, device=device, model_type="adult", input_dim=input_dim)

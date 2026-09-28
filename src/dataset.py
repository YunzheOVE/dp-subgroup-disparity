"""Dataset utilities for imbalanced MNIST experiments.

Follows the sampling method from:
"Disparate Impact in Differential Privacy from Gradient Misalignment" (arXiv:2206.07737)
"""

import random
from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms


class TensorDataset(Dataset):
    """Simple in-memory dataset of image and label tensors."""

    def __init__(self, images: torch.Tensor, labels: torch.Tensor):
        self.images = images
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.images[index], self.labels[index]


def get_mnist_subsampled(
    data_dir: Path,
    seed: int,
    keep_eight: float = 0.09,
) -> Tuple[TensorDataset, TensorDataset]:
    """Loads MNIST and downsamples digit 8 on the training split using the paper's method.

    For each training sample labeled 8, it is retained if random.random() <= keep_eight.
    All other training digits (0-7, 9) and the full test set remain 100% intact.
    Images are scaled to [0.0, 1.0].
    """
    raw_train = datasets.MNIST(root=str(data_dir), train=True, download=True)
    raw_test = datasets.MNIST(root=str(data_dir), train=False, download=True)

    # Scale images to [0, 1] float tensors of shape (N, 1, 28, 28)
    train_images = raw_train.data.unsqueeze(1).float() / 255.0
    train_labels = raw_train.targets

    test_images = raw_test.data.unsqueeze(1).float() / 255.0
    test_labels = raw_test.targets

    # Subsample training set using the author's linear filtering with random.seed
    random.seed(seed)
    keep_indices = []
    for idx, label in enumerate(train_labels):
        if label.item() == 8:
            if random.random() <= keep_eight:
                keep_indices.append(idx)
        else:
            keep_indices.append(idx)

    subsampled_train_images = train_images[keep_indices]
    subsampled_train_labels = train_labels[keep_indices]

    train_dataset = TensorDataset(subsampled_train_images, subsampled_train_labels)
    test_dataset = TensorDataset(test_images, test_labels)

    return train_dataset, test_dataset


def get_loaders(
    train_dataset: TensorDataset,
    test_dataset: TensorDataset,
    batch_size: int = 256,
) -> Tuple[DataLoader, DataLoader]:
    """Creates PyTorch DataLoaders with batch size 256 for train and test."""
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

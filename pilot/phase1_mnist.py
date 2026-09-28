"""Phase 1: train a non-private MNIST classifier with rare digit 8."""

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--keep-eight", type=float, default=0.09)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 0 < args.keep_eight <= 1:
        parser.error("--keep-eight must be between 0 and 1")
    if args.epochs < 1:
        parser.error("--epochs must be positive")

    torch.manual_seed(args.seed)
    root = Path(__file__).resolve().parent
    data_dir = root / "data"
    train = datasets.MNIST(data_dir, train=True, download=True, transform=transforms.ToTensor())
    test = datasets.MNIST(data_dir, train=False, download=True, transform=transforms.ToTensor())

    eight = torch.where(train.targets == 8)[0]
    other = torch.where(train.targets != 8)[0]
    chosen = eight[torch.rand(len(eight)) < args.keep_eight]
    indices = torch.cat((other, chosen)).sort().values.tolist()
    counts = torch.bincount(train.targets[indices], minlength=10).tolist()
    assert counts[8] == len(chosen)
    assert sum(counts) == len(indices)
    print("Training examples per digit:", counts)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = nn.Sequential(
        nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(), nn.Linear(32 * 7 * 7, 10),
    ).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    train_loader = DataLoader(Subset(train, indices), batch_size=128, shuffle=True)
    test_loader = DataLoader(test, batch_size=512)

    for epoch in range(args.epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = nn.functional.cross_entropy(model(images), labels)
            loss.backward()
            optimizer.step()
        print(f"Finished epoch {epoch + 1}/{args.epochs}")

    model.eval()
    correct = torch.zeros(10, dtype=torch.long)
    total = torch.zeros(10, dtype=torch.long)
    loss_sum = torch.zeros(10, dtype=torch.double)
    with torch.no_grad():
        for images, labels in test_loader:
            logits = model(images.to(device))
            predictions = logits.argmax(dim=1).cpu()
            losses = nn.functional.cross_entropy(logits, labels.to(device), reduction="none").cpu()
            total += torch.bincount(labels, minlength=10)
            correct += torch.bincount(labels[predictions == labels], minlength=10)
            loss_sum += torch.bincount(labels, weights=losses.double(), minlength=10)

    accuracy = (correct / total).tolist()
    mean_loss = (loss_sum / total).tolist()
    result = {
        "seed": args.seed,
        "keep_eight": args.keep_eight,
        "epochs": args.epochs,
        "device": device,
        "training_count_by_digit": counts,
        "test_count_by_digit": total.tolist(),
        "accuracy_by_digit": accuracy,
        "loss_by_digit": mean_loss,
        "overall_accuracy": correct.sum().item() / total.sum().item(),
    }
    print("Test accuracy by digit:", [f"{a:.1%}" for a in accuracy])
    print("Test loss by digit:", [f"{x:.3f}" for x in mean_loss])
    print(f"Overall test accuracy: {result['overall_accuracy']:.1%}")
    path = root / "phase1_results.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("Saved:", path)


if __name__ == "__main__":
    main()


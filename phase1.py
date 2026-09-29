"""Phase 1: Non-private baseline training on imbalanced MNIST.

Follows the protocol in plan.md and the paper:
"Disparate Impact in Differential Privacy from Gradient Misalignment" (arXiv:2206.07737)

Settings:
- 2-layer CNN (97,114 params) with Tanh activations and no pooling.
- Standard SGD with lr=0.01, momentum=0, batch size 256, 60 epochs.
- Digit 8 retained with 9% probability; all other classes 100%; full test set.
- Seeds 0 to 4 with saved initial weights and dataset indices for exact pairing with Phase 2 and 4.
"""

import argparse
import json
import platform
import time
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import torch
from torch import nn
import torchvision
import opacus

from src.dataset import get_mnist_subsampled, get_loaders
from src.models import get_model, PaperCNN


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 1: Non-private baseline training")
    parser.add_argument("--seed", type=int, default=0, help="Random seed (0 to 4)")
    parser.add_argument("--all-seeds", action="store_true", help="Run all 5 seeds (0 to 4) sequentially")
    parser.add_argument("--epochs", type=int, default=60, help="Number of training epochs (default: 60)")
    parser.add_argument("--lr", type=float, default=0.01, help="SGD learning rate (default: 0.01)")
    parser.add_argument("--momentum", type=float, default=0.0, help="SGD momentum (default: 0.0)")
    parser.add_argument("--batch-size", type=int, default=256, help="Train/test batch size (default: 256)")
    parser.add_argument("--keep-eight", type=float, default=0.09, help="Retention probability for digit 8 (default: 0.09)")
    parser.add_argument("--device", type=str, default="auto", help="Compute device: 'cuda', 'cpu', or 'auto'")
    parser.add_argument("--data-dir", type=str, default="data", help="Directory for MNIST data")
    parser.add_argument("--output-dir", type=str, default="results", help="Directory to save results and weights")
    return parser.parse_args()


def train_single_seed(
    seed: int,
    epochs: int,
    lr: float,
    momentum: float,
    batch_size: int,
    keep_eight: float,
    device: str,
    data_dir: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    """Trains Phase 1 for a single random seed and returns the evaluation metrics."""
    print("=" * 70)
    print(f"PHASE 1 (NON-PRIVATE): SEED {seed}")
    print("=" * 70)

    # 1. Deterministic seeding
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 2. Dataset loading and subsampling
    train_dataset, test_dataset, keep_indices = get_mnist_subsampled(
        data_dir=data_dir,
        seed=seed,
        keep_eight=keep_eight,
    )
    train_loader, test_loader = get_loaders(train_dataset, test_dataset, batch_size=batch_size)

    # Count training samples per digit
    train_labels = train_dataset.labels
    counts = torch.bincount(train_labels, minlength=10).tolist()
    print("Training examples per digit (0–9):")
    for digit, count in enumerate(counts):
        tag = " (RARE)" if digit == 8 else (" (CONTROL)" if digit == 2 else "")
        print(f"  Digit {digit}: {count}{tag}")
    print(f"  Total training samples: {len(train_dataset)}")

    # 3. Model initialization
    model = get_model(seed=seed, device=device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model architecture: PaperCNN ({total_params:,} parameters)")

    # Save initial weights for exact pairing across Phase 1, Phase 2, and Phase 4
    output_dir.mkdir(parents=True, exist_ok=True)
    initial_weights_path = output_dir / f"initial_weights_seed_{seed}.pt"
    torch.save(model.state_dict(), initial_weights_path)
    print(f"Saved initial weights to: {initial_weights_path.name}")

    # 4. Optimizer and loss function
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    criterion = nn.CrossEntropyLoss()

    # 5. Training loop
    start_time = time.time()
    print(f"\nStarting training: {epochs} epochs, lr={lr}, momentum={momentum}, batch_size={batch_size}...")

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(labels)

        epoch_loss = running_loss / len(train_dataset)

        # Log every 10 epochs or final epoch
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            elapsed = time.time() - start_time
            print(f"  Epoch {epoch:2d}/{epochs:2d} | Train Loss: {epoch_loss:.4f} | Time: {elapsed:.1f}s")

    total_training_time = time.time() - start_time

    # 6. Evaluation on test set
    model.eval()
    correct = torch.zeros(10, dtype=torch.long)
    total = torch.zeros(10, dtype=torch.long)
    loss_sum = torch.zeros(10, dtype=torch.double)

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            logits = model(images)
            predictions = logits.argmax(dim=1).cpu()
            losses = nn.functional.cross_entropy(logits, labels.to(device), reduction="none").cpu()

            total += torch.bincount(labels, minlength=10)
            correct += torch.bincount(labels[predictions == labels], minlength=10)
            loss_sum += torch.bincount(labels, weights=losses.double(), minlength=10)

    accuracy_by_digit = (correct / total).tolist()
    loss_by_digit = (loss_sum / total).tolist()
    overall_accuracy = correct.sum().item() / total.sum().item()
    overall_loss = loss_sum.sum().item() / total.sum().item()

    # 7. Print summary metrics
    print("\n" + "-" * 70)
    print(f"RESULTS FOR SEED {seed}:")
    print(f"  Overall Accuracy:      {overall_accuracy * 100:.2f}% (Loss: {overall_loss:.4f})")
    print(f"  Control Digit 2 Acc:   {accuracy_by_digit[2] * 100:.2f}% (Loss: {loss_by_digit[2]:.4f}) [Paper Target: ~98.0%]")
    print(f"  Rare Digit 8 Acc:      {accuracy_by_digit[8] * 100:.2f}% (Loss: {loss_by_digit[8]:.4f}) [Paper Target: ~84.3%]")
    print("-" * 70)

    # 8. Save structured result JSON
    result_data = {
        "phase": 1,
        "method": "non-private",
        "seed": seed,
        "epochs": epochs,
        "learning_rate": lr,
        "momentum": momentum,
        "batch_size": batch_size,
        "keep_eight": keep_eight,
        "training_time_seconds": round(total_training_time, 2),
        "device": device,
        "software_versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "opacus": opacus.__version__,
        },
        "training_counts_by_digit": counts,
        "test_counts_by_digit": total.tolist(),
        "accuracy_by_digit": accuracy_by_digit,
        "loss_by_digit": loss_by_digit,
        "overall_accuracy": overall_accuracy,
        "overall_loss": overall_loss,
        "digit_2_accuracy": accuracy_by_digit[2],
        "digit_8_accuracy": accuracy_by_digit[8],
        "digit_2_loss": loss_by_digit[2],
        "digit_8_loss": loss_by_digit[8],
        "keep_indices": keep_indices,
    }

    result_file = output_dir / f"phase1_seed_{seed}.json"
    result_file.write_text(json.dumps(result_data, indent=2), encoding="utf-8")
    print(f"Results saved to: {result_file}\n")

    return result_data


def main():
    args = parse_args()

    # Determine device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Using device: {device} ({torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'})")

    data_dir = Path(args.data_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    seeds = [0, 1, 2, 3, 4] if args.all_seeds else [args.seed]
    all_results = []

    for seed in seeds:
        res = train_single_seed(
            seed=seed,
            epochs=args.epochs,
            lr=args.lr,
            momentum=args.momentum,
            batch_size=args.batch_size,
            keep_eight=args.keep_eight,
            device=device,
            data_dir=data_dir,
            output_dir=output_dir,
        )
        all_results.append(res)

    # If multiple seeds were run, compute and display mean +/- std summary
    if len(all_results) > 1:
        d2_accs = [r["digit_2_accuracy"] * 100 for r in all_results]
        d8_accs = [r["digit_8_accuracy"] * 100 for r in all_results]
        overall_accs = [r["overall_accuracy"] * 100 for r in all_results]

        summary = {
            "phase": 1,
            "seeds": seeds,
            "digit_2_accuracy_mean": np.mean(d2_accs),
            "digit_2_accuracy_std": np.std(d2_accs, ddof=1),
            "digit_8_accuracy_mean": np.mean(d8_accs),
            "digit_8_accuracy_std": np.std(d8_accs, ddof=1),
            "overall_accuracy_mean": np.mean(overall_accs),
            "overall_accuracy_std": np.std(overall_accs, ddof=1),
        }

        summary_file = output_dir / "phase1_summary.json"
        summary_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        print("=" * 70)
        print("PHASE 1 MULTI-SEED SUMMARY (SEEDS 0–4):")
        print("=" * 70)
        print(f"  Digit 2 (Control) Accuracy: {summary['digit_2_accuracy_mean']:.2f}% +/- {summary['digit_2_accuracy_std']:.2f}% [Paper: 98.0% +/- 0.1%]")
        print(f"  Digit 8 (Rare) Accuracy:    {summary['digit_8_accuracy_mean']:.2f}% +/- {summary['digit_8_accuracy_std']:.2f}% [Paper: 84.3% +/- 1.1%]")
        print(f"  Overall Accuracy:           {summary['overall_accuracy_mean']:.2f}% +/- {summary['overall_accuracy_std']:.2f}%")
        print(f"Summary saved to: {summary_file}")
        print("=" * 70)


if __name__ == "__main__":
    main()

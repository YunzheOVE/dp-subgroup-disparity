"""Phase 2: Vanilla DP-SGD training on imbalanced MNIST.

Follows the protocol in plan.md and the paper:
"Disparate Impact in Differential Privacy from Gradient Misalignment" (arXiv:2206.07737)

Settings:
- 2-layer CNN (80,522 params) with Tanh activations and no pooling.
- Paired data selection and initial weights from Phase 1 for seeds 0 to 4.
- Standard DP-SGD via Opacus with:
  - Clipping norm C = 1.0
  - Noise multiplier sigma = 0.8 (fixed, no target epsilon tuning)
  - Accountant = RDP (Renyi Differential Privacy)
  - Delta = 1e-6
  - Learning rate = 0.01, momentum = 0, batch size = 256, 60 epochs
  - Target achieved epsilon ≈ 5.90
- Calculates privacy cost relative to Phase 1:
  Privacy Cost = Phase 1 Accuracy - Phase 2 Accuracy
"""

import argparse
import json
import platform
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np
import torch
from torch import nn
import torchvision
from opacus import PrivacyEngine

from src.dataset import get_mnist_subsampled, get_loaders
from src.models import get_model, PaperCNN


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 2: Vanilla DP-SGD training")
    parser.add_argument("--seed", type=int, default=0, help="Random seed (0 to 4)")
    parser.add_argument("--all-seeds", action="store_true", help="Run all 5 seeds (0 to 4) sequentially")
    parser.add_argument("--epochs", type=int, default=60, help="Number of training epochs (default: 60)")
    parser.add_argument("--lr", type=float, default=0.01, help="SGD learning rate (default: 0.01)")
    parser.add_argument("--momentum", type=float, default=0.0, help="SGD momentum (default: 0.0)")
    parser.add_argument("--batch-size", type=int, default=256, help="Train/test batch size (default: 256)")
    parser.add_argument("--max-grad-norm", "-C", type=float, default=1.0, help="Clipping norm C (default: 1.0)")
    parser.add_argument("--sigma", type=float, default=0.8, help="Noise multiplier sigma (default: 0.8)")
    parser.add_argument("--delta", type=float, default=1e-6, help="Target delta for DP accounting (default: 1e-6)")
    parser.add_argument("--accountant", type=str, default="rdp", help="DP accountant type: 'rdp' or 'prv' (default: rdp)")
    parser.add_argument("--keep-eight", type=float, default=0.09, help="Retention probability for digit 8 (default: 0.09)")
    parser.add_argument("--device", type=str, default="auto", help="Compute device: 'cuda', 'cpu', or 'auto'")
    parser.add_argument("--data-dir", type=str, default="data", help="Directory for MNIST data")
    parser.add_argument("--output-dir", type=str, default="results", help="Directory to save results")
    return parser.parse_args()


def load_phase1_pairing(seed: int, output_dir: Path) -> Tuple[Optional[List[int]], Optional[Dict[str, Any]]]:
    """Attempts to load Phase 1 sampled indices and initial weights for exact pairing."""
    phase1_file = output_dir / f"phase1_seed_{seed}.json"
    initial_weights_file = output_dir / f"initial_weights_seed_{seed}.pt"

    indices = None
    phase1_data = None
    if phase1_file.exists():
        try:
            phase1_data = json.loads(phase1_file.read_text(encoding="utf-8"))
            indices = phase1_data.get("keep_indices", None)
            print(f"Loaded paired training indices from Phase 1: {phase1_file.name}")
        except Exception as e:
            print(f"Warning: Could not read {phase1_file}: {e}")

    initial_weights_path = initial_weights_file if initial_weights_file.exists() else None
    if initial_weights_path:
        print(f"Loaded paired initial weights from Phase 1: {initial_weights_file.name}")
    else:
        print(f"Note: No saved weights file found at {initial_weights_file.name}. Initializing model deterministically with seed {seed}.")

    return indices, phase1_data


def train_single_seed_dpsgd(
    seed: int,
    epochs: int,
    lr: float,
    momentum: float,
    batch_size: int,
    max_grad_norm: float,
    sigma: float,
    delta: float,
    accountant: str,
    keep_eight: float,
    device: str,
    data_dir: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    """Runs Phase 2 Vanilla DP-SGD for a single seed and returns evaluation metrics."""
    print("=" * 70)
    print(f"PHASE 2 (VANILLA DP-SGD): SEED {seed}")
    print("=" * 70)

    # 1. Deterministic seeding
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 2. Check for Phase 1 pairing (indices and weights)
    paired_indices, phase1_data = load_phase1_pairing(seed, output_dir)

    train_dataset, test_dataset, keep_indices = get_mnist_subsampled(
        data_dir=data_dir,
        seed=seed,
        keep_eight=keep_eight,
        indices=paired_indices,
    )
    train_loader, test_loader = get_loaders(train_dataset, test_dataset, batch_size=batch_size)

    # Verify counts
    counts = torch.bincount(train_dataset.labels, minlength=10).tolist()
    print(f"Training set: {len(train_dataset)} samples (Digit 8: {counts[8]}, Digit 2: {counts[2]})")

    # 3. Model setup
    model = get_model(seed=seed, device=device)
    initial_weights_file = output_dir / f"initial_weights_seed_{seed}.pt"
    if initial_weights_file.exists():
        model.load_state_dict(torch.load(initial_weights_file, map_location=device))

    # 4. Standard optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    criterion = nn.CrossEntropyLoss()

    # 5. Attach Opacus PrivacyEngine
    try:
        privacy_engine = PrivacyEngine(accountant=accountant)
    except TypeError:
        # Fallback for older Opacus signatures
        privacy_engine = PrivacyEngine()

    print(f"\nAttaching Opacus PrivacyEngine:")
    print(f"  Accountant:       {accountant.upper()}")
    print(f"  Clipping Norm C:  {max_grad_norm}")
    print(f"  Noise Multiplier: {sigma}")
    print(f"  Target Delta:     {delta}")

    model, optimizer, train_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        noise_multiplier=sigma,
        max_grad_norm=max_grad_norm,
    )

    # 6. Training loop
    start_time = time.time()
    print(f"\nStarting DP-SGD training: {epochs} epochs, batch_size={batch_size}...")

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
            current_eps = privacy_engine.get_epsilon(delta=delta)
            elapsed = time.time() - start_time
            print(f"  Epoch {epoch:2d}/{epochs:2d} | Train Loss: {epoch_loss:.4f} | Current eps: {current_eps:.2f} | Time: {elapsed:.1f}s")

    total_training_time = time.time() - start_time
    achieved_epsilon = privacy_engine.get_epsilon(delta=delta)

    # 7. Evaluation on test set
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

    # 8. Privacy cost calculation relative to Phase 1
    privacy_cost_by_digit = None
    cost_digit_2 = None
    cost_digit_8 = None
    disparity_gap = None

    if phase1_data is not None and "accuracy_by_digit" in phase1_data:
        p1_accs = phase1_data["accuracy_by_digit"]
        privacy_cost_by_digit = [p1 - p2 for p1, p2 in zip(p1_accs, accuracy_by_digit)]
        cost_digit_2 = privacy_cost_by_digit[2]
        cost_digit_8 = privacy_cost_by_digit[8]
        disparity_gap = cost_digit_8 - cost_digit_2

    # 9. Print summary metrics
    print("\n" + "-" * 70)
    print(f"PHASE 2 RESULTS FOR SEED {seed}:")
    print(f"  Achieved Privacy:      epsilon = {achieved_epsilon:.2f}, delta = {delta}")
    print(f"  Overall Accuracy:      {overall_accuracy * 100:.2f}% (Loss: {overall_loss:.4f})")
    print(f"  Control Digit 2 Acc:   {accuracy_by_digit[2] * 100:.2f}% (Loss: {loss_by_digit[2]:.4f}) [Paper Target: ~89.0%]")
    print(f"  Rare Digit 8 Acc:      {accuracy_by_digit[8] * 100:.2f}% (Loss: {loss_by_digit[8]:.4f}) [Paper Target: ~26.3%]")

    if cost_digit_8 is not None:
        print(f"\n  Privacy Cost (π = Phase 1 Acc - Phase 2 Acc):")
        print(f"    Control Digit 2 Cost (π₂): {cost_digit_2 * 100:.2f}% [Paper Target: ~8.9%]")
        print(f"    Rare Digit 8 Cost (π₈):    {cost_digit_8 * 100:.2f}% [Paper Target: ~57.9%]")
        print(f"    Disparity Gap (π₈ - π₂):   {disparity_gap * 100:.2f}% [Paper Target: ~48.9%]")
    print("-" * 70)

    # 10. Save structured result JSON
    result_data = {
        "phase": 2,
        "method": "dpsgd",
        "seed": seed,
        "epochs": epochs,
        "learning_rate": lr,
        "momentum": momentum,
        "batch_size": batch_size,
        "clipping_norm_C": max_grad_norm,
        "noise_multiplier_sigma": sigma,
        "target_delta": delta,
        "accountant": accountant,
        "achieved_epsilon": achieved_epsilon,
        "training_time_seconds": round(total_training_time, 2),
        "device": device,
        "software_versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
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
        "privacy_cost_by_digit": privacy_cost_by_digit,
        "privacy_cost_digit_2": cost_digit_2,
        "privacy_cost_digit_8": cost_digit_8,
        "disparity_gap": disparity_gap,
    }

    result_file = output_dir / f"phase2_seed_{seed}.json"
    result_file.write_text(json.dumps(result_data, indent=2), encoding="utf-8")
    print(f"Results saved to: {result_file}\n")

    return result_data


def main():
    args = parse_args()

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
        res = train_single_seed_dpsgd(
            seed=seed,
            epochs=args.epochs,
            lr=args.lr,
            momentum=args.momentum,
            batch_size=args.batch_size,
            max_grad_norm=args.max_grad_norm,
            sigma=args.sigma,
            delta=args.delta,
            accountant=args.accountant,
            keep_eight=args.keep_eight,
            device=device,
            data_dir=data_dir,
            output_dir=output_dir,
        )
        all_results.append(res)

    # Multi-seed aggregated summary
    if len(all_results) > 1:
        d2_accs = [r["digit_2_accuracy"] * 100 for r in all_results]
        d8_accs = [r["digit_8_accuracy"] * 100 for r in all_results]
        overall_accs = [r["overall_accuracy"] * 100 for r in all_results]
        eps_list = [r["achieved_epsilon"] for r in all_results]

        summary = {
            "phase": 2,
            "seeds": seeds,
            "achieved_epsilon_mean": float(np.mean(eps_list)),
            "digit_2_accuracy_mean": float(np.mean(d2_accs)),
            "digit_2_accuracy_std": float(np.std(d2_accs, ddof=1)),
            "digit_8_accuracy_mean": float(np.mean(d8_accs)),
            "digit_8_accuracy_std": float(np.std(d8_accs, ddof=1)),
            "overall_accuracy_mean": float(np.mean(overall_accs)),
            "overall_accuracy_std": float(np.std(overall_accs, ddof=1)),
        }

        # Privacy cost statistics if Phase 1 was available
        cost_d2 = [r["privacy_cost_digit_2"] * 100 for r in all_results if r["privacy_cost_digit_2"] is not None]
        cost_d8 = [r["privacy_cost_digit_8"] * 100 for r in all_results if r["privacy_cost_digit_8"] is not None]
        gaps = [r["disparity_gap"] * 100 for r in all_results if r["disparity_gap"] is not None]

        if cost_d8:
            summary["privacy_cost_digit_2_mean"] = float(np.mean(cost_d2))
            summary["privacy_cost_digit_2_std"] = float(np.std(cost_d2, ddof=1))
            summary["privacy_cost_digit_8_mean"] = float(np.mean(cost_d8))
            summary["privacy_cost_digit_8_std"] = float(np.std(cost_d8, ddof=1))
            summary["disparity_gap_mean"] = float(np.mean(gaps))
            summary["disparity_gap_std"] = float(np.std(gaps, ddof=1))

        summary_file = output_dir / "phase2_summary.json"
        summary_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        print("=" * 70)
        print("PHASE 2 MULTI-SEED SUMMARY (SEEDS 0–4):")
        print("=" * 70)
        print(f"  Achieved Epsilon:           {summary['achieved_epsilon_mean']:.2f} (Delta: {args.delta}) [Paper: 5.90]")
        print(f"  Digit 2 (Control) Accuracy: {summary['digit_2_accuracy_mean']:.2f}% +/- {summary['digit_2_accuracy_std']:.2f}% [Paper: 89.0% +/- 0.1%]")
        print(f"  Digit 8 (Rare) Accuracy:    {summary['digit_8_accuracy_mean']:.2f}% +/- {summary['digit_8_accuracy_std']:.2f}% [Paper: 26.3% +/- 0.4%]")
        print(f"  Overall Accuracy:           {summary['overall_accuracy_mean']:.2f}% +/- {summary['overall_accuracy_std']:.2f}%")

        if cost_d8:
            print(f"\n  Privacy Cost Metrics (Relative to Phase 1):")
            print(f"    Control Digit 2 Cost (π₂): {summary['privacy_cost_digit_2_mean']:.2f}% +/- {summary['privacy_cost_digit_2_std']:.2f}% [Paper: 8.9% +/- 0.1%]")
            print(f"    Rare Digit 8 Cost (π₈):    {summary['privacy_cost_digit_8_mean']:.2f}% +/- {summary['privacy_cost_digit_8_std']:.2f}% [Paper: 57.9% +/- 1.3%]")
            print(f"    Disparity Gap (π₈ - π₂):   {summary['disparity_gap_mean']:.2f}% +/- {summary['disparity_gap_std']:.2f}% [Paper: 48.9% +/- 1.3%]")

        print(f"\nSummary saved to: {summary_file}")
        print("=" * 70)


if __name__ == "__main__":
    main()

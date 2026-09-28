"""Phase 2: Train a differentially private (DP-SGD) MNIST classifier using Opacus.

Investigates utility disparity and the privacy cost on the rare digit 8 compared
to majority classes and the control digit 2.
"""

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from opacus import PrivacyEngine


def get_args():
    parser = argparse.ArgumentParser(description="Phase 2: DP-SGD on imbalanced MNIST")
    parser.add_argument("--keep-eight", type=float, default=0.09, help="Retention probability for digit 8")
    parser.add_argument("--epochs", type=int, default=8, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="SGD momentum")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--max-grad-norm", "-C", type=float, default=1.0, help="Clipping threshold C")
    parser.add_argument("--target-epsilon", type=float, default=5.90, help="Target epsilon (default: 5.90)")
    parser.add_argument("--target-delta", type=float, default=1e-6, help="Target delta (default: 1e-6)")
    parser.add_argument("--noise-multiplier", "--sigma", type=float, default=None,
                        help="Optional explicit noise multiplier sigma (overrides target-epsilon)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--baseline", type=str, default="phase1_results.json",
                        help="Path to Phase 1 baseline JSON for privacy cost comparison")
    parser.add_argument("--output", type=str, default="phase2_results.json", help="Output JSON path")
    args = parser.parse_args()

    if not 0 < args.keep_eight <= 1:
        parser.error("--keep-eight must be between 0 and 1")
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    if args.max_grad_norm <= 0:
        parser.error("--max-grad-norm must be positive")
    return args


def create_model():
    return nn.Sequential(
        nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(), nn.Linear(32 * 7 * 7, 10),
    )


def main():
    args = get_args()
    torch.manual_seed(args.seed)

    root = Path(__file__).resolve().parent
    data_dir = root / "data"

    print("=" * 70)
    print("PHASE 2: DIFFERENTIAL PRIVACY (DP-SGD) ON IMBALANCED MNIST")
    print("=" * 70)

    # 1. Dataset setup matching Phase 1
    train = datasets.MNIST(data_dir, train=True, download=True, transform=transforms.ToTensor())
    test = datasets.MNIST(data_dir, train=False, download=True, transform=transforms.ToTensor())

    eight = torch.where(train.targets == 8)[0]
    other = torch.where(train.targets != 8)[0]
    chosen = eight[torch.rand(len(eight)) < args.keep_eight]
    indices = torch.cat((other, chosen)).sort().values.tolist()
    counts = torch.bincount(train.targets[indices], minlength=10).tolist()
    print("Training examples per digit:", counts)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device} ({torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'})")

    # 2. Model & Optimizer
    model = create_model().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    train_loader = DataLoader(Subset(train, indices), batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test, batch_size=512)

    # 3. Attach Opacus Privacy Engine
    privacy_engine = PrivacyEngine()
    if args.noise_multiplier is not None:
        print(f"Attaching PrivacyEngine with explicit sigma={args.noise_multiplier}, C={args.max_grad_norm}...")
        model, optimizer, train_loader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            noise_multiplier=args.noise_multiplier,
            max_grad_norm=args.max_grad_norm,
        )
    else:
        print(f"Attaching PrivacyEngine targeting epsilon={args.target_epsilon}, delta={args.target_delta}, C={args.max_grad_norm} over {args.epochs} epochs...")
        model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            target_epsilon=args.target_epsilon,
            target_delta=args.target_delta,
            epochs=args.epochs,
            max_grad_norm=args.max_grad_norm,
        )

    accountant_name = privacy_engine.accountant.mechanism()
    computed_sigma = optimizer.noise_multiplier
    print(f"Privacy Accountant: {accountant_name}")
    print(f"Calibrated Noise Multiplier (sigma): {computed_sigma:.4f}")
    print(f"Clipping Norm (C): {args.max_grad_norm}")
    print("-" * 70)

    # 4. Training loop
    for epoch in range(args.epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = nn.functional.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()

        curr_eps = privacy_engine.get_epsilon(args.target_delta)
        print(f"Finished epoch {epoch + 1}/{args.epochs} | Epsilon achieved: {curr_eps:.2f} (delta={args.target_delta})")

    final_eps = privacy_engine.get_epsilon(args.target_delta)
    print("-" * 70)
    print(f"Training Complete. Final Privacy Guarantee: (eps={final_eps:.4f}, delta={args.target_delta})")
    print("-" * 70)

    # 5. Evaluation
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
    overall_acc = correct.sum().item() / total.sum().item()

    print("Test accuracy by digit:", [f"{a:.1%}" for a in accuracy])
    print("Test loss by digit:", [f"{x:.3f}" for x in mean_loss])
    print(f"Overall test accuracy: {overall_acc:.1%}")

    # 6. Compare with Phase 1 Baseline & Compute Privacy Cost
    baseline_path = root / args.baseline
    privacy_cost = None
    baseline_acc = None

    if baseline_path.exists():
        try:
            baseline_data = json.loads(baseline_path.read_text(encoding="utf-8"))
            baseline_acc = baseline_data.get("accuracy_by_digit")
            if baseline_acc and len(baseline_acc) == 10:
                privacy_cost = [b - d for b, d in zip(baseline_acc, accuracy)]
                print("\n" + "=" * 75)
                print("PRIVACY COST ANALYSIS (Acc_non_private - Acc_DP)")
                print("=" * 75)
                print(f"{'Digit':<6} | {'Non-Private':<12} | {'DP-SGD Acc':<12} | {'Privacy Cost':<14} | {'Role':<18}")
                print("-" * 75)
                for d in range(10):
                    role = "Rare Subgroup (9%)" if d == 8 else ("Control Digit" if d == 2 else "Majority")
                    cost_str = f"{privacy_cost[d]:+.1%}"
                    print(f"{d:<6} | {baseline_acc[d]:<12.1%} | {accuracy[d]:<12.1%} | {cost_str:<14} | {role:<18}")
                print("-" * 75)
                print(f"Overall Accuracy: Non-Private {baseline_data.get('overall_accuracy', 0):.1%} -> DP-SGD {overall_acc:.1%}")
                print(f"Digit 8 Privacy Cost vs Digit 2 (Control) Privacy Cost: {privacy_cost[8]:+.1%} vs {privacy_cost[2]:+.1%}")
                print("=" * 75)
        except Exception as e:
            print(f"Warning: could not process baseline {baseline_path}: {e}")

    # 7. Output result JSON
    result = {
        "seed": args.seed,
        "keep_eight": args.keep_eight,
        "epochs": args.epochs,
        "device": device,
        "target_epsilon": args.target_epsilon,
        "target_delta": args.target_delta,
        "achieved_epsilon": final_eps,
        "achieved_delta": args.target_delta,
        "clipping_threshold_C": args.max_grad_norm,
        "noise_multiplier_sigma": float(computed_sigma),
        "privacy_accountant": accountant_name,
        "lr": args.lr,
        "momentum": args.momentum,
        "batch_size": args.batch_size,
        "training_count_by_digit": counts,
        "test_count_by_digit": total.tolist(),
        "accuracy_by_digit": accuracy,
        "loss_by_digit": mean_loss,
        "overall_accuracy": overall_acc,
        "non_private_accuracy_by_digit": baseline_acc,
        "privacy_cost_by_digit": privacy_cost,
    }

    out_path = root / args.output
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nSaved Phase 2 results to: {out_path}")


if __name__ == "__main__":
    main()

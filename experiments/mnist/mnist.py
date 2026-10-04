"""Unified MNIST Experiment Runner.

Provides a unified interface matching adult.py for multi-dataset consistency:
- Supports --phase 1|2|3|4|all, --seed, and --all-seeds
- Interoperates with phase1.py, phase2.py, and phase4.py
- Reads and writes to results/mnist/
"""

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Also ensure local experiments/mnist is in sys.path for phase1/phase2/phase4 imports
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from phase1 import train_single_seed as run_phase1
from phase2 import train_single_seed_dpsgd as run_phase2
from phase4 import train_single_seed_global_adapt as run_phase4


def main():
    parser = argparse.ArgumentParser(description="Unified MNIST DP-SGD Experiment Runner")
    parser.add_argument("--phase", type=str, default="all", choices=["1", "2", "3", "4", "all"], help="Phase to execute")
    parser.add_argument("--seed", type=int, default=0, help="Random seed (0 to 4)")
    parser.add_argument("--all-seeds", action="store_true", help="Run across all 5 seeds (0 to 4)")
    parser.add_argument("--epochs", type=int, default=60, help="Training epochs (default: 60)")
    parser.add_argument("--lr", type=float, default=0.01, help="SGD learning rate (default: 0.01)")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size (default: 256)")
    parser.add_argument("--keep-eight", type=float, default=0.09, help="Retention probability for digit 8 (default: 0.09)")
    parser.add_argument("--data-dir", type=str, default=str(REPO_ROOT / "data"), help="Directory for MNIST data")
    parser.add_argument("--output-dir", type=str, default=str(REPO_ROOT / "results" / "mnist"), help="Directory for results")
    parser.add_argument("--device", type=str, default="auto", help="Device: 'cuda', 'cpu', or 'auto'")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    seeds = list(range(5)) if args.all_seeds else [args.seed]

    for s in seeds:
        print(f"\n=======================================================")
        print(f"             PROCESSING MNIST EXPERIMENT: SEED {s}")
        print(f"=======================================================")

        if args.phase in ("1", "all"):
            run_phase1(
                seed=s,
                epochs=args.epochs,
                lr=args.lr,
                momentum=0.0,
                batch_size=args.batch_size,
                keep_eight=args.keep_eight,
                device=args.device,
                data_dir=data_dir,
                output_dir=output_dir,
            )

        if args.phase in ("2", "all"):
            run_phase2(
                seed=s,
                epochs=args.epochs,
                lr=args.lr,
                momentum=0.0,
                batch_size=args.batch_size,
                max_grad_norm=1.0,
                sigma=0.8,
                delta=1e-6,
                accountant="rdp",
                keep_eight=args.keep_eight,
                device=args.device,
                data_dir=data_dir,
                output_dir=output_dir,
            )

        if args.phase in ("4", "all"):
            run_phase4(
                seed=s,
                epochs=args.epochs,
                lr=0.1,  # Global-Adapt uses lr=0.1 for MNIST
                momentum=0.0,
                batch_size=args.batch_size,
                max_grad_norm=1.0,
                sigma=0.8,
                delta=1e-6,
                accountant="rdp",
                keep_eight=args.keep_eight,
                device=args.device,
                data_dir=data_dir,
                output_dir=output_dir,
                initial_Z=50.0,
                bits_noise=10.0,
                lr_Z=0.1,
                threshold=0.7,
            )


if __name__ == "__main__":
    main()

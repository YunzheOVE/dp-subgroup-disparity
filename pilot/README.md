# Pilot Study: Imbalanced MNIST DP-SGD

This directory documents the initial pilot experiment conducted on MNIST prior to scaling to the full 60-epoch, 5-seed benchmark.

---

## Purpose
1. **Feasibility test:** Verify that Opacus DP-SGD runs cleanly on the local environment and GPU.
2. **Early hypothesis validation:** Test whether differential privacy causes an immediate disparate impact on a rare class (digit 8 with 9% retention) compared to common classes (e.g., digit 2).

---

## Pilot Configuration
- **Model:** 2-layer CNN with ReLU and MaxPool (~20.5k parameters):
  `Conv(1->16, 3x3) -> ReLU -> MaxPool(2) -> Conv(16->32, 3x3) -> ReLU -> MaxPool(2) -> Linear(1568, 10)`
- **Training:** SGD with learning rate 0.1, momentum 0.9, batch size 128, 8 epochs.
- **Data:** Digit 8 kept with 9% retention probability (~504 training images); full test set.
- **Seed:** 42.

---

## Pilot Results Summary

| Experiment | Epsilon (ε) | Delta (δ) | Noise (σ) | Overall Acc | Control (Digit 2) Acc | Rare (Digit 8) Acc | Digit 8 Privacy Cost |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Phase 1 (Non-private)** | - | - | - | 98.3% | 99.2% | 91.0% | - |
| **Phase 2 (DP-SGD eps 8.0)** | 7.99 | 1e-6 | 0.50 | 92.9% | 94.5% | 70.0% | **20.9%** |
| **Phase 2 (DP-SGD eps 5.9)** | 5.90 | 1e-6 | 0.59 | 90.4% | 93.1% | 61.2% | **29.8%** |
| **Phase 2 (DP-SGD eps 3.0)** | 3.00 | 1e-6 | 0.81 | 85.1% | 88.7% | 48.6% | **42.4%** |

---

## Key Takeaway
Even with shallow 8-epoch training, the privacy cost on digit 8 is dramatically higher than on control digit 2 (20.9% vs 4.7% at epsilon ≈ 8.0). This confirmed the disparate impact phenomenon and justified progressing to the paper's standardized 60-epoch, 5-seed protocol detailed in [plan.md](../plan.md).

---

## Files
- `phase1_mnist.py`: Fast 8-epoch non-private training script.
- `phase2_dpsgd.py`: Fast 8-epoch DP-SGD training script using Opacus.
- `phase1_results.json`: Output metrics for the non-private baseline.
- `phase2_results*.json`: Output metrics across tested epsilon budgets.

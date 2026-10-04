# Experiment Log: MNIST DP-SGD Disparity Study

This log tracks experimental runs across seeds 0–4 following the protocol defined in [plan.md](plan.md), investigating differential privacy disparity on the rare digit 8 compared to control digit 2.

---

## Benchmark Summary Table (Seeds 0–4)

| Seed | Phase | Method | Epochs | Batch Size | lr | Noise (σ) | Clip (C) | Achieved (ε, δ) | Digit 2 Acc | Digit 8 Acc | Privacy Cost (π₈) | Notes |
| :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **0** | Phase 1 | Non-private | 60 | 256 | 0.01 | - | - | Non-private | 97.87% | 87.27% | - | Overall Acc: 97.21% |
| **0** | Phase 2 | Vanilla DP-SGD | 60 | 256 | 0.01 | 0.8 | 1.0 | ε = 5.90, 1e-6 | 89.15% | 24.33% | 62.94% | Overall Acc: 85.30%, π₂ = 8.72% |
| **0** | Phase 4 | DPSGD-Global-Adapt | 60 | 256 | 0.10 | 0.8 | 1.0 (Z=31.8) | ε = 5.91, 1e-6 | 92.54% | 67.86% | 19.40% | Overall Acc: 91.90%, π₂ = 5.33%, Gain: +43.53% |
| **1** | Phase 1 | Non-private | 60 | 256 | 0.01 | - | - | Non-private | 97.38% | 87.17% | - | Overall Acc: 96.98% |
| **1** | Phase 2 | Vanilla DP-SGD | 60 | 256 | 0.01 | 0.8 | 1.0 | ε = 5.90, 1e-6 | 89.15% | 23.00% | 64.17% | Overall Acc: 85.22%, π₂ = 8.24% |
| **1** | Phase 4 | DPSGD-Global-Adapt | 60 | 256 | 0.10 | 0.8 | 1.0 (Z=39.4) | ε = 5.91, 1e-6 | 91.76% | 68.79% | 18.38% | Overall Acc: 91.85%, π₂ = 5.62%, Gain: +45.79% |
| **2** | Phase 1 | Non-private | 60 | 256 | 0.01 | - | - | Non-private | 97.67% | 85.93% | - | Overall Acc: 97.08% |
| **2** | Phase 2 | Vanilla DP-SGD | 60 | 256 | 0.01 | 0.8 | 1.0 | ε = 5.90, 1e-6 | 89.05% | 28.64% | 57.29% | Overall Acc: 85.64%, π₂ = 8.62% |
| **2** | Phase 4 | DPSGD-Global-Adapt | 60 | 256 | 0.10 | 0.8 | 1.0 (Z=41.0) | ε = 5.91, 1e-6 | 90.79% | 66.94% | 18.99% | Overall Acc: 91.08%, π₂ = 6.88%, Gain: +38.30% |
| **3** | Phase 1 | Non-private | 60 | 256 | 0.01 | - | - | Non-private | 97.77% | 84.09% | - | Overall Acc: 96.86% |
| **3** | Phase 2 | Vanilla DP-SGD | 60 | 256 | 0.01 | 0.8 | 1.0 | ε = 5.90, 1e-6 | 88.95% | 25.36% | 58.73% | Overall Acc: 85.46%, π₂ = 8.82% |
| **3** | Phase 4 | DPSGD-Global-Adapt | 60 | 256 | 0.10 | 0.8 | 1.0 (Z=49.0) | ε = 5.91, 1e-6 | 91.57% | 65.40% | 18.69% | Overall Acc: 91.36%, π₂ = 6.20%, Gain: +40.04% |
| **4** | Phase 1 | Non-private | 60 | 256 | 0.01 | - | - | Non-private | 97.97% | 87.17% | - | Overall Acc: 97.29% |
| **4** | Phase 2 | Vanilla DP-SGD | 60 | 256 | 0.01 | 0.8 | 1.0 | ε = 5.90, 1e-6 | 88.86% | 23.10% | 64.07% | Overall Acc: 85.00%, π₂ = 9.11% |
| **4** | Phase 4 | DPSGD-Global-Adapt | 60 | 256 | 0.10 | 0.8 | 1.0 (Z=31.3) | ε = 5.91, 1e-6 | 90.70% | 67.04% | 20.12% | Overall Acc: 91.15%, π₂ = 7.27%, Gain: +43.94% |
| **Mean** | Phase 1 | Non-private | 60 | 256 | 0.01 | - | - | Non-private | **97.73% ± 0.22%** | **86.32% ± 1.37%** | - | Overall: 97.08% ± 0.17% |
| **Mean** | Phase 2 | Vanilla DP-SGD | 60 | 256 | 0.01 | 0.8 | 1.0 | **ε = 5.90, 1e-6** | **89.03% ± 0.13%** | **24.89% ± 2.31%** | **61.44% ± 3.21%** | Overall: 85.32% ± 0.24%, π₂ = 8.70% ± 0.32% |
| **Mean** | Phase 4 | DPSGD-Global-Adapt | 60 | 256 | 0.10 | 0.8 | 1.0 (Z_init=50) | **ε = 5.91, 1e-6** | **91.47% ± 0.76%** | **67.21% ± 1.25%** | **19.12% ± 0.68%** | Overall: 91.47% ± 0.39%, π₂ = 6.26% ± 0.82% |

---

### Phase 1 (Non-Private Baseline) Detailed Metrics

| Seed | Control (Digit 2) Acc | Rare (Digit 8) Acc | Overall Test Acc | Test Loss (Digit 2) | Test Loss (Digit 8) | Training Samples (Digit 8) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Seed 0** | 97.87% | 87.27% | 97.21% | 0.0733 | 0.4078 | 537 |
| **Seed 1** | 97.38% | 87.17% | 96.98% | 0.0809 | 0.4248 | 530 |
| **Seed 2** | 97.67% | 85.93% | 97.08% | 0.0746 | 0.4239 | 527 |
| **Seed 3** | 97.77% | 84.09% | 96.86% | 0.0733 | 0.4863 | 517 |
| **Seed 4** | 97.97% | 87.17% | 97.29% | 0.0559 | 0.4285 | 513 |
| **Mean ± Std** | **97.73% ± 0.22%** | **86.32% ± 1.37%** | **97.08% ± 0.17%** | **0.0716** | **0.4343** | **~525** |

---

### Phase 2 (Vanilla DP-SGD) Detailed Metrics

Settings: C = 1.0, σ = 0.8, δ = 10⁻⁶, achieved ε = 5.90 across all seeds.

| Seed | Control (Digit 2) Acc | Rare (Digit 8) Acc | Overall Test Acc | Privacy Cost π₂ | Privacy Cost π₈ | Disparity Gap (π₈ − π₂) | Test Loss (Digit 2) | Test Loss (Digit 8) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Seed 0** | 89.15% | 24.33% | 85.30% | 8.72% | 62.94% | 54.21% | 0.6346 | 2.6295 |
| **Seed 1** | 89.15% | 23.00% | 85.22% | 8.24% | 64.17% | 55.93% | 0.6566 | 2.6699 |
| **Seed 2** | 89.05% | 28.64% | 85.64% | 8.62% | 57.29% | 48.67% | 0.6620 | 2.5553 |
| **Seed 3** | 88.95% | 25.36% | 85.46% | 8.82% | 58.73% | 49.91% | 0.6566 | 2.7099 |
| **Seed 4** | 88.86% | 23.10% | 85.00% | 9.11% | 64.07% | 54.96% | 0.6544 | 2.6927 |
| **Mean ± Std** | **89.03% ± 0.13%** | **24.89% ± 2.31%** | **85.32% ± 0.24%** | **8.70% ± 0.32%** | **61.44% ± 3.21%** | **52.74% ± 3.24%** | **0.6528** | **2.6515** |

---

## Phase 3: Disparity Evaluation & All-Digit Comparison

In accordance with [plan.md](plan.md), Phase 3 evaluates the disparity between the non-private baseline (Phase 1) and vanilla DP-SGD (Phase 2):
- **Privacy Cost (π) = Acc(Phase 1) − Acc(Phase 2)**
- **Disparity Gap = π₈ − π₂**

### All-Digit Comparison Table (Digits 0–9 across Seeds 0–4)

| Digit | Role / Representation | Phase 1 Non-Private Acc | Phase 2 DP-SGD Acc | Privacy Cost (π) |
| :---: | :--- | :---: | :---: | :---: |
| **0** | Majority | 99.27% ± 0.18% | 97.88% ± 0.09% | 1.39% ± 0.21% |
| **1** | Majority | 99.33% ± 0.10% | 97.81% ± 0.04% | 1.52% ± 0.11% |
| **2** | **Control Class** | **97.73% ± 0.22%** | **89.03% ± 0.13%** | **8.70% ± 0.32%** |
| **3** | Majority | 98.77% ± 0.19% | 90.81% ± 0.19% | 7.96% ± 0.35% |
| **4** | Majority | 97.94% ± 0.35% | 90.45% ± 0.63% | 7.49% ± 0.88% |
| **5** | Majority | 98.27% ± 0.30% | 85.25% ± 0.35% | 13.03% ± 0.52% |
| **6** | Majority | 98.31% ± 0.43% | 94.41% ± 0.16% | 3.90% ± 0.55% |
| **7** | Majority | 97.32% ± 0.44% | 90.31% ± 0.22% | 7.00% ± 0.60% |
| **8** | **Rare Subgroup (9%)** | **86.32% ± 1.37%** | **24.89% ± 2.31%** | **61.44% ± 3.21%** |
| **9** | Majority | 97.19% ± 0.29% | 89.51% ± 0.19% | 7.67% ± 0.18% |
| **Overall** | All 10 Classes | **97.08% ± 0.17%** | **85.32% ± 0.24%** | **11.76% ± 0.21%** |

### Key Findings (Phase 3 Synthesis)
1. **Target Disparity Confirmation:**
   - Control Digit 2 Privacy Cost (π₂): **8.70% ± 0.32%** (Paper: 8.9% ± 0.1%)
   - Rare Digit 8 Privacy Cost (π₈): **61.44% ± 3.21%** (Paper: 57.9% ± 1.3%)
   - Disparity Gap (π₈ − π₂): **52.74% ± 3.24%** (Paper: 48.9% ± 1.3%)
2. **Subgroup Vulnerability:**
   - Digit 8's privacy cost is **over 7× larger** than control Digit 2 and **over 40× larger** than digits 0 and 1.
   - While majority classes retain 85%–98% test accuracy, rare digit 8 drops from 86.32% to near-random chance at 24.89%.

---

### Phase 4 (DPSGD-Global-Adapt) Detailed Metrics

Settings: C = 1.0, σ = 0.8, Z_init = 50.0, τ = 0.7, σ₂ = 10.0, η_Z = 0.1, lr = 0.10, δ = 10⁻⁶, achieved ε = 5.91 across all seeds.

| Seed | Control (Digit 2) Acc | Rare (Digit 8) Acc | Overall Test Acc | Privacy Cost π₂ | Privacy Cost π₈ | Disparity Gap (π₈ − π₂) | Rare Digit 8 Gain vs Phase 2 | Final Z | Test Loss (Digit 2) | Test Loss (Digit 8) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Seed 0** | 92.54% | 67.86% | 91.90% | 5.33% | 19.40% | 14.08% | +43.53% | 31.78 | 0.3120 | 1.1342 |
| **Seed 1** | 91.76% | 68.79% | 91.85% | 5.62% | 18.38% | 12.76% | +45.79% | 39.41 | 0.3209 | 1.1025 |
| **Seed 2** | 90.79% | 66.94% | 91.08% | 6.88% | 18.99% | 12.11% | +38.30% | 40.99 | 0.3727 | 1.2118 |
| **Seed 3** | 91.57% | 65.40% | 91.36% | 6.20% | 18.69% | 12.48% | +40.04% | 48.99 | 0.3474 | 1.1782 |
| **Seed 4** | 90.70% | 67.04% | 91.15% | 7.27% | 20.12% | 12.86% | +43.94% | 31.26 | 0.3393 | 1.1979 |
| **Mean ± Std** | **91.47% ± 0.76%** | **67.21% ± 1.25%** | **91.47% ± 0.39%** | **6.26% ± 0.82%** | **19.12% ± 0.68%** | **12.86% ± 0.74%** | **+42.32% ± 3.06%** | **38.49** | **0.3385** | **1.1649** |

---

## Final Benchmark Comparison: Paper Target vs Replicated Results

Comprehensive 3-way evaluation across all phases compared to published Table 2 benchmarks in arXiv:2206.07737:

| Method / Phase | Achieved Privacy (ε, δ) | Control (Digit 2) Acc | Rare (Digit 8) Acc | Overall Model Acc | Digit 2 Privacy Cost (π₂) | Digit 8 Privacy Cost (π₈) | Disparity Gap (π₈ − π₂) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Phase 1: Non-Private (Ours)** | ∞ | **97.73% ± 0.22%** | **86.32% ± 1.37%** | **97.08% ± 0.17%** | Baseline | Baseline | Baseline |
| *Paper Non-Private Target* | ∞ | *98.0% ± 0.1%* | *84.3% ± 1.1%* | *~98.0%* | *Baseline* | *Baseline* | *Baseline* |
| **Phase 2: Vanilla DP-SGD (Ours)** | **ε = 5.90, 10⁻⁶** | **89.03% ± 0.13%** | **24.89% ± 2.31%** | **85.32% ± 0.24%** | **8.70% ± 0.32%** | **61.44% ± 3.21%** | **52.74% ± 3.24%** |
| *Paper Vanilla DP-SGD Target* | *ε ≈ 5.90, 10⁻⁶* | *89.0% ± 0.1%* | *26.3% ± 0.4%* | *~85.5%* | *8.9% ± 0.1%* | *57.9% ± 1.3%* | *48.9% ± 1.3%* |
| **Phase 4: Global-Adapt (Ours)** | **ε = 5.91, 10⁻⁶** | **91.47% ± 0.76%** | **67.21% ± 1.25%** | **91.47% ± 0.39%** | **6.26% ± 0.82%** | **19.12% ± 0.68%** | **12.86% ± 0.74%** |
| *Paper Global-Adapt Target* | *ε ≈ 5.90, 10⁻⁶* | *92.0% ± 0.2%* | *65.5% ± 1.2%* | *~91.0%* | *6.0% ± 0.2%* | *18.8% ± 0.9%* | *12.8% ± 0.8%* |

---

## Published Paper Reference Metrics (arXiv:2206.07737, Table 2)

- **Non-private Baseline:** Digit 2 Acc: 98.0% ± 0.1% | Digit 8 Acc: 84.3% ± 1.1%
- **Vanilla DP-SGD (ε ≈ 5.90, δ = 1e-6):** Digit 2 Acc: 89.0% ± 0.1% | Digit 8 Acc: 26.3% ± 0.4% | Privacy Cost (π₈): 57.9% ± 1.3%
- **DPSGD-Global-Adapt:** Digit 2 Acc: 92.0% ± 0.2% | Digit 8 Acc: 65.5% ± 1.2% | Privacy Cost (π₈): 18.8% ± 0.9%

---

## Pilot Experiment Reference (8 Epochs, Seed 42)

Full details and scripts are preserved in [`pilot/`](pilot/README.md).
- **Non-private:** Digit 2 Acc: 99.2% | Digit 8 Acc: 91.0%
- **DP-SGD (ε ≈ 8.0):** Digit 2 Acc: 94.5% | Digit 8 Acc: 70.0% (Privacy cost: 20.9%)
- **DP-SGD (ε ≈ 5.9):** Digit 2 Acc: 93.1% | Digit 8 Acc: 61.2% (Privacy cost: 29.8%)
- **DP-SGD (ε ≈ 3.0):** Digit 2 Acc: 88.7% | Digit 8 Acc: 48.6% (Privacy cost: 42.4%)

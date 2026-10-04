# Adult Dataset DP-SGD Disparity Study: Plan

This experiment investigates whether Differential Privacy (DP-SGD) creates disparate impact across demographic groups (Male vs. Female) on the UCI Adult Census dataset, and whether `DPSGD-Global-Adapt` resolves it.

---

## 1. Experiment Overview

- **Dataset**: UCI Adult Census (45,222 complete cases; 80% train / 20% test).
- **Subgroups**: Male (Class 0) vs. Female (Class 1), balanced in training (~11,700 per group).
- **Prediction Target**: Binary classification (Income > $50K).
- **Model Architecture**: 2-layer MLP (`Linear(98, 256) → Tanh → Linear(256, 256) → Tanh → Linear(256, 2)`; 91,650 parameters).
- **Evaluation Seeds**: Seeds 0–4 (paired data splits and frozen initial weights).

---

## 2. Training Hyperparameters

All configurations match the published paper benchmarks (*Esipova et al., ICLR 2023*):

| Hyperparameter | Phase 1: Baseline | Phase 2: Vanilla DP-SGD | Phase 4: Global-Adapt |
| :--- | :---: | :---: | :---: |
| **Epochs** | 20 | 20 | 20 |
| **Batch Size** | 256 | 256 | 256 |
| **Optimizer** | SGD | SGD | SGD |
| **Learning Rate** | 0.01 | 0.01 | 0.20 |
| **Clipping Norm (C)** | None | 0.5 | 0.5 |
| **Gradient Noise (σ)** | 0.0 | 1.0 | 1.0 |
| **Adaptation Threshold (τ)** | N/A | N/A | 1.0 |
| **Count Noise (σ₂)** | N/A | N/A | 10.0 |
| **Initial Bound (Z)** | N/A | N/A | 50.0 |
| **Bound Step Size (η_Z)** | N/A | N/A | 0.1 |
| **Target Privacy (ε, δ)** | None | ε ≈ 3.41, δ = 10⁻⁶ | ε ≈ 3.42, δ = 10⁻⁶ |

---

## 3. The Four Phases

1. **Phase 1 (Non-private Baseline)**:
   - Establish benchmark accuracy and false-negative rates for Male and Female cohorts before privacy is applied.
2. **Phase 2 (Vanilla DP-SGD)**:
   - Apply standard DP-SGD with static per-sample gradient clipping and noise.
   - Measure the performance drop on both groups and determine the disparity gap.
3. **Phase 3 (Disparity Evaluation)**:
   - Calculate **Accuracy Loss = Baseline Acc − DP-SGD Acc** for each group.
   - Calculate **Disparity Gap = Male Loss − Female Loss** in percentage points.
   - Calculate **Missed High-Income Rate = 100 × FN / (TP + FN)** (False Negative Rate on > $50K earners).
4. **Phase 4 (DPSGD-Global-Adapt)**:
   - Train with dynamic per-layer gradient scaling to prevent gradient misalignment between groups.
   - Verify if accuracy is restored and the disparity gap is closed at matching privacy levels.

---

## 4. Execution Commands

From the repository root:

```powershell
# Pre-flight validation:
.\.venv\Scripts\python.exe experiments/adult/adult.py --self-test

# Run full 5-seed benchmark across all phases:
.\.venv\Scripts\python.exe experiments/adult/adult.py --phase all --all-seeds
```

Results are saved to [`results/adult/`](../../results/adult/) and summarized in [`adult_experiment_log.md`](adult_experiment_log.md).

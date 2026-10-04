# Differential Privacy Subgroup Disparity & Mitigation Benchmarks

This repository replicates and evaluates the findings of *"Disparate Impact in Differential Privacy from Gradient Misalignment"* ([arXiv:2206.07737](https://arxiv.org/abs/2206.07737), ICLR 2023).

We study how Differential Privacy (DP-SGD) creates disparate impact against rare subgroups or demographic cohorts, and benchmark algorithmic mitigation using `DPSGD-Global-Adapt`.

---

## 1. Experimental Benchmarks

| Benchmark | Domain | Subgroup Evaluated | Status | Plan | Results Log | Main Runner |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **MNIST** | Computer Vision (Images) | Rare Digit 8 (9% subsampled) vs. Control Digit 2 | **Completed** (Seeds 0–4) | [plan.md](experiments/mnist/plan.md) | [experiment_log.md](experiments/mnist/experiment_log.md) | `experiments/mnist/mnist.py` |
| **UCI Adult** | Tabular (Census Data) | Male vs. Female income disparity + FNR | **Completed** (Seeds 0–4) | [adult_plan.md](experiments/adult/adult_plan.md) | [adult_experiment_log.md](experiments/adult/adult_experiment_log.md) | `experiments/adult/adult.py` |

---

## 2. Key Findings Summary

Across both Vision and Tabular domains, standard DP-SGD causes severe disparity against underrepresented or minority groups, which `DPSGD-Global-Adapt` resolves:

- **MNIST (Rare Digit 8)**:
  - Non-private Baseline: 86.3% test accuracy.
  - Vanilla DP-SGD (ε = 5.90): Drops to **24.9%** (near random guess), creating a **52.7 pp disparity gap** relative to majority digits.
  - Global-Adapt (ε = 5.91): Recovers to **67.2%** (+42.3 pp gain), shrinking the disparity gap to 12.9 pp.
- **UCI Adult (Income Prediction)**:
  - Non-private Baseline: Male 80.6% / Female 92.2% accuracy.
  - Vanilla DP-SGD (ε = 3.41): Drops male accuracy by 10.7 pp vs. only 3.6 pp for females (**7.07 pp disparity gap**), and misses > 95% of high earners.
  - Global-Adapt (ε = 3.42): Restores accuracy to **80.6% male / 92.3% female**, shrinking the disparity gap to **0.10 pp** (effectively zero).

---

## 3. Repository Architecture

```text
research/
├── README.md                      # Project overview and quickstart
├── requirements.txt               # Pinned dependencies
│
├── experiments/                   # Dataset-isolated experiment packages
│   ├── mnist/                     # MNIST benchmark
│   │   ├── mnist.py               # Unified MNIST runner
│   │   ├── plan.md                # Experiment protocol
│   │   └── experiment_log.md      # Results summary across seeds 0–4
│   │
│   └── adult/                     # UCI Adult benchmark
│       ├── adult.py               # Unified Adult runner
│       ├── adult_plan.md          # Experiment protocol
│       └── adult_experiment_log.md# Results summary across seeds 0–4
│
├── data/                          # Cached data (data/MNIST/ and data/adult/)
├── results/                       # Generated checkpoints, summaries, and plots
└── src/                           # Shared modules (models, dataset loaders, Global-Adapt)
```

---

## 4. Quickstart & Reproduction

Run commands from the repository root using the shared environment:

```powershell
# 1. MNIST Benchmark (Phases 1, 2, 4 across all 5 seeds):
.\.venv\Scripts\python.exe experiments/mnist/mnist.py --phase all --all-seeds

# 2. Adult Benchmark (Phases 1 to 4 across all 5 seeds):
.\.venv\Scripts\python.exe experiments/adult/adult.py --phase all --all-seeds
```

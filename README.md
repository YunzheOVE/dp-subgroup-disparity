# Differential Privacy Subgroup Disparity & Mitigation Benchmarks

This repository replicates and analyzes the findings of *"Disparate Impact in Differential Privacy from Gradient Misalignment"* ([arXiv:2206.07737](https://arxiv.org/abs/2206.07737), ICLR 2023). We study how Differential Privacy (DP-SGD) disproportionately affects rare subgroups or specific demographic cohorts and evaluate algorithmic mitigations (specifically `DPSGD-Global-Adapt`).

---

## Benchmarks & Datasets

The repository groups dataset-specific scripts, protocols, and experimental logs into dedicated directories under `experiments/`:

| Benchmark | Data Domain | Subgroup Studied | Status | Plan & Protocol | Results Log | Runner Script |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **MNIST** | Computer Vision (Images) | Rare Digit 8 (9% subsampled) vs Control Digit 2 | **Completed** (Seeds 0–4) | [plan.md](experiments/mnist/plan.md) | [experiment_log.md](experiments/mnist/experiment_log.md) | `experiments/mnist/mnist.py` |
| **UCI Adult** | Tabular Census Data | Male vs. Female income disparity + FNR | **Ready to Run** | [adult_plan.md](experiments/adult/adult_plan.md) | [adult_experiment_log.md](experiments/adult/adult_experiment_log.md) | `experiments/adult/adult.py` |
| *Future (e.g. Dutch, CelebA)* | Tabular / Face Attributes | Demographic / Attribute subgroups | *Planned* | `experiments/<dataset>/plan.md` | `experiments/<dataset>/log.md` | `experiments/<dataset>/run.py` |

---

## Standardized Repository Architecture

```text
research/
├── README.md                      # Unified research hub and overview
├── requirements.txt               # Pinned environment dependencies (PyTorch, Opacus, pandas, matplotlib)
│
├── experiments/                   # Dataset-isolated experiment packages
│   ├── mnist/                     # MNIST benchmark
│   │   ├── mnist.py               # Unified MNIST runner (--phase, --seed, --all-seeds)
│   │   ├── phase1.py              # Phase 1: Non-private baseline
│   │   ├── phase2.py              # Phase 2: Vanilla DP-SGD
│   │   ├── phase4.py              # Phase 4: DPSGD-Global-Adapt
│   │   ├── plan.md                # MNIST experiment protocol
│   │   └── experiment_log.md      # MNIST benchmark results across seeds 0–4
│   │
│   └── adult/                     # UCI Adult benchmark
│       ├── adult.py               # Unified Adult runner (--self-test, --prepare-only, --phase, --all-seeds)
│       ├── adult_plan.md          # Adult experiment protocol
│       └── adult_experiment_log.md# Adult benchmark results log & targets
│
├── data/                          # Dataset storage (isolated per benchmark)
│   ├── MNIST/                     # Raw MNIST image data
│   └── adult/                     # Verified adult.data & adult.test from UCI archive
│
├── results/                       # Experimental outputs (isolated per benchmark)
│   ├── mnist/                     # All MNIST JSON metrics and checkpoints (Seeds 0–4)
│   └── adult/                     # Adult manifests, model checkpoints, summary tables, and plots
│
└── src/                           # Shared modular framework
    ├── __init__.py
    ├── models.py                  # Paper architectures: PaperCNN (MNIST) & PaperAdultMLP (Adult)
    ├── global_adapt.py            # Reusable DPSGD-Global-Adapt optimizer & PrivacyEngine
    ├── adult_dataset.py           # UCI Adult parser, standardizer, and group balancer
    ├── dataset.py                 # MNIST subsampling pipeline
    └── metrics.py                 # Standard confusion matrix, accuracy, and FNR evaluators
```

---

## Quickstart & Execution

All experiments run in the shared virtual environment (`.venv`) from the repository root:

### 1. MNIST Benchmark
```powershell
# Run a single seed across all phases:
.\.venv\Scripts\python.exe experiments/mnist/mnist.py --phase all --seed 0

# Run all 5 seeds (0 to 4):
.\.venv\Scripts\python.exe experiments/mnist/mnist.py --phase all --all-seeds
```

### 2. UCI Adult Benchmark
```powershell
# 1. Run pipeline integrity verification checks (Section 9)
.\.venv\Scripts\python.exe experiments/adult/adult.py --self-test

# 2. Run Seed 0 validation fit
.\.venv\Scripts\python.exe experiments/adult/adult.py --phase all --seed 0

# 3. Run all 5 seeds sequentially (skips already completed runs)
.\.venv\Scripts\python.exe experiments/adult/adult.py --phase all --all-seeds --skip-existing

# 4. Generate aggregate summary table and publication figures
.\.venv\Scripts\python.exe experiments/adult/adult.py --phase 3 --all-seeds
```

---

## Key Experimental Findings (MNIST Summary)

- **Vanilla DP-SGD Disparity:** Applying standard DP-SGD ($\epsilon = 5.90$) causes accuracy on rare digit 8 to collapse from **86.32% down to 24.89%** (a **61.44% privacy cost**), while control digit 2 only loses 8.70% (disparity gap of **52.74%**).
- **Global-Adapt Mitigation:** `DPSGD-Global-Adapt` restores rare digit 8 accuracy to **67.21%** (a **+42.32% recovery**), shrinking the disparity gap from **52.74% to 12.86%** under the exact same privacy budget ($\epsilon = 5.91$).

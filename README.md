# MNIST DP-SGD Subgroup Disparity Study

This repository replicates and analyzes the findings of *"Disparate Impact in Differential Privacy from Gradient Misalignment"* ([arXiv:2206.07737](https://arxiv.org/abs/2206.07737)). We study how Differential Privacy (DP-SGD) disproportionately affects rare subgroups (digit 8 subsampled to 9% retention) compared to well-represented classes (digit 2 control).

TLDR: DPSGD-Global-Adapt is not a method for stronger privacy, but a vastly better method for equitable privacy. It proves that deep learning models do not have to discard underrepresented subgroups in order to guarantee rigorous differential privacy.

---

## Repository Structure

```text
research/
├── plan.md                # Active experimental plan and protocol (paper-aligned)
├── experiment_log.md      # Summary log of benchmark runs across seeds 0–4
├── README.md              # Project documentation and guide
├── requirements.txt       # Environment dependencies
├── data/                  # Local dataset storage (MNIST raw files, gitignored)
├── phase1.py              # Phase 1: Non-private baseline runner (seeds 0–4)
├── phase2.py              # Phase 2: Vanilla DP-SGD runner (seeds 0–4)
├── phase4.py              # Phase 4: DPSGD-Global-Adapt runner (seeds 0–4)
├── src/                   # Experiment implementation package (seeds 0–4)
│   ├── __init__.py
│   ├── dataset.py         # Subsampling pipeline (exact 9% retention for digit 8)
│   ├── models.py          # Paper's exact 97.1k parameter CNN (Tanh, no pooling)
│   └── global_adapt.py    # DPSGD-Global-Adapt optimizer & privacy engine
├── results/               # Output JSON evaluation metrics for seeds 0–4
└── pilot/                 # Initial 8-epoch exploratory pilot study (seed 42)
    ├── README.md          # Pilot study summary, setup, and findings
    ├── phase1_mnist.py    # Fast pilot baseline script
    ├── phase2_dpsgd.py    # Fast pilot DP-SGD script
    └── *.json             # Pilot output metrics
```

---

## Experimental Protocol

See [plan.md](plan.md) for full specifications:
- **Dataset:** MNIST with digit 8 kept at 9% retention probability (~500 images) during training; full test set.
- **Model:** 2-layer CNN with Tanh activations and no pooling (97,114 parameters, matching the authors' MNIST script and model code). Appendix B.3 reports 80,522 parameters for MNIST; that figure conflicts with the released CNN configuration.
- **Training:** Cross-entropy loss, standard SGD (learning rate 0.01, momentum 0, batch size 256, 60 epochs).
- **Phases:**
  1. *Phase 1 (Non-private baseline):* Standard SGD.
  2. *Phase 2 (Vanilla DP-SGD):* Per-example clipping C=1.0, fixed noise multiplier σ=0.8, RDP accountant, δ=1e-6 (achieving ε ≈ 5.9).
  3. *Phase 3 (Disparity Evaluation):* Privacy cost = Phase 1 Acc - Phase 2 Acc for digits 8 and 2 across seeds 0–4.
  4. *Phase 4 (DPSGD-Global-Adapt):* Adaptive clipping mitigation (lr=0.1, strict bound Z=50, threshold τ=0.7).

---

## How to Run Experiments

### Phase 1: Non-Private Baseline
Run a single seed (e.g., Seed 0):
```bash
python phase1.py --seed 0
```
Run all 5 paired seeds (0 through 4) sequentially with aggregated summary:
```bash
python phase1.py --all-seeds
```

### Phase 2: Vanilla DP-SGD
Run a single seed paired with Phase 1:
```bash
python phase2.py --seed 0
```
Run all 5 paired seeds (0 through 4) sequentially with privacy cost comparison:
```bash
python phase2.py --all-seeds
```

### Phase 4: DPSGD-Global-Adapt
Run a single seed paired with Phase 1 and 2:
```bash
python phase4.py --seed 0
```
Run all 5 paired seeds (0 through 4) sequentially with disparity reduction analysis:
```bash
python phase4.py --all-seeds
```
Outputs and initial model weights will be saved automatically to `results/`.

---

## Published Target Benchmarks (Table 2, arXiv:2206.07737)

- **Non-private Baseline:**
  - Control Digit 2 Accuracy: 98.0% ± 0.1%
  - Rare Digit 8 Accuracy: 84.3% ± 1.1%
- **Vanilla DP-SGD (ε ≈ 5.90, δ = 1e-6):**
  - Control Digit 2 Accuracy: 89.0% ± 0.1%
  - Rare Digit 8 Accuracy: 26.3% ± 0.4%
  - Disparity Gap (π₂,₈): 48.9% ± 1.3%
- **DPSGD-Global-Adapt:**
  - Rare Digit 8 Accuracy: 65.5% ± 1.2%

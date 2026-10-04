# Adult DP-SGD Disparity Study: Experiment Log

This log tracks experimental runs across seeds 0–4 for the **UCI Adult tabular dataset** following the protocol in [adult_plan.md](adult_plan.md), investigating differential privacy disparate impact on demographic subgroups (Male vs. Female) and evaluating `DPSGD-Global-Adapt`.

---

## Published Target Benchmarks (Table 4, arXiv:2206.07737)

The paper reports the following reference values (Mean ± Standard Error across 5 seeds):

| Method / Phase | Achieved Privacy (ε, δ) | Male Accuracy (%) | Female Accuracy (%) | Male Accuracy Loss (pp) | Female Accuracy Loss (pp) | Published Loss Gap (pp) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Phase 1: Non-private** | ∞ | 80.5% ± 0.4% | 92.2% ± 0.1% | Baseline | Baseline | Baseline |
| **Phase 2: DP-SGD** | ε ≈ 3.41, 10⁻⁶ | 69.9% ± 0.4% | 88.5% ± 0.1% | 10.6 ± 0.3 | 3.6 ± 0.1 | 6.9 ± 0.3 |
| **Phase 4: Global-Adapt** | ε ≈ 3.41, 10⁻⁶ | 80.7% ± 0.4% | 92.3% ± 0.1% | -0.1 ± 0.1 | -0.1 ± 0.1 | 0.0 ± 0.1 |

*Note on Published Loss Gap:* In the paper, Male participants lose substantially more accuracy under DP-SGD (10.6 pp loss vs. 3.6 pp for Female). Global-Adapt eliminates this performance drop.

---

## Dataset Protocol & Verification

- **Source:** [UCI Adult Dataset Archive](https://archive.ics.uci.edu/dataset/2/adult) (`adult.data` and `adult.test`)
- **Raw Records:** 48,842 (32,561 in `adult.data`, 16,281 in `adult.test`)
- **Complete Cases:** 45,222 records after dropping missing values (`?`)
  - Male records: 30,527
  - Female records: 14,695
- **Subgroup Balancing:** Female retention $p=1.0$, Male retention $p = 14695 / 30527 \approx 0.481$. Average retained records: ~29,355 (~14,695 per group).
- **Features:** 98 one-hot encoded and standardized features (continuous columns standardized with sample standard deviation `ddof=1` across the complete-case pool).
- **Split:** Deterministic 80% train (23,484 rows) and 20% test (5,871 rows).

---

## Verified Execution Commands

All commands are executed from the workspace root:

```powershell
# 1. Run pipeline integrity checks
.\.venv\Scripts\python.exe adult.py --self-test

# 2. Prepare pairing manifest and frozen initial weights for Seed 0
.\.venv\Scripts\python.exe adult.py --prepare-only --seed 0

# 3. Run all phases for Seed 0 (Validation run)
.\.venv\Scripts\python.exe adult.py --phase all --seed 0

# 4. Run all phases across all 5 seeds (0 to 4)
.\.venv\Scripts\python.exe adult.py --phase all --all-seeds --skip-existing

# 5. Regenerate aggregate summary tables and publication plots
.\.venv\Scripts\python.exe adult.py --phase 3 --all-seeds
```

---

## Experimental Results

*(Results will be populated upon completing training runs across seeds 0–4)*

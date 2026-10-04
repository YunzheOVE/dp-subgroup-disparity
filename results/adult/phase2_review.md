# Adult Phase 2: review checkpoint

Completed DP-SGD seeds 0–4 with the existing `adult.py`, 20 epochs each. Values are mean ± standard error across seeds.

| Group | Accuracy (%) | Missed per 100 actual higher-income records |
|---|---:|---:|
| Male | 69.86 ± 0.33 | 94.95 ± 0.70 |
| Female | 88.54 ± 0.06 | 98.43 ± 0.43 |

Overall accuracy: **79.13%**. Every run reports **ε = 3.4078 at δ = 10⁻⁶**; this is a per-run training budget, not a combined guarantee for all released models or raw-data preprocessing.

The model misses almost all actual higher-income records despite retaining high overall accuracy for the female group. This is why the missed-prediction measure matters alongside accuracy. Formal baseline comparisons are reserved for Phase 3.

Verified: unchanged data and initial weights from Phase 1; test IDs/labels/groups align across both phases; finite checkpoints; predictions reproduce all group metrics; reported privacy budgets reproduce from recorded steps and the installed accountant's sampling bound.

Accounting note: for seeds 1 and 2, the library converts 93 nominal batches to 92 Poisson batches due to floating-point rounding. Actual sampling uses q=1/93; Opacus accounts conservatively at q=1/92. All five runs perform 1,840 steps. Both rates are recorded in `phase2_summary.json`; existing training behavior was retained.

Published configuration: clipping C=0.5, gradient noise multiplier=1, SGD learning rate=0.01, RDP accounting. Training RNG is `seed + 2000`; Phase 1 used `seed + 1000`. Data, splits, and initial weights are paired; batch membership/random draws are not identical. Historical benchmark and full-pool preprocessing limitations remain as documented in the plan.

![Phase 2 results](C:/Users/Yun/OneDrive/Documents/bdap/research/results/adult/phase2_dpsgd.png)

Stop after Phase 2 for user review. Phase 3 comparisons and Phase 4 Adult training have not been run.

Reproduce from the repository root:

```powershell
.\.venv\Scripts\python.exe experiments/adult/adult.py --phase 2 --all-seeds --skip-existing
.\.venv\Scripts\python.exe experiments/adult/review_phase2.py
```

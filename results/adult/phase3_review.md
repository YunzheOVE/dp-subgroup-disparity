# Adult Phase 3: review checkpoint

Compared five paired seeds using the existing `adult.py --phase 3`; no model training or settings changes. Values are seed averages. Changes show mean ± standard error of the paired changes.

| Group | Accuracy without privacy → DP-SGD | Accuracy loss (pp) | Missed per 100 without privacy → DP-SGD | Additional missed per 100 |
|---|---:|---:|---:|---:|
| Male | 80.57% → 69.86% | 10.71 ± 0.33 | 38.57 → 94.95 | 56.38 ± 1.14 |
| Female | 92.18% → 88.54% | 3.65 ± 0.08 | 52.23 → 98.43 | 46.20 ± 0.84 |

Male accuracy loss exceeds female accuracy loss by **7.07 ± 0.28 percentage points** (male minus female). Every seed has this same direction. The mean per-seed absolute gap and the absolute gap between mean losses therefore both equal 7.07 pp here; they are stored separately because they need not agree in general.

Accuracy loss = non-private accuracy − DP-SGD accuracy. Additional missed predictions = 100 × (DP-SGD FNR − baseline FNR), among actual higher-income records. Use percentage points for accuracy loss and per-100 records for missed predictions. SE is computed from each seed's paired change, not by subtracting error bars. Sample SD is also saved.

Practical takeaway: privacy reduces prediction quality unevenly across groups, and overall accuracy can hide failures to recognize higher-income records. Female records retain higher accuracy but have the higher final missed-prediction rate; male records experience the larger increase after adding privacy. The smaller final male/female FNR gap comes from both rates approaching failure, so it should not be presented as a fairness improvement.

The [paper's Table 4](https://arxiv.org/html/2206.07737v2) reports accuracy losses of 10.6 ± 0.3 pp for males, 3.6 ± 0.1 pp for females, and a 6.9 ± 0.3 pp loss gap. Our losses and gap closely reproduce that pattern. The paper does not report Adult FNR targets; this is our added evaluation. We have not isolated gradient misalignment as the cause in our experiments.

Verified: earlier source/checkpoint/prediction fingerprints; matching test IDs, labels, and groups; shared initialization hashes; recomputed confusion matrices and paired changes for all five seeds; summaries agree with prior audits. Both charts use the checked summary values. No new privacy mechanism is run during this analysis; training budgets remain per-run ε=3.4078 at δ=10⁻⁶ on the frozen benchmark representation.

Limits: historical US Adult data motivates a general privacy/fairness question, not a measured claim about NYC. Full-pool preprocessing follows the released protocol and limits independent deployment interpretation. Income-classification errors do not directly measure denied jobs, loans, or benefits.

![adult accuracy by group](C:/Users/Yun/OneDrive/Documents/bdap/research/results/adult/adult_accuracy_by_group.png)

![adult missed high income by group](C:/Users/Yun/OneDrive/Documents/bdap/research/results/adult/adult_missed_high_income_by_group.png)

Stop after Phase 3 for user review. Phase 4 Adult training is pending; its results will be added to the comparison after that phase. The runner's full 15-model report remains deferred until Phase 4.

Reproduce from the repository root:

```powershell
.\.venv\Scripts\python.exe experiments/adult/adult.py --phase 3 --all-seeds
.\.venv\Scripts\python.exe experiments/adult/review_phase3.py
```

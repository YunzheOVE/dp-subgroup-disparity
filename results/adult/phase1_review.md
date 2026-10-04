# Adult Phase 1: review checkpoint

Completed seeds 0–4 using the existing `adult.py`, 20 epochs each. Values below are mean ± standard error across seeds.

| Group | Accuracy (%) | Missed per 100 actual higher-income records |
|---|---:|---:|
| Male | 80.57 ± 0.39 | 38.57 ± 1.18 |
| Female | 92.18 ± 0.09 | 52.23 ± 0.85 |

Overall accuracy: **86.33%**.

Female records have higher overall accuracy but a higher missed-prediction rate among actual higher-income records. The income-label distribution differs between groups, so accuracy alone gives an incomplete picture. These are baseline errors; we cannot attribute them to differential privacy.

The paper reports baseline accuracy of 80.5 ± 0.4% for males and 92.2 ± 0.1% for females. Our baseline is close, with no parameter tuning to force a match.

Verified: cached data counts; frozen initialization checksums; disjoint split IDs; all five finite checkpoints; predictions reproduce every saved group confusion matrix, accuracy, and FNR. The same pairing files and initial weights are available for the later phases.

Implementation notes: the existing runner follows the published Adult settings but is a local implementation, not an unchanged execution of the authors' repository. It uses training seed `seed + 1000` and a large finite clipping bound (`1e9`) with zero noise. Full-pool preprocessing before the split follows the released benchmark and limits interpretation as independent deployment evaluation. Adult is historical US census data, not a sample of NYC or actual eligibility decisions.

![Phase 1 baseline](C:/Users/Yun/OneDrive/Documents/bdap/research/results/adult/phase1_baseline.png)

Phase 2, Phase 3 comparisons, and Phase 4 Adult training have not been run. Stop here for user review.

Reproduce from the repository root:

```powershell
.\.venv\Scripts\python.exe experiments/adult/adult.py --phase 1 --all-seeds --skip-existing
.\.venv\Scripts\python.exe experiments/adult/review_phase1.py
```

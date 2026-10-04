# Adult DP-SGD disparity study: experiment log

## Status — October 4, 2026

**Phases 1 and 2 complete:** five non-private baseline runs and five vanilla DP-SGD runs, seeds 0–4, 20 epochs each. The user requested one phase at a time and selected the existing [adult.py](adult.py) implementation. Training code was reused without changes. Phase 3 comparisons and Phase 4 Adult training remain pending.

Read the [Phase 1 review](../../results/adult/phase1_review.md), [baseline figure](../../results/adult/phase1_baseline.png), and [per-seed table](../../results/adult/phase1_summary.csv).

Read the [Phase 2 review](../../results/adult/phase2_review.md), [DP-SGD figure](../../results/adult/phase2_dpsgd.png), and [per-seed table](../../results/adult/phase2_summary.csv).

## Phase 1: without privacy

Use the published Adult MLP: 98 inputs, two 256-unit hidden layers with Tanh, and two output logits; 91,650 parameters. All five runs use mean cross-entropy, SGD with learning rate 0.01, no momentum or weight decay, no scheduler, and nominal batch size 256. The zero-noise Opacus wrapper succeeded in every run.

| Group | Accuracy (%) | Missed per 100 actual higher-income records | Actual higher-income records (%) |
|---|---:|---:|---:|
| Male | 80.57 ± 0.39 | 38.57 ± 1.18 | 31.55 |
| Female | 92.18 ± 0.09 | 52.23 ± 0.85 | 11.64 |

Accuracy and missed-prediction values are **mean ± standard error across five seeds**. The last column is the mean test-set prevalence across seeds. Overall accuracy is **86.33 ± 0.26%**. Full sample standard deviations are also saved in the summary JSON.

The female group has higher overall accuracy yet misses more records among people actually earning above $50,000. Higher-income records are much less frequent in that group. Accuracy alone gives an incomplete picture; both accuracy and the false-negative rate are useful. These are errors before differential privacy is applied, so Phase 1 does not establish a privacy effect or a causal explanation for the group differences.

## Phase 2: vanilla DP-SGD

Use the same data, splits, feature columns, model, and saved initial weights as Phase 1. Train with SGD learning rate 0.01, clipping norm C=0.5, gradient noise multiplier 1.0, Poisson sampling, RDP accounting, and delta=1e-6. No settings were tuned and the existing training implementation was retained.

| Group | Accuracy (%) | Missed per 100 actual higher-income records |
|---|---:|---:|
| Male | 69.86 ± 0.33 | 94.95 ± 0.70 |
| Female | 88.54 ± 0.06 | 98.43 ± 0.43 |

Values are mean ± standard error across five seeds. Overall accuracy is **79.13 ± 0.22%**. Every run reports **epsilon=3.4078045905 at delta=1e-6**, close to the paper's 3.41. This is a per-run training budget on the frozen representation; it does not compose all released models or establish end-to-end privacy for raw-data preprocessing and published diagnostics.

The model misses almost all actual higher-income records, despite high overall accuracy for the female group. This supports showing the missed-prediction measure alongside accuracy. Formal paired baseline differences and privacy-cost calculations are reserved for Phase 3; these Phase 2 values alone do not isolate a causal mechanism.

### Phase 2 accounting and result checks

All five runs perform 1,840 actual optimizer steps. The installed Opacus 1.6.0 loader rounds the Poisson epoch length for seeds 1 and 2 from 93 nominal batches to 92: floating-point inversion of `1/93` is slightly below 93 before integer conversion. Sampling still uses q=1/93, while the gradient accountant hook uses the conservative bound q=1/92. The other seeds use q=1/92 for both. Training behavior was preserved.

| Seed | Actual sampling q | Accountant q | Optimizer steps | Expected batch size | Epsilon |
|---|---:|---:|---:|---:|---:|
| 0 | 1/92 | 1/92 | 1840 | 255 | 3.4078045905 |
| 1 | 1/93 | 1/92 | 1840 | 256 | 3.4078045905 |
| 2 | 1/93 | 1/92 | 1840 | 256 | 3.4078045905 |
| 3 | 1/92 | 1/92 | 1840 | 255 | 3.4078045905 |
| 4 | 1/92 | 1/92 | 1840 | 255 | 3.4078045905 |

The independent audit verified unchanged data and initial weights; matching test IDs, labels, and groups across Phases 1 and 2; finite checkpoints and probabilities; and exact recomputation of group confusion matrices, accuracy, and FNR from saved predictions. A synthetic one-step loader check reconstructed the installed sampling and accountant-hook rates. Rebuilding RDP accounting from those rates and actual steps reproduced every reported budget. Repeated epsilon queries did not add accountant steps.

Both sampling rates, per-run epsilon, settings, source fingerprints, and checkpoint/prediction/initialization hashes are saved in [phase2_summary.json](../../results/adult/phase2_summary.json). Total recorded Phase 2 training time is 37.36 seconds, excluding preparation and final evaluation.

## Data and pairing verification

Source: [UCI Adult](https://archive.ics.uci.edu/dataset/2/adult), using cached `adult.data` and `adult.test`. The pooled raw files contain 48,842 records; removing incomplete rows leaves 45,222 records: 30,527 males and 14,695 females. Retain all female rows and sample male rows with probability 14695/30527, as in the released protocol. Income labels remain imbalanced.

| Seed | Train records | Test records | Features | Phase 1 optimizer steps |
|---|---:|---:|---:|---:|
| 0 | 23,484 | 5,871 | 98 | 1840 |
| 1 | 23,594 | 5,898 | 98 | 1860 |
| 2 | 23,594 | 5,898 | 98 | 1860 |
| 3 | 23,543 | 5,885 | 98 | 1840 |
| 4 | 23,529 | 5,882 | 98 | 1840 |

Each seed has its own balanced sample and 80/20 split. Phases 1 and 2 used the same saved pairing manifest and initial weights within each seed. Retain those files for Phase 4.

The existing self-test passed. A separate Phase 1 audit then verified all five baseline runs:

- Prepared data metadata match their frozen manifests; initialization file hashes match.
- Train/test row IDs are disjoint and saved prediction IDs, labels, and groups align with the actual test dataset.
- Checkpoints contain finite parameters and saved probabilities are finite and valid.
- Recomputing confusion matrices, accuracy, and FNR from predictions exactly reproduces every saved group metric.

The audit and hashes are saved in [phase1_summary.json](../../results/adult/phase1_summary.json). It records test/train IDs, source fingerprints, model/prediction hashes, software versions, and run sizes.

## Published reference values

Published means ± standard errors from [Table 4 of the paper](https://arxiv.org/html/2206.07737v2):

| Method | Male accuracy (%) | Female accuracy (%) | Male accuracy loss (pp) | Female accuracy loss (pp) | Published loss gap (pp) |
|---|---:|---:|---:|---:|---:|
| Non-private | 80.5 ± 0.4 | 92.2 ± 0.1 | Baseline | Baseline | Baseline |
| DP-SGD | 69.9 ± 0.4 | 88.5 ± 0.1 | 10.6 ± 0.3 | 3.6 ± 0.1 | 6.9 ± 0.3 |
| Global-Adapt | 80.7 ± 0.4 | 92.3 ± 0.1 | -0.1 ± 0.1 | -0.1 ± 0.1 | 0.0 ± 0.1 |

Our Phase 1 and Phase 2 accuracies are close to the published baseline and DP-SGD values. No settings were tuned to recover those values. The table contains published reference values; our results appear in the phase sections above. Global-Adapt has not yet been run on Adult. The paper supplies no Adult FNR target; that evaluation is our addition.

## Implementation and interpretation notes

The existing runner is a local implementation of the published configuration, rather than an unchanged execution of the authors' repository at commit `c61a163e766fde2e634b40c8afbd82e42644f5b7`.

- Initial weights use the experiment seed. Training RNG resets to `seed + 1000` for Phase 1 and `seed + 2000` for Phase 2; Phase 4 is configured for `seed + 4000`. Pairing covers the data and initial weights, not identical batch membership or noise draws.
- The baseline uses a finite Opacus clipping bound of `1e9`, rather than the authors' `sys.float_info.max`, with zero noise. This baseline makes no differential privacy claim.
- Numeric standardization and categorical encoding use the pooled data before the split, following the released protocol. This limits interpretation as an independent deployment evaluation.
- Sex is used for balancing and as a predictor.
- Adult is historical US census data, not a representative NYC sample. Income-classification errors are not measurements of actual denied jobs, loans, or benefits.
- Seeds have different retained samples/splits. Results are averaged at the seed level; overlapping test records across runs are not treated as independent people.

Environment: Python 3.13.1, PyTorch 2.6.0+cu124, Opacus 1.6.0, pandas 3.0.6, NumPy 2.5.2; NVIDIA GeForce RTX 4070 SUPER. Reused the shared `.venv`. Total recorded training time across the five baselines: 32.03 seconds, excluding data preparation and final evaluation.

## Executed commands

From the repository root:

```powershell
.\.venv\Scripts\python.exe experiments/adult/adult.py --self-test
.\.venv\Scripts\python.exe experiments/adult/adult.py --phase 1 --seed 0
.\.venv\Scripts\python.exe experiments/adult/adult.py --phase 1 --all-seeds --skip-existing
.\.venv\Scripts\python.exe experiments/adult/review_phase1.py
.\.venv\Scripts\python.exe experiments/adult/adult.py --phase 2 --seed 0
.\.venv\Scripts\python.exe experiments/adult/adult.py --phase 2 --all-seeds --skip-existing
.\.venv\Scripts\python.exe experiments/adult/review_phase2.py
```

For each phase, the all-seeds command reused the verified seed-0 result and trained seeds 1–4. Console logs are `results/adult/phase1_run_seed_0.log`, `phase1_run_all_seeds.log`, `phase2_run_seed_0.log`, and `phase2_run_all_seeds.log`. The runner deferred its full 15-model summary because Phase 4 runs are not present. The independent reviews produce separate Phase 1 and Phase 2 summaries and figures; Phase 3 comparisons have not been generated.

## Review checkpoint

Stop after Phase 2 for user review. Keep all five initialization files and pairing manifests. When the user requests Phase 3, compare the saved paired baseline and DP-SGD results using the definitions in [adult_plan.md](adult_plan.md). No Phase 4 Adult model has been trained.

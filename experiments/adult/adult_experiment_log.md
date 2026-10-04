# Adult DP-SGD disparity study: experiment log

## Status — October 3, 2026

**Phase 1 complete:** five non-private baseline runs, seeds 0–4, 20 epochs each. The user requested one phase at a time and selected the existing [adult.py](adult.py) implementation. Training code was reused without changes. Phases 2, 3 comparisons, and 4 Adult training remain pending user review.

Read the [Phase 1 review](../../results/adult/phase1_review.md), [baseline figure](../../results/adult/phase1_baseline.png), and [per-seed table](../../results/adult/phase1_summary.csv).

## Phase 1: without privacy

Use the published Adult MLP: 98 inputs, two 256-unit hidden layers with Tanh, and two output logits; 91,650 parameters. All five runs use mean cross-entropy, SGD with learning rate 0.01, no momentum or weight decay, no scheduler, and nominal batch size 256. The zero-noise Opacus wrapper succeeded in every run.

| Group | Accuracy (%) | Missed per 100 actual higher-income records | Actual higher-income records (%) |
|---|---:|---:|---:|
| Male | 80.57 ± 0.39 | 38.57 ± 1.18 | 31.55 |
| Female | 92.18 ± 0.09 | 52.23 ± 0.85 | 11.64 |

Accuracy and missed-prediction values are **mean ± standard error across five seeds**. The last column is the mean test-set prevalence across seeds. Overall accuracy is **86.33 ± 0.26%**. Full sample standard deviations are also saved in the summary JSON.

The female group has higher overall accuracy yet misses more records among people actually earning above $50,000. Higher-income records are much less frequent in that group. Accuracy alone gives an incomplete picture; both accuracy and the false-negative rate are useful. These are errors before differential privacy is applied, so Phase 1 does not establish a privacy effect or a causal explanation for the group differences.

## Data and pairing verification

Source: [UCI Adult](https://archive.ics.uci.edu/dataset/2/adult), using cached `adult.data` and `adult.test`. The pooled raw files contain 48,842 records; removing incomplete rows leaves 45,222 records: 30,527 males and 14,695 females. Retain all female rows and sample male rows with probability 14695/30527, as in the released protocol. Income labels remain imbalanced.

| Seed | Train records | Test records | Features | Optimizer steps |
|---|---:|---:|---:|---:|
| 0 | 23,484 | 5,871 | 98 | 1840 |
| 1 | 23,594 | 5,898 | 98 | 1860 |
| 2 | 23,594 | 5,898 | 98 | 1860 |
| 3 | 23,543 | 5,885 | 98 | 1840 |
| 4 | 23,529 | 5,882 | 98 | 1840 |

Each seed has its own balanced sample and 80/20 split. The same saved pairing manifest and initial weights must be reused by later methods within that seed.

The existing self-test passed. A separate audit then verified all five runs:

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

Our Phase 1 accuracies are close to the published baseline. No settings were tuned to recover those values. The private-method rows are reference values, not our results. The paper supplies no Adult FNR target; that evaluation is our addition.

## Implementation and interpretation notes

The existing runner is a local implementation of the published configuration, rather than an unchanged execution of the authors' repository at commit `c61a163e766fde2e634b40c8afbd82e42644f5b7`.

- Initial weights use the experiment seed; the training RNG resets to `seed + 1000`.
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
```

The all-seeds command reused the verified seed-0 result and trained seeds 1–4. Console logs are `results/adult/phase1_run_seed_0.log` and `results/adult/phase1_run_all_seeds.log`. The runner correctly deferred its full 15-model summary because private runs are not present. The independent review produces only the Phase 1 summary and figure.

## Review checkpoint

Stop after Phase 1 for user review. Keep all five initialization files and pairing manifests. When the user requests Phase 2, use the same data/model initialization and the fixed published DP-SGD settings in [adult_plan.md](adult_plan.md). No Phase 2 or Phase 4 Adult model has been trained.

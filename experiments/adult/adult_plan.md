# Adult DP-SGD disparity study

We study whether DP-SGD changes income-prediction accuracy unevenly across the Adult dataset's recorded male/female groups, and whether DPSGD-Global-Adapt reduces that difference. We also measure missed higher-income predictions to make the practical effect easy to explain.

Follow the same four phases as `plan.md`: **without privacy, vanilla DP-SGD, compare, and test DPSGD-Global-Adapt**. Run seeds 0-4: three training methods per seed, for **15 model fits**. Phase 3 analyzes the trained models. Complete seed 0 with the final settings first; it counts toward the five seeds.

Use the existing `experiments/adult/adult.py`, as requested after implementation was completed. It is a local implementation of the published configuration. Do not build a second training runner. Preserve all existing MNIST work and write Adult outputs separately. This is a reproduction with added subgroup evaluation for the midterm; Folktables, extra mitigation methods, tuning sweeps, and gradient/Hessian analysis are outside scope.

## Experiment settings

- **Data:** UCI Adult; pool the original training/test files and drop incomplete records, following the authors. Keep all complete-case female records and sample males to obtain approximately equal group sizes. Income-label imbalance remains.
- **Groups and target:** Male=0, Female=1. Predict whether income is above $50,000; this is class 1. Compare both groups on the same test split within each seed.
- **Preprocessing:** Follow the released ordering, including full-pool numeric standardization and one-hot encoding before the 80/20 split. Keep sex as a feature. Record the resulting counts, feature columns, and limitations.
- **Seeds:** 0-4. Verify identical selected data, split, feature columns, and initial weights across Phases 1, 2, and 4 within each seed.
- **Training:** Mean cross-entropy, SGD without momentum, weight decay, or a scheduler; 20 epochs; nominal training/test batch sizes of 256. Use final models.
- **Software:** Use the existing `.venv` and record its exact versions. The authors' pinned code remains the reference for configuration. Preserve the MNIST scripts, results, and environment.

Use the authors' MLP: `Linear(d,256)`, `Tanh`, `Linear(256,256)`, `Tanh`, `Linear(256,2)`, with biases and raw output logits. Use mean cross-entropy. No dropout or batch normalization.

| Setting | Phase 1: non-private | Phase 2: DP-SGD | Phase 4: Global-Adapt |
|---|---:|---:|---:|
| Epochs | 20 | 20 | 20 |
| Nominal train/test batch size | 256 | 256 | 256 |
| Optimizer | SGD | SGD | SGD |
| Learning rate | 0.01 | 0.01 | 0.2 |
| Momentum / weight decay | 0 / 0 | 0 / 0 | 0 / 0 |
| Learning-rate scheduler | None | None | None |
| Gradient clip C | None | 0.5 | 0.5 |
| Gradient noise multiplier sigma | 0 | 1.0 | 1.0 |
| Initial scaling bound Z | N/A | N/A | 50 |
| Adaptation threshold tau | N/A | N/A | 1.0 |
| Count noise multiplier sigma_2 | N/A | N/A | 10 |
| Bound update parameter eta_Z | N/A | N/A | 0.1 |
| Accountant | Not a privacy claim | RDP | RDP, including count mechanism |
| Delta | N/A | 1e-6 | 1e-6 |
| Seeds | 0-4 | 0-4 | 0-4 |

These values follow the [Adult script](https://github.com/layer6ai-labs/fair-dp/blob/c61a163e766fde2e634b40c8afbd82e42644f5b7/experiment_scripts/adult_script.sh) and [tabular config](https://github.com/layer6ai-labs/fair-dp/blob/c61a163e766fde2e634b40c8afbd82e42644f5b7/config/tabular.py). In particular, override the generic initial Z=100 with the script's Z=50. Do not carry over MNIST's C=1, sigma=0.8, tau=0.7, learning rate=0.1, or 60 epochs.

Preserve the fixed noise multiplier. The paper reports Adult epsilon around 3.41; record the actual result rather than tuning noise to force that number. Data-size and library differences can affect it.


## Four phases

### Phase 1: Without privacy

**Purpose:** Establish each group's performance before privacy is applied.

Run `experiments/adult/adult.py --phase 1` with learning rate `0.01`, 20 epochs, and the shared settings above. The existing runner uses a zero-noise Opacus baseline with a large finite bound (`1e9`) and ordinary shuffled batches. Verify that wrapping succeeds. This run makes no privacy claim.

Save the final checkpoint, pairing manifest, initial-state checksum, predictions, group/overall accuracy, test losses, and confusion matrices. These results are the reference for Phases 2 and 4.

### Phase 2: Vanilla DP-SGD

**Purpose:** Measure how performance changes when the model is trained with differential privacy.

When Phase 1 is approved, run the existing runner with `--phase 2`, using the same seed, data, model, and initial weights. Use learning rate `0.01`, clipping norm `C=0.5`, fixed gradient noise multiplier `sigma=1.0`, RDP accounting, and `delta=1e-6`.

Save the same evaluations as Phase 1 plus achieved epsilon and accounting metadata. The paper reports epsilon around 3.41; do not tune noise to force that value.

### Phase 3: Compare

**Purpose:** Find whether privacy changes errors unevenly across groups.

Use the existing runner's `--phase 3` to compare Phases 1 and 2 for each paired seed. After Phase 4, rerun the same analysis to include Global-Adapt.

- Calculate **privacy cost = Phase 1 group accuracy - private-method group accuracy**. Keep negative costs if performance improves.
- Compare male/female accuracy losses and their absolute difference. Report accuracy differences in percentage points.
- Calculate **FNR = FN/(FN+TP)** for each group. Present `100*FNR` as missed higher-income predictions per 100 actual higher-income records.
- Report mean, sample standard deviation, and standard error across the five seeds. Use paired changes and clearly labeled standard-error bars.
- Create the two presentation figures: accuracy by group and missed higher-income predictions by group. Initially compare baseline/DP-SGD; add Global-Adapt when Phase 4 is complete.

This phase does not train another model. The detailed metric definitions below specify denominators and aggregation.

### Phase 4: Test DPSGD-Global-Adapt

**Purpose:** Check whether the paper's mitigation improves performance and reduces the observed disparity.

Run the existing runner with `--phase 4` using the same paired data, model, and seeds. Use their Adult configuration: learning rate `0.2`, clipping norm `C=0.5`, gradient noise `sigma=1.0`, initial bound `Z=50`, threshold `tau=1.0`, private count noise `sigma_2=10`, and bound update parameter `eta_Z=0.1`. Train for 20 epochs.

Include the extra private count mechanism when reporting achieved epsilon. Save the same final evaluations, then refresh Phase 3's comparisons and figures.

Report changes in accuracy, missed higher-income predictions, and disparity relative to Phase 2, with Phase 1 as the common baseline. Retain mixed or negative outcomes. Because the published configuration uses a different learning rate, this compares the published methods/configurations rather than isolating clipping alone.

## Implementation and verification notes

### Existing implementation

Use `experiments/adult/adult.py` for training and its existing four-phase interface. Its helpers are `src/adult_dataset.py`, `src/models.py`, `src/metrics.py`, and `src/global_adapt.py`. The user's decision is to reuse this completed implementation, rather than import and adapt a second training framework.

Use `experiments/adult/review_phase1.py` to audit and summarize the five saved baseline runs and `experiments/adult/review_phase2.py` for the five DP-SGD runs, including privacy accounting. These scripts do not train Adult models or calculate paired privacy costs. The runner's full report waits for all three training methods; the phase reviews are therefore saved separately.

The existing environment has Python 3.13.1, PyTorch 2.6.0+cu124, Opacus 1.6.0, and pandas 3.0.6, running on an NVIDIA RTX 4070 SUPER. Record code hashes and exact versions with results. Cite the authors' algorithm/configuration and describe the training code as a local implementation, not an unchanged execution of their repository.

Run one Adult phase at a time and stop for user review. Phases 1 and 2 have completed seeds 0-4; Phase 3 comparisons and Phase 4 remain pending.

### Adult data protocol

#### Raw data and parsing

Obtain `adult.data` and `adult.test` from [UCI's Adult archive](https://archive.ics.uci.edu/static/public/2/adult.zip). Use the cached named files in `data/adult/`, as expected by the existing runner. Preserve the original files and record SHA-256 hashes and retrieval URL. Download once and reuse. Do not substitute a differently processed dataset.

Use the existing Adult parser for the 15 source columns. Verify that it skips the first comment line of `adult.test`, handles surrounding spaces, and accepts income labels with trailing periods. Track a stable row ID using original file and row position before filtering, outside the predictor columns. Concatenate `adult.data` first and `adult.test` second, as the authors do. If adding ID tracking, keep it out of the model inputs and preserve row order.

Drop `fnlwgt`. Remove rows containing `?` or missing required values. Check 48,842 raw records and 45,222 complete cases, with 30,527 males and 14,695 females. If counts differ, diagnose parsing or source data before training. Do not silently adjust the expected counts.

Source: [authors' Adult loader](https://github.com/layer6ai-labs/fair-dp/blob/c61a163e766fde2e634b40c8afbd82e42644f5b7/datasets/tabular.py).

#### Preserve the authors' preprocessing order

Verify that the existing preprocessing follows these steps in order:

1. Standardize `age`, `education_num`, `capital_gain`, `capital_loss`, and `hours_per_week` using means and sample standard deviations (`ddof=1`) from the entire complete-case pool, BEFORE balancing or splitting. Persist these statistics.
2. Map income `<=50K` to 0 and `>50K` to 1, accepting both dotted and undotted labels. Positive always means income above $50,000.
3. Map sex `Male=0`, `Female=1`; preserve an explicit group-name mapping.
4. Map race `White=0`, all other recorded race categories to 1, matching the released preprocessing.
5. Create a separate group array/column from sex.
6. Balance approximately using the sampling procedure below.
7. One-hot encode `workclass`, `education`, `marital_status`, `occupation`, `relationship`, and `native_country` on the retained pool, BEFORE splitting. Keep every dummy column, as `pd.get_dummies` does by default. Save the feature column order.
8. Shuffle the retained frame with `sample(frac=1, random_state=seed)`. Take the first `floor(0.2*N)` rows as test and the rest as train. Do not stratify or impose an exact group quota. No validation split or early stopping is used.

Remove `income` and the duplicate `protected_group` from predictors. **Keep `sex` as a predictor**, matching the released code. Convert feature matrices explicitly to float32, since modern pandas dummy columns may be Boolean. Labels and groups are int64. Derive input width from saved columns rather than hardcoding it.

The processed dataset, test row order, and feature order must be identical for all three methods within a seed. Different seeds may retain different rows, splits, and rare category columns; report that variation.

#### Group balancing

Use probabilities calculated from explicit group IDs, not the ordering of `value_counts()`:

- Female retention probability: 1.
- Male retention probability: `n_female / n_male`, approximately `14695 / 30527`.

Use a Python random generator initialized with the experiment seed. Draw once for EVERY complete-case row in its original order, including female rows, and retain when the draw is at most the appropriate probability. This reproduces the authors' random-draw sequence. Both groups have approximately 14,695 retained records on average; exact counts vary. This is group balancing, not income-label balancing. Record the number of positive and negative income labels within each group and split.

#### Explicit reproduction limitations

The primary protocol deliberately fits preprocessing before the split to follow the released benchmark. State this in the eventual Adult log: the holdout participates in feature preprocessing, which limits interpretation as an independent deployment evaluation. A future train-only preprocessing study must use a separate protocol and result directory, not silently replace this run.

Sex is used for balancing and as a feature. Global-Adapt itself does not require group labels to choose gradient scaling, but the complete Adult pipeline uses sex. Do not describe the whole experiment as never using demographic information.

### Baseline implementation detail

The authors wrap their non-private method in Opacus with zero noise, `max_grad_norm=sys.float_info.max`, and `poisson_sampling=False`. Our existing runner uses a large finite bound of `1e9`, with zero noise and ordinary shuffled batches. The five Phase 1 runs all confirmed successful Opacus wrapping. Record the actual expected batch size; its averaging differs slightly from plain SGD's actual-batch averaging. Never label this zero-noise baseline as private.

### Pairing and privacy bookkeeping

For each seed, the existing runner saves a pairing manifest and initial MLP weights. Reuse those files for Phases 1, 2, and 4 and verify the initialization checksum, selected data, split identities, and feature columns. Training RNG offsets are `seed + 1000` for Phase 1, `seed + 2000` for Phase 2, and `seed + 4000` configured for Phase 4. These differ from the authors' random-number path and must be documented. Use final epoch-20 models, never a best test epoch or a test-selected threshold.

Use ordinary shuffled batches for Phase 1 and Opacus Poisson sampling for Phases 2 and 4, matching the authors. Set `drop_last=False`. Do not require identical batch membership across non-private and private training. Pairing refers to the selected data, split, features, and initial weights.

For private runs, record both the actual Poisson sampling probability (`private_loader.sample_rate`) and the rate used by the accountant hook. Do not substitute the approximation `256/N`. Each accounting rate must equal or conservatively bound the actual sampling probability. For Phase 4, verify this separately for the gradient and count mechanisms and compose both mechanisms.

Phase 2 verified an Opacus 1.6.0 rounding detail: seeds 1 and 2 have 93 nominal batches, but the Poisson sampler converts `1/(1/93)` to 92 batches after floating-point rounding. Actual sampling remains `q=1/93`, while Opacus accounts conservatively at `q=1/92`. Seeds 0, 3, and 4 use `q=1/92` for both. All five private runs perform 1,840 optimizer steps and report epsilon 3.4078045905 at delta 1e-6. Both rates are saved in the Phase 2 summary; preserve this verified training behavior.

For Phase 4, use the existing adaptive engine in `src/global_adapt.py`. Verify that it uses the current Z for clipping/scaling and installs the updated Z for the next batch. Its count-accounting steps are queued by the optimizer and flushed by the local privacy engine. Verify its queued count steps are flushed exactly once before every epsilon query, including the final query. Record actual optimizer steps, count steps, q, and expected batch size. Gaussian noise standard deviation for summed gradients is sigma*C, not sigma alone. Do not add a second gradient-accountant step manually when Opacus already hooks it.

Report epsilon for EACH seed and method, not just its average. Phase 4's composed epsilon must include both mechanisms and should be at least its gradient-only accounting value. If budgets differ meaningfully, explain the difference rather than asserting an exact match.

Treat the result as per-run privacy accounting for training on the frozen benchmark representation. This does not establish end-to-end privacy for raw-data preprocessing, balancing, multiple released runs, or published diagnostics. These are public-data research experiments, not a deployed private data pipeline.

### Metric definitions and aggregation

For each group g and each private method, calculate:

`accuracy_loss_g = accuracy_nonprivate_g - accuracy_private_g`

Keep the signed losses, including negative values when a private method improves accuracy. Save the signed male-minus-female difference and its absolute value. Calculate these quantities from unrounded, paired seed results. Display accuracy differences in PERCENTAGE POINTS, not relative percent changes.

For aggregate reporting, save both the mean per-seed absolute loss gap and the absolute difference between the two mean group losses. These can differ when the sign changes across seeds. Label them explicitly. Use the latter for the paper-definition comparison; do not attach the former's standard error to it. Primary slides show group accuracies and false-negative rates, so no complicated gap uncertainty analysis is required.


#### Confusion matrices and missed higher-income predictions

Positive means income above $50,000. Predict by `argmax` over the two logits and save `softmax(logits)[:,1]` as the positive probability. The decision boundary is fixed at equal logits, with argmax ties choosing class 0. Do not optimize thresholds on test data.

For each group, store the confusion matrix with rows TRUE [0,1] and columns PREDICTED [0,1]: `[[TN,FP],[FN,TP]]`. Then compute:

- Accuracy: `(TP+TN)/(TP+TN+FP+FN)`.
- False-negative rate: `FN/(FN+TP)`.
- Missed higher-income predictions per 100 actual higher-income records: `100*FNR`.
- Recall/TPR and FPR from the same counts, as supporting diagnostics.
- Additional missed predictions after privacy: `100*(FNR_private-FNR_nonprivate)`.

Save denominator counts and income prevalence for every group and split. A denominator of zero produces an explicitly undefined/null rate, never zero or invalid JSON NaN. Store fractions in raw JSON and convert to percentages or per-100 values in clearly labeled summaries. The FNR measure is our addition; the original Adult table supplies no FNR target. It measures income-classification errors, not actual denied loans, jobs, or benefits.

Aggregate seed-level results with the mean, sample SD (`ddof=1`), and SE (`SD/sqrt(5)`). Use SE for charts comparing with the paper, which reports standard errors. Compute SD/SE of paired accuracy and FNR changes from each seed's change, not by subtracting independent error bars. Do not pool all seeds' test predictions as if they were independent people.

### Published reference values and differences to record

The following are published means with standard errors. They are reference values, not required outputs or pass/fail thresholds. Source: [paper, Table 4](https://arxiv.org/html/2206.07737v2).

| Method | Male accuracy (%) | Female accuracy (%) | Male accuracy loss (pp) | Female accuracy loss (pp) | Published loss gap (pp) |
|---|---:|---:|---:|---:|---:|
| Non-private | 80.5 +/- 0.4 | 92.2 +/- 0.1 | Baseline | Baseline | Baseline |
| DP-SGD | 69.9 +/- 0.4 | 88.5 +/- 0.1 | 10.6 +/- 0.3 | 3.6 +/- 0.1 | 6.9 +/- 0.3 |
| Global-Adapt | 80.7 +/- 0.4 | 92.3 +/- 0.1 | -0.1 +/- 0.1 | -0.1 +/- 0.1 | 0.0 +/- 0.1 |

Reproduce the table as reported; rounding can prevent differences between displayed means from equaling displayed costs. Calculate our values before rounding. Do not assume in advance which group will lose more performance, and do not alter the configuration to recover the target values.

Discrepancies already verified:

- The paper describes about 14,000 retained records per group; the released balancing code retains all 14,695 complete-case female rows and approximately that many males. Follow the code and report actual counts.
- Generic config says initial Z=100; the Adult script overrides it to 50. Use 50.
- The fixed, non-adaptive Global method has an Adult learning-rate discrepancy between paper and script. That extra method is outside our scope. Global-Adapt's learning rate of 0.2 agrees between the paper and script.
- Original software used PyTorch 1.11 and Opacus 1.1; our local implementation uses newer versions. The local adaptive engine draws count noise after gradient noise, unlike the original trainer. Verify its current-Z/next-Z dependency during Phase 4 and do not promise identical random-number paths or bit-for-bit results.
- The original loader uses the test set for intermediate validation. Our runner evaluates the final model after 20 epochs. Do not use test results to select settings or checkpoints.

Because learning rates differ by method, the main comparison evaluates published training configurations. It does not isolate the clipping algorithm independently from learning-rate choice.

### Checks and execution order

The existing runner's self-test passed. Both phase reviews independently check frozen data metadata and initialization checksums, train/test IDs, finite checkpoints, and recompute every group confusion matrix, accuracy, and FNR from saved predictions. Phase 2 also checks alignment with the Phase 1 test records and reconstructs the installed loader/accountant behavior and achieved epsilon.

Executed from the workspace root:

```powershell
.\.venv\Scripts\python.exe experiments/adult/adult.py --self-test
.\.venv\Scripts\python.exe experiments/adult/adult.py --phase 1 --seed 0
.\.venv\Scripts\python.exe experiments/adult/adult.py --phase 1 --all-seeds --skip-existing
.\.venv\Scripts\python.exe experiments/adult/review_phase1.py
.\.venv\Scripts\python.exe experiments/adult/adult.py --phase 2 --seed 0
.\.venv\Scripts\python.exe experiments/adult/adult.py --phase 2 --all-seeds --skip-existing
.\.venv\Scripts\python.exe experiments/adult/review_phase2.py
```

All five baseline and five DP-SGD runs are complete. Their separate summaries, charts, and reviews are in `results/adult/`. Stop after Phase 2 for user review; execute Phase 3 only when requested. Retain the same initialization and pairing artifacts. Do not mix stale results after code or protocol changes, even when using `--skip-existing`. Phase 4 still requires checking achieved epsilon, actual sampling rates, and extra count accounting.

### Deliverables and completion criteria

Write under `results/adult/`:

- `pairing_seed_<s>.json`, recording row/split IDs, feature columns, preprocessing, and each method's initial-state checksum. Save initial weights if useful; the authors' final checkpoints must be retained.
- `phase1_seed_<s>.json`, `phase2_seed_<s>.json`, `phase3_seed_<s>.json`, and `phase4_seed_<s>.json` for seeds 0-4. Phase 3 contains derived comparisons, not training output.
- Final model checkpoints for each trained method/seed and per-example predictions (stable row ID, true label, group, prediction, positive probability).
- `summary.json`, a compact `summary.csv`, and training logs. Include full settings, source commit/hashes, implementation fingerprints, software versions, device, elapsed time, actual sample rate, optimizer/count steps, and achieved privacy budgets in the appropriate metadata.
- `adult_accuracy_by_group.png` plus a vector PDF: group accuracy for all three methods, with clearly labeled SE bars and paired accuracy losses in the companion table.
- `adult_missed_high_income_by_group.png` plus a vector PDF: missed predictions per 100 actual higher-income records for all three methods, with SE bars. Use consistent method colors and male/female labels across both figures. Keep the underlying CSV values available.

The existing runner writes checkpoints, predictions, and phase JSON files directly to `results/adult/`. Phase 1 review outputs are `phase1_summary.json`, `phase1_summary.csv`, `phase1_review.md`, and `phase1_baseline.png`/`.pdf`. Phase 2 outputs are `phase2_summary.json`, `phase2_summary.csv`, `phase2_review.md`, and `phase2_dpsgd.png`/`.pdf`. The multi-method figures remain deliverables for the later comparison.

Create a separate `adult_experiment_log.md` with the protocol, dataset exploration, our results, comparison with published references, added FNR analysis, limitations, and exact run commands. Link it from README with a short Adult section without rewriting the MNIST conclusions during this task. Acknowledge the authors' code and clearly distinguish their algorithm/configuration from our reporting additions and extra evaluation. Describe this as reproduction plus subgroup analysis, not a newly invented training method.

Done means all five paired seeds have final results for all three methods, derived metrics recompute from saved counts/predictions, extra count privacy is included, the two charts match the summary data, and all discrepancies are honestly documented. Inspect the charts for readable labels and correct denominators. Implementation should remain small enough for every teammate to explain.

Current status: Phase 1 baseline and Phase 2 DP-SGD training and independent result verification are complete for all five seeds. Phase 3 comparisons and Phase 4 Adult training have not been run.

## References

Use the authors' repository at the same pinned commit already used by the MNIST project: `c61a163e766fde2e634b40c8afbd82e42644f5b7`. Script overrides take precedence over generic config defaults. Record this commit in results.

- [Paper: Esipova et al., ICLR 2023](https://arxiv.org/html/2206.07737v2), Section 6.1, Appendix B.1-B.3, and Table 4 in Appendix B.7.
- [Adult experiment script](https://github.com/layer6ai-labs/fair-dp/blob/c61a163e766fde2e634b40c8afbd82e42644f5b7/experiment_scripts/adult_script.sh): run settings and overrides.
- [Tabular config](https://github.com/layer6ai-labs/fair-dp/blob/c61a163e766fde2e634b40c8afbd82e42644f5b7/config/tabular.py): model and optimizer defaults.
- [Adult preprocessing](https://github.com/layer6ai-labs/fair-dp/blob/c61a163e766fde2e634b40c8afbd82e42644f5b7/datasets/tabular.py) and [sampling weights](https://github.com/layer6ai-labs/fair-dp/blob/c61a163e766fde2e634b40c8afbd82e42644f5b7/datasets/sample_weights.py).
- [MLP implementation](https://github.com/layer6ai-labs/fair-dp/blob/c61a163e766fde2e634b40c8afbd82e42644f5b7/models/neural_networks.py) and [model factory](https://github.com/layer6ai-labs/fair-dp/blob/c61a163e766fde2e634b40c8afbd82e42644f5b7/models/model_factory.py).
- [Training setup](https://github.com/layer6ai-labs/fair-dp/blob/c61a163e766fde2e634b40c8afbd82e42644f5b7/main.py), [loaders](https://github.com/layer6ai-labs/fair-dp/blob/c61a163e766fde2e634b40c8afbd82e42644f5b7/datasets/loaders.py), [trainer](https://github.com/layer6ai-labs/fair-dp/blob/c61a163e766fde2e634b40c8afbd82e42644f5b7/trainers/trainer.py), and [adaptive optimizer](https://github.com/layer6ai-labs/fair-dp/blob/c61a163e766fde2e634b40c8afbd82e42644f5b7/optimizers/dpsgd_global_adaptive_optimizer.py).
- [UCI Adult source and attribution](https://archive.ics.uci.edu/dataset/2/adult).

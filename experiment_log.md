# Experiment Log: MNIST DP-SGD Disparity Study

This log tracks experimental runs across seeds 0–4 following the protocol defined in [plan.md](plan.md), investigating differential privacy disparity on the rare digit 8 compared to control digit 2.

---

## Benchmark Summary Table (Seeds 0–4)

| Seed | Phase | Method | Epochs | Batch Size | lr | Noise (σ) | Clip (C) | Achieved (ε, δ) | Digit 2 Acc | Digit 8 Acc | Privacy Cost (π₈) | Notes |
| :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **0** | Phase 1 | Non-private | 60 | 256 | 0.01 | - | - | Non-private | *Pending* | *Pending* | - | Seed 0 baseline |
| **0** | Phase 2 | Vanilla DP-SGD | 60 | 256 | 0.01 | 0.8 | 1.0 | ε ≈ 5.9, 1e-6 | *Pending* | *Pending* | *Pending* | Seed 0 DP-SGD |
| **1** | Phase 1 | Non-private | 60 | 256 | 0.01 | - | - | Non-private | *Pending* | *Pending* | - | Seed 1 baseline |
| **1** | Phase 2 | Vanilla DP-SGD | 60 | 256 | 0.01 | 0.8 | 1.0 | ε ≈ 5.9, 1e-6 | *Pending* | *Pending* | *Pending* | Seed 1 DP-SGD |
| **2** | Phase 1 | Non-private | 60 | 256 | 0.01 | - | - | Non-private | *Pending* | *Pending* | - | Seed 2 baseline |
| **2** | Phase 2 | Vanilla DP-SGD | 60 | 256 | 0.01 | 0.8 | 1.0 | ε ≈ 5.9, 1e-6 | *Pending* | *Pending* | *Pending* | Seed 2 DP-SGD |
| **3** | Phase 1 | Non-private | 60 | 256 | 0.01 | - | - | Non-private | *Pending* | *Pending* | - | Seed 3 baseline |
| **3** | Phase 2 | Vanilla DP-SGD | 60 | 256 | 0.01 | 0.8 | 1.0 | ε ≈ 5.9, 1e-6 | *Pending* | *Pending* | *Pending* | Seed 3 DP-SGD |
| **4** | Phase 1 | Non-private | 60 | 256 | 0.01 | - | - | Non-private | *Pending* | *Pending* | - | Seed 4 baseline |
| **4** | Phase 2 | Vanilla DP-SGD | 60 | 256 | 0.01 | 0.8 | 1.0 | ε ≈ 5.9, 1e-6 | *Pending* | *Pending* | *Pending* | Seed 4 DP-SGD |

---

## Published Paper Reference Metrics (arXiv:2206.07737, Table 2)

- **Non-private Baseline:** Digit 2 Acc: 98.0% ± 0.1% | Digit 8 Acc: 84.3% ± 1.1%
- **Vanilla DP-SGD (ε ≈ 5.90, δ = 1e-6):** Digit 2 Acc: 89.0% ± 0.1% | Digit 8 Acc: 26.3% ± 0.4% | Privacy Cost (π₈): 57.9% ± 1.3%
- **DPSGD-Global-Adapt:** Digit 2 Acc: 92.0% ± 0.2% | Digit 8 Acc: 65.5% ± 1.2% | Privacy Cost (π₈): 18.8% ± 0.9%

---

## Pilot Experiment Reference (8 Epochs, Seed 42)

Full details and scripts are preserved in [`pilot/`](pilot/README.md).
- **Non-private:** Digit 2 Acc: 99.2% | Digit 8 Acc: 91.0%
- **DP-SGD (ε ≈ 8.0):** Digit 2 Acc: 94.5% | Digit 8 Acc: 70.0% (Privacy cost: 20.9%)
- **DP-SGD (ε ≈ 5.9):** Digit 2 Acc: 93.1% | Digit 8 Acc: 61.2% (Privacy cost: 29.8%)
- **DP-SGD (ε ≈ 3.0):** Digit 2 Acc: 88.7% | Digit 8 Acc: 48.6% (Privacy cost: 42.4%)

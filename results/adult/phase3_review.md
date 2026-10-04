# Adult Phase 3: Paired Comparison (Baseline vs. DP-SGD)

Paired evaluation across 5 seeds. Values are **Mean ± Standard Error of within-seed differences**.

| Group | Baseline Accuracy → DP-SGD Accuracy | Accuracy Loss (pp) | Baseline Missed > $50K → DP-SGD Missed |
| :--- | :---: | :---: | :---: |
| **Male** | 80.57% → 69.86% | **10.71 ± 0.33** | 38.6 → 95.0 (+56.4 per 100) |
| **Female** | 92.18% → 88.54% | **3.65 ± 0.08** | 52.2 → 98.4 (+46.2 per 100) |

### Key Takeaway
- **Disparity Gap**: Male accuracy loss exceeds female accuracy loss by **+7.07 ± 0.28 percentage points**.
- Matches *Esipova et al. (ICLR 2023)* reported gap of **6.9 ± 0.3 pp**.
- Both groups suffer massive increases in false negatives on high-income records (+46 to +56 per 100).

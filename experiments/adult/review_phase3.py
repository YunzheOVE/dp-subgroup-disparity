"""Audit the existing runner's paired Phase 3 results and report baseline vs DP-SGD."""

import csv
import json
from pathlib import Path

from review_phase1 import ROOT, OUT, sha, stats
import matplotlib.pyplot as plt
import numpy as np


def main():
    earlier = {phase: json.loads((OUT / f"phase{phase}_summary.json").read_text()) for phase in (1, 2)}
    sources = earlier[2]["source_sha256"]
    for relative, expected in sources.items():
        assert sha(ROOT / relative) == expected, relative
    rows, paired, audit = [], [], []
    for seed in range(5):
        result = {phase: json.loads((OUT / f"phase{phase}_seed_{seed}.json").read_text()) for phase in (1, 2, 3)}
        predictions = {}
        for phase in (1, 2):
            previous = next(a for a in earlier[phase]["audit"] if a["seed"] == seed)
            assert previous["checks_passed"] and result[phase]["seed"] == seed
            for field, fingerprint in (("model_checkpoint", "model_sha256"), ("predictions_file", "predictions_sha256")):
                assert sha(OUT / result[phase][field]) == previous[fingerprint]
            predictions[phase] = json.loads((OUT / result[phase]["predictions_file"]).read_text())
        assert [(p["stable_row_id"], p["true_label"], p["group_id"]) for p in predictions[1]] == [
            (p["stable_row_id"], p["true_label"], p["group_id"]) for p in predictions[2]]
        pairing = json.loads((OUT / f"pairing_seed_{seed}.json").read_text())
        assert result[1]["initial_weights_file"] == result[2]["initial_weights_file"]
        assert sha(OUT / result[2]["initial_weights_file"]) == pairing["initial_weights_sha256"]
        comparison = result[3]["phase2_dpsgd"]
        assert result[3]["seed"] == seed and comparison["epsilon"] == result[2]["achieved_epsilon"]
        assert "phase4_global_adapt" not in result[3], "This checkpoint compares Phases 1 and 2 only."
        for group_id, name in enumerate(("Male", "Female")):
            measured = {}
            for phase in (1, 2):
                group = [p for p in predictions[phase] if p["group_id"] == group_id]
                cm = [[sum(p["true_label"] == actual and p["prediction"] == predicted for p in group)
                       for predicted in (0, 1)] for actual in (0, 1)]
                saved = result[phase]["evaluation"]["groups"][name]
                assert cm == saved["confusion_matrix"] and sum(cm[1]) > 0
                accuracy, missed = 100 * (cm[0][0] + cm[1][1]) / len(group), 100 * cm[1][0] / sum(cm[1])
                assert abs(accuracy - 100 * saved["accuracy"]) < 1e-10
                assert abs(missed - saved["missed_per_100"]) < 1e-10
                assert abs(missed - 100 * saved["fnr"]) < 1e-10
                measured[phase] = (accuracy, missed)
            prefix = name.lower()
            cost, added = measured[1][0] - measured[2][0], measured[2][1] - measured[1][1]
            assert abs(cost - comparison[f"{prefix}_privacy_cost_pp"]) < 1e-10
            assert abs(added - comparison[f"{prefix}_added_missed_per_100"]) < 1e-10
            for phase, section in ((1, "phase1_baseline"), (2, "phase2_dpsgd")):
                assert abs(measured[phase][0] - 100 * result[3][section][f"{prefix}_acc"]) < 1e-10
                assert abs(measured[phase][1] - result[3][section][f"{prefix}_missed_per_100"]) < 1e-10
            rows.append({"seed": seed, "group": name, "test_records": len(group), "actual_higher_income": sum(cm[1]),
                         "baseline_accuracy_pct": measured[1][0], "dpsgd_accuracy_pct": measured[2][0],
                         "accuracy_loss_pp": cost, "baseline_missed_per_100": measured[1][1],
                         "dpsgd_missed_per_100": measured[2][1], "added_missed_per_100": added})
        signed = comparison["male_privacy_cost_pp"] - comparison["female_privacy_cost_pp"]
        assert abs(signed - comparison["disparity_gap_signed_pp"]) < 1e-10
        assert abs(abs(signed) - comparison["disparity_gap_abs_pp"]) < 1e-10
        paired.append(comparison)
        audit.append({"seed": seed, "checks_passed": True,
                      "input_sha256": {f"phase{phase}_seed_{seed}.json": sha(OUT / f"phase{phase}_seed_{seed}.json")
                                       for phase in (1, 2, 3)}, "initial_weights_sha256": pairing["initial_weights_sha256"]})
    names = ("Male", "Female")
    metrics = list(rows[0])[4:]
    groups = {name: {metric: stats([row[metric] for row in rows if row["group"] == name]) for metric in metrics}
              for name in names}
    for name in names:
        for phase, prefix in ((1, "baseline"), (2, "dpsgd")):
            for metric in ("accuracy_pct", "missed_per_100"):
                for statistic in ("mean", "sd", "se"):
                    assert abs(groups[name][f"{prefix}_{metric}"][statistic] - earlier[phase]["groups"][name][metric][statistic]) < 1e-10
    summary = {"phase": 3, "seeds": list(range(5)), "training_performed": False, "groups": groups,
               "signed_male_minus_female_accuracy_loss_pp": stats([p["disparity_gap_signed_pp"] for p in paired]),
               "mean_per_seed_absolute_accuracy_loss_gap_pp": stats([p["disparity_gap_abs_pp"] for p in paired]),
               "absolute_gap_between_mean_accuracy_losses_pp": abs(groups["Male"]["accuracy_loss_pp"]["mean"] - groups["Female"]["accuracy_loss_pp"]["mean"]),
               "all_seed_accuracy_loss_gaps_positive": all(p["disparity_gap_signed_pp"] > 0 for p in paired),
               "per_run_epsilon": earlier[2]["per_run_epsilon"], "delta": earlier[2]["delta"], "audit": audit,
               "source_sha256": {**sources, str(Path(__file__).relative_to(ROOT)): sha(__file__)},
               "earlier_summary_sha256": {f"phase{phase}_summary.json": sha(OUT / f"phase{phase}_summary.json") for phase in (1, 2)}}
    (OUT / "phase3_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    with (OUT / "phase3_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for metric, title, ylabel, filename in (
            ("accuracy_pct", "Accuracy falls more for males", "Correct predictions (%)", "adult_accuracy_by_group"),
            ("missed_per_100", "Higher-income records are missed more often",
             "Missed higher-income predictions\nper 100 actual higher-income records", "adult_missed_high_income_by_group")):
        fig, ax = plt.subplots(figsize=(9, 5.6), dpi=200)
        x = np.arange(2)
        for offset, prefix, label, color in ((-0.18, "baseline", "Without privacy", "#94A3B8"),
                                             (0.18, "dpsgd", "DP-SGD", "#3066BE")):
            means = [groups[name][f"{prefix}_{metric}"]["mean"] for name in names]
            errors = [groups[name][f"{prefix}_{metric}"]["se"] for name in names]
            bars = ax.bar(x + offset, means, width=0.34, yerr=errors, capsize=5, label=label, color=color)
            for bar, mean in zip(bars, means):
                ax.text(bar.get_x() + bar.get_width() / 2, mean - 7, f"{mean:.1f}", ha="center",
                        color="white", fontsize=13, fontweight="bold")
        ax.set_xticks(x, names, fontsize=13)
        ax.set_ylim(0, 104)
        ax.set_yticks(range(0, 101, 20))
        ax.set_ylabel(ylabel, fontsize=11)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=0.15)
        fig.suptitle(title, fontsize=17, y=0.95)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.13), ncol=2, frameon=False, fontsize=11)
        fig.text(0.5, 0.035, "Adult benchmark · Mean ± standard error across five seeds · DP-SGD: ε = 3.4078, δ = 10⁻⁶ per run",
                 ha="center", fontsize=9)
        fig.subplots_adjust(left=0.13, right=0.98, bottom=0.17, top=0.77)
        for extension in ("png", "pdf"):
            fig.savefig(OUT / f"{filename}.{extension}")
        plt.close(fig)
    table = ["| Group | Accuracy without privacy → DP-SGD | Accuracy loss (pp) | Missed per 100 without privacy → DP-SGD | Additional missed per 100 |",
             "|---|---:|---:|---:|---:|"]
    for name in names:
        g = groups[name]
        table.append(f"| {name} | {g['baseline_accuracy_pct']['mean']:.2f}% → {g['dpsgd_accuracy_pct']['mean']:.2f}% | "
                     f"{g['accuracy_loss_pp']['mean']:.2f} ± {g['accuracy_loss_pp']['se']:.2f} | "
                     f"{g['baseline_missed_per_100']['mean']:.2f} → {g['dpsgd_missed_per_100']['mean']:.2f} | "
                     f"{g['added_missed_per_100']['mean']:.2f} ± {g['added_missed_per_100']['se']:.2f} |")
    gap = summary["signed_male_minus_female_accuracy_loss_pp"]
    report = "# Adult Phase 3: review checkpoint\n\nCompared five paired seeds using the existing `adult.py --phase 3`; no model training or settings changes. Values are seed averages. Changes show mean ± standard error of the paired changes.\n\n"
    report += "\n".join(table)
    report += f"\n\nMale accuracy loss exceeds female accuracy loss by **{gap['mean']:.2f} ± {gap['se']:.2f} percentage points** (male minus female). Every seed has this same direction. The mean per-seed absolute gap and the absolute gap between mean losses therefore both equal {gap['mean']:.2f} pp here; they are stored separately because they need not agree in general.\n\n"
    report += "Accuracy loss = non-private accuracy − DP-SGD accuracy. Additional missed predictions = 100 × (DP-SGD FNR − baseline FNR), among actual higher-income records. Use percentage points for accuracy loss and per-100 records for missed predictions. SE is computed from each seed's paired change, not by subtracting error bars. Sample SD is also saved.\n\n"
    report += "Practical takeaway: privacy reduces prediction quality unevenly across groups, and overall accuracy can hide failures to recognize higher-income records. Female records retain higher accuracy but have the higher final missed-prediction rate; male records experience the larger increase after adding privacy. The smaller final male/female FNR gap comes from both rates approaching failure, so it should not be presented as a fairness improvement.\n\n"
    report += "The [paper's Table 4](https://arxiv.org/html/2206.07737v2) reports accuracy losses of 10.6 ± 0.3 pp for males, 3.6 ± 0.1 pp for females, and a 6.9 ± 0.3 pp loss gap. Our losses and gap closely reproduce that pattern. The paper does not report Adult FNR targets; this is our added evaluation. We have not isolated gradient misalignment as the cause in our experiments.\n\n"
    report += "Verified: earlier source/checkpoint/prediction fingerprints; matching test IDs, labels, and groups; shared initialization hashes; recomputed confusion matrices and paired changes for all five seeds; summaries agree with prior audits. Both charts use the checked summary values. No new privacy mechanism is run during this analysis; training budgets remain per-run ε=3.4078 at δ=10⁻⁶ on the frozen benchmark representation.\n\n"
    report += "Limits: historical US Adult data motivates a general privacy/fairness question, not a measured claim about NYC. Full-pool preprocessing follows the released protocol and limits independent deployment interpretation. Income-classification errors do not directly measure denied jobs, loans, or benefits.\n\n"
    for filename in ("adult_accuracy_by_group", "adult_missed_high_income_by_group"):
        report += f"![{filename.replace('_', ' ')}]({(OUT / (filename + '.png')).as_posix()})\n\n"
    report += "Stop after Phase 3 for user review. Phase 4 Adult training is pending; its results will be added to the comparison after that phase. The runner's full 15-model report remains deferred until Phase 4.\n\nReproduce from the repository root:\n\n```powershell\n.\\.venv\\Scripts\\python.exe experiments/adult/adult.py --phase 3 --all-seeds\n.\\.venv\\Scripts\\python.exe experiments/adult/review_phase3.py\n```\n"
    (OUT / "phase3_review.md").write_text(report, encoding="utf-8")
    print("Phase 3 paired-result audit passed for all five seeds.")
    print("\n".join(table))
    print("Signed male-minus-female accuracy loss:", gap)


if __name__ == "__main__":
    main()

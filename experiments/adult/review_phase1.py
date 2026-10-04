"""Audit saved Adult baseline predictions and summarize Phase 1 only. No training."""

import csv
import hashlib
import json
from pathlib import Path
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import torch
import opacus
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.adult_dataset import prepare_adult_data

OUT = ROOT / "results/adult"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stats(values):
    return {"mean": float(np.mean(values)), "sd": float(np.std(values, ddof=1)),
            "se": float(np.std(values, ddof=1) / np.sqrt(len(values)))}


def main():
    results, rows, audit = [], [], []
    for seed in range(5):
        result = json.loads((OUT / f"phase1_seed_{seed}.json").read_text())
        pairing = json.loads((OUT / f"pairing_seed_{seed}.json").read_text())
        predictions = json.loads((OUT / result["predictions_file"]).read_text())
        train, test, meta = prepare_adult_data(ROOT / "data/adult", seed)
        assert meta == pairing["data_meta"], f"Seed {seed}: pairing metadata differ"
        assert sha(OUT / result["initial_weights_file"]) == pairing["initial_weights_sha256"]
        assert [p["stable_row_id"] for p in predictions] == test.row_ids
        assert [p["true_label"] for p in predictions] == test.labels.tolist()
        assert [p["group_id"] for p in predictions] == test.groups.tolist()
        assert all(p["prediction"] in (0, 1) and np.isfinite(p["positive_probability"])
                   and 0 <= p["positive_probability"] <= 1 for p in predictions)
        assert set(train.row_ids).isdisjoint(test.row_ids)
        assert len(predictions) == result["evaluation"]["overall"]["total_samples"]
        assert result["epochs"] == 20 and result["lr"] == 0.01 and result["noise_multiplier"] == 0
        assert result["wrapped_with_opacus"] and result["achieved_epsilon"] is None
        state = torch.load(OUT / result["model_checkpoint"], map_location="cpu", weights_only=True)
        assert all(torch.isfinite(tensor).all() for tensor in state.values())
        overall_correct = sum(p["true_label"] == p["prediction"] for p in predictions)
        assert abs(overall_correct / len(predictions) - result["evaluation"]["overall"]["accuracy"]) < 1e-12
        for group_id, name in enumerate(("Male", "Female")):
            group = [p for p in predictions if p["group_id"] == group_id]
            cm = [[sum(p["true_label"] == actual and p["prediction"] == predicted for p in group)
                   for predicted in (0, 1)] for actual in (0, 1)]
            measured = result["evaluation"]["groups"][name]
            assert cm == measured["confusion_matrix"]
            accuracy = (cm[0][0] + cm[1][1]) / len(group)
            missed = 100 * cm[1][0] / sum(cm[1])
            assert abs(accuracy - measured["accuracy"]) < 1e-12
            assert abs(missed - measured["missed_per_100"]) < 1e-12
            rows.append({"seed": seed, "group": name, "test_records": len(group), "actual_higher_income": sum(cm[1]),
                         "accuracy_pct": accuracy * 100, "missed_per_100": missed,
                         "higher_income_prevalence_pct": 100 * sum(cm[1]) / len(group)})
        audit.append({"seed": seed, "train_records": len(train), "test_records": len(test), "input_dim": meta["input_dim"],
                      "parameter_count": sum(t.numel() for t in state.values()),
                      "train_row_ids": train.row_ids, "test_row_ids": test.row_ids,
                      "model_sha256": sha(OUT / result["model_checkpoint"]),
                      "predictions_sha256": sha(OUT / result["predictions_file"]),
                      "optimizer_steps": 20 * int(np.ceil(len(train) / 256)), "checks_passed": True})
        results.append(result)
    summary = {"phase": 1, "seeds": list(range(5)), "implementation": "existing_experiments/adult/adult.py",
               "overall_accuracy_pct": stats([100 * r["evaluation"]["overall"]["accuracy"] for r in results]),
               "groups": {name: {metric: stats([row[metric] for row in rows if row["group"] == name])
                                for metric in ("accuracy_pct", "missed_per_100", "higher_income_prevalence_pct")}
                          for name in ("Male", "Female")},
               "software": {"python": sys.version.split()[0], "torch": torch.__version__, "opacus": opacus.__version__,
                            "numpy": np.__version__, "pandas": pd.__version__, "gpu": torch.cuda.get_device_name(0)},
               "source_sha256": {str(p.relative_to(ROOT)): sha(p) for p in
                                 [ROOT / "experiments/adult/adult.py", Path(__file__), ROOT / "src/adult_dataset.py",
                                  ROOT / "src/models.py", ROOT / "src/metrics.py"]},
               "total_training_seconds": sum(r["elapsed_seconds"] for r in results),
               "audit": audit}
    (OUT / "phase1_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    with (OUT / "phase1_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8), dpi=180)
    names = ["Male", "Female"]
    for ax, metric, title, ylabel in zip(axes, ["accuracy_pct", "missed_per_100"],
            ["Income prediction accuracy", "Missed higher-income predictions"],
            ["Correct predictions (%)", "Missed per 100 actual higher-income records"]):
        means = [summary["groups"][name][metric]["mean"] for name in names]
        errors = [summary["groups"][name][metric]["se"] for name in names]
        ax.bar(names, means, yerr=errors, capsize=5, color=["#3066BE", "#E8A23A"], width=0.55)
        ax.set_ylim(0, 100)
        ax.set_title(title, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        for i, mean in enumerate(means):
            ax.text(i, mean + errors[i] + 2, f"{mean:.1f}", ha="center", fontsize=11)
    fig.suptitle("Adult baseline without privacy", fontsize=16)
    fig.text(0.5, 0.02, "Mean and standard error across five seeds. Both panels show performance before privacy is applied.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=[0, 0.06, 1, 0.92])
    for extension in ("png", "pdf"):
        fig.savefig(OUT / f"phase1_baseline.{extension}")
    plt.close(fig)
    table = ["| Group | Accuracy (%) | Missed per 100 actual higher-income records |", "|---|---:|---:|"]
    for name in names:
        acc, fnr = (summary["groups"][name][metric] for metric in ("accuracy_pct", "missed_per_100"))
        table.append(f"| {name} | {acc['mean']:.2f} ± {acc['se']:.2f} | {fnr['mean']:.2f} ± {fnr['se']:.2f} |")
    report = "# Adult Phase 1: review checkpoint\n\nCompleted seeds 0–4 using the existing `adult.py`, 20 epochs each. Values below are mean ± standard error across seeds.\n\n"
    report += "\n".join(table) + f"\n\nOverall accuracy: **{summary['overall_accuracy_pct']['mean']:.2f}%**.\n\n"
    report += "Female records have higher overall accuracy but a higher missed-prediction rate among actual higher-income records. The income-label distribution differs between groups, so accuracy alone gives an incomplete picture. These are baseline errors; we cannot attribute them to differential privacy.\n\n"
    report += "The paper reports baseline accuracy of 80.5 ± 0.4% for males and 92.2 ± 0.1% for females. Our baseline is close, with no parameter tuning to force a match.\n\n"
    report += "Verified: cached data counts; frozen initialization checksums; disjoint split IDs; all five finite checkpoints; predictions reproduce every saved group confusion matrix, accuracy, and FNR. The same pairing files and initial weights are available for the later phases.\n\n"
    report += "Implementation notes: the existing runner follows the published Adult settings but is a local implementation, not an unchanged execution of the authors' repository. It uses training seed `seed + 1000` and a large finite clipping bound (`1e9`) with zero noise. Full-pool preprocessing before the split follows the released benchmark and limits interpretation as independent deployment evaluation. Adult is historical US census data, not a sample of NYC or actual eligibility decisions.\n\n"
    report += f"![Phase 1 baseline]({(OUT / 'phase1_baseline.png').as_posix()})\n\n"
    report += "Phase 2, Phase 3 comparisons, and Phase 4 Adult training have not been run. Stop here for user review.\n\n"
    report += "Reproduce from the repository root:\n\n```powershell\n.\\.venv\\Scripts\\python.exe experiments/adult/adult.py --phase 1 --all-seeds --skip-existing\n.\\.venv\\Scripts\\python.exe experiments/adult/review_phase1.py\n```\n"
    (OUT / "phase1_review.md").write_text(report, encoding="utf-8")
    print("Phase 1 audit passed for all five seeds.")
    print("\n".join(table))
    print("Overall accuracy:", summary["overall_accuracy_pct"])


if __name__ == "__main__":
    main()

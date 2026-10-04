"""Verify and summarize saved Phase 2 DP-SGD runs. No Adult training or Phase 3 comparisons."""

import csv
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from opacus import PrivacyEngine
from opacus.accountants import RDPAccountant
import matplotlib.pyplot as plt

from review_phase1 import ROOT, OUT, sha, stats
from src.adult_dataset import prepare_adult_data

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def privacy_loader_check(size):
    """One synthetic step checks the installed library's actual sampler/accountant hook."""
    data = TensorDataset(torch.zeros(size, 1), torch.zeros(size, dtype=torch.long))
    loader = DataLoader(data, batch_size=256)
    model = nn.Linear(1, 2)
    engine = PrivacyEngine(accountant="rdp")
    model, optimizer, private = engine.make_private(
        module=model, optimizer=torch.optim.SGD(model.parameters(), lr=0.01), data_loader=loader,
        noise_multiplier=1.0, max_grad_norm=0.5, poisson_sampling=True)
    x, y = next(iter(private))
    optimizer.zero_grad()
    nn.functional.cross_entropy(model(x), y).backward()
    optimizer.step()
    noise, accountant_q, steps = engine.accountant.history[0]
    assert noise == 1.0 and steps == 1 and len(engine.accountant.history) == 1
    assert accountant_q >= private.sample_rate
    return len(private), private.sample_rate, accountant_q, optimizer.expected_batch_size


def main():
    results, rows, audit = [], [], []
    phase1_summary = json.loads((OUT / "phase1_summary.json").read_text())
    # The data/model/training code must match the source fingerprints checked in Phase 1.
    for relative, expected in phase1_summary["source_sha256"].items():
        assert sha(ROOT / relative) == expected, relative
    for seed in range(5):
        result = json.loads((OUT / f"phase2_seed_{seed}.json").read_text())
        pairing = json.loads((OUT / f"pairing_seed_{seed}.json").read_text())
        predictions = json.loads((OUT / result["predictions_file"]).read_text())
        baseline = json.loads((OUT / f"phase1_predictions_seed_{seed}.json").read_text())
        train, test, metadata = prepare_adult_data(ROOT / "data/adult", seed)
        assert metadata == pairing["data_meta"]
        assert sha(OUT / result["initial_weights_file"]) == pairing["initial_weights_sha256"]
        assert [p["stable_row_id"] for p in predictions] == test.row_ids
        assert [p["true_label"] for p in predictions] == test.labels.tolist()
        assert [p["group_id"] for p in predictions] == test.groups.tolist()
        assert [(p["stable_row_id"], p["true_label"], p["group_id"]) for p in predictions] == [
            (p["stable_row_id"], p["true_label"], p["group_id"]) for p in baseline]
        assert set(train.row_ids).isdisjoint(test.row_ids)
        assert all(p["prediction"] in (0, 1) and np.isfinite(p["positive_probability"])
                   and 0 <= p["positive_probability"] <= 1 for p in predictions)
        assert result["epochs"] == 20 and result["lr"] == 0.01 and result["clipping_norm"] == 0.5
        assert result["noise_multiplier"] == 1.0 and result["delta"] == 1e-6 and result["accountant"] == "rdp"
        batches, sampling_q, accountant_q, expected_batch = privacy_loader_check(len(train))
        assert abs(sampling_q - 1 / math.ceil(len(train) / 256)) < 1e-15
        assert abs(result["sample_rate_q"] - sampling_q) < 1e-15
        assert result["optimizer_steps"] == 20 * batches and result["expected_batch_size"] == expected_batch
        accountant = RDPAccountant()
        for _ in range(result["optimizer_steps"]):
            accountant.step(noise_multiplier=1.0, sample_rate=accountant_q)
        epsilon = accountant.get_epsilon(delta=1e-6)
        assert abs(epsilon - result["achieved_epsilon"]) < 1e-10
        assert accountant.get_epsilon(delta=1e-6) == epsilon
        state = torch.load(OUT / result["model_checkpoint"], map_location="cpu", weights_only=True)
        assert all(torch.isfinite(tensor).all() for tensor in state.values())
        accuracy = sum(p["prediction"] == p["true_label"] for p in predictions) / len(predictions)
        assert abs(accuracy - result["evaluation"]["overall"]["accuracy"]) < 1e-12
        for group_id, name in enumerate(("Male", "Female")):
            group = [p for p in predictions if p["group_id"] == group_id]
            cm = [[sum(p["true_label"] == actual and p["prediction"] == predicted for p in group)
                   for predicted in (0, 1)] for actual in (0, 1)]
            recorded = result["evaluation"]["groups"][name]
            assert cm == recorded["confusion_matrix"]
            group_accuracy, missed = (cm[0][0] + cm[1][1]) / len(group), 100 * cm[1][0] / sum(cm[1])
            assert abs(group_accuracy - recorded["accuracy"]) < 1e-12
            assert abs(missed - recorded["missed_per_100"]) < 1e-12
            rows.append({"seed": seed, "group": name, "test_records": len(group), "actual_higher_income": sum(cm[1]),
                         "accuracy_pct": group_accuracy * 100, "missed_per_100": missed, "epsilon": epsilon})
        audit.append({"seed": seed, "train_records": len(train), "test_records": len(test),
                      "sampling_q": sampling_q, "accountant_q": accountant_q, "optimizer_steps": result["optimizer_steps"],
                      "expected_batch_size": expected_batch, "recorded_epsilon": result["achieved_epsilon"],
                      "recomputed_epsilon": epsilon, "checks_passed": True,
                      "model_sha256": sha(OUT / result["model_checkpoint"]),
                      "predictions_sha256": sha(OUT / result["predictions_file"]),
                      "initial_weights_sha256": pairing["initial_weights_sha256"]})
        results.append(result)
    summary = {"phase": 2, "seeds": list(range(5)), "delta": 1e-6,
               "overall_accuracy_pct": stats([100 * r["evaluation"]["overall"]["accuracy"] for r in results]),
               "groups": {name: {metric: stats([row[metric] for row in rows if row["group"] == name])
                                for metric in ("accuracy_pct", "missed_per_100")} for name in ("Male", "Female")},
               "per_run_epsilon": [{"seed": r["seed"], "epsilon": r["achieved_epsilon"]} for r in results],
               "software": phase1_summary["software"], "audit": audit,
               "source_sha256": {**phase1_summary["source_sha256"], str(Path(__file__).relative_to(ROOT)): sha(__file__)},
               "total_training_seconds": sum(r["elapsed_seconds"] for r in results)}
    (OUT / "phase2_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    with (OUT / "phase2_summary.csv").open("w", newline="", encoding="utf-8") as handle:
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
        ax.set_ylim(0, 106 if metric == "missed_per_100" else 100)
        ax.set_yticks(range(0, 101, 20))
        ax.set_title(title, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        for i, mean in enumerate(means):
            ax.text(i, mean + errors[i] + 1.2, f"{mean:.1f}", ha="center", fontsize=11)
    fig.suptitle("Adult with vanilla DP-SGD", fontsize=16)
    fig.text(0.5, 0.02, "Mean and standard error across five seeds. Each run: ε = 3.4078, δ = 0.000001.", ha="center", fontsize=9)
    fig.tight_layout(rect=[0, 0.06, 1, 0.92])
    for extension in ("png", "pdf"):
        fig.savefig(OUT / f"phase2_dpsgd.{extension}")
    plt.close(fig)
    table = ["| Group | Accuracy (%) | Missed per 100 actual higher-income records |", "|---|---:|---:|"]
    for name in names:
        acc, fnr = (summary["groups"][name][metric] for metric in ("accuracy_pct", "missed_per_100"))
        table.append(f"| {name} | {acc['mean']:.2f} ± {acc['se']:.2f} | {fnr['mean']:.2f} ± {fnr['se']:.2f} |")
    report = "# Adult Phase 2: review checkpoint\n\nCompleted DP-SGD seeds 0–4 with the existing `adult.py`, 20 epochs each. Values are mean ± standard error across seeds.\n\n"
    report += "\n".join(table) + f"\n\nOverall accuracy: **{summary['overall_accuracy_pct']['mean']:.2f}%**. Every run reports **ε = 3.4078 at δ = 10⁻⁶**; this is a per-run training budget, not a combined guarantee for all released models or raw-data preprocessing.\n\n"
    report += "The model misses almost all actual higher-income records despite retaining high overall accuracy for the female group. This is why the missed-prediction measure matters alongside accuracy. Formal baseline comparisons are reserved for Phase 3.\n\n"
    report += "Verified: unchanged data and initial weights from Phase 1; test IDs/labels/groups align across both phases; finite checkpoints; predictions reproduce all group metrics; reported privacy budgets reproduce from recorded steps and the installed accountant's sampling bound.\n\n"
    report += "Accounting note: for seeds 1 and 2, the library converts 93 nominal batches to 92 Poisson batches due to floating-point rounding. Actual sampling uses q=1/93; Opacus accounts conservatively at q=1/92. All five runs perform 1,840 steps. Both rates are recorded in `phase2_summary.json`; existing training behavior was retained.\n\n"
    report += "Published configuration: clipping C=0.5, gradient noise multiplier=1, SGD learning rate=0.01, RDP accounting. Training RNG is `seed + 2000`; Phase 1 used `seed + 1000`. Data, splits, and initial weights are paired; batch membership/random draws are not identical. Historical benchmark and full-pool preprocessing limitations remain as documented in the plan.\n\n"
    report += f"![Phase 2 results]({(OUT / 'phase2_dpsgd.png').as_posix()})\n\nStop after Phase 2 for user review. Phase 3 comparisons and Phase 4 Adult training have not been run.\n\n"
    report += "Reproduce from the repository root:\n\n```powershell\n.\\.venv\\Scripts\\python.exe experiments/adult/adult.py --phase 2 --all-seeds --skip-existing\n.\\.venv\\Scripts\\python.exe experiments/adult/review_phase2.py\n```\n"
    (OUT / "phase2_review.md").write_text(report, encoding="utf-8")
    print("Phase 2 prediction/pairing/checkpoint/accounting audit passed for five seeds.")
    print("\n".join(table))
    print("Per-run epsilon:", summary["per_run_epsilon"])


if __name__ == "__main__":
    main()

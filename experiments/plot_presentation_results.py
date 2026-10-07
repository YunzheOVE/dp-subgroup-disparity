"""Plot the saved five-run results for the presentation; no training is run."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "tmp" / "presentation_data_build"
ASSETS = BUILD / "assets"
METHODS = ["Without privacy", "DP-SGD", "Global-Adapt"]
COLORS = ["#94a3b8", "#2f64c3", "#23876d"]


def read_json(relative_path):
    return json.loads((ROOT / relative_path).read_text(encoding="utf-8"))


def draw_chart(categories, means, errors, ylabel, path, title=None, footnote=None):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 16})
    fig, ax = plt.subplots(figsize=(9.5, 5.6 if title else 4.75), dpi=200)
    fig.subplots_adjust(left=0.12, right=0.985, bottom=0.17, top=0.73 if title else 0.81)
    x = np.arange(len(categories))
    width = 0.23
    for i, (method, color, values, uncertainty) in enumerate(zip(METHODS, COLORS, means, errors)):
        bars = ax.bar(
            x + (i - 1) * width, values, width,
            color=color, label=method, yerr=uncertainty,
            error_kw={"elinewidth": 1.2, "capsize": 3, "capthick": 1.2},
        )
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2, value - 3,
                f"{value:.1f}", ha="center", va="top",
                color="white", fontsize=17, fontweight="bold",
            )
    ax.set_ylim(0, 104)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.set_xticks(x, categories, fontsize=19)
    ax.set_ylabel(ylabel, fontsize=17, labelpad=10)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color="#eef0f3", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["bottom", "left"]].set_linewidth(0.8)
    ax.legend(
        loc="lower center", bbox_to_anchor=(0.5, 1.025),
        ncol=3, frameon=False, fontsize=17,
        columnspacing=1.6, handlelength=1.5,
    )
    if title:
        fig.suptitle(title, y=0.97, fontsize=22)
    if footnote:
        fig.text(0.5, 0.035, footnote, ha="center", fontsize=11)
    fig.canvas.draw()
    # The long labels must remain inside the exported canvas.
    bounds = fig.bbox
    renderer = fig.canvas.get_renderer()
    for label in [ax.yaxis.label, *ax.get_xticklabels(), *ax.get_yticklabels()]:
        box = label.get_window_extent(renderer)
        assert box.x0 >= 0 and box.y0 >= 0 and box.x1 <= bounds.x1 and box.y1 <= bounds.y1
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def main():
    ASSETS.mkdir(parents=True, exist_ok=True)
    mnist = [read_json(f"results/mnist/phase{phase}_summary.json") for phase in [1, 2, 4]]
    mnist_means = [[run[f"digit_{digit}_accuracy_mean"] for digit in [2, 8]] for run in mnist]
    mnist_errors = [[run[f"digit_{digit}_accuracy_std"] / np.sqrt(len(run["seeds"])) for digit in [2, 8]] for run in mnist]
    categories = ["Digit 2", "Digit 8 (rare)"]
    mnist_png = ROOT / "results/mnist/mnist_accuracy_by_digit_presentation.png"
    draw_chart(
        categories, mnist_means, mnist_errors, "Correct predictions (%)", mnist_png,
        title="MNIST accuracy by digit",
        footnote="Mean ± SE across five runs · δ = 10⁻⁶ · ε: DP-SGD 5.90; Global-Adapt 5.91",
    )
    draw_chart(categories, mnist_means, mnist_errors, "Correct predictions (%)", ASSETS / "mnist_accuracy.png")

    adult = read_json("results/adult/summary.json")
    keys = ["phase1_nonprivate", "phase2_dpsgd", "phase4_global_adapt"]
    if keys[2] not in adult:
        keys[2] = next(key for key in adult if key.startswith("phase4"))
    adult_means = [[adult[key][f"{group}_accuracy"]["mean"] for group in ["male", "female"]] for key in keys]
    adult_errors = [[adult[key][f"{group}_accuracy"]["se"] for group in ["male", "female"]] for key in keys]
    missed_means = [[adult[key][f"{group}_missed_per_100"]["mean"] for group in ["male", "female"]] for key in keys]
    missed_errors = [[adult[key][f"{group}_missed_per_100"]["se"] for group in ["male", "female"]] for key in keys]
    draw_chart(["Male", "Female"], adult_means, adult_errors, "Correct predictions (%)", ASSETS / "adult_accuracy.png")
    draw_chart(["Male", "Female"], missed_means, missed_errors, "Missed cases per 100", ASSETS / "adult_missed.png")
    metadata = {
        "method_names": METHODS, "colors": COLORS, "error_bars": "standard error across five runs",
        "mnist": {"categories": categories, "means": mnist_means, "se": mnist_errors},
        "adult_accuracy": {"categories": ["Male", "Female"], "means": adult_means, "se": adult_errors},
        "adult_missed": {"categories": ["Male", "Female"], "means": missed_means, "se": missed_errors,
                         "denominator": "100 records actually earning above $50,000, within each group"},
    }
    (BUILD / "chart_data.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Created MNIST PNG: {mnist_png}")
    print("Created three slide chart assets from saved results.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generate the final recall comparison bar chart."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
METRICS_PATH = OUTPUT_DIR / "final_evaluation_metrics.json"
PLOT_PATH = OUTPUT_DIR / "final_recall_comparison.png"


def main() -> int:
    with open(METRICS_PATH, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    methods = data["methods"]
    labels = list(methods.keys())
    recall_5 = [methods[m]["recall@5"] * 100 for m in labels]
    recall_10 = [methods[m]["recall@10"] * 100 for m in labels]

    x = range(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar([i - width / 2 for i in x], recall_5, width, label="Recall@5")
    bars2 = ax.bar([i + width / 2 for i in x], recall_10, width, label="Recall@10")

    ax.set_ylabel("Recall (%)")
    ax.set_title("Final Retrieval Method Comparison (40-query evaluation)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.legend()
    ax.set_ylim(0, max(max(recall_5), max(recall_10)) * 1.2 + 5)

    for bar in bars1 + bars2:
        height = bar.get_height()
        ax.annotate(
            f"{height:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(PLOT_PATH, dpi=150)
    print(f"Saved comparison plot to {PLOT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Generate a bar chart comparing recall percentages across retrieval methods.

Reads ``outputs/baseline_metrics.json`` and ``outputs/new_methods_metrics.json``,
plots Recall@5 and Recall@10 as percentages, and saves the figure to
``outputs/recall_comparison.png``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
BASELINE_PATH = OUTPUTS_DIR / "baseline_metrics.json"
NEW_METHODS_PATH = OUTPUTS_DIR / "new_methods_metrics.json"
CHART_PATH = OUTPUTS_DIR / "recall_comparison.png"


def _load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _build_series() -> tuple[list[str], list[float], list[float]]:
    baseline = _load_json(BASELINE_PATH)
    new_methods = _load_json(NEW_METHODS_PATH)

    labels: list[str] = []
    recall_at_5: list[float] = []
    recall_at_10: list[float] = []

    label_map = {
        "dense_only": "Dense-only",
        "hybrid_rrf": "Hybrid RRF",
        "multi_index_routed": "Multi-index routed",
        "hnsw": "HNSW-only",
    }
    for key, meta in baseline["baselines"].items():
        labels.append(label_map.get(key, key))
        recall_at_5.append(meta["recall@5_pct"])
        recall_at_10.append(meta["recall@10_pct"])

    new_label_map = {
        "dense": "Dense",
        "bm25": "BM25",
        "tfidf": "TF-IDF",
        "rrf": "RRF",
        "aar": "AAR",
        "mmr": "MMR",
        "cross_encoder": "Cross-Encoder",
    }
    for key, meta in new_methods["methods"].items():
        labels.append(new_label_map.get(key, key))
        recall_at_5.append(meta["recall@5"] * 100)
        recall_at_10.append(meta["recall@10"] * 100)

    return labels, recall_at_5, recall_at_10


def main() -> int:
    if not BASELINE_PATH.exists():
        print(f"Missing baseline metrics: {BASELINE_PATH}", file=sys.stderr)
        return 1
    if not NEW_METHODS_PATH.exists():
        print(f"Missing new methods metrics: {NEW_METHODS_PATH}", file=sys.stderr)
        return 1

    labels, r5, r10 = _build_series()

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 0.9), 6))
    bars1 = ax.bar(x - width / 2, r5, width, label="Recall@5", color="steelblue")
    bars2 = ax.bar(x + width / 2, r10, width, label="Recall@10", color="darkorange")

    ax.set_ylabel("Recall (%)")
    ax.set_title("Retrieval Recall Comparison (40-query evaluation set)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.legend()
    ax.set_ylim(0, max(max(r5), max(r10)) * 1.2 + 1)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # Annotate bars with their values.
    for bars in (bars1, bars2):
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.annotate(
                    f"{height:.1f}%",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )

    fig.tight_layout()
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(CHART_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved recall comparison chart to {CHART_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

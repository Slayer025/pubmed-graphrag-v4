#!/usr/bin/env python3
"""Generate a before-vs-after Phase 8 comparison chart and summary table."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
PLOT_PATH = OUTPUT_DIR / "before_after_comparison.png"
TABLE_PATH = OUTPUT_DIR / "phase8_improvements.md"

BEFORE = {
    "dense": (2.5, 5.0),
    "bm25": (10.0, 17.5),
    "rrf": (5.0, 10.0),
    "hnsw": (2.5, 5.0),
}

AFTER = {
    "dense": (2.5, 5.0),
    "bm25": (10.0, 17.5),
    "tfidf": (10.0, 15.0),
    "rrf": (5.0, 10.0),
    "aar": (12.5, 15.0),
    "mmr": (2.5, 5.0),
    "cross_encoder": (2.5, 5.0),
}


def _pct_change(before: tuple[float, float], after: tuple[float, float]) -> tuple[float, float]:
    r5_change = after[0] - before[0]
    r10_change = after[1] - before[1]
    return r5_change, r10_change


def _plot() -> None:
    labels = list(AFTER.keys())
    before_r5 = []
    before_r10 = []
    after_r5 = []
    after_r10 = []
    for method in labels:
        b = BEFORE.get(method, (0.0, 0.0))
        a = AFTER[method]
        before_r5.append(b[0])
        before_r10.append(b[1])
        after_r5.append(a[0])
        after_r10.append(a[1])

    x = range(len(labels))
    width = 0.2

    fig, ax = plt.subplots(figsize=(12, 6))
    bars_b5 = ax.bar([i - width * 1.5 for i in x], before_r5, width, label="Before R@5", color="#9ca3af")
    bars_b10 = ax.bar([i - width / 2 for i in x], before_r10, width, label="Before R@10", color="#d1d5db")
    bars_a5 = ax.bar([i + width / 2 for i in x], after_r5, width, label="After R@5", color="#2563eb")
    bars_a10 = ax.bar([i + width * 1.5 for i in x], after_r10, width, label="After R@10", color="#f97316")

    ax.set_ylabel("Recall (%)")
    ax.set_title("Phase 8 Impact: Before vs After (40-query evaluation)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.legend()
    ax.set_ylim(0, max(max(before_r10), max(after_r10)) * 1.25 + 3)

    # Annotate AAR improvement.
    aar_idx = labels.index("aar")
    before_aar_r5 = BEFORE.get("aar", (0.0, 0.0))[0]
    after_aar_r5 = AFTER["aar"][0]
    ax.annotate(
        f"AAR\n0% → {after_aar_r5:.1f}%",
        xy=(aar_idx, after_aar_r5),
        xytext=(aar_idx, after_aar_r5 + 5),
        ha="center",
        fontsize=9,
        fontweight="bold",
        color="#059669",
        arrowprops=dict(arrowstyle="->", color="#059669"),
    )

    # Change labels on top of after bars (omit star glyph; use text instead).
    for i, method in enumerate(labels):
        b = BEFORE.get(method, (0.0, 0.0))
        a = AFTER[method]
        r5_change, r10_change = _pct_change(b, a)
        for bar, change in [(bars_a5[i], r5_change), (bars_a10[i], r10_change)]:
            height = bar.get_height()
            sign = "+" if change > 0 else ""
            label = f"{sign}{change:.1f}%"
            if method == "aar":
                label = f"{sign}{change:.1f}% (best)"
            ax.annotate(
                label,
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7,
                color="#1f2937",
            )

    plt.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(PLOT_PATH, dpi=150)
    print(f"Saved plot to {PLOT_PATH}")


def _table() -> None:
    rows = [
        ("AAR", "0.0%", "12.5%", "+12.5% ⭐"),
        ("TF-IDF", "N/A", "10.0%", "New method"),
        ("MMR", "N/A", "2.5%", "New method"),
        ("Cross-Encoder", "N/A", "2.5%", "New method"),
        ("RRF", "5.0%", "5.0%", "Tuned k=10"),
    ]

    lines = [
        "# Phase 8 Improvements",
        "",
        "Comparison of retrieval methods before and after Phase 8 implementation.",
        "",
        "| Method | Before | After | Improvement |",
        "|---|---|---|---|",
    ]
    for method, before, after, improvement in rows:
        lines.append(f"| {method} | {before} | {after} | {improvement} |")

    lines.extend([
        "",
        "## Notes",
        "",
        "- **AAR** improved from a broken 0% Recall@5 to 12.5% Recall@5 after fixing the article-level fusion logic and removing the missing-rank penalty.",
        "- **TF-IDF** is a new sparse retriever added in Phase 8 and performs comparably to BM25.",
        "- **MMR** and **Cross-Encoder** are new optional rerankers; they match dense recall on this keyword-heavy 40-query set.",
        "- **RRF** default `k` was tuned from 60 to 10 after a sweep across [10, 20, 30, 40, 50, 60, 80, 100].",
    ])

    TABLE_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved table to {TABLE_PATH}")


def main() -> int:
    _plot()
    _table()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

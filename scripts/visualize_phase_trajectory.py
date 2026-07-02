#!/usr/bin/env python3
"""Visualize performance trajectory across phases for video demo."""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Create output directory
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# Phase 2 Baseline Data
phase2_data = {
    "Dense": 2.5,
    "BM25": 10.0,
    "Hybrid RRF": 5.0,
}

# Phase 8 Improvements Data
phase8_data = {
    "Dense": 2.5,  # unchanged
    "BM25": 10.0,  # unchanged
    "TF-IDF": 10.0,  # new
    "Hybrid RRF": 5.0,  # tuned
    "AAR": 12.5,  # fixed from 0% *
    "MMR": 2.5,  # new
    "Cross-Encoder": 2.5,  # new
}

# Create figure with subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Phase 2 Chart
methods_p2 = list(phase2_data.keys())
values_p2 = list(phase2_data.values())
colors_p2 = ['#3498db', '#e74c3c', '#9b59b6']

bars1 = ax1.bar(methods_p2, values_p2, color=colors_p2, edgecolor='black', linewidth=1.5)
ax1.set_ylabel('Recall@5 (%)', fontsize=12, fontweight='bold')
ax1.set_title('Phase 2 Baseline', fontsize=14, fontweight='bold', pad=20)
ax1.set_ylim(0, 15)
ax1.grid(axis='y', alpha=0.3, linestyle='--')

# Add value labels on bars
for bar, val in zip(bars1, values_p2):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
             f'{val}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

# Phase 8 Chart
methods_p8 = list(phase8_data.keys())
values_p8 = list(phase8_data.values())
colors_p8 = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6', '#f39c12', '#1abc9c', '#e67e22']

# Highlight AAR as the best performer
highlight_colors = []
for method in methods_p8:
    if method == "AAR":
        highlight_colors.append('#f1c40f')  # Gold for best performer
    else:
        highlight_colors.append('#95a5a6')  # Gray for others

bars2 = ax2.bar(methods_p8, values_p8, color=highlight_colors, edgecolor='black', linewidth=1.5)
ax2.set_ylabel('Recall@5 (%)', fontsize=12, fontweight='bold')
ax2.set_title('Phase 8 Improvements', fontsize=14, fontweight='bold', pad=20)
ax2.set_ylim(0, 15)
ax2.grid(axis='y', alpha=0.3, linestyle='--')
ax2.tick_params(axis='x', rotation=45)

# Add value labels on bars
for bar, val in zip(bars2, values_p8):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
             f'{val}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Add star annotation for AAR
aar_idx = methods_p8.index("AAR")
ax2.annotate('* Best Performer',
             xy=(aar_idx, values_p8[aar_idx]),
             xytext=(aar_idx + 0.8, values_p8[aar_idx] + 2),
             arrowprops=dict(arrowstyle='->', color='black', lw=2),
             fontsize=12, fontweight='bold', color='#d35400')

# Add main title
fig.suptitle('PubMed GraphRAG: Performance Trajectory Across Phases',
             fontsize=16, fontweight='bold', y=0.98)

# Add subtitle
fig.text(0.5, 0.92, 'Recall@5 Comparison: Phase 2 Baseline → Phase 8 Improvements',
         ha='center', fontsize=12, style='italic')

plt.tight_layout(rect=[0, 0, 1, 0.90])

# Save the figure
output_path = OUTPUT_DIR / "phase_trajectory.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"[OK] Saved trajectory chart to: {output_path}")

# Also create a combined timeline chart
fig2, ax3 = plt.subplots(figsize=(14, 8))

# Timeline data
phases = ['Phase 2\nBaseline', 'Phase 8\nImprovements']
methods_timeline = {
    'Dense': [2.5, 2.5],
    'BM25': [10.0, 10.0],
    'TF-IDF': [0, 10.0],  # 0 in phase 2, 10 in phase 8
    'AAR': [0, 12.5],  # 0 in phase 2, 12.5 in phase 8
}

x = np.arange(len(phases))
width = 0.2
colors_timeline = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']

for i, (method, values) in enumerate(methods_timeline.items()):
    offset = (i - len(methods_timeline)/2 + 0.5) * width
    bars = ax3.bar(x + offset, values, width, label=method,
                   color=colors_timeline[i], edgecolor='black', linewidth=1.5)

    # Add value labels
    for bar, val in zip(bars, values):
        if val > 0:
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f'{val}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

ax3.set_ylabel('Recall@5 (%)', fontsize=12, fontweight='bold')
ax3.set_title('Performance Evolution: Phase 2 → Phase 8', fontsize=14, fontweight='bold', pad=20)
ax3.set_xticks(x)
ax3.set_xticklabels(phases, fontsize=12, fontweight='bold')
ax3.set_ylim(0, 15)
ax3.legend(loc='upper left', fontsize=10, framealpha=0.9)
ax3.grid(axis='y', alpha=0.3, linestyle='--')

# Add annotation for AAR improvement
ax3.annotate('AAR: 0% → 12.5%\n* Major Breakthrough',
             xy=(1, 12.5),
             xytext=(1.3, 13.5),
             arrowprops=dict(arrowstyle='->', color='black', lw=2),
             fontsize=11, fontweight='bold', color='#d35400',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))

plt.tight_layout()

# Save timeline chart
output_path2 = OUTPUT_DIR / "phase_evolution.png"
plt.savefig(output_path2, dpi=300, bbox_inches='tight')
print(f"[OK] Saved evolution chart to: {output_path2}")

print("\n[OK] Visualization complete!")
print(f"  - Phase trajectory: {output_path}")
print(f"  - Phase evolution: {output_path2}")

# %% [markdown]
# # Visualization
#
# Generate all figures for the report.

# %% Setup
import sys
sys.path.insert(0, "..")
import json
import numpy as np
from pathlib import Path
from src.config import RESULTS_DIR
from src.visualization import (
    plot_radar_chart,
    plot_degradation_curves,
    plot_error_propagation,
    plot_confusion_matrices,
)

OUTPUT_DIR = Path("../report/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## 1. Radar Chart: Task Performance Comparison

# %%
# Replace with actual scores from evaluation
categories = ["Summarization", "Sentiment", "Keywords", "Intent"]
cascade_scores = [0.72, 0.80, 0.65, 0.75]  # placeholder
direct_scores = [0.78, 0.82, 0.70, 0.76]    # placeholder

fig = plot_radar_chart(categories, cascade_scores, direct_scores,
                       title="Speech Understanding: Cascade vs Direct")
fig.savefig(OUTPUT_DIR / "radar_chart.png", dpi=150, bbox_inches="tight")
fig.savefig(OUTPUT_DIR / "radar_chart.pdf", bbox_inches="tight")
print("Radar chart saved.")

# %% [markdown]
# ## 2. Degradation Curves

# %%
snr_levels = ["Clean", "20dB", "10dB", "0dB"]
cascade_degradation = [0.80, 0.78, 0.65, 0.45]  # placeholder
direct_degradation = [0.82, 0.81, 0.75, 0.62]    # placeholder

fig = plot_degradation_curves(
    "ROUGE-L F1", snr_levels,
    cascade_degradation, direct_degradation,
    title="Summarization Quality vs Babble Noise Level"
)
fig.savefig(OUTPUT_DIR / "degradation_curves.png", dpi=150, bbox_inches="tight")
fig.savefig(OUTPUT_DIR / "degradation_curves.pdf", bbox_inches="tight")
print("Degradation curves saved.")

# %% [markdown]
# ## 3. Error Propagation Scatter

# %%
wer_vals = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35]  # placeholder
summary_scores = [0.85, 0.82, 0.78, 0.72, 0.65, 0.58, 0.50, 0.42]  # placeholder

fig = plot_error_propagation(wer_vals, summary_scores,
                             task_name="Summarization")
fig.savefig(OUTPUT_DIR / "error_propagation.png", dpi=150, bbox_inches="tight")
fig.savefig(OUTPUT_DIR / "error_propagation.pdf", bbox_inches="tight")
print("Error propagation plot saved.")

# %% [markdown]
# ## 4. Confusion Matrices (Sentiment)

# %%
cascade_cm = np.array([[15, 2, 1], [3, 12, 2], [1, 1, 13]])  # placeholder
direct_cm = np.array([[16, 1, 1], [2, 13, 1], [1, 0, 15]])    # placeholder
labels = ["Positive", "Negative", "Neutral"]

fig = plot_confusion_matrices(cascade_cm, direct_cm, labels)
fig.savefig(OUTPUT_DIR / "confusion_matrices.png", dpi=150, bbox_inches="tight")
fig.savefig(OUTPUT_DIR / "confusion_matrices.pdf", bbox_inches="tight")
print("Confusion matrices saved.")

# %% [markdown]
# ## 5. Cost-Latency Bar Chart

# %%
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Latency
methods = ["Cascade\n(ASR+LLM)", "Direct\n(GPT-4o Audio)"]
latencies = [2.3, 3.1]
ax1.bar(methods, latencies, color=["#2563EB", "#DC2626"], alpha=0.85)
ax1.set_ylabel("Latency (seconds)", fontsize=12)
ax1.set_title("Average Inference Latency", fontsize=13, fontweight="bold")

# Cost
costs = [0.004, 0.018]
ax2.bar(methods, costs, color=["#2563EB", "#DC2626"], alpha=0.85)
ax2.set_ylabel("Cost per Task (USD)", fontsize=12)
ax2.set_title("API Cost per Task Call", fontsize=13, fontweight="bold")

plt.suptitle("Cost-Latency Comparison", fontsize=15, fontweight="bold")
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "cost_latency.png", dpi=150, bbox_inches="tight")
fig.savefig(OUTPUT_DIR / "cost_latency.pdf", bbox_inches="tight")
print("Cost-latency chart saved.")

print("\nAll visualizations saved to report/figures/")

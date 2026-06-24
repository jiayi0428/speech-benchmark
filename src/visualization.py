"""Visualization functions for the speech benchmark."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional, Tuple
from matplotlib.figure import Figure

sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1.2)

COLORS = {
    "cascade": "#2563EB",
    "direct": "#DC2626",
    "optional": "#7C3AED",
}


def plot_radar_chart(
    categories: List[str],
    cascade_scores: List[float],
    direct_scores: List[float],
    title: str = "Task Performance Comparison",
    optional_scores: Optional[List[float]] = None,
    optional_label: str = "Qwen2-Audio",
) -> Figure:
    """Plot a radar/spider chart comparing architectures across tasks."""
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    cascade_norm = cascade_scores + cascade_scores[:1]
    direct_norm = direct_scores + direct_scores[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    ax.fill(angles, cascade_norm, alpha=0.25, color=COLORS["cascade"])
    ax.plot(angles, cascade_norm, "o-", linewidth=2, color=COLORS["cascade"],
            label="Cascade (ASR+LLM)")

    ax.fill(angles, direct_norm, alpha=0.25, color=COLORS["direct"])
    ax.plot(angles, direct_norm, "o-", linewidth=2, color=COLORS["direct"],
            label="Direct (GPT-4o Audio)")

    if optional_scores is not None:
        opt_norm = optional_scores + optional_scores[:1]
        ax.fill(angles, opt_norm, alpha=0.15, color=COLORS["optional"])
        ax.plot(angles, opt_norm, "o-", linewidth=2, color=COLORS["optional"],
                label=optional_label)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=8)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=10)
    plt.tight_layout()
    return fig


def plot_degradation_curves(
    metric_name: str,
    x_labels: List[str],
    cascade_scores: List[float],
    direct_scores: List[float],
    title: Optional[str] = None,
) -> Figure:
    """Plot grouped bar chart showing performance degradation by noise level."""
    if title is None:
        title = f"Robustness: {metric_name} under Acoustic Degradation"

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(x_labels))
    width = 0.35

    bars1 = ax.bar(x - width / 2, cascade_scores, width,
                   label="Cascade (ASR+LLM)", color=COLORS["cascade"],
                   alpha=0.85, edgecolor="white", linewidth=0.5)
    bars2 = ax.bar(x + width / 2, direct_scores, width,
                   label="Direct (GPT-4o Audio)", color=COLORS["direct"],
                   alpha=0.85, edgecolor="white", linewidth=0.5)

    for bar in bars1:
        h = bar.get_height()
        ax.annotate(f"{h:.2f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points",
                     ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        h = bar.get_height()
        ax.annotate(f"{h:.2f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points",
                     ha="center", va="bottom", fontsize=8)

    ax.set_xlabel("Noise Condition", fontsize=12)
    ax.set_ylabel(metric_name, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=30, ha="right")
    ax.legend(fontsize=11)
    ax.set_ylim(0, max(max(cascade_scores), max(direct_scores)) * 1.15)
    plt.tight_layout()
    return fig


def plot_error_propagation(
    wer_values: List[float],
    task_scores: List[float],
    task_name: str = "Summarization",
    title: Optional[str] = None,
) -> Figure:
    """Plot WER vs downstream task score with regression line."""
    if title is None:
        title = f"ASR Error Propagation: WER vs {task_name} Score"

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(wer_values, task_scores, alpha=0.6, s=60,
               color=COLORS["cascade"], edgecolors="white",
               linewidth=0.5, zorder=5)

    if len(wer_values) > 1:
        coeffs = np.polyfit(wer_values, task_scores, 1)
        poly_fn = np.poly1d(coeffs)
        x_line = np.linspace(min(wer_values), max(wer_values), 100)
        r_val = np.corrcoef(wer_values, task_scores)[0, 1]
        ax.plot(x_line, poly_fn(x_line), "--", color="#991B1B", linewidth=2,
                label=f"Trend (r={r_val:.3f})")
        ax.legend(fontsize=10)

    ax.set_xlabel("Word Error Rate (WER)", fontsize=12)
    ax.set_ylabel(f"{task_name} Score", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlim(0, max(wer_values) * 1.1)
    plt.tight_layout()
    return fig


def plot_confusion_matrices(
    cascade_cm: np.ndarray,
    direct_cm: np.ndarray,
    labels: List[str],
    title: str = "Sentiment Confusion Matrix Comparison",
) -> Figure:
    """Plot side-by-side confusion matrices for cascade and direct."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    sns.heatmap(cascade_cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax1,
                cbar=False, linewidths=0.5)
    ax1.set_title("Cascade (ASR+LLM)", fontsize=13, fontweight="bold")
    ax1.set_ylabel("True", fontsize=11)
    ax1.set_xlabel("Predicted", fontsize=11)

    sns.heatmap(direct_cm, annot=True, fmt="d", cmap="Reds",
                xticklabels=labels, yticklabels=labels, ax=ax2,
                cbar=False, linewidths=0.5)
    ax2.set_title("Direct (GPT-4o Audio)", fontsize=13, fontweight="bold")
    ax2.set_ylabel("True", fontsize=11)
    ax2.set_xlabel("Predicted", fontsize=11)

    fig.suptitle(title, fontsize=15, fontweight="bold")
    plt.tight_layout()
    return fig

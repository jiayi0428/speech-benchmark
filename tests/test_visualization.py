"""Tests for visualization module."""
import numpy as np
from pathlib import Path
from src.visualization import (
    plot_radar_chart,
    plot_degradation_curves,
    plot_error_propagation,
)


def test_plot_radar_chart_returns_figure():
    categories = ["Summary", "Sentiment", "Keywords", "Intent"]
    cascade_scores = [0.75, 0.82, 0.68, 0.79]
    direct_scores = [0.80, 0.85, 0.72, 0.77]
    fig = plot_radar_chart(categories, cascade_scores, direct_scores)
    assert hasattr(fig, "savefig")


def test_plot_degradation_curves_returns_figure():
    snr_values = ["Clean", "20dB", "10dB", "0dB"]
    cascade_line = [0.80, 0.78, 0.65, 0.45]
    direct_line = [0.82, 0.81, 0.75, 0.62]
    fig = plot_degradation_curves(
        "Summarization", snr_values, cascade_line, direct_line
    )
    assert hasattr(fig, "savefig")


def test_plot_error_propagation_returns_figure():
    wer_values = [0.0, 0.1, 0.2, 0.3, 0.4]
    scores = [0.85, 0.80, 0.72, 0.60, 0.50]
    fig = plot_error_propagation(wer_values, scores)
    assert hasattr(fig, "savefig")

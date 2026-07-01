"""Compare Cascade and Direct summarization under deterministic white noise."""
from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np

from evaluate_white_noise import rouge_l


ROOT = Path(__file__).resolve().parent
RESULT_DIR = ROOT / "data" / "results" / "white_noise_v1"
CASCADE_INPUT = RESULT_DIR / "cascade_summary_raw.json"
DIRECT_INPUT = RESULT_DIR / "direct_raw.jsonl"
GROUND_TRUTH = ROOT / "data" / "ground_truth.json"
SCORES_OUTPUT = RESULT_DIR / "path_comparison_scores.csv"
SUMMARY_OUTPUT = RESULT_DIR / "path_comparison_summary.json"
FIGURE_OUTPUT = (
    RESULT_DIR / "figures" / "cascade_vs_direct_white_noise.png"
)

CONDITIONS = ["clean", "white_20db", "white_10db", "white_0db"]
LABELS = ["Clean", "20dB", "10dB", "0dB"]
CASCADE_CONDITION_MAP = {
    "clean": "clean",
    "white_20dB": "white_20db",
    "white_10dB": "white_10db",
    "white_0dB": "white_0db",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def bootstrap_mean_ci(
    values: list[float],
    *,
    seed: int,
    samples: int = 20_000,
) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(samples, len(array)))
    bootstrap_means = array[indices].mean(axis=1)
    low, high = np.quantile(bootstrap_means, [0.025, 0.975])
    return [float(low), float(high)]


def load_cascade() -> list[dict[str, Any]]:
    payload = json.loads(CASCADE_INPUT.read_text(encoding="utf-8"))
    records = []
    for group in payload.values():
        if not isinstance(group, list):
            continue
        for source in group:
            condition = CASCADE_CONDITION_MAP.get(source["condition"])
            if condition is None:
                raise ValueError(
                    f"Unknown Cascade condition: {source['condition']}"
                )
            records.append(
                {
                    "pipeline": "cascade",
                    "sample": source["sample"],
                    "condition": condition,
                    "output": source["output"],
                    "latency_seconds": float(source["latency"]),
                }
            )
    return records


def load_direct() -> list[dict[str, Any]]:
    return [
        {
            "pipeline": "direct",
            "sample": record["sample"],
            "condition": record["condition"],
            "output": record["output"],
            "latency_seconds": float(record["latency_seconds"]),
        }
        for record in read_jsonl(DIRECT_INPUT)
        if record.get("status") == "success"
        and record.get("task") == "summarization"
    ]


def validate(
    records: list[dict[str, Any]],
    samples: list[str],
) -> None:
    expected = {
        (pipeline, sample, condition)
        for pipeline in ["cascade", "direct"]
        for sample in samples
        for condition in CONDITIONS
    }
    actual = {
        (record["pipeline"], record["sample"], record["condition"])
        for record in records
    }
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"Result-key mismatch; missing={missing}, extra={extra}")
    if len(records) != len(actual):
        raise ValueError("Duplicate pipeline/sample/condition records")


def write_figure(summary: dict[str, Any]) -> None:
    os.environ.setdefault(
        "MPLCONFIGDIR",
        str(RESULT_DIR / ".matplotlib"),
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    colors = {"cascade": "#2563eb", "direct": "#dc2626"}
    display = {"cascade": "Cascade", "direct": "Direct (Qwen)"}

    for pipeline in ["cascade", "direct"]:
        quality = [
            summary["means"][pipeline]["rouge_l"][condition]
            for condition in CONDITIONS
        ]
        latency = [
            summary["means"][pipeline]["latency_seconds"][condition]
            for condition in CONDITIONS
        ]
        axes[0].plot(
            LABELS,
            quality,
            marker="o",
            linewidth=2,
            color=colors[pipeline],
            label=display[pipeline],
        )
        axes[1].plot(
            LABELS,
            latency,
            marker="o",
            linewidth=2,
            color=colors[pipeline],
            label=display[pipeline],
        )

    axes[0].set_title("Summarization quality")
    axes[0].set_ylabel("ROUGE-L")
    axes[0].set_ylim(0.35, 0.50)
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].set_title("Recorded inference latency")
    axes[1].set_ylabel("Seconds per summary")
    axes[1].set_ylim(bottom=0)
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.suptitle("Cascade vs Direct under white noise (N=8)")
    fig.tight_layout()
    fig.savefig(FIGURE_OUTPUT, dpi=180)
    plt.close(fig)


def main() -> None:
    if not CASCADE_INPUT.exists():
        raise FileNotFoundError(CASCADE_INPUT)
    ground_truth = json.loads(GROUND_TRUTH.read_text(encoding="utf-8"))
    samples = sorted(ground_truth)
    records = load_cascade() + load_direct()
    validate(records, samples)

    rows = []
    by_key = {}
    grouped_scores: dict[tuple[str, str], list[float]] = defaultdict(list)
    grouped_latency: dict[tuple[str, str], list[float]] = defaultdict(list)
    for record in records:
        score = rouge_l(
            ground_truth[record["sample"]]["summary"],
            record["output"],
        )
        row = {
            "pipeline": record["pipeline"],
            "sample": record["sample"],
            "condition": record["condition"],
            "rouge_l": score,
            "latency_seconds": record["latency_seconds"],
        }
        rows.append(row)
        by_key[
            (record["pipeline"], record["sample"], record["condition"])
        ] = row
        grouped_scores[(record["pipeline"], record["condition"])].append(
            score
        )
        grouped_latency[(record["pipeline"], record["condition"])].append(
            record["latency_seconds"]
        )

    with SCORES_OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(
            sorted(
                rows,
                key=lambda row: (
                    row["condition"],
                    row["sample"],
                    row["pipeline"],
                ),
            )
        )

    means: dict[str, dict[str, dict[str, float]]] = {}
    changes: dict[str, dict[str, float]] = {}
    for pipeline in ["cascade", "direct"]:
        score_means = {
            condition: mean(grouped_scores[(pipeline, condition)])
            for condition in CONDITIONS
        }
        latency_means = {
            condition: mean(grouped_latency[(pipeline, condition)])
            for condition in CONDITIONS
        }
        means[pipeline] = {
            "rouge_l": score_means,
            "latency_seconds": latency_means,
        }
        changes[pipeline] = {
            condition: score_means[condition] - score_means["clean"]
            for condition in CONDITIONS
            if condition != "clean"
        }

    differences = {}
    confidence_intervals = {}
    wins = {}
    latency_ratios = {}
    for condition_index, condition in enumerate(CONDITIONS):
        paired_differences = []
        direct_wins = cascade_wins = ties = 0
        for sample in samples:
            cascade_score = by_key[
                ("cascade", sample, condition)
            ]["rouge_l"]
            direct_score = by_key[
                ("direct", sample, condition)
            ]["rouge_l"]
            difference = direct_score - cascade_score
            paired_differences.append(difference)
            if abs(difference) < 1e-12:
                ties += 1
            elif difference > 0:
                direct_wins += 1
            else:
                cascade_wins += 1
        differences[condition] = mean(paired_differences)
        confidence_intervals[condition] = bootstrap_mean_ci(
            paired_differences,
            seed=42 + condition_index,
        )
        wins[condition] = {
            "direct": direct_wins,
            "cascade": cascade_wins,
            "ties": ties,
        }
        latency_ratios[condition] = (
            means["cascade"]["latency_seconds"][condition]
            / means["direct"]["latency_seconds"][condition]
        )

    summary = {
        "experiment_id": "white_noise_v1_path_comparison",
        "scope": "summarization_only",
        "sample_count": len(samples),
        "result_count": len(rows),
        "conditions": CONDITIONS,
        "means": means,
        "rouge_l_change_from_clean": changes,
        "direct_minus_cascade_rouge_l": differences,
        "paired_bootstrap_95_ci": confidence_intervals,
        "sample_wins": wins,
        "recorded_latency_ratio_cascade_over_direct": latency_ratios,
        "caveats": [
            "The Cascade file contains summarization outputs only.",
            "Latency values came from separate runs and may include different "
            "hardware, warm-up, and model-loading effects.",
            "With N=8, descriptive trends are not statistically conclusive.",
        ],
    }
    SUMMARY_OUTPUT.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_figure(summary)
    print(f"Compared {len(rows)} results")
    print(f"Summary: {SUMMARY_OUTPUT}")
    print(f"Figure: {FIGURE_OUTPUT}")


if __name__ == "__main__":
    main()

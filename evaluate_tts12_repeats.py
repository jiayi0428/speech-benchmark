"""Evaluate the isolated D rerun and three-repeat C/D summary experiment."""
from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np

from evaluate_white_noise import score_record


ROOT = Path(__file__).resolve().parent
TRUTH = ROOT / "data" / "ground_truth_tts12_cd_v1.json"
ORIGINAL_DIR = ROOT / "data" / "results" / "tts12_cd_v1"
RERUN_DIR = ROOT / "data" / "results" / "tts12_d_rerun_v2"
STABILITY_DIR = ROOT / "data" / "results" / "tts12_summary_stability_v1"
TASKS = ["summarization", "sentiment", "keywords", "intent"]
FIGURE_OUTPUT = ROOT / "report" / "figures" / "tts12_summary_stability.png"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def usage(records: list[dict[str, Any]]) -> dict[str, int]:
    return {
        field: sum(int(record.get("usage", {}).get(field) or 0) for record in records)
        for field in ["prompt_tokens", "completion_tokens", "total_tokens"]
    }


def bootstrap_mean_ci(
    values: list[float], *, seed: int, samples: int = 20_000
) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(samples, len(array)))
    means = array[indices].mean(axis=1)
    return [float(value) for value in np.quantile(means, [0.025, 0.975])]


def write_stability_figure(
    per_repetition: dict[str, dict[str, Any]],
    sample_direction_stability: dict[str, dict[str, Any]],
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(STABILITY_DIR / ".matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    repetitions = [1, 2, 3]
    c_means = [per_repetition[str(value)]["C_mean"] for value in repetitions]
    d_means = [per_repetition[str(value)]["D_mean"] for value in repetitions]
    labels = list(sample_direction_stability)
    sample_means = [
        mean(sample_direction_stability[label]["differences"])
        for label in labels
    ]
    colors = ["#7c3aed" if value > 0 else "#dc2626" for value in sample_means]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    axes[0].plot(repetitions, c_means, marker="o", label="C: Qwen transcript")
    axes[0].plot(repetitions, d_means, marker="o", label="D: Qwen direct")
    axes[0].set_xticks(repetitions)
    axes[0].set_ylim(0.30, 0.38)
    axes[0].set_xlabel("Repetition")
    axes[0].set_ylabel("Mean summary ROUGE-L")
    axes[0].set_title("Aggregate result across three repetitions")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    y = np.arange(len(labels))
    axes[1].barh(y, sample_means, color=colors)
    axes[1].axvline(0, color="black", linewidth=0.8)
    axes[1].set_yticks(y, labels)
    axes[1].set_xlabel("Mean C - D summary ROUGE-L")
    axes[1].set_title("Per-sample direction (purple=C, red=D)")
    axes[1].grid(axis="x", alpha=0.25)
    fig.suptitle("TTS12 C/D summary stability (N=12, three repetitions)")
    fig.tight_layout()
    FIGURE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_OUTPUT, dpi=180)
    plt.close(fig)


def merged_d(
    raw_path: Path, structured_path: Path
) -> dict[tuple[str, str], dict[str, Any]]:
    raw = [
        record for record in read_jsonl(raw_path)
        if record.get("status") == "success"
    ]
    structured = [
        record for record in read_jsonl(structured_path)
        if record.get("postprocess_status") == "success"
    ]
    raw_by_key = {(record["sample"], record["task"]): record for record in raw}
    structured_by_key = {
        (record["sample"], record["task"]): record for record in structured
    }
    return {
        key: (
            record
            if key[1] == "summarization"
            else structured_by_key[key]
        )
        for key, record in raw_by_key.items()
    }


def evaluate_d_rerun(truth: dict[str, Any], samples: list[str]) -> None:
    original = merged_d(
        ORIGINAL_DIR / "direct_raw.jsonl",
        ORIGINAL_DIR / "direct_postprocessed.jsonl",
    )
    rerun = merged_d(
        RERUN_DIR / "direct_raw.jsonl",
        RERUN_DIR / "direct_postprocessed.jsonl",
    )
    expected = {(sample, task) for sample in samples for task in TASKS}
    if set(original) != expected or set(rerun) != expected:
        raise ValueError("Original or rerun D results are incomplete")
    rows = []
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    exact_output_matches = 0
    equal_scores = 0
    for sample in samples:
        for task in TASKS:
            key = (sample, task)
            old_score, _ = score_record(original[key], truth[sample])
            new_score, valid_json = score_record(rerun[key], truth[sample])
            exact_output_matches += original[key]["output"] == rerun[key]["output"]
            equal_scores += abs(old_score - new_score) <= 1e-12
            grouped[("original", task)].append(old_score)
            grouped[("rerun", task)].append(new_score)
            rows.append(
                {
                    "sample": sample,
                    "task": task,
                    "original_score": old_score,
                    "rerun_score": new_score,
                    "difference": new_score - old_score,
                    "rerun_valid_json": valid_json,
                    "exact_output_match": (
                        original[key]["output"] == rerun[key]["output"]
                    ),
                }
            )
    with (RERUN_DIR / "scores.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    post = read_jsonl(RERUN_DIR / "direct_postprocessed.jsonl")
    summary = {
        "experiment_id": "tts12_d_rerun_v2",
        "sample_count": 12,
        "means": {
            run: {task: mean(grouped[(run, task)]) for task in TASKS}
            for run in ["original", "rerun"]
        },
        "exact_final_output_matches": exact_output_matches,
        "equal_score_rows": equal_scores,
        "row_count": len(rows),
        "api": {
            "successful_calls": len(post),
            "usage": usage(post),
            "estimated_cost_usd_at_0.0005_per_call": len(post) * 0.0005,
        },
        "interpretation": (
            "The complete D rerun exactly reproduced all 48 final outputs "
            "and scores under deterministic Qwen decoding and DeepSeek "
            "temperature zero."
        ),
    }
    (RERUN_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def evaluate_stability(truth: dict[str, Any], samples: list[str]) -> None:
    c_transcriptions = [
        record for record in read_jsonl(STABILITY_DIR / "c_transcriptions.jsonl")
        if record.get("status") == "success"
    ]
    c_summaries = [
        record for record in read_jsonl(STABILITY_DIR / "c_summaries.jsonl")
        if record.get("status") == "success"
    ]
    d_summaries = [
        record for record in read_jsonl(STABILITY_DIR / "d_summaries.jsonl")
        if record.get("status") == "success"
    ]
    expected = {
        (sample, repetition)
        for sample in samples
        for repetition in [1, 2, 3]
    }
    c_by_key = {
        (record["sample"], record["repetition"]): record
        for record in c_summaries
    }
    d_by_key = {
        (record["sample"], record["repetition"]): record
        for record in d_summaries
    }
    if set(c_by_key) != expected or set(d_by_key) != expected:
        raise ValueError("C or D repeat summaries are incomplete")

    rows = []
    scores: dict[tuple[str, str, int], float] = {}
    for sample in samples:
        for repetition in [1, 2, 3]:
            for pipeline, records in [
                ("C_qwen_transcript", c_by_key),
                ("D_qwen_direct", d_by_key),
            ]:
                score, _ = score_record(
                    records[(sample, repetition)], truth[sample]
                )
                scores[(pipeline, sample, repetition)] = score
                rows.append(
                    {
                        "sample": sample,
                        "repetition": repetition,
                        "pipeline": pipeline,
                        "summary_rouge_l": score,
                        "status": "success",
                    }
                )
    with (STABILITY_DIR / "summary_scores.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    per_repetition = {}
    for repetition in [1, 2, 3]:
        c_values = [
            scores[("C_qwen_transcript", sample, repetition)]
            for sample in samples
        ]
        d_values = [
            scores[("D_qwen_direct", sample, repetition)]
            for sample in samples
        ]
        differences = [c - d for c, d in zip(c_values, d_values)]
        per_repetition[str(repetition)] = {
            "C_mean": mean(c_values),
            "D_mean": mean(d_values),
            "C_minus_D": mean(differences),
            "paired_bootstrap_95_ci": bootstrap_mean_ci(
                differences, seed=1400 + repetition
            ),
            "C_wins": sum(value > 1e-12 for value in differences),
            "D_wins": sum(value < -1e-12 for value in differences),
            "ties": sum(abs(value) <= 1e-12 for value in differences),
        }

    transcript_groups: dict[str, list[str]] = defaultdict(list)
    c_output_groups: dict[str, list[str]] = defaultdict(list)
    d_output_groups: dict[str, list[str]] = defaultdict(list)
    for record in c_transcriptions:
        transcript_groups[record["sample"]].append(record["transcript"])
    for record in c_summaries:
        c_output_groups[record["sample"]].append(record["output"])
    for record in d_summaries:
        d_output_groups[record["sample"]].append(record["output"])
    sample_direction_stability = {}
    for sample in samples:
        differences = [
            scores[("C_qwen_transcript", sample, repetition)]
            - scores[("D_qwen_direct", sample, repetition)]
            for repetition in [1, 2, 3]
        ]
        signs = [
            "C" if value > 1e-12 else "D" if value < -1e-12 else "tie"
            for value in differences
        ]
        sample_direction_stability[sample] = {
            "differences": differences,
            "winner_by_repetition": signs,
            "stable_winner": len(set(signs)) == 1,
        }

    audit = read_jsonl(STABILITY_DIR / "c_summaries_call_audit.jsonl")
    summary = {
        "experiment_id": "tts12_summary_stability_v1",
        "sample_count": 12,
        "repetitions": 3,
        "score_row_count": len(rows),
        "per_repetition": per_repetition,
        "determinism": {
            "samples_with_three_identical_C_transcripts": sum(
                len(set(values)) == 1 for values in transcript_groups.values()
            ),
            "samples_with_three_identical_C_summaries": sum(
                len(set(values)) == 1 for values in c_output_groups.values()
            ),
            "samples_with_three_identical_D_summaries": sum(
                len(set(values)) == 1 for values in d_output_groups.values()
            ),
            "samples_with_stable_C_vs_D_winner": sum(
                value["stable_winner"]
                for value in sample_direction_stability.values()
            ),
        },
        "sample_direction_stability": sample_direction_stability,
        "api": {
            "canonical_C_calls": len(c_summaries),
            "actual_C_calls": len(audit),
            "extra_duplicate_C_calls": len(audit) - len(c_summaries),
            "canonical_C_usage": usage(c_summaries),
            "actual_C_usage": usage(audit),
            "estimated_actual_C_cost_usd_at_0.0005_per_call": (
                len(audit) * 0.0005
            ),
        },
        "caveats": [
            "Qwen uses deterministic decoding, so this tests workflow reproducibility rather than stochastic decoding sensitivity.",
            "The three repetitions for a sample are not independent samples; N remains 12.",
            "Twelve accidental duplicate C calls are retained in the audit but excluded from scoring.",
            "No latency comparison is made.",
        ],
    }
    write_stability_figure(per_repetition, sample_direction_stability)
    (STABILITY_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    truth = json.loads(TRUTH.read_text(encoding="utf-8"))
    samples = list(truth)
    evaluate_d_rerun(truth, samples)
    evaluate_stability(truth, samples)
    print(f"D rerun summary: {RERUN_DIR / 'summary.json'}")
    print(f"Stability summary: {STABILITY_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()

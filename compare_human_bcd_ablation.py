"""Compare B (Whisper cascade), C (Qwen transcript cascade), and D (Qwen direct)."""
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
CONFIG = ROOT / "experiments" / "human_speech_v1.json"
GROUND_TRUTH = ROOT / "data" / "ground_truth_human_v1.json"
RESULT_DIR = ROOT / "data" / "results" / "human_speech_v1"
B_INPUT = RESULT_DIR / "cascade_raw.json"
C_INPUT = RESULT_DIR / "qwen_transcript_cascade_raw.jsonl"
D_RAW_INPUT = RESULT_DIR / "direct_raw.jsonl"
D_STRUCTURED_INPUT = RESULT_DIR / "direct_postprocessed.jsonl"
TRANSCRIPTION_SUMMARY = RESULT_DIR / "qwen_transcription_summary.json"
API_AUDIT_SUMMARY = RESULT_DIR / "qwen_transcript_cascade_audit_summary.json"
SCORES_OUTPUT = RESULT_DIR / "bcd_ablation_scores.csv"
SUMMARY_OUTPUT = RESULT_DIR / "bcd_ablation_summary.json"
FIGURE_OUTPUT = ROOT / "report" / "figures" / "human_speech_bcd_ablation.png"
TASKS = ["summarization", "sentiment", "keywords", "intent"]
PIPELINES = ["B_whisper_cascade", "C_qwen_transcript", "D_qwen_direct"]
PAIR_NAMES = {
    "C_minus_B": ("C_qwen_transcript", "B_whisper_cascade"),
    "D_minus_C": ("D_qwen_direct", "C_qwen_transcript"),
    "D_minus_B": ("D_qwen_direct", "B_whisper_cascade"),
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
    means = array[indices].mean(axis=1)
    return [float(value) for value in np.quantile(means, [0.025, 0.975])]


def load_b(samples: list[str]) -> dict[tuple[str, str], dict[str, Any]]:
    payload = json.loads(B_INPUT.read_text(encoding="utf-8"))
    records = {}
    for entry in payload:
        for task in TASKS:
            records[(entry["sample"], task)] = {
                "sample": entry["sample"],
                "task": task,
                "output": entry[task]["output"],
            }
    expected = {(sample, task) for sample in samples for task in TASKS}
    if set(records) != expected:
        raise ValueError("B results do not match the expected sample/task keys")
    return records


def load_c(samples: list[str]) -> dict[tuple[str, str], dict[str, Any]]:
    records = [
        record
        for record in read_jsonl(C_INPUT)
        if record.get("status") == "success"
    ]
    by_key = {(record["sample"], record["task"]): record for record in records}
    expected = {(sample, task) for sample in samples for task in TASKS}
    if set(by_key) != expected or len(by_key) != len(records):
        raise ValueError("C results are missing or duplicated")
    return by_key


def load_d(samples: list[str]) -> dict[tuple[str, str], dict[str, Any]]:
    raw = [
        record
        for record in read_jsonl(D_RAW_INPUT)
        if record.get("status") == "success"
    ]
    structured = [
        record
        for record in read_jsonl(D_STRUCTURED_INPUT)
        if record.get("postprocess_status") == "success"
    ]
    raw_by_key = {(record["sample"], record["task"]): record for record in raw}
    structured_by_key = {
        (record["sample"], record["task"]): record for record in structured
    }
    records = {}
    for sample in samples:
        for task in TASKS:
            key = (sample, task)
            records[key] = (
                raw_by_key[key] if task == "summarization" else structured_by_key[key]
            )
    return records


def write_figure(summary: dict[str, Any]) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(RESULT_DIR / ".matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = ["Summary\nROUGE-L", "Sentiment\naccuracy", "Keyword\nF1", "Intent\naccuracy"]
    colors = ["#2563eb", "#7c3aed", "#dc2626"]
    display = ["B: Whisper cascade", "C: Qwen transcript", "D: Qwen direct"]
    x = np.arange(len(TASKS))
    width = 0.24
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for index, pipeline in enumerate(PIPELINES):
        values = [summary["means"][pipeline][task] for task in TASKS]
        axes[0].bar(
            x + (index - 1) * width,
            values,
            width,
            label=display[index],
            color=colors[index],
        )
    axes[0].set_xticks(x, labels)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("Score")
    axes[0].set_title("Human speech B/C/D ablation (N=8)")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend()

    wer_values = [
        summary["transcription"]["whisper_normalized_wer"],
        summary["transcription"]["qwen_cleaned_normalized_wer"],
    ]
    axes[1].bar(
        ["Whisper", "Qwen2-Audio"],
        wer_values,
        color=[colors[0], colors[1]],
    )
    axes[1].set_ylim(0, max(wer_values) * 1.35)
    axes[1].set_ylabel("Normalized WER (lower is better)")
    axes[1].set_title("Transcription quality")
    axes[1].grid(axis="y", alpha=0.25)
    for index, value in enumerate(wer_values):
        axes[1].text(index, value + 0.002, f"{value:.4f}", ha="center")

    fig.tight_layout()
    FIGURE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_OUTPUT, dpi=180)
    plt.close(fig)


def main() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    truth = json.loads(GROUND_TRUTH.read_text(encoding="utf-8"))
    samples = list(config["samples"])
    records = {
        "B_whisper_cascade": load_b(samples),
        "C_qwen_transcript": load_c(samples),
        "D_qwen_direct": load_d(samples),
    }

    rows = []
    scores: dict[tuple[str, str, str], float] = {}
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    sample_scores = {
        sample: {pipeline: {} for pipeline in PIPELINES} for sample in samples
    }
    for sample in samples:
        for task in TASKS:
            for pipeline in PIPELINES:
                score, valid_json = score_record(
                    records[pipeline][(sample, task)],
                    truth[sample],
                )
                scores[(pipeline, sample, task)] = score
                grouped[(pipeline, task)].append(score)
                sample_scores[sample][pipeline][task] = score
                rows.append(
                    {
                        "sample": sample,
                        "task": task,
                        "pipeline": pipeline,
                        "score": score,
                        "valid_json": valid_json,
                    }
                )

    with SCORES_OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    means = {
        pipeline: {
            task: mean(grouped[(pipeline, task)]) for task in TASKS
        }
        for pipeline in PIPELINES
    }
    pairwise = {}
    for pair_index, (pair_name, (left, right)) in enumerate(PAIR_NAMES.items()):
        pairwise[pair_name] = {}
        for task_index, task in enumerate(TASKS):
            differences = [
                scores[(left, sample, task)] - scores[(right, sample, task)]
                for sample in samples
            ]
            left_wins = sum(value > 1e-12 for value in differences)
            right_wins = sum(value < -1e-12 for value in differences)
            ties = len(differences) - left_wins - right_wins
            pairwise[pair_name][task] = {
                "mean_difference": mean(differences),
                "paired_bootstrap_95_ci": bootstrap_mean_ci(
                    differences,
                    seed=100 + pair_index * 10 + task_index,
                ),
                "left_wins": left_wins,
                "right_wins": right_wins,
                "ties": ties,
            }

    transcription = json.loads(
        TRANSCRIPTION_SUMMARY.read_text(encoding="utf-8")
    )["means"]
    audit = json.loads(API_AUDIT_SUMMARY.read_text(encoding="utf-8"))
    c_records = list(records["C_qwen_transcript"].values())
    canonical_usage = {
        field: sum(
            int(record.get("usage", {}).get(field) or 0)
            for record in c_records
        )
        for field in ["prompt_tokens", "completion_tokens", "total_tokens"]
    }
    summary = {
        "experiment_id": "human_speech_v1_ablation_bcd",
        "sample_count": len(samples),
        "score_row_count": len(rows),
        "definitions": {
            "B_whisper_cascade": "Audio -> Whisper transcript -> DeepSeek tasks",
            "C_qwen_transcript": "Audio -> Qwen transcript -> DeepSeek tasks",
            "D_qwen_direct": "Audio -> Qwen direct tasks -> DeepSeek formatting for structured tasks",
        },
        "means": means,
        "pairwise": pairwise,
        "transcription": {
            key: transcription[key]
            for key in [
                "whisper_project_wer",
                "qwen_cleaned_project_wer",
                "whisper_normalized_wer",
                "qwen_cleaned_normalized_wer",
            ]
        },
        "c_api": {
            "canonical_call_count": len(c_records),
            "canonical_usage": canonical_usage,
            "audit_call_count": audit["recorded_successful_api_calls"],
            "audit_total_usage": audit["all_call_usage"],
            "extra_duplicate_calls": audit["extra_duplicate_calls"],
            "estimated_all_call_cost_usd_at_0.0005_per_call": audit[
                "estimated_all_call_cost_usd_at_0.0005_per_call"
            ],
        },
        "sample_scores": sample_scores,
        "caveats": [
            "N=8 supports descriptive trends only, not statistical significance.",
            "B and C isolate the ASR model approximately because they use the same DeepSeek prompts, but B audio hashes are unavailable.",
            "C and D both use Qwen2-Audio, but task prompting and generation differ, so D-C is an approximate transcription-bottleneck test.",
            "Direct structured tasks include DeepSeek formatting.",
            "Latency is not compared because environments and timing boundaries differ.",
        ],
    }
    SUMMARY_OUTPUT.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_figure(summary)
    print(json.dumps(means, indent=2))
    print(f"Scores: {SCORES_OUTPUT}")
    print(f"Summary: {SUMMARY_OUTPUT}")
    print(f"Figure: {FIGURE_OUTPUT}")


if __name__ == "__main__":
    main()


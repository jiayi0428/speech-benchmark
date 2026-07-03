"""Evaluate and compare TTS12 paths A, B, C, and D."""
from __future__ import annotations

import csv
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np

from evaluate_tts_qwen_transcription import normalize_text
from evaluate_white_noise import score_record, wer
from src.transcription_utils import strip_transcription_wrapper


ROOT = Path(__file__).resolve().parent
INDEX = ROOT / "data" / "processed" / "tts12_cd_v1" / "index.json"
GROUND_TRUTH = ROOT / "data" / "ground_truth_tts12_cd_v1.json"
RESULT_DIR = ROOT / "data" / "results" / "tts12_cd_v1"
TRANSCRIPTION_INPUT = RESULT_DIR / "qwen_transcription_raw.jsonl"
C_INPUT = RESULT_DIR / "c_tasks_raw.jsonl"
D_RAW_INPUT = RESULT_DIR / "direct_raw.jsonl"
D_STRUCTURED_INPUT = RESULT_DIR / "direct_postprocessed.jsonl"
AB_INPUT = RESULT_DIR / "ab_oracle_vs_cascade.txt"
TRANSCRIPTION_SCORES = RESULT_DIR / "transcription_scores.csv"
SCORES_OUTPUT = RESULT_DIR / "scores.csv"
SUMMARY_OUTPUT = RESULT_DIR / "summary.json"
FIGURE_OUTPUT = ROOT / "report" / "figures" / "tts12_abcd_comparison.png"
TASKS = ["summarization", "sentiment", "keywords", "intent"]
PIPELINES = [
    "A_oracle",
    "B_whisper_cascade",
    "C_qwen_transcript",
    "D_qwen_direct",
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def bootstrap_mean_ci(
    values: list[float], *, seed: int, samples: int = 20_000
) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(samples, len(array)))
    means = array[indices].mean(axis=1)
    return [float(value) for value in np.quantile(means, [0.025, 0.975])]


def parse_ab() -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    text = AB_INPUT.read_text(encoding="utf-8")
    sample_pattern = re.compile(
        r"^(\w+)\s+([0-9.]+)\s+([0-9.]+)\s+[+-][0-9.]+\s+",
        re.MULTILINE,
    )
    by_sample = {
        sample: {"A_oracle": float(oracle), "B_whisper_cascade": float(cascade)}
        for sample, oracle, cascade in sample_pattern.findall(text)
        if sample != "AVG"
    }
    metric_patterns = {
        "sentiment": r"Sentiment:\s+Oracle=(\d+)%\s+Cascade=(\d+)%",
        "keywords": r"Keywords:\s+Oracle=([0-9.]+)\s+Cascade=([0-9.]+)",
        "intent": r"Intent:\s+Oracle=(\d+)%\s+Cascade=(\d+)%",
    }
    means: dict[str, dict[str, float]] = {
        "A_oracle": {"summarization": mean(v["A_oracle"] for v in by_sample.values())},
        "B_whisper_cascade": {
            "summarization": mean(v["B_whisper_cascade"] for v in by_sample.values())
        },
    }
    for task, pattern in metric_patterns.items():
        match = re.search(pattern, text)
        if not match:
            raise ValueError(f"Missing A/B aggregate metric: {task}")
        values = [float(match.group(1)), float(match.group(2))]
        if task != "keywords":
            values = [value / 100 for value in values]
        means["A_oracle"][task] = values[0]
        means["B_whisper_cascade"][task] = values[1]
    return by_sample, means


def usage(records: list[dict[str, Any]]) -> dict[str, int]:
    return {
        field: sum(int(record.get("usage", {}).get(field) or 0) for record in records)
        for field in ["prompt_tokens", "completion_tokens", "total_tokens"]
    }


def write_figure(means: dict[str, dict[str, float]]) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(RESULT_DIR / ".matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = ["Summary\nROUGE-L", "Sentiment\naccuracy", "Keyword\nF1", "Intent\naccuracy"]
    display = ["A: Oracle", "B: Whisper", "C: Qwen transcript", "D: Qwen direct"]
    colors = ["#059669", "#2563eb", "#7c3aed", "#dc2626"]
    x = np.arange(len(TASKS))
    width = 0.2
    fig, ax = plt.subplots(figsize=(11.5, 5.4))
    for index, pipeline in enumerate(PIPELINES):
        ax.bar(
            x + (index - 1.5) * width,
            [means[pipeline][task] for task in TASKS],
            width,
            label=display[index],
            color=colors[index],
        )
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score")
    ax.set_title("TTS12 four-path comparison (N=12)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=2)
    fig.tight_layout()
    FIGURE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_OUTPUT, dpi=180)
    plt.close(fig)


def main() -> None:
    index = json.loads(INDEX.read_text(encoding="utf-8"))
    truth = json.loads(GROUND_TRUTH.read_text(encoding="utf-8"))
    samples = [entry["topic"] for entry in index]
    ab_by_sample, ab_means = parse_ab()
    if set(ab_by_sample) != set(samples):
        raise ValueError("A/B summary samples do not match TTS12 inputs")

    transcriptions = [
        record for record in read_jsonl(TRANSCRIPTION_INPUT)
        if record.get("status") == "success"
    ]
    transcription_by_sample = {record["sample"]: record for record in transcriptions}
    if set(transcription_by_sample) != set(samples) or len(transcriptions) != 12:
        raise ValueError("Expected 12 unique successful Qwen transcripts")
    transcription_rows = []
    for sample in samples:
        record = transcription_by_sample[sample]
        raw = record["transcript"]
        cleaned = strip_transcription_wrapper(raw)
        transcription_rows.append(
            {
                "sample": sample,
                "normalized_wer": wer(
                    normalize_text(truth[sample]["transcript"]),
                    normalize_text(cleaned),
                ),
                "latency_seconds": record["latency_seconds"],
                "transcript_raw": raw,
                "transcript_cleaned": cleaned,
            }
        )
    with TRANSCRIPTION_SCORES.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(transcription_rows[0]))
        writer.writeheader()
        writer.writerows(transcription_rows)

    c_all = read_jsonl(C_INPUT)
    c_records = [record for record in c_all if record.get("status") == "success"]
    c_by_key = {(record["sample"], record["task"]): record for record in c_records}
    d_raw = [
        record for record in read_jsonl(D_RAW_INPUT)
        if record.get("status") == "success"
    ]
    d_structured = [
        record for record in read_jsonl(D_STRUCTURED_INPUT)
        if record.get("postprocess_status") == "success"
    ]
    d_raw_by_key = {(record["sample"], record["task"]): record for record in d_raw}
    d_structured_by_key = {
        (record["sample"], record["task"]): record for record in d_structured
    }
    expected = {(sample, task) for sample in samples for task in TASKS}
    if set(c_by_key) != expected or len(c_records) != 48:
        raise ValueError("Path C does not have 48 unique successful results")
    if set(d_raw_by_key) != expected or len(d_raw) != 48:
        raise ValueError("Path D raw results are incomplete")
    expected_structured = {
        (sample, task) for sample in samples for task in TASKS[1:]
    }
    if set(d_structured_by_key) != expected_structured or len(d_structured) != 36:
        raise ValueError("Path D structured results are incomplete")

    score_rows = []
    scores: dict[tuple[str, str, str], float] = {}
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for sample in samples:
        for pipeline in ["A_oracle", "B_whisper_cascade"]:
            score = ab_by_sample[sample][pipeline]
            scores[(pipeline, sample, "summarization")] = score
            grouped[(pipeline, "summarization")].append(score)
            score_rows.append(
                {
                    "sample": sample,
                    "task": "summarization",
                    "pipeline": pipeline,
                    "score": score,
                    "valid_json": None,
                    "source": "supplied A/B score file",
                }
            )
        for task in TASKS:
            for pipeline, record in [
                ("C_qwen_transcript", c_by_key[(sample, task)]),
                (
                    "D_qwen_direct",
                    d_raw_by_key[(sample, task)]
                    if task == "summarization"
                    else d_structured_by_key[(sample, task)],
                ),
            ]:
                score, valid_json = score_record(record, truth[sample])
                scores[(pipeline, sample, task)] = score
                grouped[(pipeline, task)].append(score)
                score_rows.append(
                    {
                        "sample": sample,
                        "task": task,
                        "pipeline": pipeline,
                        "score": score,
                        "valid_json": valid_json,
                        "source": "recomputed from raw output",
                    }
                )
    with SCORES_OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(score_rows[0]))
        writer.writeheader()
        writer.writerows(score_rows)

    means = {
        **ab_means,
        **{
            pipeline: {
                task: mean(grouped[(pipeline, task)]) for task in TASKS
            }
            for pipeline in ["C_qwen_transcript", "D_qwen_direct"]
        },
    }
    summary_pairs = {}
    for left_index, left in enumerate(PIPELINES):
        for right in PIPELINES[left_index + 1 :]:
            differences = [
                scores[(left, sample, "summarization")]
                - scores[(right, sample, "summarization")]
                for sample in samples
            ]
            summary_pairs[f"{left}_minus_{right}"] = {
                "mean_difference": mean(differences),
                "paired_bootstrap_95_ci": bootstrap_mean_ci(
                    differences, seed=1000 + len(summary_pairs)
                ),
                "left_wins": sum(value > 1e-12 for value in differences),
                "right_wins": sum(value < -1e-12 for value in differences),
                "ties": sum(abs(value) <= 1e-12 for value in differences),
            }
    cd_pairs = {}
    for task_index, task in enumerate(TASKS):
        differences = [
            scores[("C_qwen_transcript", sample, task)]
            - scores[("D_qwen_direct", sample, task)]
            for sample in samples
        ]
        cd_pairs[task] = {
            "mean_difference_C_minus_D": mean(differences),
            "paired_bootstrap_95_ci": bootstrap_mean_ci(
                differences, seed=1100 + task_index
            ),
            "C_wins": sum(value > 1e-12 for value in differences),
            "D_wins": sum(value < -1e-12 for value in differences),
            "ties": sum(abs(value) <= 1e-12 for value in differences),
        }

    summary = {
        "experiment_id": "tts12_cd_v1",
        "sample_count": 12,
        "definitions": {
            "A_oracle": "Ground-truth transcript -> DeepSeek tasks",
            "B_whisper_cascade": "Audio -> Whisper transcript -> DeepSeek tasks",
            "C_qwen_transcript": "Audio -> Qwen transcript -> DeepSeek tasks",
            "D_qwen_direct": "Audio -> Qwen direct understanding -> DeepSeek structured formatting",
        },
        "means": means,
        "summary_pairwise": summary_pairs,
        "C_minus_D": cd_pairs,
        "qwen_transcription": {
            "mean_normalized_wer": mean(
                row["normalized_wer"] for row in transcription_rows
            ),
            "by_sample": {
                row["sample"]: row["normalized_wer"] for row in transcription_rows
            },
        },
        "api": {
            "C_successful_calls": len(c_records),
            "D_successful_calls": len(d_structured),
            "successful_calls_total": len(c_records) + len(d_structured),
            "C_usage": usage(c_records),
            "D_usage": usage(d_structured),
            "connection_error_audit_rows": sum(
                record.get("status") == "error" for record in c_all
            ),
            "estimated_cost_usd_at_0.0005_per_successful_call": (
                len(c_records) + len(d_structured)
            )
            * 0.0005,
        },
        "source_precision": {
            "A_B_summarization": "sample-level scores supplied to 4 decimals",
            "A_B_structured_tasks": "aggregate-only and rounded in supplied file",
            "C_D": "recomputed from raw outputs with project scoring code",
        },
        "caveats": [
            "N=12 supports descriptive trends only, not statistical significance.",
            "A/B structured-task raw outputs and sample-level scores were not supplied, so no paired A/B structured analysis is possible.",
            "A/B structured aggregate values are rounded in the supplied text file.",
            "C and D share Qwen2-Audio but allocate semantic understanding differently.",
            "D structured tasks include DeepSeek formatting.",
            "Latency is not compared because timing boundaries and execution runs differ.",
        ],
    }
    SUMMARY_OUTPUT.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_figure(means)
    print(json.dumps(means, indent=2))
    print(json.dumps(summary["api"], indent=2))
    print(f"Summary: {SUMMARY_OUTPUT}")
    print(f"Figure: {FIGURE_OUTPUT}")


if __name__ == "__main__":
    main()

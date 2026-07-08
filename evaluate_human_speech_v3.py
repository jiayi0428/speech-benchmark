"""Evaluate human-speech-v3 paths A, B, C, and D."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np

from evaluate_tts_qwen_transcription import normalize_text
from evaluate_white_noise import score_record, wer
from src.transcription_utils import strip_transcription_wrapper


ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "experiments" / "human_speech_v3.json"
TRUTH = ROOT / "data" / "ground_truth_human_v3.json"
RESULT_DIR = ROOT / "data" / "results" / "human_speech_v3"
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


def usage(records: list[dict[str, Any]]) -> dict[str, int]:
    return {
        field: sum(int(record.get("usage", {}).get(field) or 0) for record in records)
        for field in ["prompt_tokens", "completion_tokens", "total_tokens"]
    }


def bootstrap_mean_ci(values: list[float], seed: int) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(20_000, len(array)))
    means = array[indices].mean(axis=1)
    return [float(value) for value in np.quantile(means, [0.025, 0.975])]


def load_success(path: Path, status_field: str = "status") -> list[dict[str, Any]]:
    return [
        record
        for record in read_jsonl(path)
        if record.get(status_field) == "success"
    ]


def main() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    truth = json.loads(TRUTH.read_text(encoding="utf-8"))
    samples = list(config["samples"])
    expected = {(sample, task) for sample in samples for task in TASKS}
    expected_structured = {
        (sample, task) for sample in samples for task in TASKS[1:]
    }
    a = load_success(RESULT_DIR / "a_tasks_raw.jsonl")
    b = load_success(RESULT_DIR / "b_tasks_raw.jsonl")
    c = load_success(RESULT_DIR / "c_tasks_raw.jsonl")
    d_raw = load_success(RESULT_DIR / "direct_raw.jsonl")
    d_structured = load_success(
        RESULT_DIR / "direct_postprocessed.jsonl", "postprocess_status"
    )
    keyed = {
        "A_oracle": {(r["sample"], r["task"]): r for r in a},
        "B_whisper_cascade": {(r["sample"], r["task"]): r for r in b},
        "C_qwen_transcript": {(r["sample"], r["task"]): r for r in c},
    }
    d_raw_by_key = {(r["sample"], r["task"]): r for r in d_raw}
    d_structured_by_key = {(r["sample"], r["task"]): r for r in d_structured}
    keyed["D_qwen_direct"] = {
        key: (
            d_raw_by_key[key]
            if key[1] == "summarization"
            else d_structured_by_key[key]
        )
        for key in expected
    }
    if any(set(keyed[pipeline]) != expected for pipeline in PIPELINES):
        raise ValueError("One or more path result sets are incomplete")
    if set(d_structured_by_key) != expected_structured:
        raise ValueError("D structured results are incomplete")

    rows = []
    scores = {}
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for sample in samples:
        for task in TASKS:
            for pipeline in PIPELINES:
                score, valid_json = score_record(
                    keyed[pipeline][(sample, task)], truth[sample]
                )
                scores[(pipeline, sample, task)] = score
                grouped[(pipeline, task)].append(score)
                rows.append(
                    {
                        "sample": sample,
                        "task": task,
                        "pipeline": pipeline,
                        "score": score,
                        "valid_json": valid_json,
                    }
                )
    with (RESULT_DIR / "scores.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
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
    for pair_index, (left, right) in enumerate(combinations(PIPELINES, 2)):
        pairwise[f"{left}_minus_{right}"] = {}
        for task_index, task in enumerate(TASKS):
            differences = [
                scores[(left, sample, task)] - scores[(right, sample, task)]
                for sample in samples
            ]
            pairwise[f"{left}_minus_{right}"][task] = {
                "mean_difference": mean(differences),
                "paired_bootstrap_95_ci": bootstrap_mean_ci(
                    differences, 2100 + pair_index * 10 + task_index
                ),
                "left_wins": sum(v > 1e-12 for v in differences),
                "right_wins": sum(v < -1e-12 for v in differences),
                "ties": sum(abs(v) <= 1e-12 for v in differences),
            }

    transcription = {}
    transcription_rows = []
    for name, filename in [
        ("B_whisper", "whisper_transcription_raw.jsonl"),
        ("C_qwen", "qwen_transcription_raw.jsonl"),
    ]:
        records = load_success(RESULT_DIR / filename)
        by_sample = {record["sample"]: record for record in records}
        values = {}
        for sample in samples:
            cleaned = strip_transcription_wrapper(by_sample[sample]["transcript"])
            value = wer(
                normalize_text(truth[sample]["transcript"]),
                normalize_text(cleaned),
            )
            values[sample] = value
            transcription_rows.append(
                {
                    "sample": sample,
                    "pipeline": name,
                    "normalized_wer": value,
                    "transcript_raw": by_sample[sample]["transcript"],
                    "transcript_cleaned": cleaned,
                }
            )
        transcription[name] = {
            "mean_normalized_wer": mean(values.values()),
            "by_sample": values,
        }
    with (RESULT_DIR / "transcription_scores.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(transcription_rows[0])
        )
        writer.writeheader()
        writer.writerows(transcription_rows)

    calls = {"A": a, "B": b, "C": c, "D": d_structured}
    summary = {
        "experiment_id": config["experiment_id"],
        "sample_count": len(samples),
        "score_row_count": len(rows),
        "definitions": {
            "A_oracle": "Human transcript -> DeepSeek tasks",
            "B_whisper_cascade": "Whisper large-v3 transcript -> DeepSeek tasks",
            "C_qwen_transcript": "Qwen transcript -> DeepSeek tasks",
            "D_qwen_direct": "Qwen direct System-turn -> DeepSeek structured formatting",
        },
        "means": means,
        "pairwise": pairwise,
        "transcription": transcription,
        "api": {
            path: {
                "successful_calls": len(records),
                "usage": usage(records),
            }
            for path, records in calls.items()
        },
        "estimated_cost_usd_at_0.0005_per_call": (
            sum(len(records) for records in calls.values()) * 0.0005
        ),
        "caveats": [
            "N=8 supports descriptive trends only, not statistical significance.",
            "A is an oracle-text baseline based on the supplied human transcript.",
            "All A/B/C text tasks use the same DeepSeek system prompts.",
            "D uses Qwen System-turn prompting and DeepSeek structured formatting.",
            "Latency is not compared across paths or execution environments.",
        ],
    }
    (RESULT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(means, indent=2, ensure_ascii=False))
    print(json.dumps(summary["transcription"], indent=2, ensure_ascii=False))
    print(json.dumps(summary["api"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

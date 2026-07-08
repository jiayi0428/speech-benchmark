"""Evaluate all four paths on the eight-sample human-speech dataset."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from statistics import mean
from typing import Any

from compare_human_bcd_ablation import (
    CONFIG,
    GROUND_TRUTH,
    RESULT_DIR,
    TASKS,
    bootstrap_mean_ci,
    load_b,
    load_c,
    load_d,
    read_jsonl,
)
from evaluate_white_noise import score_record


A_INPUT = RESULT_DIR / "oracle_tasks_raw.jsonl"
SCORES_OUTPUT = RESULT_DIR / "abcd_ablation_scores.csv"
SUMMARY_OUTPUT = RESULT_DIR / "abcd_ablation_summary.json"
PIPELINES = [
    "A_oracle",
    "B_whisper_cascade",
    "C_qwen_transcript",
    "D_qwen_direct",
]


def load_a(samples: list[str]) -> dict[tuple[str, str], dict[str, Any]]:
    records = [
        record
        for record in read_jsonl(A_INPUT)
        if record.get("status") == "success"
    ]
    by_key = {(record["sample"], record["task"]): record for record in records}
    expected = {(sample, task) for sample in samples for task in TASKS}
    if set(by_key) != expected or len(by_key) != len(records):
        raise ValueError("A results are missing or duplicated")
    return by_key


def usage(records: list[dict[str, Any]]) -> dict[str, int]:
    return {
        field: sum(int(record.get("usage", {}).get(field) or 0) for record in records)
        for field in ["prompt_tokens", "completion_tokens", "total_tokens"]
    }


def main() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    truth = json.loads(GROUND_TRUTH.read_text(encoding="utf-8"))
    samples = list(config["samples"])
    records = {
        "A_oracle": load_a(samples),
        "B_whisper_cascade": load_b(samples),
        "C_qwen_transcript": load_c(samples),
        "D_qwen_direct": load_d(samples),
    }

    rows = []
    scores: dict[tuple[str, str, str], float] = {}
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for sample in samples:
        for task in TASKS:
            for pipeline in PIPELINES:
                score, valid_json = score_record(
                    records[pipeline][(sample, task)],
                    truth[sample],
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
                    differences,
                    seed=500 + pair_index * 10 + task_index,
                ),
                "left_wins": sum(value > 1e-12 for value in differences),
                "right_wins": sum(value < -1e-12 for value in differences),
                "ties": sum(abs(value) <= 1e-12 for value in differences),
            }

    a_records = list(records["A_oracle"].values())
    summary = {
        "experiment_id": "human_speech_v1_ablation_abcd",
        "sample_count": len(samples),
        "score_row_count": len(rows),
        "definitions": {
            "A_oracle": "Human reference transcript -> DeepSeek tasks",
            "B_whisper_cascade": "Audio -> Whisper transcript -> DeepSeek tasks",
            "C_qwen_transcript": "Audio -> Qwen transcript -> DeepSeek tasks",
            "D_qwen_direct": (
                "Audio -> Qwen direct tasks -> DeepSeek formatting "
                "for structured tasks"
            ),
        },
        "means": means,
        "pairwise": pairwise,
        "a_api": {
            "successful_calls": len(a_records),
            "usage": usage(a_records),
            "estimated_cost_usd_at_0.0005_per_call": len(a_records) * 0.0005,
        },
        "caveats": [
            "N=8 supports descriptive trends only, not statistical significance.",
            "A uses the human reference transcript and is an oracle-text baseline.",
            "Latency is not compared because environments and timing boundaries differ.",
            "D results were generated with the historical qwen_user_task_v2 prompt.",
        ],
    }
    SUMMARY_OUTPUT.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Scores: {SCORES_OUTPUT}")
    print(f"Summary: {SUMMARY_OUTPUT}")


if __name__ == "__main__":
    main()

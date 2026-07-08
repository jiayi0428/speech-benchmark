"""Evaluate human_speech_v2 paths C and D."""
from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np

from evaluate_tts_qwen_transcription import normalize_text
from evaluate_white_noise import score_record, wer
from src.transcription_utils import strip_transcription_wrapper


ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "experiments" / "human_speech_v2.json"
TRUTH = ROOT / "data" / "ground_truth_human_v2.json"
RESULT_DIR = ROOT / "data" / "results" / "human_speech_v2"
TRANSCRIPTION_INPUT = RESULT_DIR / "qwen_transcription_raw.jsonl"
C_INPUT = RESULT_DIR / "c_tasks_raw.jsonl"
D_RAW_INPUT = RESULT_DIR / "direct_raw.jsonl"
D_STRUCTURED_INPUT = RESULT_DIR / "direct_postprocessed.jsonl"
TRANSCRIPTION_SCORES = RESULT_DIR / "transcription_scores.csv"
SCORES_OUTPUT = RESULT_DIR / "scores.csv"
SUMMARY_OUTPUT = RESULT_DIR / "summary.json"
FIGURE_OUTPUT = ROOT / "report" / "figures" / "human_speech_v2_cd.png"
TASKS = ["summarization", "sentiment", "keywords", "intent"]


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


def write_figure(means: dict[str, dict[str, float]]) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(RESULT_DIR / ".matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [
        "Summary\nROUGE-L",
        "Sentiment\naccuracy",
        "Keyword\nF1",
        "Intent\naccuracy",
    ]
    x = np.arange(len(TASKS))
    width = 0.34
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    for index, (pipeline, display, color) in enumerate(
        [
            ("C_qwen_transcript", "C: Qwen transcript", "#7c3aed"),
            ("D_qwen_direct", "D: Qwen direct", "#dc2626"),
        ]
    ):
        ax.bar(
            x + (index - 0.5) * width,
            [means[pipeline][task] for task in TASKS],
            width,
            label=display,
            color=color,
        )
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score")
    ax.set_title("Human speech V2 C/D comparison (N=10)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    FIGURE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_OUTPUT, dpi=180)
    plt.close(fig)


def main() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    truth = json.loads(TRUTH.read_text(encoding="utf-8"))
    samples = list(config["samples"])
    transcriptions = [
        record for record in read_jsonl(TRANSCRIPTION_INPUT)
        if record.get("status") == "success"
    ]
    transcription_by_sample = {record["sample"]: record for record in transcriptions}
    if set(transcription_by_sample) != set(samples) or len(transcriptions) != 10:
        raise ValueError("Expected 10 unique successful transcriptions")
    transcription_rows = []
    for sample in samples:
        record = transcription_by_sample[sample]
        cleaned = strip_transcription_wrapper(record["transcript"])
        transcription_rows.append(
            {
                "sample": sample,
                "normalized_wer": wer(
                    normalize_text(truth[sample]["transcript"]),
                    normalize_text(cleaned),
                ),
                "latency_seconds": record["latency_seconds"],
                "transcript_raw": record["transcript"],
                "transcript_cleaned": cleaned,
            }
        )
    with TRANSCRIPTION_SCORES.open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(transcription_rows[0]))
        writer.writeheader()
        writer.writerows(transcription_rows)

    c_records = [
        record for record in read_jsonl(C_INPUT)
        if record.get("status") == "success"
    ]
    d_raw = [
        record for record in read_jsonl(D_RAW_INPUT)
        if record.get("status") == "success"
    ]
    d_structured = [
        record for record in read_jsonl(D_STRUCTURED_INPUT)
        if record.get("postprocess_status") == "success"
    ]
    c_by_key = {(record["sample"], record["task"]): record for record in c_records}
    d_raw_by_key = {(record["sample"], record["task"]): record for record in d_raw}
    d_structured_by_key = {
        (record["sample"], record["task"]): record for record in d_structured
    }
    expected = {(sample, task) for sample in samples for task in TASKS}
    expected_structured = {
        (sample, task) for sample in samples for task in TASKS[1:]
    }
    if set(c_by_key) != expected or len(c_records) != 40:
        raise ValueError("Path C results are incomplete or duplicated")
    if set(d_raw_by_key) != expected or len(d_raw) != 40:
        raise ValueError("Path D raw results are incomplete or duplicated")
    if set(d_structured_by_key) != expected_structured or len(d_structured) != 30:
        raise ValueError("Path D structured results are incomplete or duplicated")

    rows = []
    scores: dict[tuple[str, str, str], float] = {}
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    sample_scores = {
        sample: {"C_qwen_transcript": {}, "D_qwen_direct": {}}
        for sample in samples
    }
    for sample in samples:
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
        for pipeline in ["C_qwen_transcript", "D_qwen_direct"]
    }
    pairwise = {}
    for task_index, task in enumerate(TASKS):
        differences = [
            scores[("C_qwen_transcript", sample, task)]
            - scores[("D_qwen_direct", sample, task)]
            for sample in samples
        ]
        pairwise[task] = {
            "mean_difference_C_minus_D": mean(differences),
            "paired_bootstrap_95_ci": bootstrap_mean_ci(
                differences, seed=1700 + task_index
            ),
            "C_wins": sum(value > 1e-12 for value in differences),
            "D_wins": sum(value < -1e-12 for value in differences),
            "ties": sum(abs(value) <= 1e-12 for value in differences),
        }
    summary = {
        "experiment_id": config["experiment_id"],
        "sample_count": 10,
        "means": means,
        "C_minus_D": pairwise,
        "qwen_transcription": {
            "mean_normalized_wer": mean(
                row["normalized_wer"] for row in transcription_rows
            ),
            "by_sample": {
                row["sample"]: row["normalized_wer"]
                for row in transcription_rows
            },
        },
        "api": {
            "C_successful_calls": len(c_records),
            "D_successful_calls": len(d_structured),
            "successful_calls_total": len(c_records) + len(d_structured),
            "C_usage": usage(c_records),
            "D_usage": usage(d_structured),
            "estimated_cost_usd_at_0.0005_per_call": (
                len(c_records) + len(d_structured)
            )
            * 0.0005,
        },
        "sample_scores": sample_scores,
        "caveats": [
            "N=10 supports descriptive trends only, not statistical significance.",
            "Sentiment and intent labels are imbalanced.",
            "D structured tasks include DeepSeek formatting.",
            "The recordings are not content-matched to the TTS datasets.",
            "Latency is not compared across separately executed paths.",
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


if __name__ == "__main__":
    main()

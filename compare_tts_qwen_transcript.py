"""Compare original-TTS B summaries and C/D four-task results."""
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
GROUND_TRUTH = ROOT / "data" / "ground_truth.json"
TTS_INDEX = ROOT / "data" / "processed" / "tts_samples" / "index.json"
RESULT_DIR = ROOT / "data" / "results" / "tts_qwen_transcript_v1"
C_INPUT = RESULT_DIR / "qwen_transcript_cascade_raw.jsonl"
D_RAW_INPUT = ROOT / "data" / "results" / "white_noise_v1" / "direct_raw.jsonl"
D_STRUCTURED_INPUT = (
    ROOT / "data" / "results" / "white_noise_v1" / "direct_postprocessed.jsonl"
)
B_SUMMARY_INPUT = (
    ROOT / "data" / "results" / "white_noise_v1" / "cascade_summary_raw.json"
)
TRANSCRIPTION_SUMMARY = RESULT_DIR / "qwen_transcription_summary.json"
SCORES_OUTPUT = RESULT_DIR / "comparison_scores.csv"
SUMMARY_OUTPUT = RESULT_DIR / "comparison_summary.json"
FIGURE_OUTPUT = ROOT / "report" / "figures" / "tts_qwen_transcript_comparison.png"
TASKS = ["summarization", "sentiment", "keywords", "intent"]


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


def load_c(samples: list[str]) -> dict[tuple[str, str], dict[str, Any]]:
    records = [
        record for record in read_jsonl(C_INPUT)
        if record.get("status") == "success"
    ]
    by_key = {(record["sample"], record["task"]): record for record in records}
    expected = {(sample, task) for sample in samples for task in TASKS}
    if len(records) != len(by_key) or set(by_key) != expected:
        raise ValueError("C results are missing or duplicated")
    return by_key


def load_d(samples: list[str]) -> dict[tuple[str, str], dict[str, Any]]:
    raw = [
        record for record in read_jsonl(D_RAW_INPUT)
        if record.get("status") == "success"
        and record.get("condition") == "clean"
    ]
    structured = [
        record for record in read_jsonl(D_STRUCTURED_INPUT)
        if record.get("postprocess_status") == "success"
        and record.get("condition") == "clean"
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


def load_b_summaries(samples: list[str]) -> dict[str, dict[str, Any]]:
    payload = json.loads(B_SUMMARY_INPUT.read_text(encoding="utf-8"))
    records = {
        record["sample"]: {
            "sample": record["sample"],
            "task": "summarization",
            "output": record["output"],
        }
        for record in payload["clean"]
    }
    if set(records) != set(samples):
        raise ValueError("B clean summaries do not match the TTS sample set")
    return records


def write_figure(summary: dict[str, Any]) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(RESULT_DIR / ".matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = ["#7c3aed", "#dc2626"]
    labels = [
        "Summary\nROUGE-L",
        "Sentiment\naccuracy",
        "Keyword\nF1",
        "Intent\naccuracy",
    ]
    x = np.arange(len(TASKS))
    width = 0.34
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.7))
    for index, pipeline in enumerate(["C_qwen_transcript", "D_qwen_direct"]):
        values = [summary["means"][pipeline][task] for task in TASKS]
        axes[0].bar(
            x + (index - 0.5) * width,
            values,
            width,
            color=colors[index],
            label=("C: Qwen transcript" if index == 0 else "D: Qwen direct"),
        )
    axes[0].set_xticks(x, labels)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("Score")
    axes[0].set_title("Original TTS: C versus D (N=8)")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend()

    summary_values = [
        summary["means"]["B_whisper_cascade"]["summarization"],
        summary["means"]["C_qwen_transcript"]["summarization"],
        summary["means"]["D_qwen_direct"]["summarization"],
    ]
    axes[1].bar(
        ["B: Whisper\ncascade", "C: Qwen\ntranscript", "D: Qwen\ndirect"],
        summary_values,
        color=["#2563eb", "#7c3aed", "#dc2626"],
    )
    axes[1].set_ylim(0, 1.05)
    axes[1].set_ylabel("Summary ROUGE-L")
    axes[1].set_title("Only shared B/C/D task")
    axes[1].grid(axis="y", alpha=0.25)
    for index, value in enumerate(summary_values):
        axes[1].text(index, value + 0.025, f"{value:.3f}", ha="center")

    fig.tight_layout()
    FIGURE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_OUTPUT, dpi=180)
    plt.close(fig)


def main() -> None:
    index = json.loads(TTS_INDEX.read_text(encoding="utf-8"))
    samples = [entry["topic"] for entry in index]
    truth = json.loads(GROUND_TRUTH.read_text(encoding="utf-8"))
    c_records = load_c(samples)
    d_records = load_d(samples)
    b_summaries = load_b_summaries(samples)

    rows = []
    scores: dict[tuple[str, str, str], float] = {}
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for sample in samples:
        for task in TASKS:
            for pipeline, records in [
                ("C_qwen_transcript", c_records),
                ("D_qwen_direct", d_records),
            ]:
                score, valid_json = score_record(records[(sample, task)], truth[sample])
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
        b_score, _ = score_record(b_summaries[sample], truth[sample])
        scores[("B_whisper_cascade", sample, "summarization")] = b_score
        grouped[("B_whisper_cascade", "summarization")].append(b_score)
        rows.append(
            {
                "sample": sample,
                "task": "summarization",
                "pipeline": "B_whisper_cascade",
                "score": b_score,
                "valid_json": None,
            }
        )

    with SCORES_OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    means = {
        pipeline: {
            task: mean(grouped[(pipeline, task)])
            for task in TASKS
            if grouped[(pipeline, task)]
        }
        for pipeline in [
            "B_whisper_cascade",
            "C_qwen_transcript",
            "D_qwen_direct",
        ]
    }
    pairwise = {}
    for task_index, task in enumerate(TASKS):
        differences = [
            scores[("C_qwen_transcript", sample, task)]
            - scores[("D_qwen_direct", sample, task)]
            for sample in samples
        ]
        c_wins = sum(value > 1e-12 for value in differences)
        d_wins = sum(value < -1e-12 for value in differences)
        pairwise[task] = {
            "mean_difference_C_minus_D": mean(differences),
            "paired_bootstrap_95_ci": bootstrap_mean_ci(
                differences, seed=700 + task_index
            ),
            "C_wins": c_wins,
            "D_wins": d_wins,
            "ties": len(differences) - c_wins - d_wins,
        }
    summary_pairwise = {}
    for left, right, name in [
        ("C_qwen_transcript", "B_whisper_cascade", "C_minus_B"),
        ("D_qwen_direct", "B_whisper_cascade", "D_minus_B"),
    ]:
        differences = [
            scores[(left, sample, "summarization")]
            - scores[(right, sample, "summarization")]
            for sample in samples
        ]
        summary_pairwise[name] = {
            "mean_difference": mean(differences),
            "paired_bootstrap_95_ci": bootstrap_mean_ci(
                differences, seed=800 + len(summary_pairwise)
            ),
        }

    transcription = json.loads(
        TRANSCRIPTION_SUMMARY.read_text(encoding="utf-8")
    )
    usage = {
        field: sum(
            int(record.get("usage", {}).get(field) or 0)
            for record in c_records.values()
        )
        for field in ["prompt_tokens", "completion_tokens", "total_tokens"]
    }
    summary = {
        "experiment_id": "tts_qwen_transcript_v1_comparison",
        "sample_count": len(samples),
        "score_row_count": len(rows),
        "definitions": {
            "B_whisper_cascade": "Audio -> Whisper transcript -> DeepSeek summary",
            "C_qwen_transcript": "Audio -> Qwen transcript -> DeepSeek tasks",
            "D_qwen_direct": "Audio -> Qwen direct tasks -> DeepSeek formatting for structured tasks",
        },
        "means": means,
        "C_minus_D": pairwise,
        "summary_only_pairwise": summary_pairwise,
        "qwen_transcription": transcription["means"],
        "c_api": {
            "successful_call_count": len(c_records),
            "usage": usage,
            "estimated_cost_usd_at_0.0005_per_call": len(c_records) * 0.0005,
        },
        "caveats": [
            "N=8 supports descriptive trends only, not statistical significance.",
            "Only summarization can be compared across B/C/D because stored B results lack sample-level structured tasks.",
            "Stored B results lack Whisper transcripts, so Whisper and Qwen WER cannot be compared on these TTS clips.",
            "B and D are reused from the earlier white-noise experiment clean condition; C was run later.",
            "D structured tasks include DeepSeek formatting.",
            "Latency is not compared because runs and timing boundaries differ.",
        ],
    }
    SUMMARY_OUTPUT.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_figure(summary)
    print(json.dumps(means, indent=2))
    print(json.dumps({"c_api": summary["c_api"]}, indent=2))
    print(f"Scores: {SCORES_OUTPUT}")
    print(f"Summary: {SUMMARY_OUTPUT}")
    print(f"Figure: {FIGURE_OUTPUT}")


if __name__ == "__main__":
    main()

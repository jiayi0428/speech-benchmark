"""Compare Cascade and Direct on the paired human_speech_v1 samples."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np

from evaluate_white_noise import parsed_structured, score_record, wer


ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "experiments" / "human_speech_v1.json"
GROUND_TRUTH = ROOT / "data" / "ground_truth_human_v1.json"
RESULT_DIR = ROOT / "data" / "results" / "human_speech_v1"
CASCADE_INPUT = RESULT_DIR / "cascade_raw.json"
DIRECT_INPUT = RESULT_DIR / "direct_raw.jsonl"
DIRECT_STRUCTURED_INPUT = RESULT_DIR / "direct_postprocessed.jsonl"
SCORES_OUTPUT = RESULT_DIR / "path_comparison_scores.csv"
SUMMARY_OUTPUT = RESULT_DIR / "path_comparison_summary.json"
FIGURE_OUTPUT = ROOT / "report" / "figures" / "human_speech_path_comparison.png"
TASKS = ["summarization", "sentiment", "keywords", "intent"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def load_cascade(samples: list[str]) -> dict[tuple[str, str], dict[str, Any]]:
    payload = json.loads(CASCADE_INPUT.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Cascade input must be a list")
    if [entry.get("sample") for entry in payload] != samples:
        raise ValueError("Cascade sample order or keys do not match experiment config")

    records = {}
    for entry in payload:
        sample = entry["sample"]
        transcripts = set()
        for task in TASKS:
            source = entry.get(task)
            if not isinstance(source, dict) or source.get("task") != task:
                raise ValueError(f"Invalid Cascade record: {sample}/{task}")
            if not str(source.get("output", "")).strip():
                raise ValueError(f"Empty Cascade output: {sample}/{task}")
            transcript = str(source.get("transcript", "")).strip()
            if not transcript:
                raise ValueError(f"Empty Cascade transcript: {sample}/{task}")
            transcripts.add(transcript)
            records[(sample, task)] = {
                "sample": sample,
                "task": task,
                "output": source["output"],
                "transcript": transcript,
                "latency_seconds": float(source["latency_seconds"]),
            }
        if len(transcripts) != 1:
            raise ValueError(f"Cascade transcript differs across tasks: {sample}")
    return records


def load_direct(
    samples: list[str],
) -> tuple[
    dict[tuple[str, str], dict[str, Any]],
    dict[tuple[str, str], dict[str, Any]],
]:
    raw = [
        record
        for record in read_jsonl(DIRECT_INPUT)
        if record.get("status") == "success"
    ]
    structured = [
        record
        for record in read_jsonl(DIRECT_STRUCTURED_INPUT)
        if record.get("postprocess_status") == "success"
    ]
    raw_by_key = {(record["sample"], record["task"]): record for record in raw}
    structured_by_key = {
        (record["sample"], record["task"]): record for record in structured
    }
    expected = {(sample, task) for sample in samples for task in TASKS}
    expected_structured = {
        (sample, task)
        for sample in samples
        for task in TASKS
        if task != "summarization"
    }
    if set(raw_by_key) != expected or len(raw_by_key) != len(raw):
        raise ValueError("Direct raw results are missing or duplicated")
    if set(structured_by_key) != expected_structured:
        raise ValueError("Direct structured results are missing")

    evaluated = {}
    for key, raw_record in raw_by_key.items():
        evaluated[key] = (
            raw_record
            if raw_record["task"] == "summarization"
            else structured_by_key[key]
        )
    return evaluated, raw_by_key


def write_figure(summary: dict[str, Any], samples: list[str]) -> None:
    os.environ.setdefault(
        "MPLCONFIGDIR",
        str(RESULT_DIR / ".matplotlib"),
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    labels = ["Summary\nROUGE-L", "Sentiment\naccuracy", "Keyword\nF1", "Intent\naccuracy"]
    cascade_values = [summary["means"]["cascade"][task] for task in TASKS]
    direct_values = [summary["means"]["direct"][task] for task in TASKS]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(len(TASKS))
    width = 0.36
    axes[0].bar(
        x - width / 2,
        cascade_values,
        width,
        label="Cascade",
        color="#2563eb",
    )
    axes[0].bar(
        x + width / 2,
        direct_values,
        width,
        label="Direct",
        color="#dc2626",
    )
    axes[0].set_xticks(x, labels)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("Score")
    axes[0].set_title("Human speech task scores (N=8)")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend()

    sample_x = np.arange(len(samples))
    cascade_summary = [
        summary["sample_scores"][sample]["cascade"]["summarization"]
        for sample in samples
    ]
    direct_summary = [
        summary["sample_scores"][sample]["direct"]["summarization"]
        for sample in samples
    ]
    axes[1].bar(
        sample_x - width / 2,
        cascade_summary,
        width,
        label="Cascade",
        color="#2563eb",
    )
    axes[1].bar(
        sample_x + width / 2,
        direct_summary,
        width,
        label="Direct",
        color="#dc2626",
    )
    axes[1].set_xticks(sample_x, samples, rotation=35, ha="right")
    axes[1].set_ylim(0, 0.75)
    axes[1].set_ylabel("ROUGE-L")
    axes[1].set_title("Per-sample summarization")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(FIGURE_OUTPUT, dpi=180)
    plt.close(fig)


def main() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    truth = json.loads(GROUND_TRUTH.read_text(encoding="utf-8"))
    samples = list(config["samples"])
    if set(truth) != set(samples):
        raise ValueError("Ground truth does not match experiment samples")

    cascade = load_cascade(samples)
    direct, direct_raw = load_direct(samples)
    rows = []
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    latency: dict[tuple[str, str], list[float]] = defaultdict(list)
    differences: dict[str, list[float]] = defaultdict(list)
    wins = {
        task: {"direct": 0, "cascade": 0, "ties": 0}
        for task in TASKS
    }
    sample_scores: dict[str, dict[str, dict[str, float]]] = {
        sample: {"cascade": {}, "direct": {}} for sample in samples
    }

    for sample in samples:
        for task in TASKS:
            cascade_record = cascade[(sample, task)]
            direct_record = direct[(sample, task)]
            cascade_score, cascade_json = score_record(
                cascade_record,
                truth[sample],
            )
            direct_score, direct_json = score_record(
                direct_record,
                truth[sample],
            )
            difference = direct_score - cascade_score
            if abs(difference) < 1e-12:
                winner = "tie"
                wins[task]["ties"] += 1
            elif difference > 0:
                winner = "direct"
                wins[task]["direct"] += 1
            else:
                winner = "cascade"
                wins[task]["cascade"] += 1
            rows.append(
                {
                    "sample": sample,
                    "task": task,
                    "cascade_score": cascade_score,
                    "direct_score": direct_score,
                    "direct_minus_cascade": difference,
                    "winner": winner,
                    "cascade_valid_json": cascade_json,
                    "direct_valid_json": direct_json,
                    "cascade_latency_seconds": cascade_record["latency_seconds"],
                    "direct_latency_seconds": direct_raw[
                        (sample, task)
                    ]["latency_seconds"],
                }
            )
            grouped[("cascade", task)].append(cascade_score)
            grouped[("direct", task)].append(direct_score)
            latency[("cascade", task)].append(
                cascade_record["latency_seconds"]
            )
            latency[("direct", task)].append(
                float(direct_raw[(sample, task)]["latency_seconds"])
            )
            differences[task].append(difference)
            sample_scores[sample]["cascade"][task] = cascade_score
            sample_scores[sample]["direct"][task] = direct_score

    with SCORES_OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    cascade_transcripts = {
        sample: cascade[(sample, "summarization")]["transcript"]
        for sample in samples
    }
    wer_by_sample = {
        sample: wer(truth[sample]["transcript"], cascade_transcripts[sample])
        for sample in samples
    }
    strict_json_rates = {}
    for pipeline, records in [("cascade", cascade), ("direct", direct)]:
        strict_json_rates[pipeline] = {}
        for task in ["sentiment", "keywords", "intent"]:
            values = [
                float(parsed_structured(records[(sample, task)]["output"])[1])
                for sample in samples
            ]
            strict_json_rates[pipeline][task] = mean(values)

    means = {
        pipeline: {
            task: mean(grouped[(pipeline, task)]) for task in TASKS
        }
        for pipeline in ["cascade", "direct"]
    }
    latency_means = {
        pipeline: {
            task: mean(latency[(pipeline, task)]) for task in TASKS
        }
        for pipeline in ["cascade", "direct"]
    }
    summary = {
        "experiment_id": "human_speech_v1_path_comparison",
        "sample_count": len(samples),
        "paired_result_count": len(rows),
        "tasks": TASKS,
        "means": means,
        "direct_minus_cascade": {
            task: mean(values) for task, values in differences.items()
        },
        "paired_bootstrap_95_ci": {
            task: bootstrap_mean_ci(values, seed=42 + index)
            for index, (task, values) in enumerate(differences.items())
        },
        "sample_wins": wins,
        "strict_json_rates_after_direct_postprocessing": strict_json_rates,
        "cascade_asr_wer": {
            "mean": mean(wer_by_sample.values()),
            "by_sample": wer_by_sample,
        },
        "recorded_latency_seconds": {
            "means_by_task": latency_means,
            "caveat": (
                "Cascade and Direct timings come from different machines and "
                "timing boundaries; do not interpret them as architecture speed."
            ),
        },
        "sample_scores": sample_scores,
        "source_provenance": {
            "cascade_raw_sha256": sha256_file(CASCADE_INPUT),
            "cascade_original_audio_paths": "external C:\\Users\\18553\\Desktop",
            "direct_audio_hashes_recorded": True,
        },
        "caveats": [
            "N=8 supports descriptive trends only, not statistical significance.",
            "Recordings contain uncontrolled environmental and audience noise.",
            "There are no matched clean recordings, so effects cannot be attributed to noise.",
            "The supplied Cascade file has no audio hashes, so exact byte identity with the Direct audio cannot be independently verified.",
            "Cascade and Direct were run in different environments; latency is not comparable.",
            "Direct structured scores include DeepSeek post-processing.",
            "ROUGE-L, WER, and keyword F1 use simple whitespace or exact-phrase matching.",
        ],
    }
    SUMMARY_OUTPUT.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_figure(summary, samples)
    print(json.dumps(means, indent=2))
    print(f"Cascade mean WER: {summary['cascade_asr_wer']['mean']:.4f}")
    print(f"Scores: {SCORES_OUTPUT}")
    print(f"Summary: {SUMMARY_OUTPUT}")
    print(f"Figure: {FIGURE_OUTPUT}")


if __name__ == "__main__":
    main()

"""Combine human-speech v1, v2, and v3 into one N=26 ABCD evaluation."""
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
OUTPUT_DIR = ROOT / "data" / "results" / "human_speech_combined_n26"
COMBINED_EXPERIMENT_ID = "human_speech_combined_n26"
NEW_CALLS = 134
TASKS = ["summarization", "sentiment", "keywords", "intent"]
PIPELINES = [
    "A_oracle",
    "B_whisper_cascade",
    "C_qwen_transcript",
    "D_qwen_direct_system",
]
DATASETS = {
    "v1": {
        "config": ROOT / "experiments" / "human_speech_v1.json",
        "truth": ROOT / "data" / "ground_truth_human_v1.json",
        "result": ROOT / "data" / "results" / "human_speech_v1",
        "a": "oracle_tasks_raw.jsonl",
        "b": "cascade_raw.json",
        "c": "qwen_transcript_cascade_raw.jsonl",
        "d": "d_system_v1",
    },
    "v2": {
        "config": ROOT / "experiments" / "human_speech_v2.json",
        "truth": ROOT / "data" / "ground_truth_human_v2.json",
        "result": ROOT / "data" / "results" / "human_speech_v2",
        "a": "a_tasks_raw.jsonl",
        "b": "b_tasks_raw.jsonl",
        "c": "c_tasks_raw.jsonl",
        "d": "d_system_v1",
    },
    "v3": {
        "config": ROOT / "experiments" / "human_speech_v3.json",
        "truth": ROOT / "data" / "ground_truth_human_v3.json",
        "result": ROOT / "data" / "results" / "human_speech_v3",
        "a": "a_tasks_raw.jsonl",
        "b": "b_tasks_raw.jsonl",
        "c": "c_tasks_raw.jsonl",
        "d": ".",
    },
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def successes(path: Path, field: str = "status") -> list[dict[str, Any]]:
    return [record for record in read_jsonl(path) if record.get(field) == "success"]


def task_map(records: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    return {(record["sample"], record["task"]): record for record in records}


def load_b_v1(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        (entry["sample"], task): {
            "sample": entry["sample"],
            "task": task,
            "transcript": entry[task]["transcript"],
            "output": entry[task]["output"],
        }
        for entry in payload
        for task in TASKS
    }


def bootstrap_mean_ci(values: list[float], seed: int) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(20_000, len(array)))
    means = array[indices].mean(axis=1)
    return [float(value) for value in np.quantile(means, [0.025, 0.975])]


def usage(records: list[dict[str, Any]]) -> dict[str, int]:
    return {
        field: sum(int(record.get("usage", {}).get(field) or 0) for record in records)
        for field in ["prompt_tokens", "completion_tokens", "total_tokens"]
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    combined_index_dir = ROOT / "data" / "processed" / COMBINED_EXPERIMENT_ID
    combined_index_dir.mkdir(parents=True, exist_ok=True)
    records: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    truths = {}
    sample_lists = {}
    combined_truth = {}
    combined_index = []
    api_records: dict[str, list[dict[str, Any]]] = {
        pipeline: [] for pipeline in PIPELINES
    }
    transcription_sources = {}

    for dataset, spec in DATASETS.items():
        config = json.loads(spec["config"].read_text(encoding="utf-8"))
        truth = json.loads(spec["truth"].read_text(encoding="utf-8"))
        samples = list(config["samples"])
        truths[dataset] = truth
        sample_lists[dataset] = samples
        result_dir = spec["result"]
        manifest = json.loads(
            (
                ROOT
                / "data"
                / "processed"
                / f"human_speech_{dataset}"
                / "audio_manifest.json"
            ).read_text(encoding="utf-8")
        )
        manifest_by_sample = {
            record["sample"]: record for record in manifest["records"]
        }
        for sample in samples:
            combined_id = f"{dataset}/{sample}"
            combined_truth[combined_id] = {
                "dataset": dataset,
                "sample": sample,
                **truth[sample],
            }
            combined_index.append(
                {
                    "combined_id": combined_id,
                    "dataset": dataset,
                    **manifest_by_sample[sample],
                }
            )

        a_records = successes(result_dir / spec["a"])
        b_map = (
            load_b_v1(result_dir / spec["b"])
            if dataset == "v1"
            else task_map(successes(result_dir / spec["b"]))
        )
        c_records = successes(result_dir / spec["c"])
        d_dir = (result_dir / spec["d"]).resolve()
        d_raw = successes(d_dir / "direct_raw.jsonl")
        d_post = successes(
            d_dir / "direct_postprocessed.jsonl", "postprocess_status"
        )
        maps = {
            "A_oracle": task_map(a_records),
            "B_whisper_cascade": b_map,
            "C_qwen_transcript": task_map(c_records),
        }
        d_raw_map = task_map(d_raw)
        d_post_map = task_map(d_post)
        maps["D_qwen_direct_system"] = {
            (sample, task): (
                d_raw_map[(sample, task)]
                if task == "summarization"
                else d_post_map[(sample, task)]
            )
            for sample in samples
            for task in TASKS
        }
        expected = {(sample, task) for sample in samples for task in TASKS}
        for pipeline in PIPELINES:
            if set(maps[pipeline]) != expected:
                raise ValueError(f"{dataset} {pipeline} is incomplete")
            for (sample, task), record in maps[pipeline].items():
                records[(dataset, sample, task, pipeline)] = record
        api_records["A_oracle"].extend(a_records)
        if dataset != "v1":
            api_records["B_whisper_cascade"].extend(list(b_map.values()))
        api_records["C_qwen_transcript"].extend(c_records)
        api_records["D_qwen_direct_system"].extend(d_post)

        if dataset == "v1":
            whisper_transcripts = {
                sample: b_map[(sample, "summarization")]["transcript"]
                for sample in samples
            }
        else:
            whisper_transcripts = {
                record["sample"]: record["transcript"]
                for record in successes(result_dir / "whisper_transcription_raw.jsonl")
            }
        qwen_transcripts = {
            record["sample"]: record["transcript"]
            for record in successes(result_dir / "qwen_transcription_raw.jsonl")
        }
        transcription_sources[dataset] = {
            "B_whisper": whisper_transcripts,
            "C_qwen": qwen_transcripts,
        }

    rows = []
    scores = {}
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    dataset_grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for dataset, samples in sample_lists.items():
        for sample in samples:
            for task in TASKS:
                for pipeline in PIPELINES:
                    score, valid_json = score_record(
                        records[(dataset, sample, task, pipeline)],
                        truths[dataset][sample],
                    )
                    scores[(dataset, sample, task, pipeline)] = score
                    grouped[(pipeline, task)].append(score)
                    dataset_grouped[(dataset, pipeline, task)].append(score)
                    rows.append(
                        {
                            "dataset": dataset,
                            "sample": sample,
                            "task": task,
                            "pipeline": pipeline,
                            "score": score,
                            "valid_json": valid_json,
                        }
                    )
    with (OUTPUT_DIR / "scores.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    means = {
        pipeline: {task: mean(grouped[(pipeline, task)]) for task in TASKS}
        for pipeline in PIPELINES
    }
    by_dataset = {
        dataset: {
            pipeline: {
                task: mean(dataset_grouped[(dataset, pipeline, task)])
                for task in TASKS
            }
            for pipeline in PIPELINES
        }
        for dataset in DATASETS
    }
    all_samples = [
        (dataset, sample)
        for dataset, samples in sample_lists.items()
        for sample in samples
    ]
    pairwise = {}
    for pair_index, (left, right) in enumerate(combinations(PIPELINES, 2)):
        pairwise[f"{left}_minus_{right}"] = {}
        for task_index, task in enumerate(TASKS):
            differences = [
                scores[(dataset, sample, task, left)]
                - scores[(dataset, sample, task, right)]
                for dataset, sample in all_samples
            ]
            pairwise[f"{left}_minus_{right}"][task] = {
                "mean_difference": mean(differences),
                "paired_bootstrap_95_ci": bootstrap_mean_ci(
                    differences, 3100 + pair_index * 10 + task_index
                ),
                "left_wins": sum(value > 1e-12 for value in differences),
                "right_wins": sum(value < -1e-12 for value in differences),
                "ties": sum(abs(value) <= 1e-12 for value in differences),
            }

    transcription_rows = []
    transcription_values: dict[str, list[float]] = defaultdict(list)
    transcription_by_dataset = {}
    for dataset, sources in transcription_sources.items():
        transcription_by_dataset[dataset] = {}
        for pipeline, transcripts in sources.items():
            values = {}
            for sample in sample_lists[dataset]:
                cleaned = strip_transcription_wrapper(transcripts[sample])
                value = wer(
                    normalize_text(truths[dataset][sample]["transcript"]),
                    normalize_text(cleaned),
                )
                values[sample] = value
                transcription_values[pipeline].append(value)
                transcription_rows.append(
                    {
                        "dataset": dataset,
                        "sample": sample,
                        "pipeline": pipeline,
                        "normalized_wer": value,
                    }
                )
            transcription_by_dataset[dataset][pipeline] = {
                "mean_normalized_wer": mean(values.values()),
                "by_sample": values,
            }
    with (OUTPUT_DIR / "transcription_scores.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(transcription_rows[0])
        )
        writer.writeheader()
        writer.writerows(transcription_rows)

    api = {
        pipeline: {
            "successful_calls_with_usage_records": len(api_records[pipeline]),
            "usage": usage(api_records[pipeline]),
        }
        for pipeline in PIPELINES
    }
    api["B_whisper_cascade"]["untracked_v1_successful_calls"] = 32
    summary = {
        "experiment_id": COMBINED_EXPERIMENT_ID,
        "sample_count": len(all_samples),
        "dataset_counts": {
            dataset: len(samples) for dataset, samples in sample_lists.items()
        },
        "score_row_count": len(rows),
        "definitions": {
            "A_oracle": "Human transcript -> DeepSeek tasks",
            "B_whisper_cascade": "Whisper large-v3 transcript -> DeepSeek tasks",
            "C_qwen_transcript": "Qwen transcript -> DeepSeek tasks",
            "D_qwen_direct_system": (
                "Qwen direct System-turn -> DeepSeek structured formatting"
            ),
        },
        "means": means,
        "means_by_dataset": by_dataset,
        "pairwise": pairwise,
        "transcription": {
            pipeline: {
                "mean_normalized_wer": mean(values)
            }
            for pipeline, values in transcription_values.items()
        },
        "transcription_by_dataset": transcription_by_dataset,
        "api": api,
        "estimated_total_task_api_calls": len(all_samples) * 15,
        "estimated_total_cost_usd_at_0.0005_per_call": len(all_samples) * 15 * 0.0005,
        "new_calls_for_current_extension": NEW_CALLS,
        "estimated_new_cost_usd_at_0.0005_per_call": NEW_CALLS * 0.0005,
        "caveats": [
            f"N={len(all_samples)} supports stronger descriptive trends but no claim of statistical significance is made.",
            "All D records use qwen_system_task_v1 after isolated v1/v2 reruns.",
            "All A/B/C text tasks use the same DeepSeek prompts.",
            "V1 B raw records do not contain API token usage metadata.",
            "Latency is not compared across paths or execution environments.",
        ],
    }
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (
        ROOT / "data" / f"ground_truth_{COMBINED_EXPERIMENT_ID}.json"
    ).write_text(
        json.dumps(combined_truth, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (combined_index_dir / "index.json").write_text(
        json.dumps(
            {
                "experiment_id": COMBINED_EXPERIMENT_ID,
                "sample_count": len(combined_index),
                "records": combined_index,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(means, indent=2, ensure_ascii=False))
    print(json.dumps(summary["transcription"], indent=2, ensure_ascii=False))
    print(f"Scores: {OUTPUT_DIR / 'scores.csv'}")
    print(f"Summary: {OUTPUT_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()

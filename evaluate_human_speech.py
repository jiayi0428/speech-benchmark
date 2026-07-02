"""Evaluate the human_speech_v1 Direct results against human annotations."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from evaluate_white_noise import parsed_structured, score_record


ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "experiments" / "human_speech_v1.json"
GROUND_TRUTH = ROOT / "data" / "ground_truth_human_v1.json"
RESULT_DIR = ROOT / "data" / "results" / "human_speech_v1"
RAW = RESULT_DIR / "direct_raw.jsonl"
POSTPROCESSED = RESULT_DIR / "direct_postprocessed.jsonl"
TASKS = ["summarization", "sentiment", "keywords", "intent"]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    truth = json.loads(GROUND_TRUTH.read_text(encoding="utf-8"))
    samples = list(config["samples"])
    expected = {(sample, task) for sample in samples for task in TASKS}

    raw = [record for record in read_jsonl(RAW) if record.get("status") == "success"]
    postprocessed = [
        record
        for record in read_jsonl(POSTPROCESSED)
        if record.get("postprocess_status") == "success"
    ]
    raw_by_key = {(record["sample"], record["task"]): record for record in raw}
    structured_by_key = {
        (record["sample"], record["task"]): record for record in postprocessed
    }
    if len(raw_by_key) != len(raw):
        raise ValueError("Duplicate successful raw result keys")
    if set(raw_by_key) != expected:
        raise ValueError(
            f"Raw result mismatch; missing={sorted(expected - set(raw_by_key))}, "
            f"extra={sorted(set(raw_by_key) - expected)}"
        )
    expected_structured = {
        (sample, task)
        for sample in samples
        for task in TASKS
        if task != "summarization"
    }
    if set(structured_by_key) != expected_structured:
        raise ValueError(
            "Post-processed result mismatch; "
            f"missing={sorted(expected_structured - set(structured_by_key))}, "
            f"extra={sorted(set(structured_by_key) - expected_structured)}"
        )
    if set(truth) != set(samples):
        raise ValueError("Ground-truth sample keys do not match experiment config")

    rows = []
    grouped: dict[str, list[float]] = defaultdict(list)
    strict_grouped: dict[str, list[float]] = defaultdict(list)
    sample_scores: dict[str, dict[str, float]] = defaultdict(dict)
    for sample in samples:
        for task in TASKS:
            raw_record = raw_by_key[(sample, task)]
            record = (
                raw_record
                if task == "summarization"
                else structured_by_key[(sample, task)]
            )
            score, valid_json = score_record(record, truth[sample])
            rows.append(
                {
                    "sample": sample,
                    "task": task,
                    "score": score,
                    "valid_json": valid_json,
                    "qwen_latency_seconds": raw_record["latency_seconds"],
                    "postprocess_latency_seconds": record.get(
                        "postprocess_latency_seconds"
                    ),
                }
            )
            grouped[task].append(score)
            sample_scores[sample][task] = score
            if valid_json is not None:
                strict_grouped[task].append(valid_json)

    with (RESULT_DIR / "scores.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    raw_structured = [
        record for record in raw if record["task"] != "summarization"
    ]
    raw_strict_count = sum(
        parsed_structured(record["output"])[1] for record in raw_structured
    )
    usage_totals = {
        field: sum(
            int(record.get("postprocess_usage", {}).get(field) or 0)
            for record in postprocessed
        )
        for field in ["prompt_tokens", "completion_tokens", "total_tokens"]
    }
    qwen_latencies = [float(record["latency_seconds"]) for record in raw]
    postprocess_latencies = [
        float(record["postprocess_latency_seconds"]) for record in postprocessed
    ]
    summary = {
        "experiment_id": config["experiment_id"],
        "pipeline": "direct",
        "condition": config["condition"],
        "sample_count": len(samples),
        "result_count": len(rows),
        "ground_truth": "human supplied",
        "means": {task: mean(scores) for task, scores in grouped.items()},
        "postprocessed_valid_json_rates": {
            task: mean(values) for task, values in strict_grouped.items()
        },
        "raw_structured_valid_json": {
            "valid": raw_strict_count,
            "total": len(raw_structured),
            "rate": raw_strict_count / len(raw_structured),
        },
        "latency_seconds": {
            "qwen_mean": mean(qwen_latencies),
            "qwen_min": min(qwen_latencies),
            "qwen_max": max(qwen_latencies),
            "postprocess_mean": mean(postprocess_latencies),
        },
        "postprocess_api": {
            "call_count": len(postprocessed),
            "provider": postprocessed[0].get("postprocess_provider"),
            "model": postprocessed[0].get("postprocess_model"),
            "usage": usage_totals,
            "estimated_cost_usd_using_project_0.0005_per_call_assumption": (
                len(postprocessed) * 0.0005
            ),
        },
        "sample_scores": sample_scores,
        "caveats": [
            "N=8 supports descriptive trends only, not statistical significance.",
            "Recordings contain uncontrolled environmental and audience noise.",
            "There is no matched clean recording, so effects cannot be attributed to noise.",
            "Structured tasks include DeepSeek post-processing; summarization does not.",
        ],
    }
    (RESULT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Evaluated {len(rows)} Direct results")
    print(json.dumps(summary["means"], indent=2))


if __name__ == "__main__":
    main()


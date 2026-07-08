"""Extend the harmonized human-speech benchmark from N=26 to N=50."""
import csv
import json
from pathlib import Path

import evaluate_human_speech_combined_n26 as evaluator


evaluator.COMBINED_EXPERIMENT_ID = "human_speech_combined_n50"
evaluator.OUTPUT_DIR = (
    Path(__file__).resolve().parent
    / "data"
    / "results"
    / evaluator.COMBINED_EXPERIMENT_ID
)
evaluator.NEW_CALLS = 360
evaluator.DATASETS["v4"] = {
    "config": evaluator.ROOT / "experiments" / "human_speech_v4.json",
    "truth": evaluator.ROOT / "data" / "ground_truth_human_v4.json",
    "result": evaluator.ROOT / "data" / "results" / "human_speech_v4",
    "a": "a_tasks_raw.jsonl",
    "b": "b_tasks_raw.jsonl",
    "c": "c_tasks_raw.jsonl",
    "d": ".",
}


if __name__ == "__main__":
    evaluator.main()
    combined_summary = json.loads(
        (evaluator.OUTPUT_DIR / "summary.json").read_text(encoding="utf-8")
    )
    v4_result_dir = evaluator.ROOT / "data" / "results" / "human_speech_v4"
    v4_api_files = {
        "A": ("a_tasks_raw.jsonl", "status"),
        "B": ("b_tasks_raw.jsonl", "status"),
        "C": ("c_tasks_raw.jsonl", "status"),
        "D": ("direct_postprocessed.jsonl", "postprocess_status"),
    }
    v4_api = {}
    for path_name, (filename, status_field) in v4_api_files.items():
        records = evaluator.successes(v4_result_dir / filename, status_field)
        v4_api[path_name] = {
            "successful_calls": len(records),
            "usage": evaluator.usage(records),
        }
    v4_summary = {
        "experiment_id": "human_speech_v4",
        "sample_count": 24,
        "means": combined_summary["means_by_dataset"]["v4"],
        "transcription": combined_summary["transcription_by_dataset"]["v4"],
        "api": v4_api,
        "estimated_cost_usd_at_0.0005_per_call": 0.18,
        "caveats": [
            "N=24 supports descriptive trends only, not statistical significance.",
            "D uses qwen_system_task_v1.",
            "Latency is not compared across paths.",
        ],
    }
    (v4_result_dir / "summary.json").write_text(
        json.dumps(v4_summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    for filename in ["scores.csv", "transcription_scores.csv"]:
        with (evaluator.OUTPUT_DIR / filename).open(
            encoding="utf-8", newline=""
        ) as source:
            rows = [
                row for row in csv.DictReader(source)
                if row["dataset"] == "v4"
            ]
        with (v4_result_dir / filename).open(
            "w", encoding="utf-8", newline=""
        ) as target:
            writer = csv.DictWriter(target, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

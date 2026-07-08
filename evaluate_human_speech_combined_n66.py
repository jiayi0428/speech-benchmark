"""Extend the harmonized human-speech benchmark from N=59 to N=66."""
import csv
import json
from pathlib import Path

import evaluate_human_speech_combined_n59 as n59
import evaluate_human_speech_combined_n26 as evaluator


evaluator.COMBINED_EXPERIMENT_ID = "human_speech_combined_n66"
evaluator.OUTPUT_DIR = (
    Path(__file__).resolve().parent
    / "data"
    / "results"
    / evaluator.COMBINED_EXPERIMENT_ID
)
evaluator.NEW_CALLS = 105
evaluator.DATASETS["v6"] = {
    "config": evaluator.ROOT / "experiments" / "human_speech_v6.json",
    "truth": evaluator.ROOT / "data" / "ground_truth_human_v6.json",
    "result": evaluator.ROOT / "data" / "results" / "human_speech_v6",
    "a": "a_tasks_raw.jsonl",
    "b": "b_tasks_raw.jsonl",
    "c": "c_tasks_raw.jsonl",
    "d": ".",
}


if __name__ == "__main__":
    # Importing n59 registers v4 and v5 into the shared evaluator before adding v6.
    evaluator.main()
    combined_summary = json.loads(
        (evaluator.OUTPUT_DIR / "summary.json").read_text(encoding="utf-8")
    )
    v6_result_dir = evaluator.ROOT / "data" / "results" / "human_speech_v6"
    v6_api_files = {
        "A": ("a_tasks_raw.jsonl", "status"),
        "B": ("b_tasks_raw.jsonl", "status"),
        "C": ("c_tasks_raw.jsonl", "status"),
        "D": ("direct_postprocessed.jsonl", "postprocess_status"),
    }
    v6_api = {}
    for path_name, (filename, status_field) in v6_api_files.items():
        records = evaluator.successes(v6_result_dir / filename, status_field)
        v6_api[path_name] = {
            "successful_calls": len(records),
            "usage": evaluator.usage(records),
        }
    v6_summary = {
        "experiment_id": "human_speech_v6",
        "sample_count": 7,
        "means": combined_summary["means_by_dataset"]["v6"],
        "transcription": combined_summary["transcription_by_dataset"]["v6"],
        "api": v6_api,
        "estimated_cost_usd_at_0.0005_per_call": 0.0525,
        "caveats": [
            "N=7 supports descriptive trends only, not statistical significance.",
            "D uses qwen_system_task_v1.",
            "Latency is not compared across paths.",
            "The source annotation had intent='qustion' for question.wav; the project ground truth normalizes it to 'question'.",
        ],
    }
    (v6_result_dir / "summary.json").write_text(
        json.dumps(v6_summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    for filename in ["scores.csv", "transcription_scores.csv"]:
        with (evaluator.OUTPUT_DIR / filename).open(
            encoding="utf-8", newline=""
        ) as source:
            rows = [
                row for row in csv.DictReader(source)
                if row["dataset"] == "v6"
            ]
        with (v6_result_dir / filename).open(
            "w", encoding="utf-8", newline=""
        ) as target:
            writer = csv.DictWriter(target, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

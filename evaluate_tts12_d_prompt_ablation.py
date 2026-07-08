"""Compare the original TTS12 Direct run with a prompt-placement ablation."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from evaluate_white_noise import score_record


ROOT = Path(__file__).resolve().parent
INDEX = ROOT / "data" / "processed" / "tts12_cd_v1" / "index.json"
GROUND_TRUTH = ROOT / "data" / "ground_truth_tts12_cd_v1.json"
ORIGINAL_DIR = ROOT / "data" / "results" / "tts12_cd_v1"
DEFAULT_VARIANT_DIR = ROOT / "data" / "results" / "tts12_d_system_prompt_v1"
TASKS = ["summarization", "sentiment", "keywords", "intent"]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def successful_records(result_dir: Path) -> dict[tuple[str, str], dict[str, Any]]:
    raw = [
        record
        for record in read_jsonl(result_dir / "direct_raw.jsonl")
        if record.get("status") == "success"
    ]
    structured = [
        record
        for record in read_jsonl(result_dir / "direct_postprocessed.jsonl")
        if record.get("postprocess_status") == "success"
    ]
    raw_by_key = {(record["sample"], record["task"]): record for record in raw}
    structured_by_key = {
        (record["sample"], record["task"]): record for record in structured
    }
    result = {}
    for key, record in raw_by_key.items():
        result[key] = record if key[1] == "summarization" else structured_by_key[key]
    return result


def usage(records: list[dict[str, Any]]) -> dict[str, int]:
    return {
        field: sum(int(record.get("usage", {}).get(field) or 0) for record in records)
        for field in ["prompt_tokens", "completion_tokens", "total_tokens"]
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant-dir", type=Path, default=DEFAULT_VARIANT_DIR)
    args = parser.parse_args()
    variant_dir = args.variant_dir.resolve()

    samples = [
        entry["topic"]
        for entry in json.loads(INDEX.read_text(encoding="utf-8"))
    ]
    truth = json.loads(GROUND_TRUTH.read_text(encoding="utf-8"))
    expected = {(sample, task) for sample in samples for task in TASKS}
    original = successful_records(ORIGINAL_DIR)
    variant = successful_records(variant_dir)
    if set(original) != expected or set(variant) != expected:
        raise ValueError("Both Direct runs must contain 48 complete task results")

    rows = []
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    exact_matches = 0
    for sample in samples:
        for task in TASKS:
            for run_name, records in [
                ("user_prompt_original", original),
                ("system_prompt_variant", variant),
            ]:
                record = records[(sample, task)]
                score, valid_json = score_record(record, truth[sample])
                grouped[(run_name, task)].append(score)
                rows.append(
                    {
                        "sample": sample,
                        "task": task,
                        "run": run_name,
                        "score": score,
                        "valid_json": valid_json,
                        "output": record["output"],
                    }
                )
            exact_matches += (
                original[(sample, task)]["output"]
                == variant[(sample, task)]["output"]
            )

    scores_path = variant_dir / "prompt_ablation_scores.csv"
    with scores_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    means = {
        run_name: {
            task: mean(grouped[(run_name, task)]) for task in TASKS
        }
        for run_name in ["user_prompt_original", "system_prompt_variant"]
    }
    postprocess_attempts = read_jsonl(variant_dir / "direct_postprocessed.jsonl")
    successful_postprocess = [
        record
        for record in postprocess_attempts
        if record.get("postprocess_status") == "success"
    ]
    summary = {
        "experiment_id": "tts12_d_system_prompt_v1",
        "sample_count": 12,
        "comparison": "Only Qwen prompt placement is intentionally changed.",
        "means": means,
        "system_minus_user": {
            task: means["system_prompt_variant"][task]
            - means["user_prompt_original"][task]
            for task in TASKS
        },
        "exact_final_output_matches": exact_matches,
        "row_count_per_run": 48,
        "api": {
            "successful_calls": len(successful_postprocess),
            "failed_connection_attempts": sum(
                record.get("postprocess_status") == "error"
                for record in postprocess_attempts
            ),
            "usage": usage(successful_postprocess),
            "estimated_cost_usd_at_0.0005_per_successful_call": (
                len(successful_postprocess) * 0.0005
            ),
        },
        "caveats": [
            "N=12 supports descriptive trends only, not statistical significance.",
            "Latency is not interpreted as an architecture-speed difference.",
        ],
    }
    summary_path = variant_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Scores: {scores_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()

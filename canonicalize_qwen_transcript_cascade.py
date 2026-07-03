"""Preserve the API audit and canonicalize duplicate ablation-C responses."""
from __future__ import annotations

import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "experiments" / "human_speech_v1.json"
RESULT_DIR = ROOT / "data" / "results" / "human_speech_v1"
INPUT = RESULT_DIR / "qwen_transcript_cascade_raw.jsonl"
AUDIT = RESULT_DIR / "qwen_transcript_cascade_call_audit.jsonl"
AUDIT_SUMMARY = RESULT_DIR / "qwen_transcript_cascade_audit_summary.json"
TASKS = ["summarization", "sentiment", "keywords", "intent"]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def usage_totals(records: list[dict[str, Any]]) -> dict[str, int]:
    return {
        field: sum(int(record.get("usage", {}).get(field) or 0) for record in records)
        for field in ["prompt_tokens", "completion_tokens", "total_tokens"]
    }


def main() -> None:
    if AUDIT.exists():
        raise FileExistsError(
            f"{AUDIT} already exists; refusing to overwrite the preserved audit"
        )
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    samples = list(config["samples"])
    records = read_jsonl(INPUT)
    shutil.copy2(INPUT, AUDIT)

    first_success: dict[tuple[str, str], dict[str, Any]] = {}
    counts: Counter[tuple[str, str]] = Counter()
    for record in records:
        if record.get("status") != "success":
            continue
        key = (record["sample"], record["task"])
        counts[key] += 1
        first_success.setdefault(key, record)

    expected = {(sample, task) for sample in samples for task in TASKS}
    if set(first_success) != expected:
        raise ValueError(
            f"Canonical key mismatch; missing={sorted(expected - set(first_success))}, "
            f"extra={sorted(set(first_success) - expected)}"
        )
    canonical = [
        first_success[(sample, task)] for sample in samples for task in TASKS
    ]
    INPUT.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False) + "\n" for record in canonical
        ),
        encoding="utf-8",
    )

    duplicate_keys = {
        f"{sample}/{task}": count
        for (sample, task), count in sorted(counts.items())
        if count > 1
    }
    duplicate_groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for record in records:
        if record.get("status") == "success":
            duplicate_groups.setdefault(
                (record["sample"], record["task"]),
                [],
            ).append(record)
    differing_keys = sorted(
        f"{sample}/{task}"
        for (sample, task), group in duplicate_groups.items()
        if len(group) > 1
        and len({str(record.get("output", "")) for record in group}) > 1
    )
    summary = {
        "experiment_id": "human_speech_v1_ablation_c_api_audit",
        "selection_rule": "first successful response per sample/task key",
        "recorded_successful_api_calls": sum(counts.values()),
        "required_canonical_calls": len(canonical),
        "extra_duplicate_calls": sum(counts.values()) - len(canonical),
        "duplicate_keys": duplicate_keys,
        "duplicate_output_stability": {
            "same_output_keys": len(duplicate_keys) - len(differing_keys),
            "different_output_keys": len(differing_keys),
            "different_keys": differing_keys,
        },
        "all_call_usage": usage_totals(
            [record for record in records if record.get("status") == "success"]
        ),
        "canonical_usage": usage_totals(canonical),
        "estimated_all_call_cost_usd_at_0.0005_per_call": sum(counts.values())
        * 0.0005,
        "estimated_canonical_cost_usd_at_0.0005_per_call": len(canonical)
        * 0.0005,
        "cause": (
            "The initial command timed out while its process continued running; "
            "manual resumptions overlapped with that process."
        ),
    }
    AUDIT_SUMMARY.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Preserved audit: {AUDIT}")
    print(f"Canonical results: {INPUT}")


if __name__ == "__main__":
    main()

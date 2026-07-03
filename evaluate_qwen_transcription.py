"""Evaluate Qwen transcription and compare it with supplied Whisper transcripts."""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from statistics import mean
from typing import Any

from evaluate_white_noise import wer
from src.transcription_utils import strip_transcription_wrapper


ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "experiments" / "human_speech_v1.json"
GROUND_TRUTH = ROOT / "data" / "ground_truth_human_v1.json"
RESULT_DIR = ROOT / "data" / "results" / "human_speech_v1"
QWEN_INPUT = RESULT_DIR / "qwen_transcription_raw.jsonl"
CASCADE_INPUT = RESULT_DIR / "cascade_raw.json"
SCORES_OUTPUT = RESULT_DIR / "qwen_transcription_scores.csv"
SUMMARY_OUTPUT = RESULT_DIR / "qwen_transcription_summary.json"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def normalize_text(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", text.lower()))


def main() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    truth = json.loads(GROUND_TRUTH.read_text(encoding="utf-8"))
    cascade_payload = json.loads(CASCADE_INPUT.read_text(encoding="utf-8"))
    samples = list(config["samples"])
    qwen_records = [
        record
        for record in read_jsonl(QWEN_INPUT)
        if record.get("status") == "success"
    ]
    qwen_by_sample = {record["sample"]: record for record in qwen_records}
    cascade_by_sample = {entry["sample"]: entry for entry in cascade_payload}
    if (
        set(qwen_by_sample) != set(samples)
        or len(qwen_by_sample) != len(qwen_records)
        or set(cascade_by_sample) != set(samples)
    ):
        raise ValueError("Transcription result keys do not match experiment samples")

    rows = []
    for sample in samples:
        reference = truth[sample]["transcript"]
        qwen_transcript = qwen_by_sample[sample]["transcript"]
        qwen_cleaned = strip_transcription_wrapper(qwen_transcript)
        whisper_transcript = cascade_by_sample[sample]["summarization"]["transcript"]
        rows.append(
            {
                "sample": sample,
                "qwen_project_wer": wer(reference, qwen_transcript),
                "qwen_cleaned_project_wer": wer(reference, qwen_cleaned),
                "whisper_project_wer": wer(reference, whisper_transcript),
                "qwen_normalized_wer": wer(
                    normalize_text(reference),
                    normalize_text(qwen_transcript),
                ),
                "qwen_cleaned_normalized_wer": wer(
                    normalize_text(reference),
                    normalize_text(qwen_cleaned),
                ),
                "whisper_normalized_wer": wer(
                    normalize_text(reference),
                    normalize_text(whisper_transcript),
                ),
                "qwen_latency_seconds": qwen_by_sample[sample]["latency_seconds"],
                "qwen_transcript_raw": qwen_transcript,
                "qwen_transcript_cleaned": qwen_cleaned,
                "whisper_transcript": whisper_transcript,
            }
        )

    with SCORES_OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "experiment_id": "human_speech_v1_ablation_c_transcription",
        "sample_count": len(samples),
        "means": {
            "qwen_project_wer": mean(row["qwen_project_wer"] for row in rows),
            "qwen_cleaned_project_wer": mean(
                row["qwen_cleaned_project_wer"] for row in rows
            ),
            "whisper_project_wer": mean(
                row["whisper_project_wer"] for row in rows
            ),
            "qwen_normalized_wer": mean(
                row["qwen_normalized_wer"] for row in rows
            ),
            "qwen_cleaned_normalized_wer": mean(
                row["qwen_cleaned_normalized_wer"] for row in rows
            ),
            "whisper_normalized_wer": mean(
                row["whisper_normalized_wer"] for row in rows
            ),
            "qwen_latency_seconds": mean(
                float(row["qwen_latency_seconds"]) for row in rows
            ),
        },
        "by_sample": {
            row["sample"]: {
                key: row[key]
                for key in [
                    "qwen_project_wer",
                    "qwen_cleaned_project_wer",
                    "whisper_project_wer",
                    "qwen_normalized_wer",
                    "qwen_cleaned_normalized_wer",
                    "whisper_normalized_wer",
                ]
            }
            for row in rows
        },
        "caveats": [
            "The supplied Whisper results have no audio hashes.",
            "Project WER uses lowercase whitespace tokens and retains punctuation.",
            "Normalized WER lowercases text and removes punctuation for a secondary check.",
            "Cleaned Qwen transcripts remove only fixed meta prefixes and outer quotes; recognized words are not corrected.",
            "N=8 supports descriptive trends only.",
        ],
    }
    SUMMARY_OUTPUT.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary["means"], indent=2))
    print(f"Scores: {SCORES_OUTPUT}")
    print(f"Summary: {SUMMARY_OUTPUT}")


if __name__ == "__main__":
    main()

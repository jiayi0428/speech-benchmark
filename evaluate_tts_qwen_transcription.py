"""Evaluate Qwen transcription on the original TTS samples."""
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
INDEX = ROOT / "data" / "processed" / "tts_samples" / "index.json"
RESULT_DIR = ROOT / "data" / "results" / "tts_qwen_transcript_v1"
INPUT = RESULT_DIR / "qwen_transcription_raw.jsonl"
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
    entries = json.loads(INDEX.read_text(encoding="utf-8"))
    truth = {entry["topic"]: entry["transcript"] for entry in entries}
    samples = list(truth)
    records = [
        record
        for record in read_jsonl(INPUT)
        if record.get("status") == "success"
    ]
    by_sample = {record["sample"]: record for record in records}
    if set(by_sample) != set(samples) or len(by_sample) != len(records):
        raise ValueError("TTS transcription results are missing or duplicated")

    rows = []
    for sample in samples:
        reference = truth[sample]
        raw = by_sample[sample]["transcript"]
        cleaned = strip_transcription_wrapper(raw)
        rows.append(
            {
                "sample": sample,
                "voice": by_sample[sample]["tts_voice"],
                "project_wer_raw": wer(reference, raw),
                "project_wer_cleaned": wer(reference, cleaned),
                "normalized_wer_raw": wer(
                    normalize_text(reference),
                    normalize_text(raw),
                ),
                "normalized_wer_cleaned": wer(
                    normalize_text(reference),
                    normalize_text(cleaned),
                ),
                "latency_seconds": by_sample[sample]["latency_seconds"],
                "transcript_raw": raw,
                "transcript_cleaned": cleaned,
            }
        )

    SCORES_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with SCORES_OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "experiment_id": "tts_qwen_transcript_v1",
        "sample_count": len(samples),
        "means": {
            field: mean(float(row[field]) for row in rows)
            for field in [
                "project_wer_raw",
                "project_wer_cleaned",
                "normalized_wer_raw",
                "normalized_wer_cleaned",
                "latency_seconds",
            ]
        },
        "by_sample": {
            row["sample"]: {
                key: row[key]
                for key in [
                    "project_wer_raw",
                    "project_wer_cleaned",
                    "normalized_wer_raw",
                    "normalized_wer_cleaned",
                ]
            }
            for row in rows
        },
        "caveats": [
            "Only Qwen transcription is available; the stored TTS Cascade summary file contains no Whisper transcripts.",
            "Cleaned transcripts remove only fixed meta prefixes and outer quotes.",
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


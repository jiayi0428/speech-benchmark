"""Static checks for the original-TTS Qwen transcription experiment."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_tts_index_has_eight_unique_audio_samples() -> None:
    entries = json.loads(
        (
            ROOT / "data" / "processed" / "tts_samples" / "index.json"
        ).read_text(encoding="utf-8")
    )
    assert len(entries) == 8
    assert len({entry["topic"] for entry in entries}) == 8
    assert all((ROOT / entry["audio_path"]).is_file() for entry in entries)
    assert all(entry["transcript"].strip() for entry in entries)


def test_tts_transcription_results_are_complete_when_present() -> None:
    path = (
        ROOT
        / "data"
        / "results"
        / "tts_qwen_transcript_v1"
        / "qwen_transcription_raw.jsonl"
    )
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(records) == 8
    assert all(record["status"] == "success" for record in records)
    assert len({record["sample"] for record in records}) == 8


def test_tts_c_results_have_32_unique_successful_keys() -> None:
    path = (
        ROOT
        / "data"
        / "results"
        / "tts_qwen_transcript_v1"
        / "qwen_transcript_cascade_raw.jsonl"
    )
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(records) == 32
    assert all(record["status"] == "success" for record in records)
    assert len({(record["sample"], record["task"]) for record in records}) == 32


def test_tts_comparison_has_full_cd_and_summary_only_b() -> None:
    path = (
        ROOT
        / "data"
        / "results"
        / "tts_qwen_transcript_v1"
        / "comparison_summary.json"
    )
    summary = json.loads(path.read_text(encoding="utf-8"))
    assert summary["sample_count"] == 8
    assert summary["score_row_count"] == 72
    assert set(summary["means"]["C_qwen_transcript"]) == {
        "summarization",
        "sentiment",
        "keywords",
        "intent",
    }
    assert set(summary["means"]["D_qwen_direct"]) == {
        "summarization",
        "sentiment",
        "keywords",
        "intent",
    }
    assert set(summary["means"]["B_whisper_cascade"]) == {"summarization"}
    assert summary["c_api"]["successful_call_count"] == 32

"""Integrity checks for human-speech-v3."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_human_speech_v3_prepared_inputs_match_truth_when_present() -> None:
    manifest_path = (
        ROOT / "data" / "processed" / "human_speech_v3" / "audio_manifest.json"
    )
    truth_path = ROOT / "data" / "ground_truth_human_v3.json"
    if not manifest_path.exists() or not truth_path.exists():
        return
    config = json.loads(
        (ROOT / "experiments" / "human_speech_v3.json").read_text(
            encoding="utf-8"
        )
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    truth = json.loads(truth_path.read_text(encoding="utf-8"))
    assert len(config["samples"]) == 8
    assert set(config["samples"]) == set(truth)
    assert {record["sample"] for record in manifest["records"]} == set(truth)
    assert all(record["sample_rate"] == 16000 for record in manifest["records"])
    assert all(record["channels"] == 1 for record in manifest["records"])


def test_human_speech_v3_summary_is_complete_when_present() -> None:
    path = ROOT / "data" / "results" / "human_speech_v3" / "summary.json"
    if not path.exists():
        return
    summary = json.loads(path.read_text(encoding="utf-8"))
    assert summary["sample_count"] == 8
    assert summary["score_row_count"] == 128
    assert set(summary["means"]) == {
        "A_oracle",
        "B_whisper_cascade",
        "C_qwen_transcript",
        "D_qwen_direct",
    }
    assert sum(path["successful_calls"] for path in summary["api"].values()) == 120

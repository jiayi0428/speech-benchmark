"""Integrity checks for human-speech-v4 and the combined N=50 result."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_v4_prepared_inputs_match_truth_when_present() -> None:
    manifest_path = ROOT / "data" / "processed" / "human_speech_v4" / "audio_manifest.json"
    truth_path = ROOT / "data" / "ground_truth_human_v4.json"
    if not manifest_path.exists() or not truth_path.exists():
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    truth = json.loads(truth_path.read_text(encoding="utf-8"))
    assert len(manifest["records"]) == 24
    assert len(truth) == 24
    assert {record["sample"] for record in manifest["records"]} == set(truth)


def test_combined_n50_is_complete_when_present() -> None:
    summary_path = ROOT / "data" / "results" / "human_speech_combined_n50" / "summary.json"
    if not summary_path.exists():
        return
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["sample_count"] == 50
    assert summary["dataset_counts"] == {"v1": 8, "v2": 10, "v3": 8, "v4": 24}
    assert summary["score_row_count"] == 800
    v4_summary = json.loads(
        (
            ROOT / "data" / "results" / "human_speech_v4" / "summary.json"
        ).read_text(encoding="utf-8")
    )
    assert v4_summary["sample_count"] == 24
    assert sum(item["successful_calls"] for item in v4_summary["api"].values()) == 360

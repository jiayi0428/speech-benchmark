"""Integrity checks for human_speech_v2."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_human_speech_v2_prepared_inputs_match_truth() -> None:
    config = json.loads(
        (ROOT / "experiments" / "human_speech_v2.json").read_text(
            encoding="utf-8"
        )
    )
    manifest = json.loads(
        (
            ROOT
            / "data"
            / "processed"
            / "human_speech_v2"
            / "audio_manifest.json"
        ).read_text(encoding="utf-8")
    )
    truth = json.loads(
        (ROOT / "data" / "ground_truth_human_v2.json").read_text(
            encoding="utf-8"
        )
    )
    assert len(config["samples"]) == 10
    assert set(config["samples"]) == set(truth)
    assert {record["sample"] for record in manifest["records"]} == set(truth)
    assert all(record["sample_rate"] == 16000 for record in manifest["records"])
    assert all(record["channels"] == 1 for record in manifest["records"])
    assert all(
        (ROOT / record["audio_path"]).is_file()
        for record in manifest["records"]
    )


def test_human_speech_v2_results_are_complete_when_present() -> None:
    path = ROOT / "data" / "results" / "human_speech_v2" / "summary.json"
    if not path.exists():
        return
    summary = json.loads(path.read_text(encoding="utf-8"))
    assert summary["sample_count"] == 10
    assert summary["api"]["successful_calls_total"] == 70
    assert set(summary["means"]) == {
        "C_qwen_transcript",
        "D_qwen_direct",
    }

"""Integrity checks for the combined N=26 human-speech evaluation."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_combined_n26_summary_is_complete_when_present() -> None:
    path = (
        ROOT
        / "data"
        / "results"
        / "human_speech_combined_n26"
        / "summary.json"
    )
    if not path.exists():
        return
    summary = json.loads(path.read_text(encoding="utf-8"))
    assert summary["sample_count"] == 26
    assert summary["dataset_counts"] == {"v1": 8, "v2": 10, "v3": 8}
    assert summary["score_row_count"] == 416
    assert set(summary["means"]) == {
        "A_oracle",
        "B_whisper_cascade",
        "C_qwen_transcript",
        "D_qwen_direct_system",
    }
    index = json.loads(
        (
            ROOT
            / "data"
            / "processed"
            / "human_speech_combined_n26"
            / "index.json"
        ).read_text(encoding="utf-8")
    )
    truth = json.loads(
        (
            ROOT / "data" / "ground_truth_human_combined_n26.json"
        ).read_text(encoding="utf-8")
    )
    assert index["sample_count"] == 26
    assert len(truth) == 26
    assert {record["combined_id"] for record in index["records"]} == set(truth)

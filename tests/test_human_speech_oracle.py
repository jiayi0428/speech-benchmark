"""Integrity checks for human-speech path A."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "data" / "results" / "human_speech_v1" / "oracle_tasks_raw.jsonl"


def test_human_v1_has_eight_reference_transcripts() -> None:
    config = json.loads(
        (ROOT / "experiments" / "human_speech_v1.json").read_text(encoding="utf-8")
    )
    truth = json.loads(
        (ROOT / "data" / "ground_truth_human_v1.json").read_text(encoding="utf-8")
    )
    assert len(config["samples"]) == 8
    assert set(config["samples"]) == set(truth)
    assert all(truth[sample]["transcript"].strip() for sample in config["samples"])


def test_oracle_results_are_complete_when_present() -> None:
    if not RESULT.exists():
        return
    records = [
        json.loads(line)
        for line in RESULT.read_text(encoding="utf-8").splitlines()
        if line.strip() and json.loads(line).get("status") == "success"
    ]
    assert len(records) == 32
    assert len({(record["sample"], record["task"]) for record in records}) == 32

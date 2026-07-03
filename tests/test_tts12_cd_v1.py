"""Integrity checks for the 12-sample TTS C/D experiment."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_prepared_tts12_inputs_match_ground_truth() -> None:
    index = json.loads(
        (
            ROOT / "data" / "processed" / "tts12_cd_v1" / "index.json"
        ).read_text(encoding="utf-8")
    )
    truth = json.loads(
        (ROOT / "data" / "ground_truth_tts12_cd_v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert len(index) == 12
    assert len(truth) == 12
    assert {entry["topic"] for entry in index} == set(truth)
    assert all((ROOT / entry["audio_path"]).is_file() for entry in index)
    assert all(entry["transcript"] == truth[entry["topic"]]["transcript"] for entry in index)


def test_tts12_summary_has_four_paths_when_present() -> None:
    path = ROOT / "data" / "results" / "tts12_cd_v1" / "summary.json"
    if not path.exists():
        return
    summary = json.loads(path.read_text(encoding="utf-8"))
    assert summary["sample_count"] == 12
    assert set(summary["means"]) == {
        "A_oracle",
        "B_whisper_cascade",
        "C_qwen_transcript",
        "D_qwen_direct",
    }
    assert summary["api"]["successful_calls_total"] == 84


def test_tts12_c_and_d_results_are_complete() -> None:
    result_dir = ROOT / "data" / "results" / "tts12_cd_v1"
    c_records = [
        json.loads(line)
        for line in (result_dir / "c_tasks_raw.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    c_success = [record for record in c_records if record["status"] == "success"]
    d_raw = [
        json.loads(line)
        for line in (result_dir / "direct_raw.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    d_post = [
        json.loads(line)
        for line in (result_dir / "direct_postprocessed.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    assert len(c_success) == 48
    assert len({(record["sample"], record["task"]) for record in c_success}) == 48
    assert len(d_raw) == 48
    assert len({(record["sample"], record["task"]) for record in d_raw}) == 48
    assert len(d_post) == 36
    assert all(record["postprocess_status"] == "success" for record in d_post)

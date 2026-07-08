"""Isolation checks for TTS12 repeat experiments."""
from __future__ import annotations

from pathlib import Path
import json

from run_tts12_repeats import RERUN_DIR, STABILITY_DIR


ROOT = Path(__file__).resolve().parents[1]


def test_repeat_result_directories_are_isolated() -> None:
    original = ROOT / "data" / "results" / "tts12_cd_v1"
    assert RERUN_DIR != original
    assert STABILITY_DIR != original
    assert RERUN_DIR != STABILITY_DIR


def test_completed_repeat_summaries_are_complete_when_present() -> None:
    rerun_summary = RERUN_DIR / "summary.json"
    stability_summary = STABILITY_DIR / "summary.json"
    if not rerun_summary.exists() or not stability_summary.exists():
        return
    rerun = json.loads(rerun_summary.read_text(encoding="utf-8"))
    stability = json.loads(stability_summary.read_text(encoding="utf-8"))
    assert rerun["exact_final_output_matches"] == 48
    assert rerun["equal_score_rows"] == 48
    assert stability["score_row_count"] == 72
    assert stability["determinism"][
        "samples_with_stable_C_vs_D_winner"
    ] == 12
    assert stability["api"]["actual_C_calls"] == 48
    assert stability["api"]["extra_duplicate_C_calls"] == 12

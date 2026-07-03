"""Static and artifact checks for human-speech ablation C."""
from __future__ import annotations

import json
from pathlib import Path

from src.direct_qwen import SYSTEM_PROMPTS
from src.transcription_utils import strip_transcription_wrapper


ROOT = Path(__file__).resolve().parents[1]


def test_qwen_transcription_prompt_is_verbatim_and_non_summarizing() -> None:
    prompt = SYSTEM_PROMPTS["transcription"].lower()
    assert "verbatim" in prompt
    assert "do not summarize" in prompt
    assert "return only the transcript" in prompt


def test_transcription_wrapper_cleanup_does_not_rewrite_words() -> None:
    wrapped = "The original content of this audio is: 'exact words here.'"
    assert strip_transcription_wrapper(wrapped) == "exact words here."
    assert strip_transcription_wrapper("ordinary transcript") == "ordinary transcript"


def test_canonical_c_results_have_32_unique_keys() -> None:
    path = (
        ROOT
        / "data"
        / "results"
        / "human_speech_v1"
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


def test_bcd_summary_is_complete_and_records_api_audit() -> None:
    path = (
        ROOT
        / "data"
        / "results"
        / "human_speech_v1"
        / "bcd_ablation_summary.json"
    )
    summary = json.loads(path.read_text(encoding="utf-8"))
    assert summary["sample_count"] == 8
    assert summary["score_row_count"] == 96
    assert set(summary["means"]) == {
        "B_whisper_cascade",
        "C_qwen_transcript",
        "D_qwen_direct",
    }
    assert summary["c_api"]["canonical_call_count"] == 32
    assert summary["c_api"]["audit_call_count"] == 52
    assert summary["c_api"]["extra_duplicate_calls"] == 20

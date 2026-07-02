"""Regression checks for the completed human_speech_v1 artifacts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments" / "human_speech_v1.json"
GROUND_TRUTH = ROOT / "data" / "ground_truth_human_v1.json"
MANIFEST = (
    ROOT / "data" / "processed" / "human_speech_v1" / "audio_manifest.json"
)
RESULT_DIR = ROOT / "data" / "results" / "human_speech_v1"


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_human_config_ground_truth_and_manifest_match() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    truth = json.loads(GROUND_TRUTH.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

    assert len(config["samples"]) == 8
    assert len(config["tasks"]) == 4
    assert list(truth) == config["samples"]
    assert [record["sample"] for record in manifest["records"]] == config["samples"]
    assert all(record["sample_rate"] == 16000 for record in manifest["records"])
    assert all(record["channels"] == 1 for record in manifest["records"])
    assert all(not record["denoising_applied"] for record in manifest["records"])
    assert all(not record["normalization_applied"] for record in manifest["records"])
    assert all(
        sha256_file(ROOT / record["audio_path"]) == record["output_sha256"]
        for record in manifest["records"]
    )


def test_human_direct_results_are_complete() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    raw = read_jsonl(RESULT_DIR / "direct_raw.jsonl")
    successful = [record for record in raw if record.get("status") == "success"]
    keys = {(record["sample"], record["task"]) for record in successful}
    expected = {
        (sample, task)
        for sample in config["samples"]
        for task in config["tasks"]
    }

    assert len(successful) == 32
    assert len(keys) == 32
    assert keys == expected
    assert all(record["output"].strip() for record in successful)


def test_human_postprocessing_and_summary_are_complete() -> None:
    postprocessed = read_jsonl(RESULT_DIR / "direct_postprocessed.jsonl")
    successful = [
        record
        for record in postprocessed
        if record.get("postprocess_status") == "success"
    ]
    summary = json.loads(
        (RESULT_DIR / "summary.json").read_text(encoding="utf-8")
    )

    assert len(successful) == 24
    assert len({(record["sample"], record["task"]) for record in successful}) == 24
    assert summary["sample_count"] == 8
    assert summary["result_count"] == 32
    assert summary["postprocess_api"]["call_count"] == 24
    assert summary["raw_structured_valid_json"]["valid"] == 0
    assert all(
        rate == 1.0
        for rate in summary["postprocessed_valid_json_rates"].values()
    )


def test_human_path_comparison_is_complete_and_caveated() -> None:
    cascade = json.loads(
        (RESULT_DIR / "cascade_raw.json").read_text(encoding="utf-8")
    )
    comparison = json.loads(
        (RESULT_DIR / "path_comparison_summary.json").read_text(
            encoding="utf-8"
        )
    )

    assert len(cascade) == 8
    assert comparison["sample_count"] == 8
    assert comparison["paired_result_count"] == 32
    assert set(comparison["means"]) == {"cascade", "direct"}
    assert all(
        low <= 0 <= high
        for low, high in comparison["paired_bootstrap_95_ci"].values()
    )
    assert "different machines" in comparison["recorded_latency_seconds"]["caveat"]

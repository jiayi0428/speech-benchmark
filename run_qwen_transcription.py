"""Run Qwen2-Audio verbatim transcription for human_speech_v1."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "experiments" / "human_speech_v1.json"
MANIFEST = ROOT / "data" / "processed" / "human_speech_v1" / "audio_manifest.json"
OUTPUT = (
    ROOT
    / "data"
    / "results"
    / "human_speech_v1"
    / "qwen_transcription_raw.jsonl"
)
PROMPT_VERSION = "qwen_verbatim_transcription_v1"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def append_jsonl(record: dict[str, Any]) -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-items", type=int)
    parser.add_argument("--model-path", type=Path)
    args = parser.parse_args()

    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    samples = list(config["samples"])
    records_by_sample = {
        record["sample"]: record for record in manifest["records"]
    }
    if set(records_by_sample) != set(samples):
        raise ValueError("Audio manifest does not match experiment samples")

    completed = {
        record["sample"]
        for record in read_jsonl(OUTPUT)
        if record.get("status") == "success"
    }
    pending = [sample for sample in samples if sample not in completed]
    print(
        f"Qwen transcription scope: {len(samples)} samples; "
        f"completed={len(completed)}; pending={len(pending)}"
    )
    if args.dry_run:
        return

    if args.model_path is not None:
        model_path = args.model_path.resolve()
        if not model_path.is_dir():
            raise FileNotFoundError(model_path)
        os.environ["QWEN_MODEL_PATH"] = str(model_path)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    from src.direct_qwen import MODEL_ID, QwenAudioPipeline

    pipeline = QwenAudioPipeline()
    written = 0
    for sample in pending:
        if args.max_items is not None and written >= args.max_items:
            break
        audio_record = records_by_sample[sample]
        audio_path = ROOT / audio_record["audio_path"]
        base = {
            "experiment_id": "human_speech_v1_ablation_c",
            "pipeline": "qwen_transcription",
            "sample": sample,
            "condition": config["condition"],
            "task": "transcription",
            "audio_path": audio_record["audio_path"],
            "audio_sha256": audio_record["output_sha256"],
            "source_sha256": audio_record["source_sha256"],
            "speech_model": MODEL_ID,
            "prompt_version": PROMPT_VERSION,
        }
        try:
            result = pipeline.run(str(audio_path), "transcription")
            record = {
                **base,
                "status": "success",
                "transcript": result["output"],
                "latency_seconds": result["latency_seconds"],
            }
            print(f"[OK] qwen transcription {sample}")
        except Exception as exc:
            record = {
                **base,
                "status": "error",
                "stage": "transcription",
                "error": repr(exc),
            }
            print(f"[ERROR] qwen transcription {sample}: {exc}")
        append_jsonl(record)
        written += 1
    print(f"Recorded {written} new transcription results in {OUTPUT}")


if __name__ == "__main__":
    main()


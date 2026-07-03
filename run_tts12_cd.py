"""Run local Qwen stages for the 12-sample TTS C/D experiment."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
INDEX = ROOT / "data" / "processed" / "tts12_cd_v1" / "index.json"
RESULT_DIR = ROOT / "data" / "results" / "tts12_cd_v1"
TASKS = ["summarization", "sentiment", "keywords", "intent"]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage", required=True, choices=["transcription", "direct"]
    )
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--max-items", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    entries = json.loads(INDEX.read_text(encoding="utf-8"))
    if len(entries) != 12 or len({entry["topic"] for entry in entries}) != 12:
        raise ValueError("Expected exactly 12 unique prepared TTS samples")
    output = RESULT_DIR / (
        "qwen_transcription_raw.jsonl"
        if args.stage == "transcription"
        else "direct_raw.jsonl"
    )
    existing = [
        record for record in read_jsonl(output)
        if record.get("status") == "success"
    ]
    if args.stage == "transcription":
        completed = {record["sample"] for record in existing}
        pending = [
            (entry, "transcription")
            for entry in entries
            if entry["topic"] not in completed
        ]
    else:
        completed = {(record["sample"], record["task"]) for record in existing}
        pending = [
            (entry, task)
            for entry in entries
            for task in TASKS
            if (entry["topic"], task) not in completed
        ]
    if len(completed) != len(existing):
        raise ValueError(f"Duplicate successful {args.stage} records detected")
    print(
        f"TTS12 {args.stage}: total="
        f"{12 if args.stage == 'transcription' else 48}; "
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
    from src.direct_qwen import MODEL_ID, PROMPT_VERSION, QwenAudioPipeline

    pipeline = QwenAudioPipeline()
    written = 0
    for entry, task in pending:
        if args.max_items is not None and written >= args.max_items:
            break
        sample = entry["topic"]
        audio_path = ROOT / entry["audio_path"]
        base = {
            "experiment_id": "tts12_cd_v1",
            "pipeline": (
                "qwen_transcription"
                if args.stage == "transcription"
                else "qwen_direct"
            ),
            "sample": sample,
            "condition": "clean",
            "task": task,
            "audio_path": entry["audio_path"],
            "audio_sha256": entry["audio_sha256"],
            "speech_model": MODEL_ID,
            "prompt_version": (
                "qwen_verbatim_transcription_v1"
                if task == "transcription"
                else PROMPT_VERSION
            ),
        }
        try:
            result = pipeline.run(str(audio_path), task)
            record = {
                **base,
                "status": "success",
                ("transcript" if task == "transcription" else "output"):
                    result["output"],
                "latency_seconds": result["latency_seconds"],
            }
            print(f"[OK] {args.stage} {sample} {task}")
        except Exception as exc:
            record = {
                **base,
                "status": "error",
                "stage": "qwen_inference",
                "error": repr(exc),
            }
            print(f"[ERROR] {args.stage} {sample} {task}: {exc}")
        append_jsonl(output, record)
        written += 1
    print(f"Recorded {written} new records in {output}")


if __name__ == "__main__":
    main()

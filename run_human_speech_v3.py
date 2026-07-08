"""Run local Whisper/Qwen stages for human-speech-v3 paths B, C, and D."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "experiments" / "human_speech_v3.json"
MANIFEST = ROOT / "data" / "processed" / "human_speech_v3" / "audio_manifest.json"
RESULT_DIR = ROOT / "data" / "results" / "human_speech_v3"
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
        "--stage", required=True, choices=["whisper", "qwen-transcription", "direct"]
    )
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--result-dir", type=Path, default=RESULT_DIR)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--allow-model-download", action="store_true")
    parser.add_argument("--max-items", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.resolve().read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.resolve().read_text(encoding="utf-8"))
    result_dir = args.result_dir.resolve()
    records = {record["sample"]: record for record in manifest["records"]}
    samples = list(config["samples"])
    if set(records) != set(samples) or not samples:
        raise ValueError("Expected matching non-empty human-speech records")
    output = result_dir / {
        "whisper": "whisper_transcription_raw.jsonl",
        "qwen-transcription": "qwen_transcription_raw.jsonl",
        "direct": "direct_raw.jsonl",
    }[args.stage]
    successful = [
        record for record in read_jsonl(output)
        if record.get("status") == "success"
    ]
    if args.stage == "direct":
        completed = {(record["sample"], record["task"]) for record in successful}
        pending = [
            (sample, task)
            for sample in samples
            for task in TASKS
            if (sample, task) not in completed
        ]
    else:
        completed = {record["sample"] for record in successful}
        task = "whisper_transcription" if args.stage == "whisper" else "transcription"
        pending = [(sample, task) for sample in samples if sample not in completed]
    if len(completed) != len(successful):
        raise ValueError(f"Duplicate successful {args.stage} records")
    total = len(samples) * len(TASKS) if args.stage == "direct" else len(samples)
    print(
        f"{config['experiment_id']} {args.stage}: total={total}; "
        f"completed={len(completed)}; pending={len(pending)}"
    )
    if args.dry_run:
        return

    if not args.allow_model_download:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    if args.stage == "whisper":
        from faster_whisper import WhisperModel
        from src.config import (
            WHISPER_COMPUTE_TYPE,
            WHISPER_DEVICE,
            WHISPER_MODEL,
        )

        pipeline = WhisperModel(
            WHISPER_MODEL,
            device=WHISPER_DEVICE,
            compute_type=WHISPER_COMPUTE_TYPE,
        )
        model_id = WHISPER_MODEL
    else:
        if args.model_path is not None:
            model_path = args.model_path.resolve()
            if not model_path.is_dir():
                raise FileNotFoundError(model_path)
            os.environ["QWEN_MODEL_PATH"] = str(model_path)
        from src.direct_qwen import MODEL_ID, PROMPT_VERSION, QwenAudioPipeline

        pipeline = QwenAudioPipeline(
            prompt_mode="user" if args.stage == "qwen-transcription" else "system"
        )
        model_id = MODEL_ID

    written = 0
    for sample, task in pending:
        if args.max_items is not None and written >= args.max_items:
            break
        audio_record = records[sample]
        audio_path = ROOT / audio_record["audio_path"]
        base = {
            "experiment_id": config["experiment_id"],
            "sample": sample,
            "condition": config["condition"],
            "task": task,
            "audio_path": audio_record["audio_path"],
            "audio_sha256": audio_record["output_sha256"],
            "source_sha256": audio_record["source_sha256"],
            "speech_model": model_id,
        }
        try:
            if args.stage == "whisper":
                started = time.perf_counter()
                segments, _ = pipeline.transcribe(str(audio_path), beam_size=5)
                transcript = " ".join(segment.text for segment in segments).strip()
                latency = time.perf_counter() - started
                record = {
                    **base,
                    "pipeline": "whisper_transcription",
                    "prompt_version": "none",
                    "status": "success",
                    "transcript": transcript,
                    "latency_seconds": round(latency, 3),
                }
            else:
                result = pipeline.run(str(audio_path), task)
                record = {
                    **base,
                    "pipeline": (
                        "qwen_transcription"
                        if args.stage == "qwen-transcription"
                        else "qwen_direct"
                    ),
                    "prompt_version": (
                        "qwen_verbatim_transcription_v1"
                        if args.stage == "qwen-transcription"
                        else PROMPT_VERSION
                    ),
                    "prompt_mode": (
                        "user" if args.stage == "qwen-transcription" else "system"
                    ),
                    "status": "success",
                    ("transcript" if args.stage == "qwen-transcription" else "output"):
                        result["output"],
                    "latency_seconds": result["latency_seconds"],
                }
            print(f"[OK] {args.stage} {sample} {task}")
        except Exception as exc:
            record = {
                **base,
                "pipeline": args.stage,
                "status": "error",
                "error": repr(exc),
            }
            print(f"[ERROR] {args.stage} {sample} {task}: {exc}")
        append_jsonl(output, record)
        written += 1
    print(f"Recorded {written} new records in {output}")


if __name__ == "__main__":
    main()

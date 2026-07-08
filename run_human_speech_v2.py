"""Run local Qwen stages for human_speech_v2 paths C and D."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "experiments" / "human_speech_v2.json"
MANIFEST = ROOT / "data" / "processed" / "human_speech_v2" / "audio_manifest.json"
RESULT_DIR = ROOT / "data" / "results" / "human_speech_v2"
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
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    records = {record["sample"]: record for record in manifest["records"]}
    samples = list(config["samples"])
    if set(records) != set(samples) or len(samples) != 10:
        raise ValueError("Expected 10 matching human-speech-v2 records")
    output = RESULT_DIR / (
        "qwen_transcription_raw.jsonl"
        if args.stage == "transcription"
        else "direct_raw.jsonl"
    )
    successful = [
        record for record in read_jsonl(output)
        if record.get("status") == "success"
    ]
    if args.stage == "transcription":
        completed = {record["sample"] for record in successful}
        pending = [
            (sample, "transcription")
            for sample in samples
            if sample not in completed
        ]
    else:
        completed = {(record["sample"], record["task"]) for record in successful}
        pending = [
            (sample, task)
            for sample in samples
            for task in TASKS
            if (sample, task) not in completed
        ]
    if len(completed) != len(successful):
        raise ValueError(f"Duplicate successful {args.stage} records")
    print(
        f"human_speech_v2 {args.stage}: total="
        f"{10 if args.stage == 'transcription' else 40}; "
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

    pipeline = QwenAudioPipeline(
        prompt_mode="user" if args.stage == "transcription" else "system"
    )
    written = 0
    for sample, task in pending:
        if args.max_items is not None and written >= args.max_items:
            break
        audio_record = records[sample]
        base = {
            "experiment_id": config["experiment_id"],
            "pipeline": (
                "qwen_transcription"
                if task == "transcription"
                else "qwen_direct"
            ),
            "sample": sample,
            "condition": config["condition"],
            "task": task,
            "audio_path": audio_record["audio_path"],
            "audio_sha256": audio_record["output_sha256"],
            "source_sha256": audio_record["source_sha256"],
            "speech_model": MODEL_ID,
            "prompt_version": (
                "qwen_verbatim_transcription_v1"
                if task == "transcription"
                else PROMPT_VERSION
            ),
        }
        try:
            result = pipeline.run(str(ROOT / audio_record["audio_path"]), task)
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

"""Run isolated local Qwen rerun and summary-stability experiments."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
INDEX = ROOT / "data" / "processed" / "tts12_cd_v1" / "index.json"
RERUN_DIR = ROOT / "data" / "results" / "tts12_d_rerun_v2"
STABILITY_DIR = ROOT / "data" / "results" / "tts12_summary_stability_v1"
TASKS = ["summarization", "sentiment", "keywords", "intent"]
REPETITIONS = [1, 2, 3]


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


def scope(
    stage: str, entries: list[dict[str, Any]]
) -> tuple[Path, list[tuple[dict[str, Any], str, int]]]:
    if stage == "d-rerun":
        return (
            RERUN_DIR / "direct_raw.jsonl",
            [(entry, task, 1) for entry in entries for task in TASKS],
        )
    if stage == "stability-c":
        return (
            STABILITY_DIR / "c_transcriptions.jsonl",
            [
                (entry, "transcription", repetition)
                for entry in entries
                for repetition in REPETITIONS
            ],
        )
    return (
        STABILITY_DIR / "d_summaries.jsonl",
        [
            (entry, "summarization", repetition)
            for entry in entries
            for repetition in REPETITIONS
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        required=True,
        choices=["d-rerun", "stability-c", "stability-d"],
    )
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--max-items", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    entries = json.loads(INDEX.read_text(encoding="utf-8"))
    if len(entries) != 12:
        raise ValueError("Expected 12 prepared TTS samples")
    output, full_scope = scope(args.stage, entries)
    successful = [
        record for record in read_jsonl(output)
        if record.get("status") == "success"
    ]
    completed = {
        (record["sample"], record["task"], record["repetition"])
        for record in successful
    }
    if len(completed) != len(successful):
        raise ValueError(f"Duplicate successful records in {output}")
    pending = [
        item
        for item in full_scope
        if (item[0]["topic"], item[1], item[2]) not in completed
    ]
    print(
        f"{args.stage}: total={len(full_scope)}; "
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
        prompt_mode="user" if args.stage == "stability-c" else "system"
    )
    written = 0
    for entry, task, repetition in pending:
        if args.max_items is not None and written >= args.max_items:
            break
        sample = entry["topic"]
        base = {
            "experiment_id": (
                "tts12_d_rerun_v2"
                if args.stage == "d-rerun"
                else "tts12_summary_stability_v1"
            ),
            "pipeline": (
                "qwen_transcription"
                if args.stage == "stability-c"
                else "qwen_direct"
            ),
            "sample": sample,
            "condition": "clean",
            "task": task,
            "repetition": repetition,
            "audio_path": entry["audio_path"],
            "audio_sha256": entry["audio_sha256"],
            "speech_model": MODEL_ID,
            "prompt_version": (
                "qwen_verbatim_transcription_v1"
                if task == "transcription"
                else PROMPT_VERSION
            ),
            "generation_config": {
                "do_sample": False,
                "temperature": None,
                "max_new_tokens": 256,
            },
        }
        try:
            result = pipeline.run(str(ROOT / entry["audio_path"]), task)
            record = {
                **base,
                "status": "success",
                ("transcript" if task == "transcription" else "output"):
                    result["output"],
                "latency_seconds": result["latency_seconds"],
            }
            print(f"[OK] {args.stage} {sample} {task} rep={repetition}")
        except Exception as exc:
            record = {
                **base,
                "status": "error",
                "stage": "qwen_inference",
                "error": repr(exc),
            }
            print(
                f"[ERROR] {args.stage} {sample} {task} "
                f"rep={repetition}: {exc}"
            )
        append_jsonl(output, record)
        written += 1
    print(f"Recorded {written} new records in {output}")


if __name__ == "__main__":
    main()

"""Run the local Direct path on the human_speech_v1 dataset."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "experiments" / "human_speech_v1.json"
MANIFEST = ROOT / "data" / "processed" / "human_speech_v1" / "audio_manifest.json"
RESULT_DIR = ROOT / "data" / "results" / "human_speech_v1"
DEFAULT_OUTPUT = RESULT_DIR / "direct_raw.jsonl"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
    return records


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()


def result_key(record: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(record["pipeline"]),
        str(record["sample"]),
        str(record["condition"]),
        str(record["task"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-items", type=int)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=["summarization", "sentiment", "keywords", "intent"],
    )
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    if not MANIFEST.exists():
        raise FileNotFoundError(
            f"Missing {MANIFEST}. Run prepare_human_speech.py first."
        )
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    records_by_sample = {
        record["sample"]: record for record in manifest["records"]
    }
    samples = list(config["samples"])
    tasks = list(args.tasks or config["tasks"])
    condition = str(config["condition"])
    missing_samples = sorted(set(samples) - set(records_by_sample))
    if missing_samples:
        raise ValueError(f"Manifest is missing samples: {missing_samples}")

    existing = read_jsonl(DEFAULT_OUTPUT)
    completed = {
        result_key(record)
        for record in existing
        if record.get("status") == "success"
    }
    pending_keys = [
        ("direct", sample, condition, task)
        for sample in samples
        for task in tasks
        if ("direct", sample, condition, task) not in completed
    ]
    total = len(samples) * len(tasks)
    print(
        f"Scope: {len(samples)} samples x {len(tasks)} tasks = {total}; "
        f"pending={len(pending_keys)}"
    )
    if args.dry_run:
        return

    if args.model_path is not None:
        model_path = args.model_path.resolve()
        if not model_path.is_dir():
            raise FileNotFoundError(model_path)
        os.environ["QWEN_MODEL_PATH"] = str(model_path)

    # This experiment must never trigger an implicit model download.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    from src.direct_qwen import MODEL_ID, PROMPT_VERSION, QwenAudioPipeline

    pipeline = QwenAudioPipeline()
    written = 0
    for sample in samples:
        audio_record = records_by_sample[sample]
        audio_path = ROOT / audio_record["audio_path"]
        for task in tasks:
            key = ("direct", sample, condition, task)
            if key in completed:
                continue
            if args.max_items is not None and written >= args.max_items:
                print(f"Recorded {written} new results in {DEFAULT_OUTPUT}")
                return
            base = {
                "experiment_id": config["experiment_id"],
                "pipeline": "direct",
                "sample": sample,
                "condition": condition,
                "task": task,
                "audio_path": audio_record["audio_path"],
                "audio_sha256": audio_record["output_sha256"],
                "source_sha256": audio_record["source_sha256"],
            }
            try:
                result = pipeline.run(str(audio_path), task)
                output_record = {
                    **base,
                    "status": "success",
                    "speech_model": MODEL_ID,
                    "prompt_version": PROMPT_VERSION,
                    "output": result["output"],
                    "latency_seconds": result["latency_seconds"],
                }
                print(f"[OK] direct {sample} {task}")
            except Exception as exc:
                output_record = {
                    **base,
                    "status": "error",
                    "stage": "inference",
                    "speech_model": MODEL_ID,
                    "prompt_version": PROMPT_VERSION,
                    "error": repr(exc),
                }
                print(f"[ERROR] direct {sample} {task}: {exc}")
            append_jsonl(DEFAULT_OUTPUT, output_record)
            written += 1

    print(f"Recorded {written} new results in {DEFAULT_OUTPUT}")


if __name__ == "__main__":
    main()

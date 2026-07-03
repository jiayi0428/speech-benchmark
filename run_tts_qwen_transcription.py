"""Run Qwen2-Audio verbatim transcription on the original eight TTS clips."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
INDEX = ROOT / "data" / "processed" / "tts_samples" / "index.json"
RESULT_DIR = ROOT / "data" / "results" / "tts_qwen_transcript_v1"
OUTPUT = RESULT_DIR / "qwen_transcription_raw.jsonl"
PROMPT_VERSION = "qwen_verbatim_transcription_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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

    entries = json.loads(INDEX.read_text(encoding="utf-8"))
    samples = [entry["topic"] for entry in entries]
    entry_by_sample = {entry["topic"]: entry for entry in entries}
    if len(samples) != 8 or len(entry_by_sample) != 8:
        raise ValueError("Expected exactly eight unique TTS samples")

    completed = {
        record["sample"]
        for record in read_jsonl(OUTPUT)
        if record.get("status") == "success"
    }
    pending = [sample for sample in samples if sample not in completed]
    print(
        f"TTS Qwen transcription scope: {len(samples)}; "
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
        entry = entry_by_sample[sample]
        audio_path = ROOT / entry["audio_path"]
        base = {
            "experiment_id": "tts_qwen_transcript_v1",
            "pipeline": "qwen_transcription",
            "sample": sample,
            "condition": "clean",
            "task": "transcription",
            "audio_path": audio_path.relative_to(ROOT).as_posix(),
            "audio_sha256": sha256_file(audio_path),
            "speech_model": MODEL_ID,
            "prompt_version": PROMPT_VERSION,
            "tts_voice": entry["speaker"],
        }
        try:
            result = pipeline.run(str(audio_path), "transcription")
            record = {
                **base,
                "status": "success",
                "transcript": result["output"],
                "latency_seconds": result["latency_seconds"],
            }
            print(f"[OK] TTS qwen transcription {sample}")
        except Exception as exc:
            record = {
                **base,
                "status": "error",
                "stage": "transcription",
                "error": repr(exc),
            }
            print(f"[ERROR] TTS qwen transcription {sample}: {exc}")
        append_jsonl(record)
        written += 1
    print(f"Recorded {written} new TTS transcription results in {OUTPUT}")


if __name__ == "__main__":
    main()


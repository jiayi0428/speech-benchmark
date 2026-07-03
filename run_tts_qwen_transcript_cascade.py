"""Run DeepSeek tasks on Qwen transcripts of the original TTS clips."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from src.transcription_utils import strip_transcription_wrapper


ROOT = Path(__file__).resolve().parent
INDEX = ROOT / "data" / "processed" / "tts_samples" / "index.json"
RESULT_DIR = ROOT / "data" / "results" / "tts_qwen_transcript_v1"
TRANSCRIPT_INPUT = RESULT_DIR / "qwen_transcription_raw.jsonl"
OUTPUT = RESULT_DIR / "qwen_transcript_cascade_raw.jsonl"
TASKS = ["summarization", "sentiment", "keywords", "intent"]
PROMPT_VERSION = "cascade_text_tasks_v1"


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
    args = parser.parse_args()

    entries = json.loads(INDEX.read_text(encoding="utf-8"))
    samples = [entry["topic"] for entry in entries]
    transcripts = [
        record
        for record in read_jsonl(TRANSCRIPT_INPUT)
        if record.get("status") == "success"
    ]
    transcript_by_sample = {record["sample"]: record for record in transcripts}
    if set(transcript_by_sample) != set(samples) or len(transcripts) != len(samples):
        raise ValueError("TTS Qwen transcripts are missing or duplicated")

    successful = [
        record
        for record in read_jsonl(OUTPUT)
        if record.get("status") == "success"
    ]
    completed = {(record["sample"], record["task"]) for record in successful}
    if len(completed) != len(successful):
        raise ValueError("Duplicate successful TTS C results detected")
    pending = [
        (sample, task)
        for sample in samples
        for task in TASKS
        if (sample, task) not in completed
    ]
    print(
        f"TTS C DeepSeek scope: {len(samples)} samples x {len(TASKS)} tasks "
        f"= {len(samples) * len(TASKS)}; completed={len(completed)}; "
        f"pending={len(pending)}"
    )
    if args.dry_run:
        return

    from openai import OpenAI
    from src.cascade import SYSTEM_PROMPTS
    from src.config import (
        TEXT_LLM_API_KEY,
        TEXT_LLM_BASE_URL,
        TEXT_LLM_MODEL,
        TEXT_LLM_PROVIDER,
    )

    if TEXT_LLM_PROVIDER != "deepseek" or not TEXT_LLM_API_KEY:
        raise RuntimeError(
            "TTS ablation C requires DEEPSEEK_API_KEY and refuses other providers."
        )
    client = OpenAI(
        api_key=TEXT_LLM_API_KEY,
        base_url=TEXT_LLM_BASE_URL,
    )

    written = 0
    for sample, task in pending:
        if args.max_items is not None and written >= args.max_items:
            break
        source = transcript_by_sample[sample]
        raw_transcript = source["transcript"]
        transcript = strip_transcription_wrapper(raw_transcript)
        started = time.perf_counter()
        base = {
            "experiment_id": "tts_qwen_transcript_v1",
            "pipeline": "qwen_transcript_cascade",
            "sample": sample,
            "condition": "clean",
            "task": task,
            "audio_path": source["audio_path"],
            "audio_sha256": source["audio_sha256"],
            "speech_model": source["speech_model"],
            "transcription_prompt_version": source["prompt_version"],
            "transcript_raw": raw_transcript,
            "transcript": transcript,
            "transcription_latency_seconds": source["latency_seconds"],
            "text_prompt_version": PROMPT_VERSION,
            "text_llm_provider": TEXT_LLM_PROVIDER,
            "text_llm_model": TEXT_LLM_MODEL,
        }
        try:
            response = client.chat.completions.create(
                model=TEXT_LLM_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPTS[task]},
                    {"role": "user", "content": transcript},
                ],
                temperature=0.0,
            )
            usage = response.usage
            llm_latency = time.perf_counter() - started
            record = {
                **base,
                "status": "success",
                "output": response.choices[0].message.content,
                "llm_latency_seconds": round(llm_latency, 3),
                "latency_seconds": round(
                    float(source["latency_seconds"]) + llm_latency,
                    3,
                ),
                "usage": {
                    "prompt_tokens": getattr(usage, "prompt_tokens", None),
                    "completion_tokens": getattr(usage, "completion_tokens", None),
                    "total_tokens": getattr(usage, "total_tokens", None),
                },
            }
            print(f"[OK] TTS C {sample} {task}")
        except Exception as exc:
            record = {
                **base,
                "status": "error",
                "stage": "text_llm",
                "error": repr(exc),
            }
            print(f"[ERROR] TTS C {sample} {task}: {exc}")
        append_jsonl(record)
        written += 1
        time.sleep(0.3)
    print(f"Recorded {written} new TTS C results in {OUTPUT}")


if __name__ == "__main__":
    main()

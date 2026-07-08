"""Run DeepSeek stages for human_speech_v2 paths C and D."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from postprocess_tts12_cd import D_PROMPTS
from src.transcription_utils import strip_transcription_wrapper


ROOT = Path(__file__).resolve().parent
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
    parser.add_argument("--stage", required=True, choices=["c", "d"])
    parser.add_argument("--max-items", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    input_path = RESULT_DIR / (
        "qwen_transcription_raw.jsonl" if args.stage == "c" else "direct_raw.jsonl"
    )
    output_path = RESULT_DIR / (
        "c_tasks_raw.jsonl" if args.stage == "c" else "direct_postprocessed.jsonl"
    )
    source = [
        record for record in read_jsonl(input_path)
        if record.get("status") == "success"
    ]
    if args.stage == "c":
        if len(source) != 10:
            raise ValueError("Path C requires 10 successful transcripts")
        scope = [(record, task) for record in source for task in TASKS]
    else:
        if len(source) != 40:
            raise ValueError("Path D requires 40 successful task results")
        scope = [
            (record, record["task"])
            for record in source
            if record["task"] in D_PROMPTS
        ]
    successful = [
        record for record in read_jsonl(output_path)
        if (
            record.get("status") == "success"
            if args.stage == "c"
            else record.get("postprocess_status") == "success"
        )
    ]
    completed = {(record["sample"], record["task"]) for record in successful}
    if len(completed) != len(successful):
        raise ValueError(f"Duplicate successful path {args.stage.upper()} records")
    pending = [
        (record, task)
        for record, task in scope
        if (record["sample"], task) not in completed
    ]
    total = 40 if args.stage == "c" else 30
    print(
        f"human_speech_v2 paid {args.stage.upper()}: total={total}; "
        f"completed={len(completed)}; pending={len(pending)}"
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
        raise RuntimeError("human_speech_v2 requires DEEPSEEK_API_KEY")
    client = OpenAI(api_key=TEXT_LLM_API_KEY, base_url=TEXT_LLM_BASE_URL)
    written = 0
    for source_record, task in pending:
        if args.max_items is not None and written >= args.max_items:
            break
        started = time.perf_counter()
        if args.stage == "c":
            raw_transcript = source_record["transcript"]
            transcript = strip_transcription_wrapper(raw_transcript)
            messages = [
                {"role": "system", "content": SYSTEM_PROMPTS[task]},
                {"role": "user", "content": transcript},
            ]
            base = {
                **source_record,
                "pipeline": "qwen_transcript_cascade",
                "task": task,
                "transcript_raw": raw_transcript,
                "transcript": transcript,
                "transcription_latency_seconds": source_record["latency_seconds"],
                "text_prompt_version": "cascade_text_tasks_v1",
            }
        else:
            messages = [
                {
                    "role": "user",
                    "content": (
                        f"{D_PROMPTS[task]}\n\n"
                        f"ANALYSIS:\n{source_record['output'][:500]}"
                    ),
                }
            ]
            base = {**source_record, "raw_output": source_record["output"]}
        try:
            response = client.chat.completions.create(
                model=TEXT_LLM_MODEL,
                messages=messages,
                temperature=0.0,
            )
            usage = response.usage
            latency = time.perf_counter() - started
            usage_payload = {
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
                "total_tokens": getattr(usage, "total_tokens", None),
            }
            if args.stage == "c":
                record = {
                    **base,
                    "status": "success",
                    "output": response.choices[0].message.content,
                    "llm_latency_seconds": round(latency, 3),
                    "usage": usage_payload,
                }
            else:
                record = {
                    **base,
                    "output": response.choices[0].message.content,
                    "postprocess_status": "success",
                    "postprocess_latency_seconds": round(latency, 3),
                    "usage": usage_payload,
                }
            print(f"[OK] path {args.stage.upper()} {source_record['sample']} {task}")
        except Exception as exc:
            record = {
                **base,
                (
                    "status"
                    if args.stage == "c"
                    else "postprocess_status"
                ): "error",
                "error": repr(exc),
            }
            print(
                f"[ERROR] path {args.stage.upper()} "
                f"{source_record['sample']} {task}: {exc}"
            )
        append_jsonl(output_path, record)
        written += 1
        time.sleep(0.3)
    print(f"Recorded {written} new records in {output_path}")


if __name__ == "__main__":
    main()

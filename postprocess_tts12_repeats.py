"""Run isolated paid DeepSeek stages for TTS12 repeat experiments."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from postprocess_tts12_cd import D_PROMPTS
from src.transcription_utils import strip_transcription_wrapper


ROOT = Path(__file__).resolve().parent
RERUN_DIR = ROOT / "data" / "results" / "tts12_d_rerun_v2"
STABILITY_DIR = ROOT / "data" / "results" / "tts12_summary_stability_v1"


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
        "--stage", required=True, choices=["d-rerun", "stability-c"]
    )
    parser.add_argument("--max-items", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.stage == "d-rerun":
        input_path = RERUN_DIR / "direct_raw.jsonl"
        output_path = RERUN_DIR / "direct_postprocessed.jsonl"
        source = [
            record for record in read_jsonl(input_path)
            if record.get("status") == "success"
            and record["task"] in D_PROMPTS
        ]
        expected = 36
    else:
        input_path = STABILITY_DIR / "c_transcriptions.jsonl"
        output_path = STABILITY_DIR / "c_summaries.jsonl"
        source = [
            record for record in read_jsonl(input_path)
            if record.get("status") == "success"
        ]
        expected = 36
    if len(source) != expected:
        raise ValueError(f"Expected {expected} source records for {args.stage}")
    successful = [
        record for record in read_jsonl(output_path)
        if (
            record.get("postprocess_status") == "success"
            if args.stage == "d-rerun"
            else record.get("status") == "success"
        )
    ]
    completed = {
        (record["sample"], record["task"], record["repetition"])
        for record in successful
    }
    if len(completed) != len(successful):
        raise ValueError(f"Duplicate successful records in {output_path}")
    pending = []
    for record in source:
        output_task = (
            "summarization" if args.stage == "stability-c" else record["task"]
        )
        if (record["sample"], output_task, record["repetition"]) not in completed:
            pending.append(record)
    print(
        f"{args.stage} paid stage: total={expected}; "
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
        raise RuntimeError("Repeat experiments require DEEPSEEK_API_KEY")
    client = OpenAI(api_key=TEXT_LLM_API_KEY, base_url=TEXT_LLM_BASE_URL)
    written = 0
    for source_record in pending:
        if args.max_items is not None and written >= args.max_items:
            break
        started = time.perf_counter()
        if args.stage == "d-rerun":
            task = source_record["task"]
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
        else:
            task = "summarization"
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
            if args.stage == "d-rerun":
                record = {
                    **base,
                    "output": response.choices[0].message.content,
                    "postprocess_status": "success",
                    "postprocess_latency_seconds": round(latency, 3),
                    "usage": usage_payload,
                }
            else:
                record = {
                    **base,
                    "status": "success",
                    "output": response.choices[0].message.content,
                    "llm_latency_seconds": round(latency, 3),
                    "usage": usage_payload,
                }
            print(
                f"[OK] {args.stage} {source_record['sample']} "
                f"{task} rep={source_record['repetition']}"
            )
        except Exception as exc:
            record = {
                **base,
                (
                    "postprocess_status"
                    if args.stage == "d-rerun"
                    else "status"
                ): "error",
                "error": repr(exc),
            }
            print(
                f"[ERROR] {args.stage} {source_record['sample']} "
                f"{task} rep={source_record['repetition']}: {exc}"
            )
        append_jsonl(output_path, record)
        written += 1
        time.sleep(0.3)
    print(f"Recorded {written} new records in {output_path}")


if __name__ == "__main__":
    main()

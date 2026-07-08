"""Run DeepSeek stages for human-speech-v3 paths A, B, C, and D."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from postprocess_tts12_cd import D_PROMPTS
from src.transcription_utils import strip_transcription_wrapper


ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "experiments" / "human_speech_v3.json"
TRUTH = ROOT / "data" / "ground_truth_human_v3.json"
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
    parser.add_argument("--stage", required=True, choices=["a", "b", "c", "d"])
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--truth", type=Path, default=TRUTH)
    parser.add_argument("--result-dir", type=Path, default=RESULT_DIR)
    parser.add_argument("--max-items", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.resolve().read_text(encoding="utf-8"))
    truth_path = args.truth.resolve()
    result_dir = args.result_dir.resolve()
    samples = list(config["samples"])
    output_path = result_dir / {
        "a": "a_tasks_raw.jsonl",
        "b": "b_tasks_raw.jsonl",
        "c": "c_tasks_raw.jsonl",
        "d": "direct_postprocessed.jsonl",
    }[args.stage]
    if args.stage == "a":
        truth = json.loads(truth_path.read_text(encoding="utf-8"))
        scope = [
            (
                {
                    "experiment_id": config["experiment_id"],
                    "sample": sample,
                    "condition": config["condition"],
                    "transcript": truth[sample]["transcript"],
                    "pipeline": "oracle_transcript",
                },
                task,
            )
            for sample in samples
            for task in TASKS
        ]
    else:
        input_path = result_dir / {
            "b": "whisper_transcription_raw.jsonl",
            "c": "qwen_transcription_raw.jsonl",
            "d": "direct_raw.jsonl",
        }[args.stage]
        source = [
            record
            for record in read_jsonl(input_path)
            if record.get("status") == "success"
        ]
        expected = len(samples) * len(TASKS) if args.stage == "d" else len(samples)
        if len(source) != expected:
            raise ValueError(f"Path {args.stage.upper()} source is incomplete")
        scope = (
            [
                (record, task)
                for record in source
                for task in TASKS
            ]
            if args.stage in {"b", "c"}
            else [
                (record, record["task"])
                for record in source
                if record["task"] in D_PROMPTS
            ]
        )
    successful = [
        record for record in read_jsonl(output_path)
        if (
            record.get("postprocess_status") == "success"
            if args.stage == "d"
            else record.get("status") == "success"
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
    total = (
        len(samples) * (len(TASKS) - 1)
        if args.stage == "d"
        else len(samples) * len(TASKS)
    )
    print(
        f"human_speech_v3 paid {args.stage.upper()}: total={total}; "
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
        raise RuntimeError("human_speech_v3 requires DEEPSEEK_API_KEY")
    client = OpenAI(api_key=TEXT_LLM_API_KEY, base_url=TEXT_LLM_BASE_URL)
    written = 0
    for source_record, task in pending:
        if args.max_items is not None and written >= args.max_items:
            break
        started = time.perf_counter()
        if args.stage == "d":
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
            raw_transcript = source_record["transcript"]
            transcript = strip_transcription_wrapper(raw_transcript)
            messages = [
                {"role": "system", "content": SYSTEM_PROMPTS[task]},
                {"role": "user", "content": transcript},
            ]
            base = {
                **source_record,
                "pipeline": {
                    "a": "oracle_transcript",
                    "b": "whisper_cascade",
                    "c": "qwen_transcript_cascade",
                }[args.stage],
                "task": task,
                "transcript_raw": raw_transcript,
                "transcript": transcript,
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
            if args.stage == "d":
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
            print(f"[OK] path {args.stage.upper()} {source_record['sample']} {task}")
        except Exception as exc:
            record = {
                **base,
                ("postprocess_status" if args.stage == "d" else "status"): "error",
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

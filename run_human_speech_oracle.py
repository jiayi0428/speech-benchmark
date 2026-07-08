"""Run path A: human reference transcript -> DeepSeek tasks."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "experiments" / "human_speech_v1.json"
GROUND_TRUTH = ROOT / "data" / "ground_truth_human_v1.json"
RESULT_DIR = ROOT / "data" / "results" / "human_speech_v1"
OUTPUT = RESULT_DIR / "oracle_tasks_raw.jsonl"
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
    parser.add_argument("--max-items", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    truth = json.loads(GROUND_TRUTH.read_text(encoding="utf-8"))
    samples = list(config["samples"])
    if len(samples) != 8 or set(samples) != set(truth):
        raise ValueError("Expected exactly 8 matching human-speech samples")

    existing = [
        record
        for record in read_jsonl(OUTPUT)
        if record.get("status") == "success"
    ]
    completed = {(record["sample"], record["task"]) for record in existing}
    if len(completed) != len(existing):
        raise ValueError("Duplicate successful path A records")
    pending = [
        (sample, task)
        for sample in samples
        for task in TASKS
        if (sample, task) not in completed
    ]
    print(
        f"Human speech path A: total=32; completed={len(completed)}; "
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
        raise RuntimeError("Path A requires DEEPSEEK_API_KEY")
    client = OpenAI(api_key=TEXT_LLM_API_KEY, base_url=TEXT_LLM_BASE_URL)
    written = 0
    for sample, task in pending:
        if args.max_items is not None and written >= args.max_items:
            break
        transcript = truth[sample]["transcript"]
        started = time.perf_counter()
        base = {
            "experiment_id": "human_speech_v1_oracle_a",
            "pipeline": "oracle_transcript",
            "sample": sample,
            "condition": config["condition"],
            "task": task,
            "transcript": transcript,
            "text_prompt_version": "cascade_text_tasks_v1",
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
            latency = time.perf_counter() - started
            usage = response.usage
            record = {
                **base,
                "status": "success",
                "output": response.choices[0].message.content,
                "llm_latency_seconds": round(latency, 3),
                "usage": {
                    "prompt_tokens": getattr(usage, "prompt_tokens", None),
                    "completion_tokens": getattr(usage, "completion_tokens", None),
                    "total_tokens": getattr(usage, "total_tokens", None),
                },
            }
            print(f"[OK] path A {sample} {task}")
        except Exception as exc:
            record = {
                **base,
                "status": "error",
                "error": repr(exc),
            }
            print(f"[ERROR] path A {sample} {task}: {exc}")
        append_jsonl(OUTPUT, record)
        written += 1
        time.sleep(0.3)
    print(f"Recorded {written} new records in {OUTPUT}")


if __name__ == "__main__":
    main()

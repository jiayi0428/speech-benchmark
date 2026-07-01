"""Convert Qwen structured-task answers to JSON through the text LLM."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
RESULT_DIR = ROOT / "data" / "results" / "white_noise_v1"
INPUT = RESULT_DIR / "direct_raw.jsonl"
OUTPUT = RESULT_DIR / "direct_postprocessed.jsonl"

PROMPTS = {
    "sentiment": (
        "Below is an AI analysis of an audio clip. Based on this analysis, "
        "classify the speaker sentiment as exactly one of: positive, negative, neutral. "
        'Return ONLY JSON: {"sentiment": "<label>", "confidence": <float>}'
    ),
    "keywords": (
        "Below is an AI analysis of an audio clip. Based on this analysis, "
        "extract 5-7 key phrases. "
        'Return ONLY JSON list: ["keyword1", "keyword2", ...]'
    ),
    "intent": (
        "Below is an AI analysis of an audio clip. Based on this analysis, "
        "classify the speaker intent as: inform, persuade, entertain, question, describe. "
        'Return ONLY JSON: {"intent": "<label>", "confidence": <float>}'
    ),
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def key(record: dict[str, Any]) -> tuple[str, str, str]:
    return record["sample"], record["condition"], record["task"]


def append(record: dict[str, Any]) -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-items", type=int)
    args = parser.parse_args()

    from openai import OpenAI
    from src.config import (
        TEXT_LLM_API_KEY,
        TEXT_LLM_BASE_URL,
        TEXT_LLM_MODEL,
    )

    if not TEXT_LLM_API_KEY:
        raise RuntimeError(
            "Post-processing requires DEEPSEEK_API_KEY or OPENAI_API_KEY in .env"
        )
    client = OpenAI(
        api_key=TEXT_LLM_API_KEY,
        base_url=TEXT_LLM_BASE_URL,
    )
    source = [
        record
        for record in read_jsonl(INPUT)
        if record.get("status") == "success"
        and record.get("task") in PROMPTS
    ]
    completed = {
        key(record)
        for record in read_jsonl(OUTPUT)
        if record.get("status") == "success"
    }
    written = 0
    for record in source:
        if key(record) in completed:
            continue
        if args.max_items is not None and written >= args.max_items:
            break
        started = time.perf_counter()
        try:
            response = client.chat.completions.create(
                model=TEXT_LLM_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": (
                            f"{PROMPTS[record['task']]}\n\n"
                            f"ANALYSIS:\n{record['output'][:500]}"
                        ),
                    }
                ],
                temperature=0.0,
            )
            structured = response.choices[0].message.content
            output_record = {
                **record,
                "raw_output": record["output"],
                "output": structured,
                "postprocess_status": "success",
                "postprocess_latency_seconds": round(
                    time.perf_counter() - started,
                    3,
                ),
            }
            print(
                f"[OK] {record['sample']} {record['condition']} "
                f"{record['task']}"
            )
        except Exception as exc:
            output_record = {
                **record,
                "raw_output": record["output"],
                "postprocess_status": "error",
                "error": repr(exc),
            }
            print(f"[ERROR] {key(record)}: {exc}")
        append(output_record)
        written += 1
        time.sleep(0.3)
    print(f"Recorded {written} post-processed results in {OUTPUT}")


if __name__ == "__main__":
    main()

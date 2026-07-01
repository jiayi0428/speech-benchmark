"""Run the white-noise benchmark with checkpointed JSONL output."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "experiments" / "white_noise_v1.json"
MANIFEST = ROOT / "data" / "processed" / "white_noise_v1" / "audio_manifest.json"
RESULT_DIR = ROOT / "data" / "results" / "white_noise_v1"


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()


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


def result_key(record: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(record["pipeline"]),
        str(record["sample"]),
        str(record["condition"]),
        str(record["task"]),
    )


def select_scope(
    config: dict[str, Any],
    smoke: bool,
) -> tuple[list[str], list[str], list[str]]:
    if smoke:
        smoke_config = config["smoke"]
        return (
            list(smoke_config["samples"]),
            list(smoke_config["conditions"]),
            list(smoke_config["tasks"]),
        )
    return (
        list(config["samples"]),
        [condition["name"] for condition in config["conditions"]],
        list(config["tasks"]),
    )


def ordered_audio_records(
    manifest: dict[str, Any],
    samples: Iterable[str],
    conditions: Iterable[str],
) -> list[dict[str, Any]]:
    by_key = {
        (record["sample"], record["condition"]): record
        for record in manifest["records"]
    }
    selected = []
    for sample in samples:
        for condition in conditions:
            key = (sample, condition)
            if key not in by_key:
                raise KeyError(f"Missing prepared audio: {key}")
            selected.append(by_key[key])
    return selected


def run_cascade(
    audio_records: list[dict[str, Any]],
    tasks: list[str],
    output: Path,
    completed: set[tuple[str, str, str, str]],
    max_items: int | None,
) -> int:
    from src.config import TEXT_LLM_API_KEY, TEXT_LLM_MODEL, WHISPER_MODEL

    if not TEXT_LLM_API_KEY:
        raise RuntimeError(
            "Cascade requires DEEPSEEK_API_KEY or OPENAI_API_KEY in .env"
        )

    from src.cascade import CascadePipeline

    pipeline = CascadePipeline()
    written = 0

    for audio_record in audio_records:
        pending_tasks = [
            task
            for task in tasks
            if (
                "cascade",
                audio_record["sample"],
                audio_record["condition"],
                task,
            )
            not in completed
        ]
        if not pending_tasks:
            continue

        audio_path = ROOT / audio_record["audio_path"]
        try:
            asr_start = time.perf_counter()
            transcript = pipeline.transcribe(str(audio_path))
            asr_latency = time.perf_counter() - asr_start
        except Exception as exc:
            for task in pending_tasks:
                record = {
                    **base_record("cascade", audio_record, task),
                    "status": "error",
                    "stage": "asr",
                    "error": repr(exc),
                }
                append_jsonl(output, record)
                written += 1
            continue

        for task in pending_tasks:
            if max_items is not None and written >= max_items:
                return written
            started = time.perf_counter()
            try:
                answer = pipeline._call_llm(transcript, task)
                llm_latency = time.perf_counter() - started
                record = {
                    **base_record("cascade", audio_record, task),
                    "status": "success",
                    "asr_model": WHISPER_MODEL,
                    "text_llm_model": TEXT_LLM_MODEL,
                    "transcript": transcript,
                    "output": answer,
                    "asr_latency_seconds": round(asr_latency, 3),
                    "llm_latency_seconds": round(llm_latency, 3),
                    "latency_seconds": round(asr_latency + llm_latency, 3),
                }
                print(
                    f"[OK] cascade {audio_record['sample']} "
                    f"{audio_record['condition']} {task}"
                )
            except Exception as exc:
                record = {
                    **base_record("cascade", audio_record, task),
                    "status": "error",
                    "stage": "llm",
                    "asr_model": WHISPER_MODEL,
                    "text_llm_model": TEXT_LLM_MODEL,
                    "transcript": transcript,
                    "error": repr(exc),
                }
                print(
                    f"[ERROR] cascade {audio_record['sample']} "
                    f"{audio_record['condition']} {task}: {exc}"
                )
            append_jsonl(output, record)
            written += 1

    return written


def run_qwen(
    audio_records: list[dict[str, Any]],
    tasks: list[str],
    output: Path,
    completed: set[tuple[str, str, str, str]],
    max_items: int | None,
) -> int:
    from src.direct_qwen import MODEL_ID, PROMPT_VERSION, QwenAudioPipeline

    pipeline = QwenAudioPipeline()
    written = 0
    for audio_record in audio_records:
        audio_path = ROOT / audio_record["audio_path"]
        for task in tasks:
            key = (
                "qwen",
                audio_record["sample"],
                audio_record["condition"],
                task,
            )
            if key in completed:
                continue
            if max_items is not None and written >= max_items:
                return written
            try:
                result = pipeline.run(str(audio_path), task)
                record = {
                    **base_record("qwen", audio_record, task),
                    "status": "success",
                    "speech_model": MODEL_ID,
                    "prompt_version": PROMPT_VERSION,
                    "output": result["output"],
                    "latency_seconds": result["latency_seconds"],
                }
                print(
                    f"[OK] qwen {audio_record['sample']} "
                    f"{audio_record['condition']} {task}"
                )
            except Exception as exc:
                record = {
                    **base_record("qwen", audio_record, task),
                    "status": "error",
                    "stage": "inference",
                    "speech_model": MODEL_ID,
                    "prompt_version": PROMPT_VERSION,
                    "error": repr(exc),
                }
                print(
                    f"[ERROR] qwen {audio_record['sample']} "
                    f"{audio_record['condition']} {task}: {exc}"
                )
            append_jsonl(output, record)
            written += 1
    return written


def base_record(
    pipeline: str,
    audio_record: dict[str, Any],
    task: str,
) -> dict[str, Any]:
    return {
        "experiment_id": audio_record["experiment_id"],
        "pipeline": pipeline,
        "sample": audio_record["sample"],
        "condition": audio_record["condition"],
        "seed": audio_record["seed"],
        "task": task,
        "audio_path": audio_record["audio_path"],
        "audio_sha256": audio_record["output_sha256"],
        "requested_snr_db": audio_record["requested_snr_db"],
        "measured_snr_db": audio_record["measured_snr_db"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline", choices=["cascade", "qwen"], required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-items", type=int)
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=["summarization", "sentiment", "keywords", "intent"],
        help="Run only the selected tasks.",
    )
    parser.add_argument(
        "--output-name",
        help="Write to this JSONL filename inside the experiment result directory.",
    )
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    if not MANIFEST.exists():
        raise FileNotFoundError(
            f"Missing {MANIFEST}. Run prepare_white_noise.py first."
        )
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    samples, conditions, tasks = select_scope(config, args.smoke)
    if args.tasks:
        tasks = list(args.tasks)
    audio_records = ordered_audio_records(manifest, samples, conditions)
    total = len(audio_records) * len(tasks)

    if args.output_name:
        output_name = Path(args.output_name).name
        if output_name != args.output_name or not output_name.endswith(".jsonl"):
            raise ValueError("--output-name must be a plain .jsonl filename")
    elif args.smoke:
        output_name = (
            "cascade_smoke_raw.jsonl"
            if args.pipeline == "cascade"
            else "direct_smoke_raw.jsonl"
        )
    else:
        output_name = (
            "cascade_raw.jsonl"
            if args.pipeline == "cascade"
            else "direct_raw.jsonl"
        )
    output = RESULT_DIR / output_name
    existing = read_jsonl(output)
    completed = {
        result_key(record)
        for record in existing
        if record.get("status") == "success"
    }
    pending = sum(
        (
            args.pipeline,
            audio_record["sample"],
            audio_record["condition"],
            task,
        )
        not in completed
        for audio_record in audio_records
        for task in tasks
    )
    print(
        f"Scope: {len(samples)} samples × {len(conditions)} conditions × "
        f"{len(tasks)} tasks = {total}; pending={pending}"
    )
    if args.dry_run:
        return

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    if args.pipeline == "cascade":
        written = run_cascade(
            audio_records,
            tasks,
            output,
            completed,
            args.max_items,
        )
    else:
        written = run_qwen(
            audio_records,
            tasks,
            output,
            completed,
            args.max_items,
        )
    print(f"Recorded {written} new results in {output}")


if __name__ == "__main__":
    main()

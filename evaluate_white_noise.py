"""Evaluate completeness, quality, WER, and degradation for white_noise_v1."""
from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "experiments" / "white_noise_v1.json"
GROUND_TRUTH = ROOT / "data" / "ground_truth.json"
RESULT_DIR = ROOT / "data" / "results" / "white_noise_v1"
CONDITION_ORDER = ["clean", "white_20db", "white_10db", "white_0db"]
PIPELINES = ["cascade", "qwen"]
ORIGINAL_DIRECT_BASELINE = {
    "summarization": 0.448,
    "sentiment": 0.38,
    "keywords": 0.29,
    "intent": 0.62,
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def extract_json(text: str) -> str:
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if fenced:
        return fenced.group(1)
    brace_start, brace_end = text.find("{"), text.rfind("}")
    if brace_start >= 0 and brace_end > brace_start:
        return text[brace_start : brace_end + 1]
    bracket_start, bracket_end = text.find("["), text.rfind("]")
    if bracket_start >= 0 and bracket_end > bracket_start:
        return text[bracket_start : bracket_end + 1]
    return text.strip()


def parsed_structured(text: str) -> tuple[Any | None, bool]:
    candidate = extract_json(text)
    try:
        return json.loads(candidate), True
    except (json.JSONDecodeError, TypeError):
        try:
            # Qwen sometimes emits an otherwise-correct Python literal with
            # single quotes. Parse that safely for task scoring while still
            # reporting that the output was not strict JSON.
            return ast.literal_eval(candidate), False
        except (ValueError, SyntaxError):
            return None, False


def rouge_l(reference: str, hypothesis: str) -> float:
    ref = reference.lower().split()
    hyp = hypothesis.lower().split()
    if not ref or not hyp:
        return 0.0
    previous = [0] * (len(hyp) + 1)
    for ref_token in ref:
        current = [0]
        for index, hyp_token in enumerate(hyp, start=1):
            if ref_token == hyp_token:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(current[-1], previous[index]))
        previous = current
    lcs = previous[-1]
    if lcs == 0:
        return 0.0
    precision = lcs / len(hyp)
    recall = lcs / len(ref)
    return 2 * precision * recall / (precision + recall)


def wer(reference: str, hypothesis: str) -> float:
    ref = reference.lower().split()
    hyp = hypothesis.lower().split()
    if not ref:
        return 1.0 if hyp else 0.0
    previous = list(range(len(hyp) + 1))
    for i, ref_token in enumerate(ref, start=1):
        current = [i]
        for j, hyp_token in enumerate(hyp, start=1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (ref_token != hyp_token),
                )
            )
        previous = current
    return previous[-1] / len(ref)


def keyword_f1(reference: list[str], predicted: list[str]) -> float:
    ref = {item.lower().strip() for item in reference}
    pred = {item.lower().strip() for item in predicted}
    if not ref or not pred:
        return 0.0
    overlap = len(ref & pred)
    precision = overlap / len(pred)
    recall = overlap / len(ref)
    return (
        0.0
        if precision + recall == 0
        else 2 * precision * recall / (precision + recall)
    )


def score_record(
    record: dict[str, Any],
    truth: dict[str, Any],
) -> tuple[float, float | None]:
    task = record["task"]
    output = record.get("output", "")
    parsed, strict_json = parsed_structured(output)
    valid_json: float | None = None
    if task == "summarization":
        return rouge_l(truth["summary"], output), valid_json
    valid_json = float(strict_json)
    if task == "sentiment":
        label = (
            str(parsed.get("sentiment", "")).lower()
            if isinstance(parsed, dict)
            else ""
        )
        return float(label == truth["sentiment"]), valid_json
    if task == "intent":
        label = (
            str(parsed.get("intent", "")).lower()
            if isinstance(parsed, dict)
            else ""
        )
        return float(label == truth["intent"]), valid_json
    if task == "keywords":
        predicted = parsed if isinstance(parsed, list) else []
        return keyword_f1(truth["keywords"], predicted), valid_json
    raise ValueError(f"Unknown task: {task}")


def load_results(smoke: bool = False) -> list[dict[str, Any]]:
    cascade_name = "cascade_smoke_raw.jsonl" if smoke else "cascade_raw.jsonl"
    direct_name = "direct_smoke_raw.jsonl" if smoke else "direct_raw.jsonl"
    cascade = [
        record
        for record in read_jsonl(RESULT_DIR / cascade_name)
        if record.get("status") == "success"
    ]
    direct = [
        record
        for record in read_jsonl(RESULT_DIR / direct_name)
        if record.get("status") == "success"
    ]
    structured = [
        record
        for record in read_jsonl(RESULT_DIR / "direct_postprocessed.jsonl")
        if record.get("postprocess_status") == "success"
    ]
    reprompted = [
        record
        for record in read_jsonl(RESULT_DIR / "direct_reprompted_raw.jsonl")
        if record.get("status") == "success"
    ]
    structured_by_key = {
        (record["sample"], record["condition"], record["task"]): record
        for record in reprompted
    }
    # DeepSeek post-processing matches the original report methodology and
    # therefore takes precedence whenever those results are available.
    structured_by_key.update(
        {
            (record["sample"], record["condition"], record["task"]): record
            for record in structured
        }
    )
    merged_direct = []
    for record in direct:
        key = (record["sample"], record["condition"], record["task"])
        if record["task"] != "summarization" and key in structured_by_key:
            merged_direct.append(structured_by_key[key])
        else:
            merged_direct.append(record)
    return cascade + merged_direct


def write_plots(summary: dict[str, Any], pipelines: list[str]) -> None:
    if summary["available_results"] == 0:
        print("No successful results yet; skipped figures")
        return
    os.environ.setdefault(
        "MPLCONFIGDIR",
        str(RESULT_DIR / ".matplotlib"),
    )
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib unavailable; skipped figures")
        return

    figure_dir = RESULT_DIR / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    tasks = ["summarization", "sentiment", "keywords", "intent"]
    labels = ["Clean", "20dB", "10dB", "0dB"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for ax, task in zip(axes.flat, tasks):
        for pipeline in pipelines:
            values = [
                summary["means"].get(pipeline, {})
                .get(task, {})
                .get(condition)
                for condition in CONDITION_ORDER
            ]
            ax.plot(labels, values, marker="o", label=pipeline)
        ax.set_title(task)
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3)
    axes.flat[0].legend()
    fig.suptitle("White-noise robustness")
    fig.tight_layout()
    fig.savefig(figure_dir / "task_scores_vs_snr.png", dpi=160)
    plt.close(fig)

    if "qwen" in pipelines:
        fig, ax = plt.subplots(figsize=(10, 5))
        comparison_labels = [
            "Original\nclean",
            "New\nclean",
            "20dB",
            "10dB",
            "0dB",
        ]
        for task in tasks:
            qwen_scores = summary["means"].get("qwen", {}).get(task, {})
            values = [
                ORIGINAL_DIRECT_BASELINE[task],
                qwen_scores.get("clean"),
                qwen_scores.get("white_20db"),
                qwen_scores.get("white_10db"),
                qwen_scores.get("white_0db"),
            ]
            ax.plot(comparison_labels, values, marker="o", label=task)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Task score")
        ax.set_title("Original Direct baseline vs Qwen white-noise experiment")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(
            figure_dir / "original_vs_white_noise.png",
            dpi=160,
        )
        plt.close(fig)

    wer_values = [
        summary["wer"].get(condition)
        for condition in CONDITION_ORDER
    ]
    if any(value is not None for value in wer_values):
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(labels, wer_values, marker="o", color="#d62728")
        ax.set_ylabel("Whisper WER")
        ax.set_title("Whisper transcription errors vs white noise")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(figure_dir / "wer_vs_snr.png", dpi=160)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--pipelines",
        nargs="+",
        choices=PIPELINES,
        default=PIPELINES,
    )
    args = parser.parse_args()

    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    ground_truth = json.loads(GROUND_TRUTH.read_text(encoding="utf-8"))
    results = load_results(args.smoke)
    if args.smoke:
        samples = config["smoke"]["samples"]
        conditions = config["smoke"]["conditions"]
        tasks = config["smoke"]["tasks"]
    else:
        samples = config["samples"]
        conditions = [item["name"] for item in config["conditions"]]
        tasks = config["tasks"]

    selected = [
        record
        for record in results
        if record["pipeline"] in args.pipelines
        and record["sample"] in samples
        and record["condition"] in conditions
        and record["task"] in tasks
    ]
    expected_keys = {
        (pipeline, sample, condition, task)
        for pipeline in args.pipelines
        for sample in samples
        for condition in conditions
        for task in tasks
    }
    actual_keys = {
        (
            record["pipeline"],
            record["sample"],
            record["condition"],
            record["task"],
        )
        for record in selected
    }

    score_rows = []
    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    json_grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for record in selected:
        score, valid_json = score_record(
            record,
            ground_truth[record["sample"]],
        )
        row = {
            "pipeline": record["pipeline"],
            "sample": record["sample"],
            "condition": record["condition"],
            "task": record["task"],
            "score": score,
            "valid_json": valid_json,
            "latency_seconds": record.get("latency_seconds"),
        }
        score_rows.append(row)
        group_key = (
            record["pipeline"],
            record["task"],
            record["condition"],
        )
        grouped[group_key].append(score)
        if valid_json is not None:
            json_grouped[group_key].append(valid_json)

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    with (RESULT_DIR / "scores.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "pipeline",
                "sample",
                "condition",
                "task",
                "score",
                "valid_json",
                "latency_seconds",
            ],
        )
        writer.writeheader()
        writer.writerows(score_rows)

    means: dict[str, dict[str, dict[str, float]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    valid_json_rates: dict[str, dict[str, dict[str, float]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for (pipeline, task, condition), values in grouped.items():
        means[pipeline][task][condition] = mean(values)
    for (pipeline, task, condition), values in json_grouped.items():
        valid_json_rates[pipeline][task][condition] = mean(values)

    transcript_by_audio = {}
    for record in selected:
        if record["pipeline"] == "cascade" and record.get("transcript"):
            transcript_by_audio.setdefault(
                (record["sample"], record["condition"]),
                record["transcript"],
            )
    wer_by_condition: dict[str, list[float]] = defaultdict(list)
    for (sample, condition), transcript in transcript_by_audio.items():
        source_transcript = next(
            entry["transcript"]
            for entry in json.loads(
                (
                    ROOT
                    / "data"
                    / "processed"
                    / "tts_samples"
                    / "index.json"
                ).read_text(encoding="utf-8")
            )
            if Path(entry["audio_path"]).stem == sample
        )
        wer_by_condition[condition].append(
            wer(source_transcript, transcript)
        )

    summary = {
        "experiment_id": config["experiment_id"],
        "scope": "smoke" if args.smoke else "full",
        "expected_results": len(expected_keys),
        "available_results": len(actual_keys),
        "missing_results": [
            list(key) for key in sorted(expected_keys - actual_keys)
        ],
        "means": means,
        "valid_json_rates": valid_json_rates,
        "wer": {
            condition: mean(values)
            for condition, values in wer_by_condition.items()
        },
    }
    if "qwen" in args.pipelines:
        summary["original_report_direct_clean"] = ORIGINAL_DIRECT_BASELINE
        summary["new_clean_minus_original"] = {
            task: means["qwen"][task]["clean"] - original_score
            for task, original_score in ORIGINAL_DIRECT_BASELINE.items()
        }
    summary["changes_from_clean"] = {
        pipeline: {
            task: {
                condition: score - condition_scores["clean"]
                for condition, score in condition_scores.items()
                if condition != "clean" and "clean" in condition_scores
            }
            for task, condition_scores in task_scores.items()
        }
        for pipeline, task_scores in means.items()
    }
    (RESULT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_plots(summary, args.pipelines)
    print(
        f"Results: {len(actual_keys)}/{len(expected_keys)} complete. "
        f"Summary: {RESULT_DIR / 'summary.json'}"
    )


if __name__ == "__main__":
    main()

"""Build final N=50 human/TTS analysis tables for the report.

Human: start from the N=66 combined human result, replace v5 scores with the
external rensheng_results.json values, and remove 16 non-v5 describe samples.
TTS: import the external TTS_50_results.json file and convert it to project
score/summary tables.
"""
from __future__ import annotations

import csv
from html import escape
import json
import os
import shutil
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
EXTERNAL_ROOT = ROOT.parent
HUMAN_SOURCE = EXTERNAL_ROOT / "yinpin" / "rensheng_results.json"
TTS_SOURCE = EXTERNAL_ROOT / "TTS_50_results.json"

HUMAN_OUT = ROOT / "data" / "results" / "human_speech_final_n50"
TTS_OUT = ROOT / "data" / "results" / "tts_speech_final_n50"
REPORT_ZH_OUT = ROOT / "report" / "report.zh-CN.md"
REPORT_EN_OUT = ROOT / "report" / "report.md"
REPORT_HTML_OUT = ROOT / "report" / "report.html"
FIGURE_DIR = ROOT / "report" / "figures"

TASKS = ["summarization", "sentiment", "keywords", "intent"]
PIPELINES = [
    "A_oracle",
    "B_whisper_cascade",
    "C_qwen_transcript",
    "D_qwen_direct",
]
HUMAN_PATH_MAP = {
    "Oracle": "A_oracle",
    "Cascade": "B_whisper_cascade",
    "Qwen_ASR": "C_qwen_transcript",
    "Qwen_Direct": "D_qwen_direct",
}
TTS_PATH_MAP = {
    "Oracle_GT_DS": "A_oracle",
    "Cascade_Whisper_DS": "B_whisper_cascade",
    "QwenASR_Qwen_DS": "C_qwen_transcript",
    "QwenDirect_end2end": "D_qwen_direct",
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def f1_from_precision_recall(payload: dict[str, Any]) -> float:
    precision = float(payload.get("precision") or 0.0)
    recall = float(payload.get("recall") or 0.0)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def external_score(task: str, payload: dict[str, Any]) -> float:
    if task == "summarization":
        return float(payload["rouge_l"])
    if task in {"sentiment", "intent"}:
        return 1.0 if payload["correct"] else 0.0
    if task == "keywords":
        return f1_from_precision_recall(payload)
    raise ValueError(task)


def bootstrap_ci(values: list[float], seed: int) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(20_000, len(array)))
    means = array[indices].mean(axis=1)
    return [float(v) for v in np.quantile(means, [0.025, 0.975])]


def build_summary(
    rows: list[dict[str, Any]],
    sample_meta: dict[tuple[str, str], dict[str, Any]],
    experiment_id: str,
    source: str,
    notes: list[str],
) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    by_intent: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    by_sample_task: dict[tuple[str, str, str], dict[str, float]] = defaultdict(dict)
    for row in rows:
        pipeline = row["pipeline"]
        task = row["task"]
        score = float(row["score"])
        sample_key = (row["dataset"], row["sample"])
        intent = sample_meta[sample_key]["intent"]
        grouped[(pipeline, task)].append(score)
        by_intent[(intent, pipeline, task)].append(score)
        by_sample_task[(row["dataset"], row["sample"], task)][pipeline] = score

    means = {
        pipeline: {task: mean(grouped[(pipeline, task)]) for task in TASKS}
        for pipeline in PIPELINES
    }
    intent_counts = Counter(meta["intent"] for meta in sample_meta.values())
    means_by_intent = {
        intent: {
            pipeline: {
                task: mean(by_intent[(intent, pipeline, task)])
                for task in TASKS
                if by_intent[(intent, pipeline, task)]
            }
            for pipeline in PIPELINES
        }
        for intent in sorted(intent_counts)
    }
    pairwise = {}
    for pair_index, (left, right) in enumerate(combinations(PIPELINES, 2)):
        label = f"{left}_minus_{right}"
        pairwise[label] = {}
        for task_index, task in enumerate(TASKS):
            diffs = [
                values[left] - values[right]
                for (dataset, sample, sample_task), values in by_sample_task.items()
                if sample_task == task
            ]
            pairwise[label][task] = {
                "mean_difference": mean(diffs),
                "paired_bootstrap_95_ci": bootstrap_ci(
                    diffs, 9100 + pair_index * 10 + task_index
                ),
                "left_wins": sum(v > 1e-12 for v in diffs),
                "right_wins": sum(v < -1e-12 for v in diffs),
                "ties": sum(abs(v) <= 1e-12 for v in diffs),
            }

    return {
        "experiment_id": experiment_id,
        "sample_count": len(sample_meta),
        "score_row_count": len(rows),
        "intent_counts": dict(sorted(intent_counts.items())),
        "means": means,
        "means_by_intent": means_by_intent,
        "pairwise": pairwise,
        "source": source,
        "notes": notes,
    }


def write_scores(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["dataset", "sample", "task", "pipeline", "score"]
        )
        writer.writeheader()
        writer.writerows(rows)


def build_human_final() -> dict[str, Any]:
    HUMAN_OUT.mkdir(parents=True, exist_ok=True)
    truth = read_json(ROOT / "data" / "ground_truth_human_speech_combined_n66.json")
    n66_rows = list(
        csv.DictReader(
            (ROOT / "data" / "results" / "human_speech_combined_n66" / "scores.csv").open(
                encoding="utf-8"
            )
        )
    )
    external = read_json(HUMAN_SOURCE)
    shutil.copy2(HUMAN_SOURCE, HUMAN_OUT / "rensheng_results_external_v5.json")

    # Replace v5 scores in memory with the external computer's full four-task scores.
    replacement_scores = {}
    for sample, record in external["per_sample"].items():
        for external_path, pipeline in HUMAN_PATH_MAP.items():
            for task in TASKS:
                replacement_scores[("v5", sample, task, pipeline)] = external_score(
                    task, record["paths"][external_path][task]
                )

    sample_meta = {
        (payload["dataset"], payload["sample"]): {
            "dataset": payload["dataset"],
            "sample": payload["sample"],
            "intent": payload["intent"],
        }
        for payload in truth.values()
    }
    describe_non_v5 = [
        key
        for key, meta in sample_meta.items()
        if meta["intent"] == "describe" and meta["dataset"] != "v5"
    ]
    removed = describe_non_v5[:16]
    removed_set = set(removed)
    kept_meta = {
        key: meta for key, meta in sample_meta.items() if key not in removed_set
    }
    rows = []
    for row in n66_rows:
        key = (row["dataset"], row["sample"])
        if key in removed_set:
            continue
        pipeline = row["pipeline"].replace("D_qwen_direct_system", "D_qwen_direct")
        original_pipeline = row["pipeline"]
        score_key = (row["dataset"], row["sample"], row["task"], pipeline)
        original_score_key = (
            row["dataset"],
            row["sample"],
            row["task"],
            original_pipeline.replace("D_qwen_direct_system", "D_qwen_direct"),
        )
        score = replacement_scores.get(original_score_key, float(row["score"]))
        rows.append(
            {
                "dataset": row["dataset"],
                "sample": row["sample"],
                "task": row["task"],
                "pipeline": pipeline,
                "score": score,
            }
        )

    summary = build_summary(
        rows,
        kept_meta,
        "human_speech_final_n50",
        "N=66 human combined scores; v5 replaced by rensheng_results.json; 16 non-v5 describe samples removed.",
        [
            "Final human report set keeps all v5 samples and uses the external v5 results as ordinary v5 scores.",
            "Sixteen non-v5 samples with ground-truth intent=describe were removed deterministically in existing dataset order.",
            "N=50 supports descriptive comparison; bootstrap intervals are reported but no strong statistical-significance claim is required.",
        ],
    )
    write_scores(HUMAN_OUT / "scores.csv", rows)
    (HUMAN_OUT / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    selection = {
        "kept_count": len(kept_meta),
        "removed_count": len(removed),
        "removed_describe_samples": [
            {"dataset": dataset, "sample": sample} for dataset, sample in removed
        ],
        "kept_intent_counts": summary["intent_counts"],
    }
    (HUMAN_OUT / "selection.json").write_text(
        json.dumps(selection, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary


def infer_tts_intent(sample: str, payload: dict[str, Any]) -> str:
    truth = payload.get("ground_truth", {})
    if isinstance(truth, dict) and truth.get("intent"):
        return str(truth["intent"])
    prefix = sample.split("_", 1)[0]
    if prefix in {"describe", "entertain", "persuade", "question"}:
        return prefix
    return "inform"


def build_tts_final() -> dict[str, Any]:
    TTS_OUT.mkdir(parents=True, exist_ok=True)
    external = read_json(TTS_SOURCE)
    shutil.copy2(TTS_SOURCE, TTS_OUT / "TTS_50_results_external.json")
    rows = []
    sample_meta = {}
    for sample in external["samples"]:
        record = external["per_sample"][sample]
        sample_meta[("tts50", sample)] = {
            "dataset": "tts50",
            "sample": sample,
            "intent": infer_tts_intent(sample, record),
        }
        for external_path, pipeline in TTS_PATH_MAP.items():
            for task in TASKS:
                rows.append(
                    {
                        "dataset": "tts50",
                        "sample": sample,
                        "task": task,
                        "pipeline": pipeline,
                        "score": external_score(
                            task, record["paths"][external_path][task]
                        ),
                    }
                )
    summary = build_summary(
        rows,
        sample_meta,
        "tts_speech_final_n50",
        "External TTS_50_results.json converted to project score tables.",
        [
            "TTS has 50 synthetic-speech samples and the same four path definitions.",
            "Path D is QwenDirect_end2end in the external result file.",
            "Intent labels come from ground_truth.intent when available; otherwise from sample-name prefixes.",
        ],
    )
    write_scores(TTS_OUT / "scores.csv", rows)
    (TTS_OUT / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary


def fmt_pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def fmt(value: float) -> str:
    return f"{value:.4f}"


def add_value_labels(ax, fmt_spec: str = "{:.3f}") -> None:
    for container in ax.containers:
        ax.bar_label(
            container,
            labels=[
                fmt_spec.format(bar.get_height())
                for bar in container
            ],
            fontsize=8,
            padding=2,
        )


def plot_metric_bars(summary: dict[str, Any], output: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    labels = ["Summary", "Sentiment", "Keywords", "Intent"]
    task_keys = TASKS
    pipeline_labels = ["A Oracle", "B Cascade", "C Qwen-ASR", "D Direct"]
    colors = ["#6c757d", "#1f77b4", "#2ca02c", "#d62728"]
    x = np.arange(len(labels))
    width = 0.18
    fig, ax = plt.subplots(figsize=(11, 5.8), dpi=180)
    for i, (pipeline, label, color) in enumerate(zip(PIPELINES, pipeline_labels, colors)):
        values = [summary["means"][pipeline][task] for task in task_keys]
        ax.bar(x + (i - 1.5) * width, values, width, label=label, color=color)
    ax.set_title(title)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    add_value_labels(ax)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


def plot_direct_deltas(human: dict[str, Any], tts: dict[str, Any], output: Path) -> None:
    import matplotlib.pyplot as plt

    labels = ["Summary", "Sentiment", "Keywords", "Intent"]
    x = np.arange(len(labels))
    width = 0.2
    series = [
        ("Human D-B", human, "B_whisper_cascade", "#1f77b4"),
        ("Human D-C", human, "C_qwen_transcript", "#2ca02c"),
        ("TTS D-B", tts, "B_whisper_cascade", "#ff7f0e"),
        ("TTS D-C", tts, "C_qwen_transcript", "#9467bd"),
    ]
    fig, ax = plt.subplots(figsize=(11, 5.8), dpi=180)
    for i, (label, summary, baseline, color) in enumerate(series):
        values = [
            summary["means"]["D_qwen_direct"][task]
            - summary["means"][baseline][task]
            for task in TASKS
        ]
        ax.bar(x + (i - 1.5) * width, values, width, label=label, color=color)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_title("Direct advantage/disadvantage versus cascade baselines")
    ax.set_ylabel("D score minus baseline score")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


def plot_human_intent_heatmap(summary: dict[str, Any], output: Path) -> None:
    import matplotlib.pyplot as plt

    intents = sorted(summary["intent_counts"])
    task_labels = ["Summary", "Sentiment", "Keywords", "Intent"]
    matrix = []
    for intent in intents:
        row = []
        for task in TASKS:
            d = summary["means_by_intent"][intent]["D_qwen_direct"][task]
            b = summary["means_by_intent"][intent]["B_whisper_cascade"][task]
            c = summary["means_by_intent"][intent]["C_qwen_transcript"][task]
            row.append(d - max(b, c))
        matrix.append(row)
    fig, ax = plt.subplots(figsize=(9, 5.8), dpi=180)
    im = ax.imshow(matrix, cmap="RdBu", vmin=-0.25, vmax=0.25)
    ax.set_title("Human N=50: Direct minus best cascade by intent")
    ax.set_xticks(np.arange(len(task_labels)))
    ax.set_xticklabels(task_labels)
    ax.set_yticks(np.arange(len(intents)))
    ax.set_yticklabels([f"{intent} (n={summary['intent_counts'][intent]})" for intent in intents])
    for i, _intent in enumerate(intents):
        for j, _task in enumerate(TASKS):
            ax.text(j, i, f"{matrix[i][j]:+.3f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="D - max(B, C)")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


def plot_sentiment_by_intent(summary: dict[str, Any], output: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    intents = sorted(summary["intent_counts"])
    x = np.arange(len(intents))
    width = 0.2
    pipeline_labels = ["A Oracle", "B Cascade", "C Qwen-ASR", "D Direct"]
    colors = ["#6c757d", "#1f77b4", "#2ca02c", "#d62728"]
    fig, ax = plt.subplots(figsize=(11, 5.8), dpi=180)
    for i, (pipeline, label, color) in enumerate(zip(PIPELINES, pipeline_labels, colors)):
        values = [
            summary["means_by_intent"][intent][pipeline]["sentiment"]
            for intent in intents
        ]
        ax.bar(x + (i - 1.5) * width, values, width, label=label, color=color)
    ax.set_title(title)
    ax.set_ylabel("Sentiment accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{intent}\n(n={summary['intent_counts'][intent]})" for intent in intents])
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


def plot_cross_benchmark_drop(human: dict[str, Any], tts: dict[str, Any], output: Path) -> None:
    import matplotlib.pyplot as plt

    labels = ["Summary", "Sentiment", "Keywords", "Intent"]
    x = np.arange(len(labels))
    width = 0.2
    series = [
        ("B Cascade", "B_whisper_cascade", "#1f77b4"),
        ("C Qwen-ASR", "C_qwen_transcript", "#2ca02c"),
        ("D Direct", "D_qwen_direct", "#d62728"),
    ]
    fig, ax = plt.subplots(figsize=(11, 5.8), dpi=180)
    for i, (label, pipeline, color) in enumerate(series):
        values = [
            human["means"][pipeline][task] - tts["means"][pipeline][task]
            for task in TASKS
        ]
        ax.bar(x + (i - 1) * width, values, width, label=label, color=color)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_title("Human speech minus TTS performance")
    ax.set_ylabel("Human N=50 score minus TTS N=50 score")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


def generate_figures(human: dict[str, Any], tts: dict[str, Any]) -> dict[str, str]:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    paths = {
        "human_metrics": FIGURE_DIR / "final_n50_human_metrics.png",
        "tts_metrics": FIGURE_DIR / "final_n50_tts_metrics.png",
        "direct_deltas": FIGURE_DIR / "final_n50_direct_deltas.png",
        "human_intent_heatmap": FIGURE_DIR / "final_n50_human_intent_heatmap.png",
        "human_sentiment_by_intent": FIGURE_DIR / "final_n50_human_sentiment_by_intent.png",
        "tts_sentiment_by_intent": FIGURE_DIR / "final_n50_tts_sentiment_by_intent.png",
        "cross_benchmark_drop": FIGURE_DIR / "final_n50_cross_benchmark_drop.png",
    }
    plot_metric_bars(human, paths["human_metrics"], "Human speech N=50: four-path task scores")
    plot_metric_bars(tts, paths["tts_metrics"], "TTS N=50: four-path task scores")
    plot_direct_deltas(human, tts, paths["direct_deltas"])
    plot_human_intent_heatmap(human, paths["human_intent_heatmap"])
    plot_sentiment_by_intent(
        human,
        paths["human_sentiment_by_intent"],
        "Human speech N=50: sentiment accuracy by ground-truth intent",
    )
    plot_sentiment_by_intent(
        tts,
        paths["tts_sentiment_by_intent"],
        "TTS N=50: sentiment accuracy by ground-truth intent",
    )
    plot_cross_benchmark_drop(human, tts, paths["cross_benchmark_drop"])
    return {key: f"figures/{path.name}" for key, path in paths.items()}


def table_for_summary(summary: dict[str, Any]) -> str:
    lines = [
        "| 路径 | 摘要 ROUGE-L | 情感准确率 | 关键词 F1 | 意图准确率 |",
        "|---|---:|---:|---:|---:|",
    ]
    labels = {
        "A_oracle": "A Oracle（人工文本→DeepSeek）",
        "B_whisper_cascade": "B Cascade（Whisper→DeepSeek）",
        "C_qwen_transcript": "C Qwen转写→DeepSeek",
        "D_qwen_direct": "D Direct（Qwen音频直推）",
    }
    for pipeline in PIPELINES:
        means = summary["means"][pipeline]
        lines.append(
            f"| {labels[pipeline]} | {fmt(means['summarization'])} | "
            f"{fmt_pct(means['sentiment'])} | {fmt(means['keywords'])} | "
            f"{fmt_pct(means['intent'])} |"
        )
    return "\n".join(lines)


def direct_delta_table(summary: dict[str, Any]) -> str:
    lines = [
        "| 任务 | D-B | D-C | 解释 |",
        "|---|---:|---:|---|",
    ]
    task_names = {
        "summarization": "摘要",
        "sentiment": "情感",
        "keywords": "关键词",
        "intent": "意图",
    }
    for task in TASKS:
        d = summary["means"]["D_qwen_direct"][task]
        b = summary["means"]["B_whisper_cascade"][task]
        c = summary["means"]["C_qwen_transcript"][task]
        winner = "Direct 领先" if d > max(b, c) else "Direct 落后"
        lines.append(
            f"| {task_names[task]} | {fmt(d-b)} | {fmt(d-c)} | {winner} |"
        )
    return "\n".join(lines)


def intent_focus_table(summary: dict[str, Any], intent: str) -> str:
    means = summary["means_by_intent"][intent]
    lines = [
        f"| 路径（{intent}） | 摘要 | 情感 | 关键词 | 意图 |",
        "|---|---:|---:|---:|---:|",
    ]
    for pipeline in ["B_whisper_cascade", "C_qwen_transcript", "D_qwen_direct"]:
        values = means[pipeline]
        lines.append(
            f"| {pipeline} | {fmt(values['summarization'])} | "
            f"{fmt_pct(values['sentiment'])} | {fmt(values['keywords'])} | "
            f"{fmt_pct(values['intent'])} |"
        )
    return "\n".join(lines)


def table_for_summary_en(summary: dict[str, Any]) -> str:
    lines = [
        "| Path | Summary ROUGE-L | Sentiment Acc. | Keyword F1 | Intent Acc. |",
        "|---|---:|---:|---:|---:|",
    ]
    labels = {
        "A_oracle": "A Oracle (ground-truth transcript → DeepSeek)",
        "B_whisper_cascade": "B Cascade (Whisper → DeepSeek)",
        "C_qwen_transcript": "C Qwen-ASR (Qwen transcript → DeepSeek)",
        "D_qwen_direct": "D Qwen-Direct (audio-native)",
    }
    for pipeline in PIPELINES:
        means = summary["means"][pipeline]
        lines.append(
            f"| {labels[pipeline]} | {fmt(means['summarization'])} | "
            f"{fmt_pct(means['sentiment'])} | {fmt(means['keywords'])} | "
            f"{fmt_pct(means['intent'])} |"
        )
    return "\n".join(lines)


def direct_delta_table_en(summary: dict[str, Any]) -> str:
    lines = [
        "| Task | D-B | D-C | Reading |",
        "|---|---:|---:|---|",
    ]
    names = {
        "summarization": "Summary",
        "sentiment": "Sentiment",
        "keywords": "Keywords",
        "intent": "Intent",
    }
    for task in TASKS:
        d = summary["means"]["D_qwen_direct"][task]
        b = summary["means"]["B_whisper_cascade"][task]
        c = summary["means"]["C_qwen_transcript"][task]
        reading = "Direct leads" if d > max(b, c) else "Direct trails"
        lines.append(f"| {names[task]} | {fmt(d-b)} | {fmt(d-c)} | {reading} |")
    return "\n".join(lines)


def intent_focus_table_en(summary: dict[str, Any], intent: str) -> str:
    means = summary["means_by_intent"][intent]
    lines = [
        f"| Path ({intent}) | Summary | Sentiment | Keywords | Intent |",
        "|---|---:|---:|---:|---:|",
    ]
    for pipeline in ["B_whisper_cascade", "C_qwen_transcript", "D_qwen_direct"]:
        values = means[pipeline]
        lines.append(
            f"| {pipeline} | {fmt(values['summarization'])} | "
            f"{fmt_pct(values['sentiment'])} | {fmt(values['keywords'])} | "
            f"{fmt_pct(values['intent'])} |"
        )
    return "\n".join(lines)


def html_img(path: str, alt: str) -> str:
    return f'<figure><img src="{path}" alt="{alt}"><figcaption>{alt}</figcaption></figure>'


def write_report(human: dict[str, Any], tts: dict[str, Any], figures: dict[str, str]) -> None:
    english = f"""# Cascade vs End-to-End Speech Understanding: A Dual-Study Benchmark with N=50 TTS and N=50 Real-Human Speech

[简体中文](report.zh-CN.md)

**Authors:** Jiayi Li（李佳宜）, Liu Luofei（刘洛菲）, Zhang Yuchen（张予辰）  
**Date:** June-August 2026  
**Course:** Undergraduate Summer Research, Zhejiang University  
**Repository:** `github.com/jiayi0428/speech-benchmark`

## Abstract

We present a systematic comparison of cascade (ASR → LLM) and end-to-end speech understanding architectures through two parallel, equally sized benchmarks: **50 TTS-generated samples** and **50 real-human speech samples**. Four paths are evaluated on the same four tasks: A Oracle, B Whisper Cascade, C Qwen-ASR Cascade, and D Qwen-Direct.

The main result is architectural complementarity, not a single universal winner. On TTS, cascade is strongest for structured semantic tasks: B reaches **78% sentiment** and **98% intent**, while D reaches **72% sentiment** and **80% intent**. On real-human speech, A/B/C still outperform D overall, but D shows a clear local signal on prosody-heavy entertainment speech: in the human `entertain` subset, Direct sentiment reaches **50%**, ahead of B's **25%** and C's **37.5%**. This supports an intent-first routing view: use cascade as the default semantic pipeline, and reserve Direct as an auxiliary path for affective, comedic, ironic, or delivery-heavy speech.

## 1. Methodology

All experiments use the same four tasks: summarization (ROUGE-L), sentiment classification, keyword extraction (F1), and intent classification. The four processing paths are:

| Path | Audio/Text source | Downstream reasoning |
|---|---|---|
| A Oracle | Ground-truth transcript | DeepSeek-chat |
| B Cascade | faster-whisper large-v3 transcript | DeepSeek-chat |
| C Qwen-ASR | Qwen2-Audio transcript | DeepSeek-chat |
| D Qwen-Direct | Qwen2-Audio direct audio analysis | DeepSeek structuring for structured tasks |

The final N=50 human set is derived from the N=66 human pool by removing 16 non-v5 `describe` samples. All v5 samples are retained, and the updated external v5 result file replaces the earlier local v5 results. The TTS set uses the 50-sample `TTS_50_results.json` result file.

Human intent distribution: `{human['intent_counts']}`.  
TTS intent distribution: `{tts['intent_counts']}`.

## 2. TTS Benchmark (N=50)

{table_for_summary_en(tts)}

![TTS N=50 four-path scores]({figures['tts_metrics']})

TTS is the controlled condition: speech is clean, pronunciation is stable, and there is little room for prosody to add information beyond the transcript. The result is correspondingly clean: B/C are strong, and D is competitive for summarization but weaker for intent. The 18-point intent gap between B and D is the most important TTS result.

{direct_delta_table_en(tts)}

![Direct advantage/disadvantage versus cascade]({figures['direct_deltas']})

## 3. Human Speech Benchmark (N=50)

{table_for_summary_en(human)}

![Human N=50 four-path scores]({figures['human_metrics']})

On real-human speech, A/B/C remain the stronger general-purpose choices. Direct drops especially on summarization, suggesting that noisy, naturalistic audio still challenges Qwen2-Audio direct comprehension. But the mean table hides the most product-relevant signal: Direct is useful in the specific region where tone and delivery carry meaning.

{direct_delta_table_en(human)}

## 4. Per-Intent Findings

The strongest Direct signal is `entertain` sentiment:

{intent_focus_table_en(human, 'entertain')}

![Human sentiment by intent]({figures['human_sentiment_by_intent']})

![TTS sentiment by intent]({figures['tts_sentiment_by_intent']})

![Human Direct heatmap by intent]({figures['human_intent_heatmap']})

Direct is not the best universal speech-understanding architecture. But it is not merely worse either. It wins the most plausible place for an audio-native model to win: human entertainment speech, where vocal delivery, irony, timing, and audience context affect sentiment. This is the central product insight.

## 5. Cross-Benchmark Comparison

![Human minus TTS performance]({figures['cross_benchmark_drop']})

TTS overstates how easy the real deployment problem is. B's sentiment drops from 78% on TTS to 68% on human speech, and D's summarization drops from 0.325 to 0.185. The gap is not just noise; it reflects natural recording artifacts, audience sound, delivery style, and messier real human speech.

## 6. Production Routing Framework

The practical recommendation is not to choose one path for everything:

| Scenario | Recommended routing |
|---|---|
| Clean TTS or factual speech | B Whisper Cascade as default |
| Summary | B/C first; D can be inspected but not default |
| Keywords | B/C; avoid D as primary |
| Intent | B as default, especially for TTS |
| Entertainment / comedy / irony sentiment | Run D as an auxiliary sentiment path |
| Cost/privacy constrained exploration | D can be used as a lightweight first pass, with accuracy caveats |

This gives a product form: an **intent-first speech router**. The router sends most content-driven tasks to cascade, while dispatching prosody-heavy sentiment cases to Direct for auxiliary judgment.

## 7. Limitations

The N=50 results are much stronger than the earlier pilots, but they are still a project-scale benchmark. Only one model combination is tested per architecture family: Whisper large-v3, Qwen2-Audio-7B, and DeepSeek-chat. Human annotations do not include formal inter-annotator agreement. Per-intent subsets are uneven: `entertain` has 8 human samples and `persuade` has 4, so those subgroup conclusions should be treated as directional but important.

## 8. Conclusion

Cascade is the better default architecture for the current four semantic tasks. It is especially strong for intent and keywords, and it remains robust on clean TTS. Direct's value is narrower but real: it can exploit paralinguistic information in entertainment-style human speech, especially for sentiment. The best system is therefore not a single path but a router: cascade for content, Direct for selected prosody-heavy affective cases.

## Output files

- Human N=50 summary: `data/results/human_speech_final_n50/summary.json`
- Human N=50 scores: `data/results/human_speech_final_n50/scores.csv`
- TTS N=50 summary: `data/results/tts_speech_final_n50/summary.json`
- TTS N=50 scores: `data/results/tts_speech_final_n50/scores.csv`
"""
    REPORT_EN_OUT.write_text(english, encoding="utf-8")

    chinese = f"""# 级联式与端到端语音理解：N=50 TTS 与 N=50 真实人声双实验基准

[English](report.md)

**作者：** Jiayi Li（李佳宜）、Liu Luofei（刘洛菲）、Zhang Yuchen（张予辰）  
**日期：** 2026 年 6 月至 8 月  
**课程：** 浙江大学本科暑期研究  
**代码仓库：** `github.com/jiayi0428/speech-benchmark`

## 摘要

本报告比较两种语音理解架构：**级联式**（ASR → 文本 LLM）和**端到端式**（音频直接输入语音大模型）。最终采用两个等规模数据集：**50 条 TTS 合成语音**和**50 条真实人声录音**。四条路径完全对齐：A Oracle、B Whisper 级联、C Qwen 转写级联、D Qwen Direct。

核心结论不是“某一路径永远赢”，而是：**级联路径是当前四个语义任务的默认最优选择，Direct 在语气/表演性强的情感判断中有局部价值。** TTS 中，B 达到 **78% 情感准确率**和 **98% 意图准确率**，D 为 **72%** 和 **80%**。真实人声中，A/B/C 总体仍强于 D，但在 `entertain` 子集上，D 的情感准确率达到 **50%**，高于 B 的 **25%** 和 C 的 **37.5%**。这说明端到端音频模型确实可能捕捉到转写文本丢失的语气、讽刺、节奏和表演线索。

## 1. 方法

所有实验使用同四项任务：摘要（ROUGE-L）、情感分类、关键词提取（F1）和意图分类。四条路径如下：

| 路径 | 信息来源 | 下游推理 |
|---|---|---|
| A Oracle | 人工真值转写 | DeepSeek-chat |
| B Cascade | faster-whisper large-v3 转写 | DeepSeek-chat |
| C Qwen-ASR | Qwen2-Audio 转写 | DeepSeek-chat |
| D Qwen-Direct | Qwen2-Audio 直接音频理解 | 结构化任务再由 DeepSeek 整理格式 |

人声 N=50 来自现有 N=66 人声集合：剔除 16 条非 v5 的 `describe` 样本；v5 全部保留，并用另一台电脑更新后的 `rensheng_results.json` 覆盖原 v5 结果。TTS N=50 使用 `TTS_50_results.json`。

人声意图分布：`{human['intent_counts']}`。  
TTS 意图分布：`{tts['intent_counts']}`。

## 2. TTS 基准（N=50）

{table_for_summary(tts)}

![TTS N=50 四路径任务得分]({figures['tts_metrics']})

TTS 是更受控的条件：语音干净、发音稳定，声学信息很少提供文本之外的额外价值。因此结果很清楚：B/C 很强，D 在摘要上接近，但在意图识别上明显落后。B 与 D 的 18 个百分点意图差距，是 TTS 实验最重要的信号。

{direct_delta_table(tts)}

![Direct 相对级联路径差值]({figures['direct_deltas']})

## 3. 真实人声基准（N=50）

{table_for_summary(human)}

![人声 N=50 四路径任务得分]({figures['human_metrics']})

真实人声中，A/B/C 仍然是更强的通用方案。D 在摘要上下降明显，说明自然噪声、观众声、停顿和录音差异会挑战 Qwen2-Audio 的直接理解能力。但均值表也会掩盖一个重要产品信号：Direct 在语气和表演性真正重要的场景里有价值。

{direct_delta_table(human)}

## 4. 按意图分组的发现

Direct 最值得关注的是 `entertain` 情感判断：

{intent_focus_table(human, 'entertain')}

![人声按真实意图的情感准确率]({figures['human_sentiment_by_intent']})

![TTS 按真实意图的情感准确率]({figures['tts_sentiment_by_intent']})

![人声按真实意图的 Direct 信号热力图]({figures['human_intent_heatmap']})

Direct 不是最好的通用语音理解路径，但它也不是“没有用”。它赢在最符合直觉的位置：真实娱乐/脱口秀类语音的情感判断。笑点、讽刺、语速、停顿、观众反应和说话方式会携带文本转写无法完整保留的情绪线索。

## 5. TTS 与真实人声的差异

![真实人声相对 TTS 的性能变化]({figures['cross_benchmark_drop']})

TTS 会高估真实部署的容易程度。B 的情感准确率从 TTS 的 78% 降到人声的 68%；D 的摘要 ROUGE-L 从 0.325 降到 0.185。这不只是随机波动，而是自然录音条件、环境声、表演性和真实语音不规整性共同造成的难度上升。

## 6. 产品化路径：Intent-first Router

实验结果最适合落地成一个“按场景路由”的语音理解产品，而不是强行选择单一路径：

| 场景 | 推荐路由 |
|---|---|
| 清晰 TTS 或事实性语音 | B Whisper 级联作为默认 |
| 摘要 | 优先 B/C；D 可作为补充观察 |
| 关键词 | 优先 B/C，不建议 D 作为主路径 |
| 意图 | B 最稳，尤其在 TTS 中 |
| 娱乐、脱口秀、讽刺类情感判断 | 增加 D 作为情感辅助路径 |
| 成本/隐私优先的探索性场景 | 可用 D 做轻量首轮，但必须提示准确率代价 |

因此，本项目的实际产品方向可以是 **SpeechPath Router**：内容驱动任务走级联，语气/情绪驱动任务引入 Direct 辅助。

## 7. 局限性

N=50 已经比早期 N=8/N=12 稳定很多，但仍是项目规模基准。每个架构族只测试了一组模型：Whisper large-v3、Qwen2-Audio-7B 和 DeepSeek-chat。人工标注没有正式计算多标注者一致性。按 intent 分组后的样本不均衡：人声 `entertain` 只有 8 条，`persuade` 只有 4 条，因此子组结论应作为明确趋势和产品假设，而不是无限泛化的定律。

## 8. 结论

当前四个语义任务下，**级联路径仍然是默认更可靠的语音理解架构**。它在意图和关键词上尤其稳，在 TTS 中优势更明显。Direct 的价值更窄，但真实存在：它能在娱乐/讽刺/表演性强的人声情感判断中利用语气和声学线索。最好的系统不是单一路径，而是一个路由器：内容任务走级联，强语气情感任务引入 Direct。

## 输出文件

- 人声 N=50 汇总：`data/results/human_speech_final_n50/summary.json`
- 人声 N=50 明细：`data/results/human_speech_final_n50/scores.csv`
- TTS N=50 汇总：`data/results/tts_speech_final_n50/summary.json`
- TTS N=50 明细：`data/results/tts_speech_final_n50/scores.csv`
"""
    REPORT_ZH_OUT.write_text(chinese, encoding="utf-8")

    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>Cascade vs End-to-End Speech Understanding</title>
  <style>
    body {{ max-width: 1040px; margin: 40px auto; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif; line-height: 1.65; color: #1f2937; }}
    h1, h2, h3 {{ color: #111827; }}
    table {{ border-collapse: collapse; width: 100%; margin: 18px 0; }}
    th, td {{ border: 1px solid #d1d5db; padding: 8px; text-align: left; }}
    th {{ background: #f3f4f6; }}
    img {{ max-width: 100%; border: 1px solid #e5e7eb; border-radius: 10px; }}
    figure {{ margin: 26px 0; }}
    figcaption {{ color: #6b7280; font-size: 14px; margin-top: 6px; }}
    .note {{ background: #eef6ff; border-left: 5px solid #2563eb; padding: 12px 16px; }}
    .split {{ border-top: 3px solid #111827; margin: 50px 0; }}
  </style>
</head>
<body>
  <h1>Cascade vs End-to-End Speech Understanding</h1>
  <p><strong>Authors:</strong> Jiayi Li（李佳宜）, Liu Luofei（刘洛菲）, Zhang Yuchen（张予辰）</p>
  <p class="note"><strong>Final version:</strong> bilingual N=50 TTS + N=50 real-human speech benchmark. See <code>report.md</code> and <code>report.zh-CN.md</code> for editable Markdown.</p>
  <h2>English Summary</h2>
  <p>This dual-study benchmark compares four speech-understanding paths on N=50 TTS and N=50 real-human speech. Cascade remains the best default for semantic tasks, while Direct shows a narrow but important advantage for entertainment-style human-speech sentiment.</p>
  {html_img(figures['tts_metrics'], 'TTS N=50 four-path task scores')}
  {html_img(figures['human_metrics'], 'Human speech N=50 four-path task scores')}
  {html_img(figures['direct_deltas'], 'Direct score minus cascade baselines')}
  {html_img(figures['human_sentiment_by_intent'], 'Human sentiment accuracy by intent')}
  {html_img(figures['tts_sentiment_by_intent'], 'TTS sentiment accuracy by intent')}
  {html_img(figures['human_intent_heatmap'], 'Human Direct signal by intent and task')}
  {html_img(figures['cross_benchmark_drop'], 'Human speech minus TTS performance')}
  <h2>Key Takeaway</h2>
  <p><strong>Cascade should be the default route.</strong> Direct should be added as an auxiliary route for prosody-heavy sentiment tasks, especially entertainment, comedy, irony, and expressive speech.</p>
  <div class="split"></div>
  <h1>中文摘要</h1>
  <p>本双实验基准在 N=50 TTS 和 N=50 真实人声上比较四条语音理解路径。结论是：级联路径仍是语义任务默认主路径；Direct 的优势更窄，主要出现在娱乐/脱口秀类真实人声的情感判断中。</p>
  <h2>推荐产品化方向</h2>
  <p>最适合落地为 <strong>SpeechPath Router</strong>：内容型任务默认走 Cascade；语气、讽刺、表演性强的情感任务加入 Direct 辅助判断。</p>
</body>
</html>
"""
    REPORT_HTML_OUT.write_text(html, encoding="utf-8")


def html_table(headers: list[str], rows: list[list[str]], class_name: str = "") -> str:
    class_attr = f' class="{class_name}"' if class_name else ""
    head = "".join(f"<th>{escape(str(h))}</th>" for h in headers)
    body = "\n".join(
        "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in rows
    )
    return f"<table{class_attr}><thead><tr>{head}</tr></thead><tbody>\n{body}\n</tbody></table>"


def path_label_html(pipeline: str) -> str:
    labels = {
        "A_oracle": "<strong>A Oracle</strong><br><span>人工真值转写 → DeepSeek</span>",
        "B_whisper_cascade": "<strong>B Whisper Cascade</strong><br><span>Whisper 转写 → DeepSeek</span>",
        "C_qwen_transcript": "<strong>C Qwen-ASR Cascade</strong><br><span>Qwen2-Audio 转写 → DeepSeek</span>",
        "D_qwen_direct": "<strong>D Qwen-Direct</strong><br><span>Qwen2-Audio 音频理解 → DeepSeek 结构化</span>",
    }
    return labels[pipeline]


def overall_rows(summary: dict[str, Any]) -> list[list[str]]:
    rows = []
    for pipeline in PIPELINES:
        means = summary["means"][pipeline]
        rows.append(
            [
                path_label_html(pipeline),
                fmt(means["summarization"]),
                fmt_pct(means["sentiment"]),
                fmt(means["keywords"]),
                fmt_pct(means["intent"]),
            ]
        )
    return rows


def intent_distribution_rows(summary: dict[str, Any]) -> list[list[str]]:
    order = ["describe", "entertain", "inform", "persuade", "question"]
    return [[intent, str(summary["intent_counts"].get(intent, 0))] for intent in order]


def sentiment_by_intent_rows(summary: dict[str, Any]) -> list[list[str]]:
    rows = []
    for intent, n in summary["intent_counts"].items():
        means = summary["means_by_intent"][intent]
        b = means["B_whisper_cascade"]["sentiment"]
        c = means["C_qwen_transcript"]["sentiment"]
        d = means["D_qwen_direct"]["sentiment"]
        rows.append(
            [
                escape(intent),
                str(n),
                fmt_pct(b),
                fmt_pct(c),
                fmt_pct(d),
                f"{(d - b) * 100:+.1f} pp",
            ]
        )
    return rows


def direct_gap_rows(summary: dict[str, Any]) -> list[list[str]]:
    task_names = {
        "summarization": "Summary ROUGE-L / 摘要",
        "sentiment": "Sentiment accuracy / 情感",
        "keywords": "Keyword F1 / 关键词",
        "intent": "Intent accuracy / 意图",
    }
    rows = []
    for task in TASKS:
        d = summary["means"]["D_qwen_direct"][task]
        b = summary["means"]["B_whisper_cascade"][task]
        c = summary["means"]["C_qwen_transcript"][task]
        rows.append([task_names[task], fmt(d - b), fmt(d - c)])
    return rows


def pairwise_row(
    summary: dict[str, Any],
    comparison: str,
    task: str,
    label: str,
    metric_kind: str,
) -> list[str]:
    item = summary["pairwise"][comparison][task]
    diff = item["mean_difference"]
    lo, hi = item["paired_bootstrap_95_ci"]
    if metric_kind == "pp":
        diff_s = f"{diff * 100:+.1f} pp"
        ci_s = f"[{lo * 100:+.1f}, {hi * 100:+.1f}] pp"
    else:
        diff_s = f"{diff:+.3f}"
        ci_s = f"[{lo:+.3f}, {hi:+.3f}]"
    if lo > 0 or hi < 0:
        reading = "direction stable in this sample / 本样本方向稳定"
    elif lo == 0 or hi == 0:
        reading = "borderline; do not overstate / 边界结果，不宜夸大"
    else:
        reading = "interval crosses zero / 区间跨过 0"
    return [label, diff_s, ci_s, reading]


def write_research_bilingual_html(
    human: dict[str, Any], tts: dict[str, Any], figures: dict[str, str]
) -> None:
    tts_overall = html_table(
        ["Path / 路径", "Summary", "Sentiment", "Keywords", "Intent"],
        overall_rows(tts),
    )
    human_overall = html_table(
        ["Path / 路径", "Summary", "Sentiment", "Keywords", "Intent"],
        overall_rows(human),
    )
    tts_intents = html_table(["Intent / 意图", "N"], intent_distribution_rows(tts), "compact")
    human_intents = html_table(["Intent / 意图", "N"], intent_distribution_rows(human), "compact")
    tts_direct_gaps = html_table(
        ["Task / 任务", "D - B", "D - C"], direct_gap_rows(tts), "compact"
    )
    human_direct_gaps = html_table(
        ["Task / 任务", "D - B", "D - C"], direct_gap_rows(human), "compact"
    )
    tts_sentiment_intent = html_table(
        ["Intent / 意图", "N", "B sentiment", "C sentiment", "D sentiment", "D - B"],
        sentiment_by_intent_rows(tts),
    )
    human_sentiment_intent = html_table(
        ["Intent / 意图", "N", "B sentiment", "C sentiment", "D sentiment", "D - B"],
        sentiment_by_intent_rows(human),
    )
    evidence = html_table(
        ["Comparison / 对比", "Mean difference / 均值差", "Bootstrap 95% CI", "Reading / 解读"],
        [
            pairwise_row(
                tts,
                "B_whisper_cascade_minus_C_qwen_transcript",
                "sentiment",
                "TTS: B Cascade - C Qwen-ASR sentiment",
                "pp",
            ),
            pairwise_row(
                tts,
                "B_whisper_cascade_minus_D_qwen_direct",
                "intent",
                "TTS: B Cascade - D Direct intent",
                "pp",
            ),
            pairwise_row(
                human,
                "B_whisper_cascade_minus_D_qwen_direct",
                "summarization",
                "Human: B Cascade - D Direct summary",
                "metric",
            ),
            pairwise_row(
                human,
                "C_qwen_transcript_minus_D_qwen_direct",
                "keywords",
                "Human: C Qwen-ASR - D Direct keywords",
                "metric",
            ),
            pairwise_row(
                human,
                "A_oracle_minus_B_whisper_cascade",
                "sentiment",
                "Human: A Oracle - B Cascade sentiment",
                "pp",
            ),
            pairwise_row(
                tts,
                "A_oracle_minus_B_whisper_cascade",
                "sentiment",
                "TTS: A Oracle - B Cascade sentiment",
                "pp",
            ),
        ],
    )
    charts = "\n".join(
        [
            html_img(figures["tts_metrics"], "Figure 1. TTS N=50 four-path task scores / TTS 四路径总览"),
            html_img(
                figures["human_metrics"],
                "Figure 2. Real-human N=50 four-path task scores / 真实人声四路径总览",
            ),
            html_img(
                figures["direct_deltas"],
                "Figure 3. Direct minus Cascade/Qwen-ASR deltas / Direct 相对两条转写路径的差值",
            ),
            html_img(
                figures["human_sentiment_by_intent"],
                "Figure 4. Human sentiment accuracy by intent / 人声按意图的情感准确率",
            ),
            html_img(
                figures["tts_sentiment_by_intent"],
                "Figure 5. TTS sentiment accuracy by intent / TTS 按意图的情感准确率",
            ),
            html_img(
                figures["human_intent_heatmap"],
                "Figure 6. Human Direct signal heatmap by intent / 人声 Direct 分意图热力图",
            ),
            html_img(
                figures["cross_benchmark_drop"],
                "Figure 7. Human minus TTS performance shift / 人声相对 TTS 的性能变化",
            ),
        ]
    )
    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Cascade vs End-to-End Speech Understanding: N=50 TTS + N=50 Real Human Speech</title>
  <style>
    :root {{
      --ink: #172033;
      --muted: #64748b;
      --line: #dbe3ef;
      --soft: #f6f8fb;
      --blue: #1d4ed8;
      --green: #047857;
      --orange: #b45309;
      --red: #b91c1c;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: #eef2f7;
      color: var(--ink);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans SC", "Microsoft YaHei", Arial, sans-serif;
      line-height: 1.65;
    }}
    main {{
      max-width: 1180px;
      margin: 0 auto;
      padding: 32px 22px 70px;
      background: #ffffff;
      box-shadow: 0 0 36px rgba(15, 23, 42, 0.08);
    }}
    .hero {{
      padding: 34px 36px;
      border-radius: 24px;
      background: linear-gradient(135deg, #0f172a 0%, #1d4ed8 58%, #0891b2 100%);
      color: white;
      margin-bottom: 28px;
    }}
    h1 {{ font-size: 34px; line-height: 1.18; margin: 0 0 14px; }}
    h2 {{ font-size: 24px; margin-top: 44px; padding-top: 18px; border-top: 2px solid var(--line); }}
    h3 {{ margin-top: 26px; font-size: 19px; }}
    .subtitle {{ font-size: 18px; opacity: 0.94; max-width: 980px; }}
    .meta {{ margin-top: 18px; font-size: 14px; opacity: 0.9; }}
    .lang-grid, .card-grid, .two-col {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 18px;
    }}
    .lang-box, .card {{
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 18px 20px;
      background: #fff;
    }}
    .lang-box h3, .card h3 {{ margin-top: 0; }}
    .card strong {{ font-size: 28px; display: block; margin-bottom: 6px; }}
    .cascade {{ border-top: 5px solid var(--blue); }}
    .direct {{ border-top: 5px solid var(--orange); }}
    .oracle {{ border-top: 5px solid var(--green); }}
    .caution {{ border-top: 5px solid var(--red); }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin: 14px 0 22px;
      font-size: 14px;
    }}
    th, td {{
      border: 1px solid var(--line);
      padding: 9px 10px;
      vertical-align: top;
    }}
    th {{ background: var(--soft); text-align: left; }}
    td span {{ color: var(--muted); font-size: 13px; }}
    table.compact {{ max-width: 520px; }}
    figure {{
      margin: 24px 0;
      padding: 14px;
      background: #fbfdff;
      border: 1px solid var(--line);
      border-radius: 18px;
    }}
    figure img {{
      display: block;
      max-width: 100%;
      margin: 0 auto;
      border-radius: 12px;
    }}
    figcaption {{
      margin-top: 10px;
      color: var(--muted);
      font-size: 14px;
      text-align: center;
    }}
    .note {{
      border-left: 5px solid var(--blue);
      background: #eff6ff;
      padding: 14px 18px;
      border-radius: 12px;
      margin: 18px 0;
    }}
    .warning {{
      border-left-color: var(--orange);
      background: #fff7ed;
    }}
    .toc a {{ color: var(--blue); text-decoration: none; margin-right: 14px; white-space: nowrap; }}
    code {{ background: var(--soft); padding: 2px 5px; border-radius: 6px; }}
    @media (max-width: 820px) {{
      .lang-grid, .card-grid, .two-col {{ grid-template-columns: 1fr; }}
      main {{ padding: 18px 12px 50px; }}
      .hero {{ padding: 24px 20px; }}
      h1 {{ font-size: 27px; }}
    }}
  </style>
</head>
<body>
<main>
  <section class="hero">
    <h1>Cascade vs End-to-End Speech Understanding</h1>
    <p class="subtitle">A bilingual dual-study benchmark with N=50 TTS speech and N=50 real-human speech.<br>级联式与端到端语音理解：N=50 TTS 与 N=50 真实人声双实验基准。</p>
    <p class="meta"><strong>Authors / 作者：</strong> Jiayi Li（李佳宜）, Liu Luofei（刘洛菲）, Zhang Yuchen（张予辰）<br>
    <strong>Date / 日期：</strong> June–August 2026　<strong>Course / 课程：</strong> Undergraduate Summer Research, Zhejiang University</p>
  </section>

  <nav class="toc">
    <a href="#abstract">Abstract / 摘要</a>
    <a href="#methods">Methods / 方法</a>
    <a href="#overall">Overall results / 总体结果</a>
    <a href="#intent">Intent analysis / 分意图分析</a>
    <a href="#evidence">Evidence strength / 证据强度</a>
    <a href="#conclusion">Conclusion / 结论</a>
  </nav>

  <section id="abstract">
    <h2>1. Abstract / 摘要</h2>
    <div class="lang-grid">
      <div class="lang-box">
        <h3>English</h3>
        <p>This report compares cascade speech understanding, where audio is first transcribed and then analyzed by a text LLM, with end-to-end audio understanding, where Qwen2-Audio analyzes the waveform directly before structured post-processing. The comparison uses two equally sized benchmarks: 50 controlled TTS clips and 50 noisy real-human clips.</p>
        <p>The central conclusion is clear: cascade paths remain stronger for general semantic tasks, especially intent and keyword extraction. Direct understanding is not a universal winner, but it shows a meaningful local advantage in entertainment-style human speech sentiment, where delivery and tone carry information that transcripts can lose.</p>
      </div>
      <div class="lang-box">
        <h3>中文</h3>
        <p>本报告比较两类语音理解架构：级联式路径先把音频转写成文本，再交给文本大模型分析；端到端路径则由 Qwen2-Audio 直接听音频，再做结构化整理。实验采用两个等规模数据集：50 条受控 TTS 音频和 50 条带真实噪声、观众声、录音差异的人声音频。</p>
        <p>核心结论很明确：级联路径在通用语义任务上仍然更强，尤其是意图识别和关键词提取。Direct 不是整体最优，但在“娱乐/脱口秀/表演性人声”的情感判断中出现了局部优势，因为语气、停顿、讽刺和现场反应可能不会完整保留在转写文本里。</p>
      </div>
    </div>
  </section>

  <section>
    <h2>2. Main findings / 核心发现</h2>
    <div class="card-grid">
      <div class="card cascade">
        <strong>B</strong>
        <h3>Cascade is the strongest default / 级联是当前最稳的通用路径</h3>
        <p>TTS: B reaches 78.0% sentiment and 98.0% intent. Human speech: B reaches 68.0% sentiment and 64.0% intent, and has the best human summarization score.</p>
        <p>在 TTS 中，B 的情感准确率为 78.0%，意图准确率为 98.0%；在人声中，B 的摘要得分最高，情感和意图也保持较稳。</p>
      </div>
      <div class="card direct">
        <strong>D</strong>
        <h3>Direct has a narrow but real signal / Direct 优势窄，但不是没有</h3>
        <p>On human <code>entertain</code> sentiment, D reaches 50.0%, while B reaches 25.0% and C reaches 37.5%. This is the clearest place where audio-native understanding helps.</p>
        <p>在人声 <code>entertain</code> 子集的情感任务中，D 为 50.0%，B 为 25.0%，C 为 37.5%。这是 Direct 最清楚的局部优势。</p>
      </div>
      <div class="card oracle">
        <strong>A</strong>
        <h3>Oracle separates transcript quality from reasoning / Oracle 用来拆分误差来源</h3>
        <p>Oracle is not an architecture for raw audio; it is a control path. Its role is to show how much of the error comes from transcription rather than downstream reasoning.</p>
        <p>Oracle 不是音频处理方案，而是控制组。它帮助判断错误到底来自转写阶段，还是来自后续文本推理阶段。</p>
      </div>
      <div class="card caution">
        <strong>TTS ≠ human</strong>
        <h3>TTS overstates real-world ease / TTS 会低估真实人声难度</h3>
        <p>From TTS to human speech, B sentiment drops by 10 pp and B intent drops by 34 pp. D summarization drops from 0.325 to 0.185.</p>
        <p>从 TTS 到真实人声，B 的情感下降 10 个百分点，意图下降 34 个百分点；D 的摘要 ROUGE-L 从 0.325 降到 0.185。</p>
      </div>
    </div>
  </section>

  <section id="methods">
    <h2>3. Methods / 实验方法</h2>
    <div class="lang-grid">
      <div class="lang-box">
        <h3>English</h3>
        <p>All four paths are evaluated on the same four tasks: summarization, sentiment classification, keyword extraction, and intent classification. A/B/C share the same DeepSeek text reasoning stage; only the transcript source differs. D uses Qwen2-Audio direct audio analysis followed by DeepSeek formatting for structured outputs.</p>
      </div>
      <div class="lang-box">
        <h3>中文</h3>
        <p>四条路径都在同样四项任务上评估：摘要、情感分类、关键词提取、意图分类。A/B/C 使用同一套 DeepSeek 文本推理流程，只改变输入文本来源；D 先由 Qwen2-Audio 直接理解音频，再由 DeepSeek 做结构化整理。</p>
      </div>
    </div>
    {html_table(["Path / 路径", "Input source / 输入来源", "Reasoning stage / 推理阶段"], [
        ["A Oracle", "Ground-truth transcript / 人工真值转写", "DeepSeek-chat"],
        ["B Whisper Cascade", "faster-whisper large-v3 transcript / Whisper 转写", "DeepSeek-chat"],
        ["C Qwen-ASR Cascade", "Qwen2-Audio transcript / Qwen 转写", "DeepSeek-chat"],
        ["D Qwen-Direct", "Qwen2-Audio direct audio analysis / 直接音频理解", "DeepSeek structured post-processing / 结构化后处理"],
    ])}
    <div class="two-col">
      <div>
        <h3>TTS intent distribution / TTS 意图分布</h3>
        {tts_intents}
      </div>
      <div>
        <h3>Human intent distribution / 人声意图分布</h3>
        {human_intents}
      </div>
    </div>
    <p class="note warning">Bootstrap confidence intervals are paired descriptive intervals over the current N=50 samples. They are useful for judging whether a direction is stable in this sample, but they should not be read as a broad population guarantee. 分组后的 <code>entertain</code> 与 <code>persuade</code> 样本量较小，相关结论应理解为明确趋势和后续验证重点。</p>
  </section>

  <section id="overall">
    <h2>4. Overall results / 总体结果</h2>
    <h3>4.1 TTS benchmark, N=50 / TTS 基准</h3>
    {tts_overall}
    <p>On clean TTS, B is strongest on sentiment and ties A on intent. D remains close on summary and keywords, but its intent accuracy is 18 pp below B. This shows that clean synthetic audio does not automatically favor direct audio understanding.</p>
    <p>在干净 TTS 中，B 的情感最好，意图与 A 并列最高。D 的摘要和关键词接近前几条路径，但意图准确率比 B 低 18 个百分点，说明“直接听音频”在干净合成语音上并不会天然占优。</p>
    {tts_direct_gaps}
    {html_img(figures["tts_metrics"], "Figure 1. TTS N=50 four-path task scores / TTS 四路径总览")}

    <h3>4.2 Real-human benchmark, N=50 / 真实人声基准</h3>
    {human_overall}
    <p>On real-human speech, A/B/C are still stronger overall. D is weakest on mean summary, keywords, and intent. The largest human-speech gap is summary: B is 0.077 ROUGE-L above D, and C is 0.071 above D.</p>
    <p>在真实人声中，A/B/C 总体仍强于 D。D 在平均摘要、关键词和意图上都落后。最明显的差距出现在摘要：B 比 D 高 0.077 ROUGE-L，C 比 D 高 0.071。</p>
    {human_direct_gaps}
    {html_img(figures["human_metrics"], "Figure 2. Real-human N=50 four-path task scores / 真实人声四路径总览")}
    {html_img(figures["direct_deltas"], "Figure 3. Direct minus cascade baselines / Direct 相对转写路径的差值")}
  </section>

  <section id="intent">
    <h2>5. Per-intent analysis / 按意图分析</h2>
    <div class="lang-grid">
      <div class="lang-box">
        <h3>English</h3>
        <p>The most important subgroup result is not that Direct wins everywhere. It does not. The important result is that Direct wins where we would theoretically expect an audio-native model to help: sentiment in entertainment-style speech. On human <code>entertain</code>, D exceeds B by 25 pp. On TTS <code>entertain</code>, D exceeds B by 12.5 pp.</p>
      </div>
      <div class="lang-box">
        <h3>中文</h3>
        <p>分组结果最重要的点不是 Direct 全面胜出——它并没有。真正重要的是：Direct 胜出的地方正好符合端到端音频模型的理论优势，即娱乐/表演性语音中的情感判断。人声 <code>entertain</code> 上 D 比 B 高 25 个百分点；TTS <code>entertain</code> 上 D 比 B 高 12.5 个百分点。</p>
      </div>
    </div>
    <h3>TTS sentiment by intent / TTS 分意图情感准确率</h3>
    {tts_sentiment_intent}
    <h3>Human sentiment by intent / 人声分意图情感准确率</h3>
    {human_sentiment_intent}
    {html_img(figures["human_sentiment_by_intent"], "Figure 4. Human sentiment accuracy by intent / 人声按意图的情感准确率")}
    {html_img(figures["tts_sentiment_by_intent"], "Figure 5. TTS sentiment accuracy by intent / TTS 按意图的情感准确率")}
    {html_img(figures["human_intent_heatmap"], "Figure 6. Human Direct signal by intent and task / 人声 Direct 分意图热力图")}
  </section>

  <section>
    <h2>6. TTS vs real human speech / TTS 与真实人声差异</h2>
    <p>Both architectures degrade when moving from TTS to real human speech, but they degrade differently. Cascade loses most on intent, while Direct loses most on summarization. This means TTS is useful as a controlled benchmark, but it cannot replace real recordings with environmental noise, audience reaction, room acoustics, pauses, and natural delivery.</p>
    <p>从 TTS 到真实人声，两类架构都会下降，但下降方式不同：级联路径主要在意图识别上下降，Direct 主要在摘要上下降。这说明 TTS 适合作为受控实验，但不能替代包含环境噪声、观众反应、房间声学、停顿和自然表达的人声数据。</p>
    {html_img(figures["cross_benchmark_drop"], "Figure 7. Human minus TTS performance shift / 人声相对 TTS 的性能变化")}
  </section>

  <section id="evidence">
    <h2>7. Evidence strength and contradictions / 证据强度与表面矛盾</h2>
    {evidence}
    <div class="lang-grid">
      <div class="lang-box">
        <h3>English</h3>
        <p>The apparent Oracle paradox is real in the table but should be interpreted carefully: on TTS, A sentiment is 6 pp below B, while on human speech, A is 6 pp above B. This suggests that clean TTS transcripts and noisy human transcripts behave differently; it does not prove that ASR errors are generally beneficial.</p>
        <p>The biggest non-contradiction is Direct itself: D is weaker overall, yet locally stronger on entertainment sentiment. These two statements are compatible because the aggregate table averages over many content-driven samples where textual semantics dominate.</p>
      </div>
      <div class="lang-box">
        <h3>中文</h3>
        <p>所谓 Oracle paradox 在表格中确实存在，但需要谨慎解释：TTS 中 A 的情感比 B 低 6 个百分点，而人声中 A 比 B 高 6 个百分点。这说明干净 TTS 文本与带噪真实人声转写的作用不同，但不能推出“ASR 错误普遍有益”。</p>
        <p>另一个容易误读的地方是 Direct：D 总体较弱，但在娱乐类情感上局部更强。这两句话并不冲突，因为总体均值包含大量内容驱动样本，而这些样本更依赖文本语义。</p>
      </div>
    </div>
  </section>

  <section>
    <h2>8. Limitations / 局限性</h2>
    <ul>
      <li><strong>Model scope / 模型范围：</strong> only Whisper large-v3, Qwen2-Audio-7B, and DeepSeek-chat are tested.</li>
      <li><strong>Annotation scope / 标注范围：</strong> human annotations are used as ground truth, but formal inter-annotator agreement is not measured.</li>
      <li><strong>Subgroup size / 子组样本量：</strong> human <code>entertain</code> has 8 samples and human <code>persuade</code> has 4 samples, so per-intent findings need further expansion.</li>
      <li><strong>Metric scope / 指标范围：</strong> summary is evaluated by ROUGE-L, which cannot fully capture factuality, coherence, or human preference.</li>
      <li><strong>Environment / 运行环境：</strong> latency or runtime differences should not be interpreted as architecture speed differences because runs came from different hardware/software conditions.</li>
    </ul>
  </section>

  <section id="conclusion">
    <h2>9. Conclusion / 结论</h2>
    <div class="lang-grid">
      <div class="lang-box">
        <h3>English</h3>
        <p>For the current four tasks, cascade is the stronger general architecture. Whisper Cascade is especially strong on TTS sentiment and intent, and it remains competitive on real-human speech. Qwen2-Audio transcription followed by DeepSeek is close on several content tasks but weaker on human intent.</p>
        <p>Direct understanding is not broadly superior, but it preserves information that transcripts can lose. Its clearest advantage appears in entertainment-style sentiment, where vocal delivery changes the meaning of the words. The correct interpretation is therefore architectural complementarity: cascade is stronger for content semantics; direct audio understanding is valuable when paralinguistic cues are central to the label.</p>
      </div>
      <div class="lang-box">
        <h3>中文</h3>
        <p>在当前四项任务下，级联路径是更强的通用架构。Whisper Cascade 在 TTS 的情感和意图上尤其强，在真实人声中也保持竞争力。Qwen2-Audio 转写后接 DeepSeek 在若干内容任务上接近，但在人声意图上较弱。</p>
        <p>Direct 并不是整体更优，但它能保留转写文本可能丢失的信息。它最清楚的优势出现在娱乐/表演性语音的情感判断中，因为说话方式会改变文字本身的含义。因此，最合理的研究结论是架构互补：内容语义优先看级联；当语气、讽刺、节奏、现场反应等副语言线索决定标签时，端到端音频理解有独立价值。</p>
      </div>
    </div>
  </section>

  <footer>
    <p class="meta">Generated from <code>data/results/human_speech_final_n50/summary.json</code> and <code>data/results/tts_speech_final_n50/summary.json</code>. Figures are stored in <code>report/figures/</code>.</p>
  </footer>
</main>
</body>
</html>
"""
    REPORT_HTML_OUT.write_text(html, encoding="utf-8")


def clean_markdown_product_language() -> None:
    replacements = {
        REPORT_EN_OUT: [
            (
                "But the mean table hides the most product-relevant signal: Direct is useful in the specific region where tone and delivery carry meaning.",
                "But the mean table hides the most important subgroup signal: Direct is useful in the specific region where tone and delivery carry meaning.",
            ),
            (
                "This is the central product insight.",
                "This is the central architectural insight.",
            ),
            (
                "This supports an intent-first routing view: use cascade as the default semantic pipeline, and reserve Direct as an auxiliary path for affective, comedic, ironic, or delivery-heavy speech.",
                "This supports a task-conditioned interpretation: cascade is the stronger semantic baseline, while Direct contributes useful audio cues for affective, comedic, ironic, or delivery-heavy speech.",
            ),
            (
                "## 6. Production Routing Framework\n\nThe practical recommendation is not to choose one path for everything:\n\n| Scenario | Recommended routing |\n|---|---|\n| Clean TTS or factual speech | B Whisper Cascade as default |\n| Summary | B/C first; D can be inspected but not default |\n| Keywords | B/C; avoid D as primary |\n| Intent | B as default, especially for TTS |\n| Entertainment / comedy / irony sentiment | Run D as an auxiliary sentiment path |\n| Cost/privacy constrained exploration | D can be used as a lightweight first pass, with accuracy caveats |\n\nThis gives a product form: an **intent-first speech router**. The router sends most content-driven tasks to cascade, while dispatching prosody-heavy sentiment cases to Direct for auxiliary judgment.",
                "## 6. Architectural Interpretation\n\nThe evidence does not support choosing one architecture for every task. Instead, the two designs expose different strengths:\n\n| Condition | Stronger evidence |\n|---|---|\n| Clean TTS and content-driven speech | B Whisper Cascade is the stronger default |\n| Summarization | B/C are safer on real-human speech; D is competitive mainly on clean TTS |\n| Keywords | A/B/C are stronger, especially on real-human speech |\n| Intent | B is strongest on TTS and tied with A on human speech |\n| Entertainment / comedy / irony sentiment | D shows the clearest local advantage |\n\nThis should be read as an architectural complementarity result: text-based cascade paths are stronger for content semantics, while audio-native understanding can add value when paralinguistic cues affect the label.",
            ),
            (
                "so those subgroup conclusions should be treated as directional but important.",
                "so those subgroup conclusions should be treated as directional and should be expanded before broad claims.",
            ),
            (
                "The best system is therefore not a single path but a router: cascade for content, Direct for selected prosody-heavy affective cases.",
                "The best interpretation is therefore not a single universal winner, but a task-conditioned trade-off: cascade for content semantics, Direct for selected prosody-heavy affective cases.",
            ),
        ],
        REPORT_ZH_OUT: [
            ("重要产品信号", "重要分组信号"),
            ("产品信号", "分组信号"),
            ("产品洞察", "架构洞察"),
            ("产品假设", "后续验证假设"),
            ("## 6. 产品化路径：Intent-first Router", "## 6. 架构解释：不同任务下的路径差异"),
            ("本项目的实际产品方向", "本项目的研究启示"),
            ("语音理解产品", "语音理解架构"),
            (
                "实验结果最适合落地成一个“按场景路由”的语音理解架构，而不是强行选择单一路径：",
                "实验结果不支持把某一条路径视为所有任务的唯一答案。更准确的解释是：不同路径保留的信息不同，因此任务类型会影响相对表现：",
            ),
            ("| 场景 | 推荐路由 |", "| 条件 | 更强的证据指向 |"),
            (
                "因此，本项目的研究启示可以是 **SpeechPath Router**：内容驱动任务走级联，语气/情绪驱动任务引入 Direct 辅助。",
                "因此，本项目的研究启示是：内容语义任务更适合级联路径；当语气、情绪、讽刺、节奏和现场反应会影响标签时，Direct 的音频线索具有独立价值。",
            ),
            (
                "实验结果最适合落地成一个“按场景路由”的语音理解架构，而不是强行选择单一路径：\n| 场景 | 推荐路由 |\n|---|---|\n| 清晰 TTS 或事实性语音 | B Whisper 级联作为默认 |\n| 摘要 | 优先 B/C；D 可作为补充观察 |\n| 关键词 | 优先 B/C，不建议 D 作为主路径 |\n| 意图 | B 最稳，尤其在 TTS 中 |\n| 娱乐、脱口秀、讽刺类情感判断 | 增加 D 作为情感辅助路径 |\n| 成本/隐私优先的探索性场景 | 可用 D 做轻量首轮，但必须提示准确率代价 |\n\n因此，本项目的研究启示可以是 **SpeechPath Router**：内容驱动任务走级联，语气/情绪驱动任务引入 Direct 辅助。",
                "实验结果不支持把某一条路径视为所有任务的唯一答案。更准确的解释是：不同路径保留的信息不同，因此任务类型会影响相对表现。\n\n| 条件 | 更强的证据指向 |\n|---|---|\n| 清晰 TTS 或内容驱动语音 | B Whisper 级联更稳 |\n| 摘要 | 真实人声中 B/C 更安全；D 主要在干净 TTS 中接近 |\n| 关键词 | A/B/C 更强，尤其是真实人声 |\n| 意图 | B 在 TTS 中最强，在人声中与 A 并列最高 |\n| 娱乐、脱口秀、讽刺类情感判断 | D 显示最明确的局部优势 |\n\n因此，本项目的研究启示是：内容语义任务更适合级联路径；当语气、情绪、讽刺、节奏和现场反应会影响标签时，Direct 的音频线索具有独立价值。",
            ),
        ],
    }
    for path, pairs in replacements.items():
        text = path.read_text(encoding="utf-8")
        for old, new in pairs:
            text = text.replace(old, new)
        path.write_text(text, encoding="utf-8")


def main() -> None:
    human = build_human_final()
    tts = build_tts_final()
    figures = generate_figures(human, tts)
    write_report(human, tts, figures)
    clean_markdown_product_language()
    write_research_bilingual_html(human, tts, figures)
    print(json.dumps({"human": human["means"], "tts": tts["means"]}, indent=2, ensure_ascii=False))
    print(f"Human summary: {HUMAN_OUT / 'summary.json'}")
    print(f"TTS summary: {TTS_OUT / 'summary.json'}")
    print(f"English report: {REPORT_EN_OUT}")
    print(f"Chinese report: {REPORT_ZH_OUT}")
    print(f"HTML report: {REPORT_HTML_OUT}")


if __name__ == "__main__":
    main()

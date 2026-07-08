"""Build final N=50 human/TTS analysis tables for the report.

Human: start from the N=66 combined human result, replace v5 scores with the
external rensheng_results.json values, and remove 16 non-v5 describe samples.
TTS: import the external TTS_50_results.json file and convert it to project
score/summary tables.
"""
from __future__ import annotations

import csv
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
REPORT_OUT = ROOT / "report" / "final_n50_report.zh-CN.md"
REPORT_EN_OUT = ROOT / "report" / "final_n50_report.md"
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


def generate_figures(human: dict[str, Any], tts: dict[str, Any]) -> dict[str, str]:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    paths = {
        "human_metrics": FIGURE_DIR / "final_n50_human_metrics.png",
        "tts_metrics": FIGURE_DIR / "final_n50_tts_metrics.png",
        "direct_deltas": FIGURE_DIR / "final_n50_direct_deltas.png",
        "human_intent_heatmap": FIGURE_DIR / "final_n50_human_intent_heatmap.png",
    }
    plot_metric_bars(human, paths["human_metrics"], "Human speech N=50: four-path task scores")
    plot_metric_bars(tts, paths["tts_metrics"], "TTS N=50: four-path task scores")
    plot_direct_deltas(human, tts, paths["direct_deltas"])
    plot_human_intent_heatmap(human, paths["human_intent_heatmap"])
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


def write_report(human: dict[str, Any], tts: dict[str, Any], figures: dict[str, str]) -> None:
    report = f"""# N=50 人声与 TTS 四路径语音理解报告

**作者：** Jiayi Li（李佳宜）、Liu Luofei（刘洛菲）、Zhang Yuchen（张予辰）  
**日期：** 2026-07-08  
**任务：** 摘要、情感、关键词、意图  
**路径：** A Oracle、B Whisper 级联、C Qwen 转写级联、D Qwen Direct

## 1. 数据口径

本报告采用两个各 N=50 的集合：

- **人声 N=50：** 从现有 N=66 人声结果中剔除 16 条真实意图为 `describe` 的非 v5 样本；v5 的 9 条样本全部保留，并使用 `rensheng_results.json` 中另一台电脑的四任务结果作为 v5 最终结果。
- **TTS N=50：** 使用 `TTS_50_results.json` 中的 50 条 Microsoft Edge TTS 样本结果。

人声最终意图分布：`{human['intent_counts']}`。  
TTS 最终意图分布：`{tts['intent_counts']}`。

## 2. 人声 N=50 结果

{table_for_summary(human)}

![人声 N=50 四路径任务得分]({figures['human_metrics']})

人声中，A/B/C 仍然整体强于 D。最重要的是：D 并不是全面失败，它在部分情感场景有信号；但作为通用四任务方案，显式转写后交给 DeepSeek 的级联路径更可靠。

{direct_delta_table(human)}

![Direct 相对级联路径的差值]({figures['direct_deltas']})

### 2.1 人声路径优劣

- **A Oracle** 是文本上限。它通常最高，说明任务本身更偏文本语义理解，而不是单纯声学识别。
- **B Whisper 级联** 是最稳的实用路径。它在摘要和意图上非常强，且不依赖 Qwen2-Audio 的直接任务遵循能力。
- **C Qwen 转写级联** 证明问题不只是 Whisper/Qwen 的 ASR 差异。即使用 Qwen 做转写，再让 DeepSeek 做文本任务，结果仍然明显强于 Qwen Direct。
- **D Direct** 的优势集中在少数情感判断场景，尤其是 v5 中的脱口秀/娱乐类语音；但在关键词和意图上明显吃亏。

### 2.2 人声中特定 intent 的 Direct 信号

`entertain` 是 Direct 最值得继续追的场景：

{intent_focus_table(human, 'entertain')}

![人声按真实 intent 的 Direct 信号]({figures['human_intent_heatmap']})

在人声 `entertain` 子集上，D 的情感准确率高于 B/C。这支持一个更积极的判断：**当语气、笑点、讽刺和表演性比纯文本更重要时，端到端音频模型可能捕捉到级联转写丢失的线索。**

但是，D 在同一 `entertain` 子集的摘要、关键词和意图上仍然落后。也就是说，Direct 的现有优势更像是“声学情绪线索优势”，不是完整语义理解优势。

## 3. TTS N=50 结果

{table_for_summary(tts)}

![TTS N=50 四路径任务得分]({figures['tts_metrics']})

TTS 的结论更直接：**级联路径明显更强，尤其是意图识别。** B 的意图准确率达到 98%，D 为 80%。在合成、清晰、无环境噪声的语音里，Direct 没有从声学信息中获得额外收益，反而暴露出任务遵循和标签稳定性问题。

{direct_delta_table(tts)}

## 4. 人声 vs TTS 的对照

两组 N=50 指向同一个主结论：**如果目标是摘要、情感、关键词、意图这四个语义任务，级联路线目前优于端到端 Direct。**

但二者也有差异：

- **TTS：** 声音干净、语义清楚，级联优势更干脆。B/C 通过转写把问题还原成文本理解，DeepSeek 很擅长这类任务。
- **人声：** 包含真实语气、表演、停顿、观众感和录音差异。D 在 `entertain` 情感上出现优势，说明 Direct 并非没有价值。

因此，更强的结论不是“Direct 没用”，而是：

> **Direct 不适合作为当前四个语义任务的默认主路径；但它值得作为情绪、语气、讽刺、表演性语音的补充路径继续研究。**

## 5. 四条路径的最终判断

| 路径 | 优势 | 劣势 | 建议定位 |
|---|---|---|---|
| A Oracle | 文本上限，帮助判断转写之外的任务难度 | 依赖人工转写，不是自动系统 | 上限参照 |
| B Whisper 级联 | 最稳，摘要和意图强，工程可控 | 可能丢失语气/情绪线索 | 当前主推荐路径 |
| C Qwen 转写级联 | 控制了音频模型变量，表现接近 B | ASR 稳定性略弱于 Whisper | 重要消融路径 |
| D Direct | 在娱乐类人声情感上有局部优势，流程短，少一次显式转写 | 关键词和意图弱，TTS 中无整体优势 | 情绪/语气任务的补充路径 |

## 6. 结论

本轮 N=50 人声和 N=50 TTS 的综合结果支持一个比较明确的判断：

**级联路径仍然是当前更可靠的语音理解架构。**

尤其在 TTS N=50 中，B/C 明显压过 D；在人声 N=50 中，B/C 也总体领先。Direct 的亮点集中在真实人声的 `entertain` 情感判断上，这说明音频端到端路线确实可能利用语气和表演性信息，但这种优势还没有扩展到摘要、关键词和意图。

下一步最值得做的不是继续泛泛扩大样本，而是定向扩大：

1. `entertain`、讽刺、黑色幽默、情绪反转类人声；
2. 同文本不同语气的成对样本；
3. 情感强度、讽刺识别、语气判断等更依赖声学线索的新任务。

如果这些任务中 Direct 持续领先，才能更有力地证明端到端音频理解的独特价值。

## 7. 相关输出文件

- 人声 N=50 汇总：`data/results/human_speech_final_n50/summary.json`
- 人声 N=50 明细：`data/results/human_speech_final_n50/scores.csv`
- 人声剔除清单：`data/results/human_speech_final_n50/selection.json`
- TTS N=50 汇总：`data/results/tts_speech_final_n50/summary.json`
- TTS N=50 明细：`data/results/tts_speech_final_n50/scores.csv`
- 本报告：`report/final_n50_report.zh-CN.md`
"""
    REPORT_OUT.write_text(report, encoding="utf-8")

    english = f"""# N=50 Human Speech and TTS Four-Path Speech Understanding Report

**Authors:** Jiayi Li（李佳宜）, Liu Luofei（刘洛菲）, Zhang Yuchen（张予辰）  
**Date:** 2026-07-08  
**Tasks:** summarization, sentiment, keywords, intent  
**Paths:** A Oracle, B Whisper Cascade, C Qwen Transcript Cascade, D Qwen Direct

## 1. Dataset protocol

This report uses two matched N=50 evaluation sets.

- **Human speech N=50:** start from the existing N=66 human set, remove 16 non-v5 samples whose ground-truth intent is `describe`, keep all v5 samples, and replace v5 with the external `rensheng_results.json` results.
- **TTS N=50:** use the 50 Microsoft Edge TTS samples in `TTS_50_results.json`.

Human intent distribution: `{human['intent_counts']}`.  
TTS intent distribution: `{tts['intent_counts']}`.

## 2. Human speech N=50

{table_for_summary(human)}

![Human N=50 four-path scores]({figures['human_metrics']})

In human speech, A/B/C remain stronger than D overall. Direct is not useless: its clearest signal appears in sentiment analysis for entertainment-style speech. But as a general four-task architecture, explicit transcription followed by DeepSeek remains more reliable.

{direct_delta_table(human)}

![Direct deltas versus cascade baselines]({figures['direct_deltas']})

### Direct by intent

The strongest Direct signal is in `entertain` sentiment:

{intent_focus_table(human, 'entertain')}

![Human intent heatmap]({figures['human_intent_heatmap']})

This supports a positive but narrow interpretation: when tone, irony, delivery, and performance matter, an end-to-end audio model can preserve cues that a transcript may flatten. However, Direct still trails on summarization, keywords, and intent in the same subset.

## 3. TTS N=50

{table_for_summary(tts)}

![TTS N=50 four-path scores]({figures['tts_metrics']})

The TTS result is more straightforward: cascade is stronger, especially for intent. B reaches 98% intent accuracy, while D reaches 80%. In clean synthetic speech, Direct does not gain much from acoustic information and still shows weaker task-following for labels.

{direct_delta_table(tts)}

## 4. Final interpretation

The combined N=50 human and N=50 TTS evidence supports a clear practical conclusion:

**Cascade remains the better default architecture for the current semantic tasks.**

Direct should not be discarded. Its entertainment-sentiment signal suggests that audio-native models may be valuable for tasks where acoustic cues are central: irony, sarcasm, affect intensity, delivery style, laughter, hesitation, or emotional reversal. But for summarization, keywords, and intent, transcription-based pipelines are currently stronger.

## 5. Files

- Human N=50 summary: `data/results/human_speech_final_n50/summary.json`
- Human N=50 scores: `data/results/human_speech_final_n50/scores.csv`
- Human N=50 selection: `data/results/human_speech_final_n50/selection.json`
- TTS N=50 summary: `data/results/tts_speech_final_n50/summary.json`
- TTS N=50 scores: `data/results/tts_speech_final_n50/scores.csv`
- Chinese report: `report/final_n50_report.zh-CN.md`
- English report: `report/final_n50_report.md`
"""
    REPORT_EN_OUT.write_text(english, encoding="utf-8")


def main() -> None:
    human = build_human_final()
    tts = build_tts_final()
    figures = generate_figures(human, tts)
    write_report(human, tts, figures)
    print(json.dumps({"human": human["means"], "tts": tts["means"]}, indent=2, ensure_ascii=False))
    print(f"Human summary: {HUMAN_OUT / 'summary.json'}")
    print(f"TTS summary: {TTS_OUT / 'summary.json'}")
    print(f"Report: {REPORT_OUT}")


if __name__ == "__main__":
    main()

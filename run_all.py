#!/usr/bin/env python3
"""One-click reproduction: Cascade vs Direct benchmark with ground truth metrics.

Reproduces the full paper: data prep -> cascade -> direct -> evaluate -> charts.
Estimated runtime: ~1 hour (Cascade fast, Direct: Qwen2-Audio INT4 on RTX 5070).
"""
import json, sys, time, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import PROCESSED_DIR, RESULTS_DIR, TASKS, NOISE_CONDITIONS
from src.data import prepare_clip_metadata, split_dataset, inject_noise, load_audio, save_audio
from src.cascade import CascadePipeline
from src.direct_qwen import QwenAudioPipeline
from src.evaluation import parse_sentiment_json, parse_keywords_json, parse_intent_json, paired_ttest, bootstrap_ci
from src.visualization import plot_radar_chart, plot_degradation_curves, plot_error_propagation, plot_confusion_matrices
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT = Path(__file__).parent / "report" / "figures"
OUTPUT.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# 1. Load data + ground truth
# ============================================================
print("=" * 60)
print("STEP 1: Load TTS dataset and ground truth")
print("=" * 60)

with open(PROCESSED_DIR / "tts_samples" / "index.json") as f:
    all_entries = json.load(f)
with open(Path(__file__).parent / "data" / "ground_truth.json") as f:
    ground_truth = json.load(f)

clips = [prepare_clip_metadata(e) for e in all_entries]
splits = split_dataset(clips, dev=0, test=len(clips))  # all = test
test_clips = splits["test"]

print(f"  {len(test_clips)} TTS speech samples")
for c in test_clips:
    name = Path(c['audio_path']).stem
    gt = ground_truth.get(name, {})
    print(f"    {name:25s} {c['duration']:5.1f}s  sentiment={gt.get('sentiment','?'):8s}  intent={gt.get('intent','?')}")

with open(PROCESSED_DIR / "test_clean.json", "w") as f:
    json.dump(test_clips, f, indent=2)

# ============================================================
# 2. Cascade pipeline
# ============================================================
print(f"\n{'='*60}")
print("STEP 2: Cascade Pipeline (faster-whisper + DeepSeek)")
print("=" * 60)

cascade = CascadePipeline()
cascade_results = []
for clip in test_clips:
    cr = {}
    name = Path(clip["audio_path"]).stem
    for task in TASKS:
        print(f"  [{task:12s}] {name}...", end=" ", flush=True)
        result = cascade.run(clip["audio_path"], task)
        cr[task] = result
        print(f"({result['latency_seconds']:.1f}s)")
    cr["clip_id"] = clip["audio_path"]
    cr["sample"] = name
    cascade_results.append(cr)

with open(RESULTS_DIR / "cascade_clean.json", "w") as f:
    json.dump(cascade_results, f, indent=2)
print(f"  Cascade complete: {len(cascade_results)} samples x {len(TASKS)} tasks")

# ============================================================
# 3. Direct pipeline
# ============================================================
print(f"\n{'='*60}")
print("STEP 3: Direct Pipeline (Qwen2-Audio-7B, local INT4)")
print("=" * 60)

qwen = QwenAudioPipeline()
direct_results = []
for clip in test_clips:
    dr = {}
    name = Path(clip["audio_path"]).stem
    for task in TASKS:
        print(f"  [{task:12s}] {name}...", end=" ", flush=True)
        result = qwen.run(clip["audio_path"], task)
        dr[task] = result
        print(f"({result['latency_seconds']:.1f}s)")
    dr["clip_id"] = clip["audio_path"]
    dr["sample"] = name
    direct_results.append(dr)

with open(RESULTS_DIR / "direct_clean.json", "w") as f:
    json.dump(direct_results, f, indent=2)
print(f"  Direct complete: {len(direct_results)} samples x {len(TASKS)} tasks")

# ============================================================
# 4. Noise robustness experiment (3 samples x white noise 10dB & 0dB)
# ============================================================
print(f"\n{'='*60}")
print("STEP 4: Noise Robustness (white noise 10dB & 0dB)")
print("=" * 60)

noise_test = test_clips[:4]  # first 4 samples for noise experiment
noise_conditions = {"clean": {}, "white_10db": {"noise_type": "white", "snr_db": 10, "seed": 42},
                     "white_0db": {"noise_type": "white", "snr_db": 0, "seed": 42}}
noise_results = {"cascade": {}, "direct": {}}

for cond_name, noise_kwargs in noise_conditions.items():
    print(f"\n  [{cond_name}]")
    # Create temp noisy audio files
    noisy_paths = {}
    for clip in noise_test:
        name = Path(clip["audio_path"]).stem
        if noise_kwargs:
            audio, sr = load_audio(clip["audio_path"])
            noisy = inject_noise(audio, sr, **noise_kwargs)
            tmp = Path(clip["audio_path"]).parent / f"_noise_{cond_name}_{name}.wav"
            save_audio(noisy, str(tmp), sr)
            noisy_paths[name] = str(tmp)
        else:
            noisy_paths[name] = clip["audio_path"]

    # Cascade on noise
    c_noise = []
    for clip in noise_test:
        name = Path(clip["audio_path"]).stem
        print(f"    Cascade summarization [{name}]...", end=" ", flush=True)
        result = cascade.run(noisy_paths[name], "summarization")
        c_noise.append({"sample": name, "output": result["output"], "latency": result["latency_seconds"]})
        print(f"({result['latency_seconds']:.1f}s)")
    noise_results["cascade"][cond_name] = c_noise

    # Direct on noise
    d_noise = []
    for clip in noise_test:
        name = Path(clip["audio_path"]).stem
        print(f"    Direct summarization [{name}]...", end=" ", flush=True)
        result = qwen.run(noisy_paths[name], "summarization")
        d_noise.append({"sample": name, "output": result["output"], "latency": result["latency_seconds"]})
        print(f"({result['latency_seconds']:.1f}s)")
    noise_results["direct"][cond_name] = d_noise

    # Cleanup temp files
    if noise_kwargs:
        for name, path in noisy_paths.items():
            if "_noise_" in path:
                Path(path).unlink(missing_ok=True)

with open(RESULTS_DIR / "noise_robustness.json", "w") as f:
    json.dump(noise_results, f, indent=2)
print("  Noise experiment complete.")

# ============================================================
# 5. Evaluate with GROUND TRUTH metrics
# ============================================================
print(f"\n{'='*60}")
print("STEP 5: Evaluation (Ground Truth Metrics)")
print("=" * 60)

# ---- 5a. Sentiment Accuracy ----
print("\n  --- Sentiment ---")
c_sent_ok = 0; d_sent_ok = 0; total = 0
for cr, dr in zip(cascade_results, direct_results):
    name = cr["sample"]
    gt_sent = ground_truth.get(name, {}).get("sentiment", "unknown")
    if gt_sent == "unknown": continue
    total += 1

    c_out = cr.get("sentiment", {}).get("output", "")
    c_parsed = parse_sentiment_json(c_out)
    if c_parsed["sentiment"] == gt_sent: c_sent_ok += 1
    print(f"    [{name}] GT={gt_sent:8s}  Cascade={c_parsed['sentiment']:8s}  Direct={d_out[:30] if (d_out := dr.get('sentiment',{}).get('output','')) else '?'}")

    d_out = dr.get("sentiment", {}).get("output", "")
    d_parsed = parse_sentiment_json(d_out)
    if d_parsed["sentiment"] == gt_sent: d_sent_ok += 1

c_sent_acc = c_sent_ok / total if total else 0
d_sent_acc = d_sent_ok / total if total else 0
print(f"    Cascade Accuracy: {c_sent_acc:.1%} ({c_sent_ok}/{total})")
print(f"    Direct  Accuracy: {d_sent_acc:.1%} ({d_sent_ok}/{total})")

# ---- 5b. Intent Accuracy ----
print("\n  --- Intent ---")
c_int_ok = 0; d_int_ok = 0; total_i = 0
for cr, dr in zip(cascade_results, direct_results):
    name = cr["sample"]
    gt_int = ground_truth.get(name, {}).get("intent", "unknown")
    if gt_int == "unknown": continue
    total_i += 1
    c_out = cr.get("intent", {}).get("output", "")
    c_parsed = parse_intent_json(c_out)
    if c_parsed["intent"] == gt_int: c_int_ok += 1
    d_out = dr.get("intent", {}).get("output", "")
    d_parsed = parse_intent_json(d_out)
    if d_parsed["intent"] == gt_int: d_int_ok += 1
    print(f"    [{name}] GT={gt_int:10s}  Cascade={c_parsed['intent']:10s}  Direct={d_parsed['intent']:10s}")

c_int_acc = c_int_ok / total_i if total_i else 0
d_int_acc = d_int_ok / total_i if total_i else 0
print(f"    Cascade Accuracy: {c_int_acc:.1%} ({c_int_ok}/{total_i})")
print(f"    Direct  Accuracy: {d_int_acc:.1%} ({d_int_ok}/{total_i})")

# ---- 5c. Keyword F1 ----
print("\n  --- Keywords ---")
from src.evaluation import compute_keyword_f1
c_kw_f1s = []; d_kw_f1s = []
for cr, dr in zip(cascade_results, direct_results):
    name = cr["sample"]
    gt_kw = ground_truth.get(name, {}).get("keywords", [])
    if not gt_kw: continue
    c_out = cr.get("keywords", {}).get("output", "")
    c_kw = parse_keywords_json(c_out)
    c_f1, _, _ = compute_keyword_f1(gt_kw, c_kw)
    c_kw_f1s.append(c_f1)
    d_out = dr.get("keywords", {}).get("output", "")
    d_kw = parse_keywords_json(d_out)
    d_f1, _, _ = compute_keyword_f1(gt_kw, d_kw)
    d_kw_f1s.append(d_f1)
    print(f"    [{name}] GT={len(gt_kw)}kw  Cascade F1={c_f1:.2f}  Direct F1={d_f1:.2f}")

c_kw_f1 = np.mean(c_kw_f1s) if c_kw_f1s else 0
d_kw_f1 = np.mean(d_kw_f1s) if d_kw_f1s else 0
print(f"    Cascade Avg F1: {c_kw_f1:.2f}")
print(f"    Direct  Avg F1: {d_kw_f1:.2f}")

# ---- 5d. Summarization ROUGE-L (approximate) ----
print("\n  --- Summarization (ROUGE-L) ---")
def rouge_l_approx(reference, hypothesis):
    """Simple ROUGE-L approximation: longest common subsequence ratio."""
    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()
    # LCS via DP
    m, n = len(ref_words), len(hyp_words)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(1, m+1):
        for j in range(1, n+1):
            dp[i][j] = dp[i-1][j-1] + 1 if ref_words[i-1] == hyp_words[j-1] else max(dp[i-1][j], dp[i][j-1])
    lcs = dp[m][n]
    if m == 0 or n == 0: return 0.0
    precision = lcs / n if n > 0 else 0
    recall = lcs / m if m > 0 else 0
    if precision + recall == 0: return 0.0
    return 2 * precision * recall / (precision + recall)

c_rouge = []; d_rouge = []
for cr, dr in zip(cascade_results, direct_results):
    name = cr["sample"]
    gt_sum = ground_truth.get(name, {}).get("summary", "")
    if not gt_sum: continue
    c_out = cr.get("summarization", {}).get("output", "")
    d_out = dr.get("summarization", {}).get("output", "")
    c_r = rouge_l_approx(gt_sum, c_out)
    d_r = rouge_l_approx(gt_sum, d_out)
    c_rouge.append(c_r); d_rouge.append(d_r)
    print(f"    [{name}] Cascade ROUGE-L={c_r:.3f}  Direct ROUGE-L={d_r:.3f}")

c_rouge_avg = np.mean(c_rouge) if c_rouge else 0
d_rouge_avg = np.mean(d_rouge) if d_rouge else 0
print(f"    Cascade Avg ROUGE-L: {c_rouge_avg:.3f}")
print(f"    Direct  Avg ROUGE-L: {d_rouge_avg:.3f}")

# ---- 5e. Noise robustness scores ----
print("\n  --- Noise Robustness ---")
robustness_c = {}
robustness_d = {}
for cond_name in noise_conditions:
    c_entries = noise_results["cascade"].get(cond_name, [])
    d_entries = noise_results["direct"].get(cond_name, [])
    c_rouges = []; d_rouges = []
    for ce, de in zip(c_entries, d_entries):
        name = ce["sample"]
        gt_sum = ground_truth.get(name, {}).get("summary", "")
        if gt_sum:
            c_rouges.append(rouge_l_approx(gt_sum, ce["output"]))
            d_rouges.append(rouge_l_approx(gt_sum, de["output"]))
    robustness_c[cond_name] = np.mean(c_rouges) if c_rouges else 0
    robustness_d[cond_name] = np.mean(d_rouges) if d_rouges else 0
    print(f"    [{cond_name:12s}] Cascade ROUGE-L={robustness_c[cond_name]:.3f}  Direct ROUGE-L={robustness_d[cond_name]:.3f}")

# ---- 5f. Statistical tests ----
print("\n  --- Statistics ---")
all_c_lat = [r[t]["latency_seconds"] for r in cascade_results for t in TASKS]
all_d_lat = [r[t]["latency_seconds"] for r in direct_results for t in TASKS]
n = min(len(all_c_lat), len(all_d_lat))
ttest = paired_ttest(all_c_lat[:n], all_d_lat[:n])
print(f"  Latency: Cascade {np.mean(all_c_lat):.1f}s  Direct {np.mean(all_d_lat):.1f}s  ({np.mean(all_d_lat)/np.mean(all_c_lat):.1f}x)")
print(f"  t-test: t={ttest['statistic']:.2f}, p={ttest['p_value']:.4f}, d={ttest['cohens_d']:.2f}")

# ============================================================
# 6. Generate Charts
# ============================================================
print(f"\n{'='*60}")
print("STEP 6: Generate Charts")
print("=" * 60)

# Radar chart
fig = plot_radar_chart(
    ["Summarization", "Sentiment", "Keywords", "Intent"],
    [c_rouge_avg, c_sent_acc, c_kw_f1, c_int_acc],
    [d_rouge_avg, d_sent_acc, d_kw_f1, d_int_acc],
    title="Cascade vs Direct: Ground Truth Evaluation"
)
fig.savefig(OUTPUT / "radar_chart.png", dpi=150, bbox_inches="tight")
fig.savefig(OUTPUT / "radar_chart.pdf", bbox_inches="tight")
print("  [OK] radar_chart")

# Latency bar
fig, ax = plt.subplots(figsize=(8,5))
methods = ["Cascade\n(Whisper + DeepSeek)", "Direct\n(Qwen2-Audio local)"]
ax.bar(methods, [np.mean(all_c_lat), np.mean(all_d_lat)], color=["#2563EB","#DC2626"], alpha=0.85, edgecolor="white")
for i,v in enumerate([np.mean(all_c_lat), np.mean(all_d_lat)]):
    ax.text(i, v+5, f"{v:.1f}s", ha="center", fontsize=14, fontweight="bold")
ax.set_ylabel("Latency (seconds)", fontsize=12)
ax.set_title(f"Latency: Cascade {np.mean(all_d_lat)/np.mean(all_c_lat):.1f}x faster", fontsize=14, fontweight="bold")
plt.tight_layout()
fig.savefig(OUTPUT / "latency_comparison.png", dpi=150, bbox_inches="tight")
fig.savefig(OUTPUT / "latency_comparison.pdf", bbox_inches="tight")
print("  [OK] latency_comparison")

# Cost bar
fig, ax = plt.subplots(figsize=(8,5))
ax.bar(methods, [0.0005, 0], color=["#2563EB","#DC2626"], alpha=0.85, edgecolor="white")
ax.text(0, 0.00055, "$0.0005/task", ha="center", fontsize=14, fontweight="bold")
ax.text(1, 0.00005, "FREE", ha="center", fontsize=14, fontweight="bold")
ax.set_ylabel("Cost (USD)", fontsize=12)
ax.set_title("API Cost Comparison", fontsize=14, fontweight="bold")
plt.tight_layout()
fig.savefig(OUTPUT / "cost_comparison.png", dpi=150, bbox_inches="tight")
fig.savefig(OUTPUT / "cost_comparison.pdf", bbox_inches="tight")
print("  [OK] cost_comparison")

# Robustness degradation curves
if len(noise_conditions) >= 2:
    conds = list(noise_conditions.keys())
    fig = plot_degradation_curves(
        "ROUGE-L", conds,
        [robustness_c[c] for c in conds],
        [robustness_d[c] for c in conds],
        title="Robustness: Summarization Quality vs White Noise"
    )
    fig.savefig(OUTPUT / "degradation_curves.png", dpi=150, bbox_inches="tight")
    fig.savefig(OUTPUT / "degradation_curves.pdf", bbox_inches="tight")
    print("  [OK] degradation_curves")

# ============================================================
# 7. Summary
# ============================================================
summary = {
    "dataset": "TTS (Microsoft Edge, 8 English voices)",
    "samples": len(test_clips),
    "ground_truth": "Manually annotated by author",
    "cascade_latency": float(np.mean(all_c_lat)),
    "direct_latency": float(np.mean(all_d_lat)),
    "latency_ratio": float(np.mean(all_d_lat) / np.mean(all_c_lat)),
    "t_statistic": float(ttest["statistic"]),
    "p_value": float(ttest["p_value"]),
    "cohens_d": float(ttest["cohens_d"]),
    "cascade": {
        "sentiment_accuracy": round(c_sent_acc, 3),
        "intent_accuracy": round(c_int_acc, 3),
        "keyword_f1": round(c_kw_f1, 3),
        "summarization_rouge_l": round(c_rouge_avg, 3),
    },
    "direct": {
        "sentiment_accuracy": round(d_sent_acc, 3),
        "intent_accuracy": round(d_int_acc, 3),
        "keyword_f1": round(d_kw_f1, 3),
        "summarization_rouge_l": round(d_rouge_avg, 3),
    },
    "noise_robustness": {
        "cascade": {c: round(v, 3) for c, v in robustness_c.items()},
        "direct": {c: round(v, 3) for c, v in robustness_d.items()},
    },
}
with open(RESULTS_DIR / "final_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"\n{'='*60}")
print("ALL DONE — Final Results")
print("=" * 60)
print(json.dumps(summary, indent=2))
print(f"\nCharts: {OUTPUT}/")
print("Report: report/report.md")
print("README: README.md")

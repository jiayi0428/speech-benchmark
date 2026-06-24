# %% [markdown]
# # Evaluation
#
# Compute metrics and statistical tests on experiment results.

# %% Setup
import sys
sys.path.insert(0, "..")
import json
import numpy as np
from pathlib import Path
from src.config import RESULTS_DIR, TASKS
from src.evaluation import (
    compute_wer,
    parse_sentiment_json,
    parse_keywords_json,
    parse_intent_json,
    compute_keyword_f1,
    compute_sentiment_accuracy,
    paired_ttest,
    bootstrap_ci,
)

# %% [markdown]
# ## Load Results

# %%
def load_results(name):
    path = RESULTS_DIR / f"{name}.json"
    if not path.exists():
        print(f"WARNING: {path} not found. Run the experiment notebook first.")
        return []
    with open(path) as f:
        return json.load(f)

cascade_clean = load_results("cascade_clean")
direct_clean = load_results("direct_clean")
cascade_rob = load_results("cascade_robustness")
direct_rob = load_results("direct_robustness")

print(f"Loaded: {len(cascade_clean)} cascade clean, {len(direct_clean)} direct clean")
print(f"Loaded: {len(cascade_rob)} cascade robust, {len(direct_rob)} direct robust")

# %% [markdown]
# ## Compute WER (Cascade Only)

# %%
def compute_all_wer(cascade_results):
    """Extract WER from cascade results (requires ground truth transcript)."""
    wer_values = []
    for result in cascade_results:
        for task_name, task_result in result.items():
            if isinstance(task_result, dict) and "transcript" in task_result:
                # WER needs reference - use empty string if not available
                ref = ""  # would be clip["transcript"] in real pipeline
                hyp = task_result["transcript"]
                wer = compute_wer(ref, hyp)
                wer_values.append(wer)
    if wer_values:
        print(f"Mean WER: {np.mean(wer_values):.3f}")
        print(f"Median WER: {np.median(wer_values):.3f}")
    return wer_values

wer_values = compute_all_wer(cascade_clean)

# %% [markdown]
# ## Parse and Score Sentiment

# %%
def evaluate_sentiment(cascade_results, direct_results):
    """Parse sentiment outputs and compute accuracy."""
    cascade_preds, direct_preds = [], []

    for results in cascade_results:
        for task_name, task_result in results.items():
            if isinstance(task_result, dict) and task_result.get("task") == "sentiment":
                parsed = parse_sentiment_json(task_result.get("output", ""))
                cascade_preds.append(parsed["sentiment"])

    for results in direct_results:
        for task_name, task_result in results.items():
            if isinstance(task_result, dict) and task_result.get("task") == "sentiment":
                parsed = parse_sentiment_json(task_result.get("output", ""))
                direct_preds.append(parsed["sentiment"])

    print(f"Cascade sentiment: {len(cascade_preds)} predictions")
    print(f"Direct sentiment: {len(direct_preds)} predictions")

    # Count sentiment distribution
    from collections import Counter
    print(f"Cascade distribution: {Counter(cascade_preds)}")
    print(f"Direct distribution: {Counter(direct_preds)}")

    return cascade_preds, direct_preds

cascade_sent, direct_sent = evaluate_sentiment(cascade_clean, direct_clean)

# %% [markdown]
# ## Parse and Score Keywords

# %%
def evaluate_keywords(cascade_results, direct_results):
    """Parse keyword outputs."""
    cascade_kw, direct_kw = [], []

    for results in cascade_results:
        for task_name, task_result in results.items():
            if isinstance(task_result, dict) and task_result.get("task") == "keywords":
                parsed = parse_keywords_json(task_result.get("output", ""))
                cascade_kw.append(parsed)

    for results in direct_results:
        for task_name, task_result in results.items():
            if isinstance(task_result, dict) and task_result.get("task") == "keywords":
                parsed = parse_keywords_json(task_result.get("output", ""))
                direct_kw.append(parsed)

    print(f"Cascade avg keywords per clip: {np.mean([len(k) for k in cascade_kw]):.1f}")
    print(f"Direct avg keywords per clip: {np.mean([len(k) for k in direct_kw]):.1f}")

    return cascade_kw, direct_kw

cascade_kw, direct_kw = evaluate_keywords(cascade_clean, direct_clean)

# %% [markdown]
# ## Robustness Analysis

# %%
def compute_degradation_curves(cascade_rob, direct_rob):
    """Group results by noise condition and compute per-condition scores."""
    conditions = ["clean", "babble_20db", "babble_10db", "babble_0db",
                  "white_20db", "white_10db", "white_0db",
                  "reverb_0.5s", "reverb_1.0s", "reverb_1.5s"]

    cascade_by_cond = {c: [] for c in conditions}
    direct_by_cond = {c: [] for c in conditions}

    for result in cascade_rob:
        cond = result.get("noise_condition", "clean")
        if cond in cascade_by_cond:
            cascade_by_cond[cond].append(result)

    for result in direct_rob:
        cond = result.get("noise_condition", "clean")
        if cond in direct_by_cond:
            direct_by_cond[cond].append(result)

    for cond in conditions:
        c_count = len(cascade_by_cond[cond])
        d_count = len(direct_by_cond[cond])
        if c_count > 0 or d_count > 0:
            print(f"  {cond}: {c_count} cascade, {d_count} direct")

    return cascade_by_cond, direct_by_cond

cascade_cond, direct_cond = compute_degradation_curves(cascade_rob, direct_rob)

# %% [markdown]
# ## Statistical Tests

# %%
# Example: compare latency between cascade and direct
cascade_latencies = []
for results in cascade_clean:
    for task_name, task_result in results.items():
        if isinstance(task_result, dict):
            cascade_latencies.append(task_result.get("latency_seconds", 0))

direct_latencies = []
for results in direct_clean:
    for task_name, task_result in results.items():
        if isinstance(task_result, dict):
            direct_latencies.append(task_result.get("latency_seconds", 0))

if len(cascade_latencies) >= 5 and len(direct_latencies) >= 5:
    # Trim to same length
    n = min(len(cascade_latencies), len(direct_latencies))
    test_result = paired_ttest(cascade_latencies[:n], direct_latencies[:n])
    print(f"Latency comparison (paired t-test):")
    print(f"  Cascade mean: {np.mean(cascade_latencies[:n]):.3f}s")
    print(f"  Direct mean: {np.mean(direct_latencies[:n]):.3f}s")
    print(f"  t = {test_result['statistic']:.3f}, p = {test_result['p_value']:.4f}")
    print(f"  Cohen's d = {test_result['cohens_d']:.3f}")

    diffs = [a - b for a, b in zip(cascade_latencies[:n], direct_latencies[:n])]
    ci_low, ci_high = bootstrap_ci(diffs)
    print(f"  95% CI for difference: [{ci_low:.3f}, {ci_high:.3f}]")

print("\nEvaluation complete!")

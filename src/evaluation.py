"""Evaluation metrics and statistical tests for the speech benchmark."""
import json
import re
import numpy as np
from typing import Dict, List, Tuple, Any
from scipy import stats as scipy_stats


# --- WER ---

def compute_wer(reference: str, hypothesis: str) -> float:
    """Compute Word Error Rate between reference and hypothesis.

    Uses Levenshtein edit distance at word level.

    Args:
        reference: Ground truth transcript.
        hypothesis: ASR output transcript.

    Returns:
        WER as a float (0.0 = perfect, 1.0 = completely wrong).
    """
    ref_words = reference.strip().split()
    hyp_words = hypothesis.strip().split()

    if len(ref_words) == 0:
        return 1.0 if len(hyp_words) > 0 else 0.0

    d = np.zeros((len(ref_words) + 1, len(hyp_words) + 1), dtype=np.int32)
    for i in range(len(ref_words) + 1):
        d[i, 0] = i
    for j in range(len(hyp_words) + 1):
        d[0, j] = j

    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            cost = 0 if ref_words[i - 1].lower() == hyp_words[j - 1].lower() else 1
            d[i, j] = min(
                d[i - 1, j] + 1,
                d[i, j - 1] + 1,
                d[i - 1, j - 1] + cost,
            )

    return float(d[len(ref_words), len(hyp_words)] / len(ref_words))


# --- JSON Parsing ---

def _extract_json(text: str) -> str:
    """Extract JSON from LLM output that may have markdown or extra text."""
    match = re.search(r'```(?:json)?\s*(.*?)\s*```', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    brace_start = text.find('{')
    brace_end = text.rfind('}')
    if brace_start != -1 and brace_end != -1:
        return text[brace_start:brace_end + 1]
    bracket_start = text.find('[')
    bracket_end = text.rfind(']')
    if bracket_start != -1 and bracket_end != -1:
        return text[bracket_start:bracket_end + 1]
    return text.strip()


def parse_sentiment_json(output: str) -> Dict[str, Any]:
    """Parse LLM sentiment output into structured dict."""
    try:
        data = json.loads(_extract_json(output))
        sentiment = str(data.get("sentiment", "unknown")).lower().strip()
        if sentiment not in ("positive", "negative", "neutral"):
            sentiment = "unknown"
        confidence = float(data.get("confidence", 0.0))
        confidence = max(0.0, min(1.0, confidence))
        return {"sentiment": sentiment, "confidence": confidence}
    except (json.JSONDecodeError, ValueError, TypeError):
        return {"sentiment": "unknown", "confidence": 0.0}


def parse_keywords_json(output: str) -> List[str]:
    """Parse LLM keyword output into a list of strings."""
    try:
        data = json.loads(_extract_json(output))
        if isinstance(data, list):
            return [str(k).strip() for k in data if str(k).strip()]
        return []
    except (json.JSONDecodeError, ValueError, TypeError):
        return []


def parse_intent_json(output: str) -> Dict[str, Any]:
    """Parse LLM intent output into structured dict."""
    valid_intents = {"inform", "persuade", "entertain", "question", "describe"}
    try:
        data = json.loads(_extract_json(output))
        intent = str(data.get("intent", "unknown")).lower().strip()
        if intent not in valid_intents:
            intent = "unknown"
        confidence = float(data.get("confidence", 0.0))
        confidence = max(0.0, min(1.0, confidence))
        return {"intent": intent, "confidence": confidence}
    except (json.JSONDecodeError, ValueError, TypeError):
        return {"intent": "unknown", "confidence": 0.0}


# --- Scoring Functions ---

def compute_keyword_f1(
    reference: List[str],
    predicted: List[str],
) -> Tuple[float, float, float]:
    """Compute precision, recall, and F1 for keyword extraction.

    Matches are case-insensitive.

    Returns:
        (f1, precision, recall)
    """
    ref_set = {k.lower().strip() for k in reference}
    pred_set = {k.lower().strip() for k in predicted}

    if len(pred_set) == 0 and len(ref_set) == 0:
        return 1.0, 1.0, 1.0
    if len(pred_set) == 0 or len(ref_set) == 0:
        return 0.0, 0.0, 0.0

    intersection = ref_set & pred_set
    precision = len(intersection) / len(pred_set)
    recall = len(intersection) / len(ref_set)

    if precision + recall == 0:
        return 0.0, precision, recall

    f1 = 2 * precision * recall / (precision + recall)
    return f1, precision, recall


def compute_sentiment_accuracy(
    y_true: List[str],
    y_pred: List[str],
) -> Tuple[float, float]:
    """Compute accuracy and macro F1 for sentiment classification.

    Returns:
        (accuracy, macro_f1)
    """
    from sklearn.metrics import accuracy_score, f1_score

    accuracy = accuracy_score(y_true, y_pred)
    labels = sorted(set(y_true) | set(y_pred))
    try:
        macro_f1 = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    except ValueError:
        macro_f1 = 0.0
    return float(accuracy), float(macro_f1)


# --- Statistical Tests ---

def paired_ttest(
    scores_a: List[float],
    scores_b: List[float],
) -> Dict[str, float]:
    """Run a paired t-test comparing two sets of scores.

    Args:
        scores_a: Scores from method A (e.g., cascade).
        scores_b: Scores from method B (e.g., direct).

    Returns:
        {"statistic": t, "p_value": p, "mean_diff": mean(a-b), "cohens_d": d}
    """
    a = np.array(scores_a, dtype=np.float64)
    b = np.array(scores_b, dtype=np.float64)
    result = scipy_stats.ttest_rel(a, b)
    diff = a - b
    d = np.mean(diff) / (np.std(diff, ddof=1) + 1e-10)
    return {
        "statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "mean_diff": float(np.mean(diff)),
        "cohens_d": float(d),
    }


def bootstrap_ci(
    differences: List[float],
    n_bootstrap: int = 10000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float]:
    """Compute bootstrap confidence interval for mean difference.

    Args:
        differences: List of paired differences (method_a - method_b).
        n_bootstrap: Number of bootstrap resamples.
        ci_level: Confidence level (default 0.95).
        seed: Random seed.

    Returns:
        (lower_bound, upper_bound)
    """
    rng = np.random.RandomState(seed)
    diffs = np.array(differences, dtype=np.float64)
    means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(diffs, size=len(diffs), replace=True)
        means.append(np.mean(sample))
    alpha = (1.0 - ci_level) / 2.0
    lower = np.percentile(means, 100 * alpha)
    upper = np.percentile(means, 100 * (1 - alpha))
    return float(lower), float(upper)

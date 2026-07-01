"""Post-process Direct (Qwen2-Audio) outputs into structured JSON via DeepSeek.

Problem: Qwen2-Audio understands audio well but doesn't follow "output JSON" instructions.
Solution: Let Qwen2-Audio output free-form text, then use DeepSeek to convert it
          into the required JSON format. Both pipelines then use the same text LLM
          for the final structured output step -- fair comparison.

Usage: python postprocess_direct.py
Input:  data/results/direct_clean.json (from run_all.py)
Output: data/results/direct_clean_structured.json + evaluation metrics
"""
import json, time
from pathlib import Path
from openai import OpenAI
from src.config import TEXT_LLM_API_KEY, TEXT_LLM_BASE_URL, TEXT_LLM_MODEL
from src.evaluation import parse_sentiment_json, parse_keywords_json, parse_intent_json, compute_keyword_f1

client = OpenAI(api_key=TEXT_LLM_API_KEY, base_url=TEXT_LLM_BASE_URL)

with open("data/results/direct_clean.json") as f: direct = json.load(f)
with open("data/ground_truth.json") as f: gt = json.load(f)

STRUCTURE_PROMPTS = {
    "sentiment": (
        'Below is an AI analysis of an audio clip. Based on this analysis, '
        'classify the speaker sentiment as exactly one of: positive, negative, neutral. '
        'Return ONLY JSON: {"sentiment": "<label>", "confidence": <float>}'
    ),
    "keywords": (
        'Below is an AI analysis of an audio clip. Based on this analysis, '
        'extract 5-7 key phrases. '
        'Return ONLY JSON list: ["keyword1", "keyword2", ...]'
    ),
    "intent": (
        'Below is an AI analysis of an audio clip. Based on this analysis, '
        'classify the speaker intent as: inform, persuade, entertain, question, describe. '
        'Return ONLY JSON: {"intent": "<label>", "confidence": <float>}'
    ),
}

print("Structuring Direct outputs via DeepSeek...\n")
for entry in direct:
    name = Path(entry["clip_id"]).stem
    for task in ["sentiment", "keywords", "intent"]:
        qwen_output = entry.get(task, {}).get("output", "")
        if not qwen_output: continue
        prompt = f"{STRUCTURE_PROMPTS[task]}\n\nANALYSIS:\n{qwen_output[:500]}"
        print(f"  [{name}] {task}...", end=" ", flush=True)
        try:
            resp = client.chat.completions.create(
                model=TEXT_LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
            structured = resp.choices[0].message.content
            entry[task]["output"] = structured
            print("OK")
        except Exception as e:
            print(f"ERROR: {e}")
        time.sleep(0.3)

with open("data/results/direct_clean_structured.json", "w") as f:
    json.dump(direct, f, indent=2)

# Evaluate against ground truth
print("\nEvaluating structured Direct outputs...\n")
with open("data/results/cascade_clean.json") as f: cascade_all = json.load(f)
c_by_id = {c["clip_id"]: c for c in cascade_all}
d_by_id = {d["clip_id"]: d for d in direct}
common = sorted(set(c_by_id) & set(d_by_id))
N = len(common)

# Sentiment
c_ok = d_ok = 0
for cid in common:
    name = Path(cid).stem
    gt_s = gt.get(name, {}).get("sentiment", "")
    c_p = parse_sentiment_json(c_by_id[cid].get("sentiment", {}).get("output", ""))
    d_p = parse_sentiment_json(d_by_id[cid].get("sentiment", {}).get("output", ""))
    if c_p["sentiment"] == gt_s: c_ok += 1
    if d_p["sentiment"] == gt_s: d_ok += 1
    print(f"  [{name}] GT={gt_s:9s} C={c_p['sentiment']:9s} D={d_p['sentiment']:9s}")
print(f"  Sentiment: Cascade={c_ok/N:.0%} Direct={d_ok/N:.0%}\n")

# Intent
c_ok = d_ok = 0
for cid in common:
    name = Path(cid).stem
    gt_i = gt.get(name, {}).get("intent", "")
    c_p = parse_intent_json(c_by_id[cid].get("intent", {}).get("output", ""))
    d_p = parse_intent_json(d_by_id[cid].get("intent", {}).get("output", ""))
    if c_p["intent"] == gt_i: c_ok += 1
    if d_p["intent"] == gt_i: d_ok += 1
    print(f"  [{name}] GT={gt_i:10s} C={c_p['intent']:10s} D={d_p['intent']:10s}")
print(f"  Intent: Cascade={c_ok/N:.0%} Direct={d_ok/N:.0%}\n")

# Keywords
c_f1s = []; d_f1s = []
for cid in common:
    name = Path(cid).stem
    gt_k = gt.get(name, {}).get("keywords", [])
    c_k = parse_keywords_json(c_by_id[cid].get("keywords", {}).get("output", ""))
    d_k = parse_keywords_json(d_by_id[cid].get("keywords", {}).get("output", ""))
    cf, _, _ = compute_keyword_f1(gt_k, c_k); df, _, _ = compute_keyword_f1(gt_k, d_k)
    c_f1s.append(cf); d_f1s.append(df)
    print(f"  [{name}] C F1={cf:.2f} D F1={df:.2f}")
print(f"  Keywords: Cascade F1={sum(c_f1s)/len(c_f1s):.2f} Direct F1={sum(d_f1s)/len(d_f1s):.2f}")

print(f"\nDone. N={N}. Results in data/results/direct_clean_structured.json")

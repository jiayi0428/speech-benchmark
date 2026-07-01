"""Blind LLM evaluation: let DeepSeek judge Cascade vs Direct outputs side-by-side.

Outputs are anonymized (System A / System B) and presented in random order
to reduce positional bias. The LLM is shown the ground truth and asked to
rate each system on accuracy, completeness, and conciseness (1-10 scale).

Usage: python blind_judge.py
Input:  data/results/cascade_clean.json, direct_clean.json, ground_truth.json
Output: data/results/blind_judge_results.json
"""
import json, time, random
from pathlib import Path
from openai import OpenAI
from src.config import TEXT_LLM_API_KEY, TEXT_LLM_BASE_URL, TEXT_LLM_MODEL

client = OpenAI(api_key=TEXT_LLM_API_KEY, base_url=TEXT_LLM_BASE_URL)

with open("data/results/cascade_clean.json") as f: cascade = json.load(f)
with open("data/results/direct_clean.json") as f: direct = json.load(f)
with open("data/ground_truth.json") as f: gt = json.load(f)

c_by_id = {c["clip_id"]: c for c in cascade}
d_by_id = {d["clip_id"]: d for d in direct}
common = sorted(set(c_by_id) & set(d_by_id))

JUDGE_PROMPT = """You are an expert evaluator of speech understanding quality.
You will see the GROUND TRUTH and two system outputs (labeled System A and System B).
Rate each on a scale of 1-10 for accuracy, completeness, and conciseness.
Then state which system is better overall and WHY in one sentence.

GROUND TRUTH: {ground_truth}

System A: {output_a}

System B: {output_b}

Respond as JSON:
{{"system_a_score": <1-10>, "system_b_score": <1-10>, "winner": "A"|"B"|"tie", "reason": "<one sentence why>"}}"""

judgments = []

for task in ["summarization", "sentiment", "keywords", "intent"]:
    print(f"\n=== {task} ===")
    for cid in common:
        name = Path(cid).stem
        gt_data = gt.get(name, {})

        if task == "summarization":
            gt_str = gt_data.get("summary", "")
        elif task == "sentiment":
            gt_str = gt_data.get("sentiment", "")
        elif task == "keywords":
            gt_str = ", ".join(gt_data.get("keywords", []))
        else:
            gt_str = gt_data.get("intent", "")

        c_out = c_by_id[cid].get(task, {}).get("output", "")
        d_out = d_by_id[cid].get(task, {}).get("output", "")

        # Randomize A/B assignment to reduce positional bias
        if random.random() < 0.5:
            output_a, output_b = c_out, d_out
            a_is = "Cascade"
        else:
            output_a, output_b = d_out, c_out
            a_is = "Direct"

        prompt = JUDGE_PROMPT.format(ground_truth=gt_str, output_a=output_a[:500], output_b=output_b[:500])

        print(f"  [{name}] ...", end=" ", flush=True)
        try:
            resp = client.chat.completions.create(
                model=TEXT_LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0)
            result_text = resp.choices[0].message.content

            # Parse JSON from response
            import re
            match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if match:
                parsed = json.loads(match.group())
            else:
                parsed = {}

            # Map back to real identities
            if a_is == "Cascade":
                c_score = parsed.get("system_a_score", 5)
                d_score = parsed.get("system_b_score", 5)
                raw_winner = parsed.get("winner", "tie")
                winner = "Cascade" if raw_winner == "A" else ("Direct" if raw_winner == "B" else "tie")
            else:
                d_score = parsed.get("system_a_score", 5)
                c_score = parsed.get("system_b_score", 5)
                raw_winner = parsed.get("winner", "tie")
                winner = "Direct" if raw_winner == "A" else ("Cascade" if raw_winner == "B" else "tie")

            judgment = {
                "sample": name, "task": task, "a_was": a_is,
                "cascade_score": c_score, "direct_score": d_score,
                "winner": winner, "reason": parsed.get("reason", ""),
                "cascade_output_preview": c_out[:80],
                "direct_output_preview": d_out[:80],
            }
            judgments.append(judgment)
            print(f"C={c_score} D={d_score} winner={winner}")

        except Exception as e:
            print(f"ERROR: {e}")
        time.sleep(0.3)

with open("data/results/blind_judge_results.json", "w") as f:
    json.dump(judgments, f, indent=2)

# Summary
wins_c = sum(1 for j in judgments if j["winner"] == "Cascade")
wins_d = sum(1 for j in judgments if j["winner"] == "Direct")
ties = sum(1 for j in judgments if j["winner"] == "tie")
print(f"\n=== Summary ===")
print(f"Cascade wins: {wins_c}, Direct wins: {wins_d}, Ties: {ties}")
for task in ["summarization", "sentiment", "keywords", "intent"]:
    tj = [j for j in judgments if j["task"] == task]
    if tj:
        c_avg = sum(j["cascade_score"] for j in tj) / len(tj)
        d_avg = sum(j["direct_score"] for j in tj) / len(tj)
        d_wins = sum(1 for j in tj if j["winner"] == "Direct")
        print(f"  {task:15s} C={c_avg:.1f} D={d_avg:.1f}  Direct wins: {d_wins}/{len(tj)}")

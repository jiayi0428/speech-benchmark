# %% [markdown]
# # Experiment Runner
#
# Run both pipelines (Cascade and Direct) over test sets.

# %% Setup
import sys
sys.path.insert(0, "..")
import json
from pathlib import Path
from tqdm import tqdm
from src.cascade import CascadePipeline
from src.data import load_audio, inject_noise, save_audio
from src.config import PROCESSED_DIR, RESULTS_DIR, TASKS, SPEECH_LLM_PROVIDER

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Initialize Pipelines

# %%
print("Loading Cascade pipeline...")
cascade = CascadePipeline()
# Auto-select Direct pipeline based on available API keys
if SPEECH_LLM_PROVIDER == "gemini":
    from src.direct_gemini import GeminiDirectPipeline
    DirectClass = GeminiDirectPipeline
elif SPEECH_LLM_PROVIDER == "openai":
    from src.direct import DirectPipeline
    DirectClass = DirectPipeline
else:
    from src.direct_qwen import QwenAudioPipeline
    DirectClass = QwenAudioPipeline

print(f"Loading Direct pipeline ({SPEECH_LLM_PROVIDER}: {DirectClass.__name__})...")
direct = DirectClass()
print("Both pipelines ready.")

# %% [markdown]
# ## Helper Functions

# %%
def run_sample_cascade(audio_path, noise_kwargs=None):
    """Run cascade pipeline on one audio file with optional noise."""
    if noise_kwargs:
        audio, sr = load_audio(audio_path)
        noisy = inject_noise(audio, sr, seed=42, **noise_kwargs)
        tmp = Path(audio_path).parent / f"_tmp_{Path(audio_path).name}"
        save_audio(noisy, str(tmp), sr)
        audio_path = str(tmp)

    results = {}
    for task in TASKS:
        result = cascade.run(audio_path, task)
        results[task] = result
    return results


def run_sample_direct(audio_path, noise_kwargs=None):
    """Run direct pipeline on one audio file with optional noise."""
    if noise_kwargs:
        audio, sr = load_audio(audio_path)
        noisy = inject_noise(audio, sr, seed=42, **noise_kwargs)
        tmp = Path(audio_path).parent / f"_tmp_{Path(audio_path).name}"
        save_audio(noisy, str(tmp), sr)
        audio_path = str(tmp)

    results = {}
    for task in TASKS:
        result = direct.run(audio_path, task)
        results[task] = result
    return results

# %% [markdown]
# ## Run Clean Test Set

# %%
with open(PROCESSED_DIR / "test_clean.json") as f:
    test_clean = json.load(f)

cascade_clean = []
direct_clean = []

for clip in tqdm(test_clean, desc="Clean test"):
    cr = run_sample_cascade(clip["audio_path"])
    cr["clip_id"] = clip["audio_path"]
    cascade_clean.append(cr)

    dr = run_sample_direct(clip["audio_path"])
    dr["clip_id"] = clip["audio_path"]
    direct_clean.append(dr)

with open(RESULTS_DIR / "cascade_clean.json", "w") as f:
    json.dump(cascade_clean, f, indent=2)
with open(RESULTS_DIR / "direct_clean.json", "w") as f:
    json.dump(direct_clean, f, indent=2)

print(f"Clean test complete: {len(cascade_clean)} cascade, {len(direct_clean)} direct")

# %% [markdown]
# ## Run Robustness Test Set

# %%
with open(PROCESSED_DIR / "test_robustness.json") as f:
    test_robustness = json.load(f)

cascade_rob = []
direct_rob = []

for variant in tqdm(test_robustness, desc="Robustness test"):
    noise = variant.get("noise_kwargs", {})
    cond = variant.get("noise_condition", "clean")

    cr = run_sample_cascade(variant["audio_path"], noise)
    cr["clip_id"] = variant["audio_path"]
    cr["noise_condition"] = cond
    cascade_rob.append(cr)

    dr = run_sample_direct(variant["audio_path"], noise)
    dr["clip_id"] = variant["audio_path"]
    dr["noise_condition"] = cond
    direct_rob.append(dr)

with open(RESULTS_DIR / "cascade_robustness.json", "w") as f:
    json.dump(cascade_rob, f, indent=2)
with open(RESULTS_DIR / "direct_robustness.json", "w") as f:
    json.dump(direct_rob, f, indent=2)

print(f"Robustness test complete: {len(cascade_rob)} cascade, {len(direct_rob)} direct")

# %% [markdown]
# ## Summary

# %%
def summarize(results, name):
    n_samples = len(results)
    if n_samples == 0:
        print(f"{name}: No results")
        return
    total_calls = sum(len(r) for r in results if isinstance(r, dict))
    print(f"{name}: {n_samples} samples, ~{total_calls} task calls")
    print(f"  Estimated API cost: ${total_calls * 0.003:.2f}")


summarize(cascade_clean, "Cascade Clean")
summarize(direct_clean, "Direct Clean")
summarize(cascade_rob, "Cascade Robustness")
summarize(direct_rob, "Direct Robustness")
print("\nAll experiments complete!")

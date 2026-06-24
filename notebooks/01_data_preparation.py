# %% [markdown]
# # Data Preparation
#
# This notebook prepares the TED-LIUM dataset for the speech understanding benchmark.

# %% Setup
import sys
sys.path.insert(0, "..")
import json
import numpy as np
import soundfile as sf
from pathlib import Path
from src.data import prepare_clip_metadata, split_dataset, generate_noise_variants, load_audio
from src.config import NOISE_CONDITIONS, PROCESSED_DIR, RAW_DIR, SAMPLE_RATE

# %% [markdown]
# ## Load or Create Dataset
#
# If TED-LIUM is available at `data/raw/tedlium/`, parse it.
# Otherwise, create a small synthetic dataset for testing the pipeline.

# %%
def load_or_create_dataset():
    """Load TED-LIUM or create synthetic test data."""
    tedlium_dir = RAW_DIR / "tedlium"
    stm_dir = tedlium_dir / "stm"

    if stm_dir.exists():
        print(f"Loading TED-LIUM from {tedlium_dir}...")
        return load_tedlium_metadata(tedlium_dir)
    else:
        print("TED-LIUM not found. Creating synthetic dataset for testing...")
        return create_synthetic_dataset(n_samples=30)


def load_tedlium_metadata(tedlium_dir):
    """Parse TED-LIUM STM files to build a clip index."""
    entries = []
    for stm_path in sorted(tedlium_dir.glob("stm/*.stm")):
        with open(stm_path) as f:
            for line in f:
                parts = line.strip().split(" ", 6)
                if len(parts) < 7:
                    continue
                filename = parts[0]
                speaker = parts[2]
                start = float(parts[3])
                end = float(parts[4])
                text = parts[6]
                audio_path = str(tedlium_dir / "sph" / f"{filename}.sph")
                entries.append({
                    "audio_path": audio_path,
                    "transcript": text,
                    "speaker": speaker,
                    "duration": end - start,
                })
    return entries


def create_synthetic_dataset(n_samples=30):
    """Create a small synthetic dataset with sine tones and placeholder text for testing."""
    synth_dir = PROCESSED_DIR / "synthetic"
    synth_dir.mkdir(parents=True, exist_ok=True)

    topics = [
        "Artificial intelligence is transforming how we work and live.",
        "Climate change requires urgent action from all nations.",
        "The future of education lies in personalized learning.",
        "Space exploration opens new frontiers for humanity.",
        "Healthy eating habits contribute to longer, happier lives.",
    ]
    entries = []
    for i in range(n_samples):
        audio_path = str(synth_dir / f"sample_{i:04d}.wav")
        duration = np.random.uniform(30, 120)
        t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)
        audio = 0.3 * np.sin(2 * np.pi * 220 * t).astype(np.float32)
        sf.write(audio_path, audio, SAMPLE_RATE)

        transcript = " ".join(np.random.choice(topics, size=np.random.randint(5, 15)))
        entries.append({
            "audio_path": audio_path,
            "transcript": transcript,
            "speaker": f"speaker_{i % 5}",
            "duration": duration,
        })
    print(f"Created {n_samples} synthetic audio samples")
    return entries


entries = load_or_create_dataset()
print(f"Total entries: {len(entries)}")

# %% [markdown]
# ## Normalize and Split

# %%
# Normalize metadata
clips = [prepare_clip_metadata(e) for e in entries]

# Filter to reasonable lengths (30s - 5min for testing)
clips = [c for c in clips if 10 <= c["duration"] <= 300]
print(f"Filtered to {len(clips)} clips")

# Split
splits = split_dataset(clips, dev=min(10, len(clips)//3), test=min(30, len(clips)*2//3))
dev_clips = splits["dev"]
test_clean_clips = splits["test"][:20] if len(splits["test"]) > 20 else splits["test"]
robustness_clips = splits["test"][:10] if len(splits["test"]) >= 10 else splits["test"]

# Generate noise variants
robustness_variants = []
for clip in robustness_clips:
    variants = generate_noise_variants(clip, NOISE_CONDITIONS)
    robustness_variants.extend(variants.values())

print(f"Dev: {len(dev_clips)} clips")
print(f"Test-Clean: {len(test_clean_clips)} clips")
print(f"Test-Robustness: {len(robustness_variants)} variants")

# %% [markdown]
# ## Save Splits

# %%
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

with open(PROCESSED_DIR / "dev.json", "w") as f:
    json.dump(dev_clips, f, indent=2)
with open(PROCESSED_DIR / "test_clean.json", "w") as f:
    json.dump(test_clean_clips, f, indent=2)
with open(PROCESSED_DIR / "test_robustness.json", "w") as f:
    json.dump(robustness_variants, f, indent=2)

print("Splits saved to data/processed/")
print("Done!")

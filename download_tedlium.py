#!/usr/bin/env python3
"""Download TED-LIUM v3 via HuggingFace datasets (faster in China via hf-mirror.com).

Usage:
    python download_tedlium.py              # 50 talks (~1GB)
    python download_tedlium.py --all         # Full dataset (slow!)
"""
import argparse
import soundfile as sf
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data" / "raw" / "tedlium"


def download_via_huggingface(n_talks: int = 50) -> None:
    """Download TED-LIUM using HuggingFace datasets and save as WAV+transcript."""
    print(f"Downloading {n_talks} TED talks via HuggingFace datasets...")
    print("(Set HF_ENDPOINT=https://hf-mirror.com for faster download in China)")
    print()

    from datasets import load_dataset

    ds = load_dataset("LIUM/tedlium", "release3", split="train", streaming=True,
                       trust_remote_code=True)

    sph_dir = DATA_DIR / "sph"
    stm_dir = DATA_DIR / "stm"
    sph_dir.mkdir(parents=True, exist_ok=True)
    stm_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for sample in ds:
        if count >= n_talks:
            break

        try:
            audio_data = sample.get("audio")
            if audio_data is None:
                continue

            array = audio_data.get("array")
            sr = audio_data.get("sampling_rate", 16000)

            if array is None:
                continue

            # Convert to float32 mono
            array = np.array(array, dtype=np.float32)
            if array.ndim > 1:
                array = array.mean(axis=0)

            talk_id = sample.get("talk_id", f"ted_{count:04d}")

            # Save audio
            audio_path = sph_dir / f"{talk_id}.wav"
            sf.write(str(audio_path), array, int(sr))

            # Save transcript in STM-like format
            transcript = sample.get("text", "")
            stm_path = stm_dir / f"{talk_id}.stm"
            with open(stm_path, "w", encoding="utf-8") as f:
                # Minimal STM format: filename channel speaker start end <> text
                speaker = sample.get("speaker_id", "unknown")
                start = sample.get("start_time", 0.0)
                end = sample.get("end_time", len(array) / sr)
                f.write(f"{talk_id} 1 {speaker} {start:.2f} {end:.2f} <,> {transcript}\n")

            count += 1
            if count % 10 == 0:
                print(f"  Downloaded {count}/{n_talks} talks...")

        except Exception as e:
            print(f"  Skipping sample: {e}")
            continue

    # Also generate metadata for data prep notebook
    import json
    entries = []
    for wav_path in sorted(sph_dir.glob("*.wav")):
        talk_id = wav_path.stem
        stm_path = stm_dir / f"{talk_id}.stm"
        transcript = ""
        speaker = "unknown"
        duration = 0.0

        if stm_path.exists():
            with open(stm_path) as f:
                line = f.readline().strip()
                parts = line.split(" ", 6)
                if len(parts) >= 7:
                    speaker = parts[2]
                    duration = float(parts[4]) - float(parts[3])
                    transcript = parts[6]

        entries.append({
            "audio_path": str(wav_path),
            "transcript": transcript,
            "speaker": speaker,
            "duration": duration or float(len(sf.read(str(wav_path))[0])) / 16000,
        })

    with open(DATA_DIR / "index.json", "w") as f:
        json.dump(entries, f, indent=2)

    print(f"\nDone! {count} talks saved to {DATA_DIR}")
    print(f"  Audio: {sph_dir} ({len(list(sph_dir.glob('*.wav')))} files)")
    print(f"  Transcripts: {stm_dir} ({len(list(stm_dir.glob('*.stm')))} files)")
    print(f"  Index: {DATA_DIR / 'index.json'}")

    # Print a sample
    if entries:
        print(f"\nSample entry:")
        e = entries[0]
        print(f"  Speaker: {e['speaker']}")
        print(f"  Duration: {e['duration']:.1f}s")
        print(f"  Transcript: {e['transcript'][:100]}...")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="Download full dataset (slow!)")
    args = parser.parse_args()

    n = 10000 if args.all else 50
    download_via_huggingface(n)


if __name__ == "__main__":
    main()

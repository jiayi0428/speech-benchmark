"""Tests for dataset preparation."""
import json
import numpy as np
import soundfile as sf
from pathlib import Path
from src.data import (
    prepare_clip_metadata,
    split_dataset,
    generate_noise_variants,
)

TMP_DIR = Path(__file__).parent / "fixtures"


def _make_fake_tedlium_index(output_dir: Path, n_files: int = 5):
    """Create a minimal fake TED-LIUM index for testing."""
    output_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for i in range(n_files):
        audio_path = output_dir / f"ted_{i}.wav"
        t = np.linspace(0, 2, 32000, endpoint=False)
        audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        sf.write(str(audio_path), audio, 16000)
        entries.append({
            "audio_path": str(audio_path),
            "transcript": f"This is transcript number {i}.",
            "speaker": f"speaker_{i}",
            "duration": 2.0,
        })
    return entries


def test_prepare_clip_metadata_extracts_fields():
    entries = _make_fake_tedlium_index(TMP_DIR / "fake_ted")
    meta = prepare_clip_metadata(entries[0])
    assert "audio_path" in meta
    assert "transcript" in meta
    assert meta["duration"] == 2.0


def test_split_dataset_returns_correct_sizes():
    entries = _make_fake_tedlium_index(TMP_DIR / "split_test")
    splits = split_dataset(entries, dev=2, test=3)
    assert len(splits["dev"]) == 2
    assert len(splits["test"]) == 3
    dev_ids = {s["audio_path"] for s in splits["dev"]}
    test_ids = {s["audio_path"] for s in splits["test"]}
    assert dev_ids.isdisjoint(test_ids)


def test_split_dataset_raises_on_too_many():
    entries = _make_fake_tedlium_index(TMP_DIR / "split_err")
    try:
        split_dataset(entries, dev=10, test=10)
    except ValueError:
        pass
    else:
        raise AssertionError("Should have raised ValueError")


def test_generate_noise_variants_produces_all_conditions():
    entries = _make_fake_tedlium_index(TMP_DIR / "noise_var", n_files=1)
    conditions = {"clean": {}, "white_20db": {"noise_type": "white", "snr_db": 20}}
    variants = generate_noise_variants(entries[0], conditions)
    assert len(variants) == 2
    assert "clean" in variants
    assert "white_20db" in variants

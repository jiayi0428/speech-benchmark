"""Tests for the fixed white-noise experiment audio generation."""
import numpy as np

from prepare_white_noise import add_white_noise


def make_signal(seconds: float = 1.0, sr: int = 16000) -> np.ndarray:
    time = np.arange(int(seconds * sr), dtype=np.float32) / sr
    return (0.5 * np.sin(2 * np.pi * 440 * time)).astype(np.float32)


def test_white_noise_is_deterministic():
    signal = make_signal()
    first, first_meta = add_white_noise(signal, snr_db=10, seed=42)
    second, second_meta = add_white_noise(signal, snr_db=10, seed=42)
    assert np.array_equal(first, second)
    assert first_meta == second_meta


def test_white_noise_matches_requested_snr():
    signal = make_signal()
    _, metadata = add_white_noise(signal, snr_db=0, seed=42)
    assert abs(metadata["measured_snr_db"]) < 1e-5


def test_white_noise_prevents_clipping():
    signal = make_signal() * 1.9
    noisy, metadata = add_white_noise(signal, snr_db=0, seed=42)
    assert np.max(np.abs(noisy)) <= 0.99901
    assert metadata["clipping_ratio"] == 0.0

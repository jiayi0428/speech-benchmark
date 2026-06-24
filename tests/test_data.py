"""Tests for data loading and noise injection."""
import numpy as np
import soundfile as sf
from pathlib import Path
from src.data import load_audio, inject_noise

TMP_DIR = Path(__file__).parent / "fixtures"
TMP_DIR.mkdir(exist_ok=True)


def _make_sine(path, duration=3.0, sr=16000, freq=440):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    audio = (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    sf.write(str(path), audio, sr)
    return path, audio, sr


def test_load_audio_returns_float32_mono():
    path, original, sr = _make_sine(TMP_DIR / "test_tone.wav")
    audio, out_sr = load_audio(str(path))
    assert out_sr == 16000
    assert audio.dtype == np.float32
    assert audio.ndim == 1


def test_load_audio_resamples_to_target():
    path, _, _ = _make_sine(TMP_DIR / "test_22k.wav", sr=22050)
    audio, out_sr = load_audio(str(path), target_sr=16000)
    assert out_sr == 16000


def test_inject_noise_returns_same_shape():
    path, original, sr = _make_sine(TMP_DIR / "test_noise_in.wav")
    audio, _ = load_audio(str(path))
    noisy = inject_noise(audio, sr, noise_type="white", snr_db=20)
    assert noisy.shape == audio.shape
    assert noisy.dtype == np.float32


def test_inject_noise_clean_is_unchanged():
    path, original, sr = _make_sine(TMP_DIR / "test_clean.wav")
    audio, _ = load_audio(str(path))
    noisy = inject_noise(audio, sr, noise_type=None)
    assert np.allclose(audio, noisy, atol=1e-6)


def test_inject_reverb_returns_same_length():
    path, original, sr = _make_sine(TMP_DIR / "test_reverb.wav")
    audio, _ = load_audio(str(path))
    reverbed = inject_noise(audio, sr, noise_type="reverb", rt60=0.5)
    assert len(reverbed) == len(audio)


def test_inject_noise_0db_makes_it_noisier():
    """At 0dB SNR, signal and noise power are equal - audio changes substantially."""
    path, original, sr = _make_sine(TMP_DIR / "test_0db.wav")
    audio, _ = load_audio(str(path))
    noisy = inject_noise(audio, sr, noise_type="white", snr_db=0)
    diff = np.mean((audio - noisy) ** 2)
    assert diff > 0.01  # Should be noticeably different

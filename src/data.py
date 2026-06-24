"""Audio loading, preprocessing, and noise injection."""
import json
import numpy as np
import librosa
import soundfile as sf
from scipy.signal import butter, lfilter
from typing import Any, Dict, List, Optional, Tuple


def load_audio(path: str, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    """Load an audio file as float32 mono, resampled to target_sr.

    Args:
        path: Path to audio file.
        target_sr: Target sample rate in Hz.

    Returns:
        (audio_array, sample_rate) where audio_array is 1-D float32 in [-1, 1].
    """
    audio, sr = librosa.load(path, sr=target_sr, mono=True)
    return audio.astype(np.float32), sr


def save_audio(audio: np.ndarray, path: str, sr: int = 16000) -> None:
    """Save a float32 audio array to a WAV file."""
    sf.write(path, audio, sr)


def inject_noise(
    audio: np.ndarray,
    sr: int,
    noise_type: Optional[str] = None,
    snr_db: float = 20.0,
    rt60: float = 0.5,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Apply acoustic degradation to an audio signal.

    Args:
        audio: 1-D float32 audio array.
        sr: Sample rate.
        noise_type: "white", "babble", "reverb", or None (no degradation).
        snr_db: Signal-to-noise ratio in dB (for white/babble).
        rt60: Reverberation time in seconds (for reverb).
        seed: Random seed for reproducible noise generation.

    Returns:
        Degraded audio array, same shape and dtype as input.
    """
    if noise_type is None:
        return audio.copy()

    rng = np.random.RandomState(seed) if seed is not None else np.random

    if noise_type in ("white", "babble"):
        return _add_noise(audio, noise_type, snr_db, rng)
    elif noise_type == "reverb":
        return _add_reverb(audio, sr, rt60, rng)
    else:
        raise ValueError(f"Unknown noise_type: {noise_type}")


def _add_noise(audio: np.ndarray, noise_type: str, snr_db: float, rng) -> np.ndarray:
    """Add white or babble noise at specified SNR."""
    signal_power = np.mean(audio ** 2)
    snr_linear = 10 ** (snr_db / 10.0)
    target_noise_power = signal_power / snr_linear

    if noise_type == "white":
        noise = rng.randn(len(audio)).astype(np.float32)
    else:  # babble: band-pass filtered noise approximates speech-shaped noise
        noise = rng.randn(len(audio)).astype(np.float32)
        b, a_coeff = butter(4, [0.2, 0.6], btype="band")
        noise = lfilter(b, a_coeff, noise).astype(np.float32)

    noise_power = np.mean(noise ** 2)
    noise = noise * np.sqrt(target_noise_power / (noise_power + 1e-10))
    return (audio + noise).astype(np.float32)


def _add_reverb(audio: np.ndarray, sr: int, rt60: float, rng) -> np.ndarray:
    """Add synthetic reverberation using an exponential decay model with diffusion."""
    decay_samples = int(rt60 * sr)
    impulse_len = min(decay_samples, sr * 2)
    impulse = np.zeros(impulse_len, dtype=np.float32)
    impulse[0] = 1.0
    decay_db_per_sample = 60.0 / max(decay_samples, 1)
    decay_linear = 10 ** (-decay_db_per_sample / 20.0)
    for i in range(1, impulse_len):
        impulse[i] = impulse[i - 1] * decay_linear
        impulse[i] += rng.randn() * 0.001

    reverbed = np.convolve(audio, impulse, mode="full")[:len(audio)]
    reverbed = reverbed / (np.max(np.abs(reverbed)) + 1e-8)
    return reverbed.astype(np.float32)


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    """Peak-normalize audio to [-1, 1]."""
    peak = np.max(np.abs(audio))
    if peak > 0:
        return audio / peak
    return audio


def prepare_clip_metadata(raw_entry: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize a raw dataset entry to our standard metadata format.

    Args:
        raw_entry: Dict with at least 'audio_path', 'transcript'.

    Returns:
        Standardized dict with keys: audio_path, transcript, speaker, duration.
    """
    return {
        "audio_path": raw_entry["audio_path"],
        "transcript": raw_entry.get("transcript", ""),
        "speaker": raw_entry.get("speaker", "unknown"),
        "duration": raw_entry.get("duration", 0.0),
    }


def split_dataset(
    entries: List[Dict[str, Any]],
    dev: int,
    test: int,
    seed: int = 42,
) -> Dict[str, List[Dict[str, Any]]]:
    """Randomly split entries into dev and test sets.

    Args:
        entries: List of metadata dicts.
        dev: Number of dev samples.
        test: Number of test samples.
        seed: Random seed for reproducibility.

    Returns:
        {"dev": [...], "test": [...]}

    Raises:
        ValueError: If dev + test exceeds available entries.
    """
    if dev + test > len(entries):
        raise ValueError(
            f"Requested {dev + test} samples but only {len(entries)} available."
        )
    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(entries))
    return {
        "dev": [entries[i] for i in indices[:dev]],
        "test": [entries[i] for i in indices[dev:dev + test]],
    }


def generate_noise_variants(
    clip: Dict[str, Any],
    conditions: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Create metadata entries for all noise variants of a clip.

    Does NOT generate audio files -- creates metadata entries describing
    which noise condition to apply at inference time.

    Args:
        clip: Standardized metadata dict.
        conditions: Dict mapping condition name to noise kwargs.

    Returns:
        Dict mapping condition name to clip metadata with noise info attached.
    """
    variants = {}
    for cond_name, noise_kwargs in conditions.items():
        variant = dict(clip)
        variant["noise_condition"] = cond_name
        variant["noise_kwargs"] = noise_kwargs
        variants[cond_name] = variant
    return variants

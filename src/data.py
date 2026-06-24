"""Audio loading, preprocessing, and noise injection."""
import numpy as np
import librosa
import soundfile as sf
from typing import Optional, Tuple


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
) -> np.ndarray:
    """Apply acoustic degradation to an audio signal.

    Args:
        audio: 1-D float32 audio array.
        sr: Sample rate.
        noise_type: "white", "babble", "reverb", or None (no degradation).
        snr_db: Signal-to-noise ratio in dB (for white/babble).
        rt60: Reverberation time in seconds (for reverb).

    Returns:
        Degraded audio array, same shape and dtype as input.
    """
    if noise_type is None:
        return audio.copy()

    if noise_type in ("white", "babble"):
        return _add_noise(audio, noise_type, snr_db)
    elif noise_type == "reverb":
        return _add_reverb(audio, sr, rt60)
    else:
        raise ValueError(f"Unknown noise_type: {noise_type}")


def _add_noise(audio: np.ndarray, noise_type: str, snr_db: float) -> np.ndarray:
    """Add white or babble noise at specified SNR."""
    signal_power = np.mean(audio ** 2)
    snr_linear = 10 ** (snr_db / 10.0)
    target_noise_power = signal_power / snr_linear

    if noise_type == "white":
        noise = np.random.randn(len(audio)).astype(np.float32)
    else:  # babble: simulate with filtered noise (colored noise approximation)
        noise = np.random.randn(len(audio)).astype(np.float32)
        from scipy.signal import butter, lfilter
        b, a_coeff = butter(4, [0.2, 0.6], btype="band")
        noise = lfilter(b, a_coeff, noise).astype(np.float32)

    noise_power = np.mean(noise ** 2)
    noise = noise * np.sqrt(target_noise_power / (noise_power + 1e-10))
    return (audio + noise).astype(np.float32)


def _add_reverb(audio: np.ndarray, sr: int, rt60: float) -> np.ndarray:
    """Add synthetic reverberation using a simple exponential decay model."""
    decay_samples = int(rt60 * sr)
    impulse_len = min(decay_samples, sr * 2)
    impulse = np.zeros(impulse_len, dtype=np.float32)
    impulse[0] = 1.0
    decay_db_per_sample = 60.0 / max(decay_samples, 1)
    decay_linear = 10 ** (-decay_db_per_sample / 20.0)
    for i in range(1, impulse_len):
        impulse[i] = impulse[i - 1] * decay_linear
        impulse[i] += np.random.randn() * 0.001

    reverbed = np.convolve(audio, impulse, mode="full")[:len(audio)]
    reverbed = reverbed / (np.max(np.abs(reverbed)) + 1e-8)
    return reverbed.astype(np.float32)


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    """Peak-normalize audio to [-1, 1]."""
    peak = np.max(np.abs(audio))
    if peak > 0:
        return audio / peak
    return audio

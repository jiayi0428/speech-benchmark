"""Tests for direct (Speech LLM) pipeline."""
import base64
import numpy as np
import soundfile as sf
from pathlib import Path
from src.direct import DirectPipeline, encode_audio_base64

TMP_DIR = Path(__file__).parent / "fixtures"


def _make_sine_audio(path, duration=1.0, sr=16000):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    audio = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    sf.write(str(path), audio, sr)
    return str(path)


def test_encode_audio_base64_returns_string():
    path = _make_sine_audio(TMP_DIR / "b64_test.wav")
    b64 = encode_audio_base64(path)
    assert isinstance(b64, str)
    decoded = base64.b64decode(b64)
    assert len(decoded) > 0


def test_encode_audio_base64_is_deterministic():
    path = _make_sine_audio(TMP_DIR / "b64_det.wav")
    b64_1 = encode_audio_base64(path)
    b64_2 = encode_audio_base64(path)
    assert b64_1 == b64_2


def test_direct_pipeline_initializes(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-dummy-key")
    pipeline = DirectPipeline()
    assert pipeline.client is not None
    assert pipeline.model == "gpt-4o-audio-preview"

"""Tests for direct (Speech LLM) pipeline."""
import base64
import numpy as np
import pytest
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


@pytest.mark.skip(reason="Requires OpenAI API key — use test_direct_gemini.py or test_direct_qwen.py instead")
def test_direct_pipeline_initializes(monkeypatch):
    """DirectPipeline (OpenAI) initializes when OPENAI_API_KEY is the only key set."""
    import importlib
    import src.config
    import src.direct

    # Clear all API keys, then set only OPENAI
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-dummy-key")

    importlib.reload(src.config)
    importlib.reload(src.direct)

    try:
        from src.direct import DirectPipeline
        pipeline = DirectPipeline()
        assert pipeline.client is not None
        # Check the model contains "audio" or "gpt" (works with either OpenAI variant)
        model_lower = pipeline.model.lower()
        assert "audio" in model_lower or "gpt" in model_lower
    finally:
        # Restore by reloading with actual .env values
        importlib.reload(src.config)
        importlib.reload(src.direct)

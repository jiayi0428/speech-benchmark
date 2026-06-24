"""Tests for cascade pipeline."""
import numpy as np
import soundfile as sf
from pathlib import Path
from src.cascade import transcribe, CascadePipeline

TMP_DIR = Path(__file__).parent / "fixtures"


def _make_silent_audio(path, duration=3.0, sr=16000):
    audio = np.zeros(int(sr * duration), dtype=np.float32)
    sf.write(str(path), audio, sr)
    return str(path)


def test_transcribe_returns_string():
    path = _make_silent_audio(TMP_DIR / "silent.wav")
    result = transcribe(path)
    assert isinstance(result, str)


def test_transcribe_handles_short_audio():
    path = _make_silent_audio(TMP_DIR / "short.wav", duration=1.0)
    result = transcribe(path)
    assert isinstance(result, str)


def test_cascade_pipeline_has_model_loaded():
    pipeline = CascadePipeline()
    assert pipeline.model is not None

"""Tests for Gemini direct pipeline."""
import os
from unittest import mock
from pathlib import Path
from src.direct_gemini import GeminiDirectPipeline, SYSTEM_PROMPTS


def test_system_prompts_have_all_tasks():
    """All 4 tasks have prompt templates."""
    for task in ["summarization", "sentiment", "keywords", "intent"]:
        assert task in SYSTEM_PROMPTS
        assert len(SYSTEM_PROMPTS[task]) > 20


def test_gemini_pipeline_initializes_with_key():
    """Pipeline initializes when GEMINI_API_KEY is set."""
    with mock.patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
        # Force re-import of config
        import importlib
        import src.config
        importlib.reload(src.config)

        try:
            pipeline = GeminiDirectPipeline()
            assert pipeline.client is not None
            assert "gemini" in pipeline.model
        finally:
            # Restore original config
            importlib.reload(src.config)

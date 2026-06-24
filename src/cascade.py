"""Cascade pipeline: faster-whisper ASR -> GPT-4o-mini text LLM."""
import time
import logging
from typing import Any

from faster_whisper import WhisperModel
from openai import OpenAI

from src.config import (
    WHISPER_MODEL,
    WHISPER_DEVICE,
    WHISPER_COMPUTE_TYPE,
    TEXT_LLM_MODEL,
    OPENAI_API_KEY,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompts per task
# ---------------------------------------------------------------------------

SYSTEM_PROMPTS: dict[str, str] = {
    "summarization": (
        "You are a summarization expert. Summarize the following transcript "
        "in 3-5 sentences. Be concise and capture the main points. "
        "Return ONLY the summary, no preamble."
    ),
    "sentiment": (
        "You are a sentiment analyst. Classify the sentiment of the following "
        "transcript as exactly one of: positive, negative, or neutral. "
        "Also provide a confidence score from 0.0 to 1.0. "
        'Return your answer as JSON: {"sentiment": "<label>", "confidence": <float>}'
    ),
    "keywords": (
        "You are a keyword extraction expert. Extract 5-10 most important "
        "keywords or key phrases from the following transcript. "
        'Return your answer as a JSON list of strings: ["keyword1", "keyword2", ...]'
    ),
    "intent": (
        "You are a speech analyst. Identify the speaker's primary intent from "
        "this transcript. Choose exactly one of: inform, persuade, entertain, "
        "question, describe. "
        'Return your answer as JSON: {"intent": "<label>", "confidence": <float>}'
    ),
}

# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class CascadePipeline:
    """ASR -> Text LLM pipeline for speech understanding."""

    def __init__(self) -> None:
        logger.info("Loading Whisper model: %s", WHISPER_MODEL)
        self.model = WhisperModel(
            WHISPER_MODEL,
            device=WHISPER_DEVICE,
            compute_type=WHISPER_COMPUTE_TYPE,
        )
        if OPENAI_API_KEY:
            self.client = OpenAI(api_key=OPENAI_API_KEY)
        else:
            self.client = None  # type: ignore[assignment]

    def transcribe(self, audio_path: str) -> str:
        """Transcribe an audio file to text.

        Args:
            audio_path: Path to the audio file.

        Returns:
            Transcribed text string.
        """
        segments, _ = self.model.transcribe(audio_path, beam_size=5)
        transcript = " ".join(segment.text for segment in segments)
        return transcript.strip()

    def _call_llm(self, transcript: str, task: str) -> str:
        """Send transcript to GPT-4o-mini for task inference."""
        if self.client is None:
            raise RuntimeError(
                "OpenAI client not initialized – OPENAI_API_KEY is not set."
            )
        system_prompt = SYSTEM_PROMPTS.get(task, SYSTEM_PROMPTS["summarization"])
        response = self.client.chat.completions.create(
            model=TEXT_LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": transcript},
            ],
            temperature=0.0,
        )
        return response.choices[0].message.content

    def run(self, audio_path: str, task: str) -> dict[str, Any]:
        """Run the full cascade pipeline: transcribe -> LLM inference.

        Args:
            audio_path: Path to the audio file.
            task: One of "summarization", "sentiment", "keywords", "intent".

        Returns:
            Dict with keys: task, transcript, output, latency_seconds.
        """
        t_start = time.time()

        transcript = self.transcribe(audio_path)
        output = self._call_llm(transcript, task)

        latency = time.time() - t_start

        return {
            "task": task,
            "transcript": transcript,
            "output": output,
            "latency_seconds": round(latency, 3),
        }


# ---------------------------------------------------------------------------
# Lazy-loaded singleton for convenience
# ---------------------------------------------------------------------------

_pipeline_instance: CascadePipeline | None = None


def transcribe(audio_path: str) -> str:
    """Transcribe audio using the shared cascade pipeline instance."""
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = CascadePipeline()
    return _pipeline_instance.transcribe(audio_path)

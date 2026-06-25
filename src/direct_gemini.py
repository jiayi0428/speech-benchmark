"""Direct pipeline: Gemini 2.5 Flash for end-to-end speech understanding."""
import time
from typing import Any, Dict

from google import genai
from google.genai import types

from src.config import SPEECH_LLM_API_KEY, SPEECH_LLM_MODEL

SYSTEM_PROMPTS: dict[str, str] = {
    "summarization": (
        "Listen to the following audio and summarize it in 3-5 sentences. "
        "Be concise and capture the main points. "
        "Return ONLY the summary, no preamble."
    ),
    "sentiment": (
        "Listen to the following audio and classify the speaker's sentiment "
        "as exactly one of: positive, negative, or neutral. "
        "Consider tone of voice, pace, and word choice. "
        'Return your answer as JSON: {"sentiment": "<label>", "confidence": <float>}'
    ),
    "keywords": (
        "Listen to the following audio and extract 5-10 most important "
        "keywords or key phrases. Consider emphasis, repetition, and topic signals. "
        'Return your answer as a JSON list of strings: ["keyword1", "keyword2", ...]'
    ),
    "intent": (
        "Listen to the following audio and identify the speaker's primary "
        "communicative intent. Choose exactly one of: inform, persuade, entertain, "
        "question, describe. Consider tone, structure, and rhetorical cues. "
        'Return your answer as JSON: {"intent": "<label>", "confidence": <float>}'
    ),
}


class GeminiDirectPipeline:
    """Gemini 2.5 Flash pipeline for end-to-end speech understanding."""

    def __init__(self) -> None:
        self.client = genai.Client(api_key=SPEECH_LLM_API_KEY)
        self.model = SPEECH_LLM_MODEL

    def run(self, audio_path: str, task: str) -> Dict[str, Any]:
        """Run direct audio inference via Gemini.

        Args:
            audio_path: Path to the audio file (WAV, MP3, etc.).
            task: One of "summarization", "sentiment", "keywords", "intent".

        Returns:
            Dict with keys: task, output, latency_seconds.
        """
        t_start = time.time()

        system_prompt = SYSTEM_PROMPTS.get(task, SYSTEM_PROMPTS["summarization"])

        # Upload audio for Gemini processing
        audio_file = self.client.files.upload(file=audio_path)

        response = self.client.models.generate_content(
            model=self.model,
            contents=[
                audio_file,
                system_prompt,
            ],
            config=types.GenerateContentConfig(temperature=0.0),
        )

        latency = time.time() - t_start

        return {
            "task": task,
            "output": response.text,
            "latency_seconds": round(latency, 3),
        }

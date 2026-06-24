"""Direct pipeline: GPT-4o Audio mode for end-to-end speech understanding."""
import base64
import time
from typing import Any, Dict

from openai import OpenAI

from src.config import OPENAI_API_KEY, SPEECH_LLM_MODEL

SYSTEM_PROMPTS = {
    "summarization": (
        "You are a summarization expert. Listen to the following audio and "
        "summarize it in 3-5 sentences. Be concise and capture the main points. "
        "Return ONLY the summary, no preamble."
    ),
    "sentiment": (
        "You are a sentiment analyst. Listen to the following audio and "
        "classify the speaker's sentiment as exactly one of: positive, negative, "
        "or neutral. Consider tone of voice, pace, and word choice. "
        'Return your answer as JSON: {"sentiment": "<label>", "confidence": <float>}'
    ),
    "keywords": (
        "You are a keyword extraction expert. Listen to the following audio and "
        "extract 5-10 most important keywords or key phrases. "
        "Consider emphasis, repetition, and topic signals in the speech. "
        'Return your answer as a JSON list of strings: ["keyword1", "keyword2", ...]'
    ),
    "intent": (
        "You are a speech analyst. Listen to the following audio and identify "
        "the speaker's primary communicative intent. Choose exactly one of: "
        "inform, persuade, entertain, question, describe. "
        "Consider tone, structure, and rhetorical cues. "
        'Return your answer as JSON: {"intent": "<label>", "confidence": <float>}'
    ),
}


def encode_audio_base64(audio_path: str) -> str:
    """Read an audio file and encode it as base64 for the OpenAI Audio API.

    Args:
        audio_path: Path to the audio file (WAV, MP3, etc.).

    Returns:
        Base64-encoded string of the audio bytes.
    """
    with open(audio_path, "rb") as f:
        audio_bytes = f.read()
    return base64.b64encode(audio_bytes).decode("utf-8")


class DirectPipeline:
    """GPT-4o Audio mode pipeline for end-to-end speech understanding."""

    def __init__(self) -> None:
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.model = SPEECH_LLM_MODEL

    def run(self, audio_path: str, task: str) -> Dict[str, Any]:
        """Run direct audio inference for a specific task.

        Args:
            audio_path: Path to the audio file (WAV, MP3, etc.).
            task: One of "summarization", "sentiment", "keywords", "intent".

        Returns:
            Dict with keys: task, output, latency_seconds.
        """
        t_start = time.time()

        audio_b64 = encode_audio_base64(audio_path)
        system_prompt = SYSTEM_PROMPTS.get(task, SYSTEM_PROMPTS["summarization"])

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_audio",
                            "input_audio": {
                                "data": audio_b64,
                                "format": "wav",
                            },
                        },
                    ],
                },
            ],
            temperature=0.0,
        )

        latency = time.time() - t_start

        return {
            "task": task,
            "output": response.choices[0].message.content,
            "latency_seconds": round(latency, 3),
        }

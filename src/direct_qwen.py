"""Direct pipeline: Qwen2-Audio-7B (local, open-source speech LLM).

Runs end-to-end speech understanding on a local GPU with INT4 quantization.
First run downloads ~14GB model from HuggingFace.
"""
import time
import warnings
from typing import Any, Dict

import librosa
import torch
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

from src.config import SAMPLE_RATE


MODEL_ID = "Qwen/Qwen2-Audio-7B-Instruct"

SYSTEM_PROMPTS: dict[str, str] = {
    "summarization": (
        "Listen to the audio and summarize it in 3-5 sentences. "
        "Be concise and capture the main points. "
        "Return ONLY the summary, no preamble."
    ),
    "sentiment": (
        "Listen to the audio and classify the speaker's sentiment "
        "as exactly one of: positive, negative, or neutral. "
        "Consider tone of voice, pace, and word choice. "
        'Return JSON: {"sentiment": "<label>", "confidence": <float>}'
    ),
    "keywords": (
        "Listen to the audio and extract 5-10 most important "
        "keywords or key phrases. Consider emphasis and repetition. "
        'Return JSON list: ["keyword1", "keyword2", ...]'
    ),
    "intent": (
        "Listen to the audio and identify the speaker's primary "
        "communicative intent. Choose exactly one of: inform, persuade, "
        "entertain, question, describe. Consider tone and structure. "
        'Return JSON: {"intent": "<label>", "confidence": <float>}'
    ),
}


class QwenAudioPipeline:
    """Local Qwen2-Audio pipeline for end-to-end speech understanding."""

    def __init__(self) -> None:
        print(f"Loading {MODEL_ID} (INT4, this may take a few minutes)...")

        self.processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
                MODEL_ID,
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )

        self.sample_rate = self.processor.feature_extractor.sampling_rate
        print(f"Qwen2-Audio loaded (sample rate: {self.sample_rate}Hz).")

    def run(self, audio_path: str, task: str) -> Dict[str, Any]:
        """Run direct audio inference via Qwen2-Audio.

        Args:
            audio_path: Path to the audio file.
            task: One of "summarization", "sentiment", "keywords", "intent".

        Returns:
            Dict with keys: task, output, latency_seconds.
        """
        t_start = time.time()

        # Load and resample audio to model's expected rate
        audio, _ = librosa.load(audio_path, sr=self.sample_rate)

        system_prompt = SYSTEM_PROMPTS.get(task, SYSTEM_PROMPTS["summarization"])

        conversation = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"type": "audio", "audio_url": None},
                {"type": "text", "text": "Please respond with ONLY the answer, no extra text."},
            ]},
        ]

        text = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True
        )
        inputs = self.processor(
            text=text,
            audios=[audio],
            return_tensors="pt",
            sampling_rate=self.sample_rate,
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                temperature=None,
            )

        response = self.processor.batch_decode(
            generated_ids[:, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )[0]

        latency = time.time() - t_start
        return {
            "task": task,
            "output": response.strip(),
            "latency_seconds": round(latency, 3),
        }

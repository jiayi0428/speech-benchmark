"""Direct pipeline: Qwen2-Audio-7B (local, open-source speech LLM).

Runs end-to-end speech understanding on a local GPU with INT4 quantization.
First run downloads ~14GB model from HuggingFace.
"""
import time
import warnings
from typing import Any, Dict

import librosa
import torch
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2AudioForConditionalGeneration,
)

from src.config import SAMPLE_RATE


MODEL_ID = "Qwen/Qwen2-Audio-7B-Instruct"

SYSTEM_PROMPTS: dict[str, str] = {
    "summarization": (
        "Listen to the audio and summarize it in 3-5 sentences. "
        "Be concise and capture the main points. "
        "Return ONLY the summary, no preamble."
    ),
    "sentiment": (
        "Listen carefully to this audio. First, describe the speaker's "
        "tone of voice, pace, emotional cues, and word choices in detail. "
        "Note whether they sound happy, angry, worried, excited, calm, sad, "
        "or neutral. Then, based on your observations, state the overall "
        "sentiment as one of: positive, negative, or neutral. "
        "Be thorough in your description before giving the final label."
    ),
    "keywords": (
        "Listen carefully to this audio. First, write down the main topics "
        "and concepts discussed. Note which terms are repeated or emphasized. "
        "Then, list the 5-10 most important keywords or key phrases from the "
        "speech. Include both specific terms and general concepts."
    ),
    "intent": (
        "Listen carefully to this audio. First, describe what the speaker "
        "is trying to achieve. Are they teaching facts? Convincing you of "
        "something? Telling a story for amusement? Describing a scene or "
        "experience? Asking deep questions? Note the speaker's tone, "
        "structure, and rhetorical strategies. Then, based on your analysis, "
        "state the primary communicative intent as one of: inform, persuade, "
        "entertain, question, describe. Explain your reasoning."
    ),
}


class QwenAudioPipeline:
    """Local Qwen2-Audio pipeline for end-to-end speech understanding."""

    def __init__(self) -> None:
        print(f"Loading {MODEL_ID} (INT4, this may take a few minutes)...")

        self.processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
                MODEL_ID,
                quantization_config=quantization_config,
                device_map={"": torch.cuda.current_device()},
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
            audio=[audio],
            return_tensors="pt",
            sampling_rate=self.sample_rate,
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=512,
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

"""Direct pipeline: Qwen2-Audio-7B (local, open-source speech LLM).

Runs end-to-end speech understanding on a local GPU with INT4 quantization.
First run downloads ~14GB model from HuggingFace.
"""
import time
import warnings
import os
from typing import Any, Dict

import librosa
import torch
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2AudioForConditionalGeneration,
)

from src.config import PROJECT_ROOT, SAMPLE_RATE


MODEL_ID = os.getenv(
    "QWEN_MODEL_PATH",
    "Qwen/Qwen2-Audio-7B-Instruct",
)
USER_PROMPT_VERSION = "qwen_user_task_v2"
SYSTEM_PROMPT_VERSION = "qwen_system_task_v1"
PROMPT_VERSION = SYSTEM_PROMPT_VERSION
DEFAULT_PROMPT_MODE = "system"
SYSTEM_USER_INSTRUCTION = "Please respond with ONLY the answer, no extra text."

SYSTEM_PROMPTS: dict[str, str] = {
    "transcription": (
        "Transcribe the speech verbatim in English. "
        "Preserve the exact wording, including repetitions, hesitations, "
        "and incomplete sentences. Do not summarize, correct, or explain. "
        "Return ONLY the transcript."
    ),
    "summarization": (
        "Listen to the audio and summarize it in 3-5 sentences. "
        "Be concise and capture the main points. "
        "Return ONLY the summary, no preamble."
    ),
    "sentiment": (
        "Listen to the audio and classify the speaker's sentiment "
        "as exactly one of: positive, negative, or neutral. "
        "Consider tone of voice, pace, and word choice. "
        'Return strict valid JSON with double quotes: '
        '{"sentiment": "<label>", "confidence": <float>}'
    ),
    "keywords": (
        "Listen to the audio and extract 5-10 most important "
        "keywords or key phrases. Consider emphasis and repetition. "
        'Return a strict valid JSON list using double quotes: '
        '["keyword1", "keyword2", ...]'
    ),
    "intent": (
        "Listen to the audio and identify the speaker's primary "
        "communicative intent. Choose exactly one of: inform, persuade, "
        "entertain, question, describe. Consider tone and structure. "
        'Return strict valid JSON with double quotes: '
        '{"intent": "<label>", "confidence": <float>}'
    ),
}

SYSTEM_TURN_PROMPTS: dict[str, str] = {
    "transcription": SYSTEM_PROMPTS["transcription"],
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


def build_conversation(
    task: str, prompt_mode: str = DEFAULT_PROMPT_MODE
) -> list[dict[str, Any]]:
    """Build a Qwen2-Audio conversation without loading the model."""
    task_prompt = SYSTEM_PROMPTS.get(task, SYSTEM_PROMPTS["summarization"])
    if prompt_mode == "user":
        return [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio_url": None},
                    {"type": "text", "text": task_prompt},
                ],
            },
        ]
    if prompt_mode == "system":
        return [
            {
                "role": "system",
                "content": SYSTEM_TURN_PROMPTS.get(
                    task, SYSTEM_TURN_PROMPTS["summarization"]
                ),
            },
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio_url": None},
                    {"type": "text", "text": SYSTEM_USER_INSTRUCTION},
                ],
            },
        ]
    raise ValueError(f"Unsupported prompt mode: {prompt_mode}")


class QwenAudioPipeline:
    """Local Qwen2-Audio pipeline for end-to-end speech understanding."""

    def __init__(self, *, prompt_mode: str = DEFAULT_PROMPT_MODE) -> None:
        if prompt_mode not in {"user", "system"}:
            raise ValueError(f"Unsupported prompt mode: {prompt_mode}")
        self.prompt_mode = prompt_mode
        print(f"Loading {MODEL_ID} (INT4, this may take a few minutes)...")

        self.processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            llm_int8_enable_fp32_cpu_offload=True,
        )
        offload_dir = PROJECT_ROOT / ".cache" / "qwen_offload"
        offload_dir.mkdir(parents=True, exist_ok=True)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
                MODEL_ID,
                quantization_config=quantization_config,
                device_map="auto",
                max_memory={0: "7GiB", "cpu": "20GiB"},
                offload_folder=str(offload_dir),
                offload_state_dict=True,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )

        self.sample_rate = self.processor.feature_extractor.sampling_rate
        print(f"Qwen2-Audio loaded (sample rate: {self.sample_rate}Hz).")

    def run(self, audio_path: str, task: str) -> Dict[str, Any]:
        """Run direct audio inference via Qwen2-Audio.

        Args:
            audio_path: Path to the audio file.
            task: One of "transcription", "summarization", "sentiment",
                "keywords", or "intent".

        Returns:
            Dict with keys: task, output, latency_seconds.
        """
        t_start = time.time()

        # Load and resample audio to model's expected rate
        audio, _ = librosa.load(audio_path, sr=self.sample_rate)

        # System-turn prompting intentionally lets Qwen produce a free-form
        # analysis for structured tasks; DeepSeek performs the final schema
        # conversion. User-turn prompting remains available for reproducing
        # the original experiment.
        conversation = build_conversation(task, self.prompt_mode)

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

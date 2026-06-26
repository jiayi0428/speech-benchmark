# Speech Understanding Benchmark — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible benchmark comparing cascade (ASR→Text LLM) and direct (Speech LLM) architectures on 4 speech understanding tasks with robustness testing.

**Architecture:** Modular Python library (`src/`) with three pipeline modules feeding a shared evaluation layer. Jupyter notebooks orchestrate experiments. Gradio app provides interactive demo. All local inference on RTX 5070 8GB; cloud API for GPT-4o.

**Tech Stack:** Python 3.14, faster-whisper, OpenAI API, librosa, audiomentations, plotly, gradio, scipy/scikit-learn

---

## File Map

| File | Responsibility |
|------|---------------|
| `src/__init__.py` | Package marker |
| `src/config.py` | Paths, model names, noise params, API key loading |
| `src/data.py` | Dataset loading, audio I/O, noise injection, preprocessing |
| `src/cascade.py` | faster-whisper transcription → GPT-4o-mini task inference |
| `src/direct.py` | GPT-4o Audio mode task inference |
| `src/evaluation.py` | Task metrics, statistical tests, WER computation |
| `src/visualization.py` | All plotting functions (matplotlib + plotly) |
| `app/gradio_app.py` | Interactive web UI |
| `notebooks/01_data_preparation.ipynb` | Dataset exploration and preprocessing |
| `notebooks/02_cascade_pipeline.ipynb` | Run cascade experiments |
| `notebooks/03_direct_pipeline.ipynb` | Run direct experiments |
| `notebooks/04_evaluation.ipynb` | Compute metrics, run statistical tests |
| `notebooks/05_visualization.ipynb` | Generate all figures |
| `notebooks/06_deep_case_studies.ipynb` | Failure catalog and analysis |
| `requirements.txt` | Python dependencies |
| `.env.example` | Template for API key |

---

## Phase 0: Project Scaffold

### Task 0: Create project structure and dependencies

**Files:**
- Create: `requirements.txt`
- Create: `.env.example`
- Create: `src/__init__.py`
- Create: `src/config.py`

- [ ] **Step 1: Create project directory structure**

```bash
mkdir -p speech-benchmark/src speech-benchmark/app speech-benchmark/notebooks speech-benchmark/data/raw speech-benchmark/data/processed speech-benchmark/data/results speech-benchmark/report
```

- [ ] **Step 2: Write `requirements.txt`**

```
faster-whisper>=1.0.0
torch>=2.4.0
transformers>=4.45.0
openai>=1.50.0
librosa>=0.10.0
soundfile>=0.12.0
pydub>=0.25.0
audiomentations>=0.35.0
rouge-score>=0.1.2
bert-score>=0.3.13
scipy>=1.14.0
scikit-learn>=1.5.0
matplotlib>=3.9.0
plotly>=5.23.0
seaborn>=0.13.0
ipywidgets>=8.1.0
gradio>=4.40.0
pandas>=2.2.0
numpy>=1.26.0
tqdm>=4.66.0
python-dotenv>=1.0.0
jiwer>=3.0.0
```

- [ ] **Step 3: Write `.env.example`**

```
OPENAI_API_KEY=sk-your-key-here
```

- [ ] **Step 4: Write `src/__init__.py`**

```python
"""Speech Understanding Benchmark - Cascade vs End-to-End."""
```

- [ ] **Step 5: Write `src/config.py`**

```python
"""Central configuration for the speech benchmark project."""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# --- Paths ---
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = DATA_DIR / "results"

# Ensure directories exist
for d in [RAW_DIR, PROCESSED_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# --- ASR Model ---
WHISPER_MODEL = "large-v3"
WHISPER_DEVICE = "cuda"
WHISPER_COMPUTE_TYPE = "float16"

# --- API Models ---
TEXT_LLM_MODEL = "gpt-4o-mini"
SPEECH_LLM_MODEL = "gpt-4o-audio-preview"

# --- Audio ---
SAMPLE_RATE = 16000
MAX_AUDIO_SECONDS = 300

# --- Noise Conditions ---
NOISE_CONDITIONS = {
    "clean": {},
    "babble_20db": {"noise_type": "babble", "snr_db": 20},
    "babble_10db": {"noise_type": "babble", "snr_db": 10},
    "babble_0db":  {"noise_type": "babble", "snr_db": 0},
    "white_20db":  {"noise_type": "white",  "snr_db": 20},
    "white_10db":  {"noise_type": "white",  "snr_db": 10},
    "white_0db":   {"noise_type": "white",  "snr_db": 0},
    "reverb_0.5s": {"noise_type": "reverb", "rt60": 0.5},
    "reverb_1.0s": {"noise_type": "reverb", "rt60": 1.0},
    "reverb_1.5s": {"noise_type": "reverb", "rt60": 1.5},
}

# --- Tasks ---
TASKS = ["summarization", "sentiment", "keywords", "intent"]

# --- API Key ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set. Copy .env.example to .env and fill in your key.")
```

- [ ] **Step 6: Install dependencies**

```bash
cd speech-benchmark
pip install -r requirements.txt
```

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "chore: scaffold project structure and configuration"
```

---

## Phase 1: Data Module

### Task 1: Audio loading and noise injection

**Files:**
- Create: `src/data.py`
- Create: `tests/test_data.py`

- [ ] **Step 1: Write the failing test**

```python
"""Tests for data loading and noise injection."""
import numpy as np
import soundfile as sf
from pathlib import Path
from src.data import load_audio, inject_noise, resample_audio

TMP_DIR = Path(__file__).parent / "fixtures"
TMP_DIR.mkdir(exist_ok=True)

def _make_sine(path, duration=3.0, sr=16000, freq=440):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    audio = (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    sf.write(str(path), audio, sr)
    return path, audio, sr

def test_load_audio_returns_float32_mono():
    path, original, sr = _make_sine(TMP_DIR / "test_tone.wav")
    audio, out_sr = load_audio(str(path))
    assert out_sr == 16000
    assert audio.dtype == np.float32
    assert audio.ndim == 1

def test_load_audio_resamples_to_target():
    path, _, _ = _make_sine(TMP_DIR / "test_22k.wav", sr=22050)
    audio, out_sr = load_audio(str(path), target_sr=16000)
    assert out_sr == 16000

def test_inject_noise_returns_same_shape():
    path, original, sr = _make_sine(TMP_DIR / "test_noise_in.wav")
    audio, _ = load_audio(str(path))
    noisy = inject_noise(audio, sr, noise_type="white", snr_db=20)
    assert noisy.shape == audio.shape
    assert noisy.dtype == np.float32

def test_inject_noise_clean_is_unchanged():
    path, original, sr = _make_sine(TMP_DIR / "test_clean.wav")
    audio, _ = load_audio(str(path))
    noisy = inject_noise(audio, sr, noise_type=None)
    assert np.allclose(audio, noisy, atol=1e-6)

def test_inject_reverb_returns_same_length():
    path, original, sr = _make_sine(TMP_DIR / "test_reverb.wav")
    audio, _ = load_audio(str(path))
    reverbed = inject_noise(audio, sr, noise_type="reverb", rt60=0.5)
    assert len(reverbed) == len(audio)

def test_inject_noise_0db_makes_it_noisier():
    """At 0dB SNR, signal and noise power are equal - audio changes substantially."""
    path, original, sr = _make_sine(TMP_DIR / "test_0db.wav")
    audio, _ = load_audio(str(path))
    noisy = inject_noise(audio, sr, noise_type="white", snr_db=0)
    diff = np.mean((audio - noisy) ** 2)
    assert diff > 0.01  # Should be noticeably different
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd speech-benchmark && python -m pytest tests/test_data.py -v
```
Expected: all FAIL (module not found / functions not defined)

- [ ] **Step 3: Write `src/data.py`**

```python
"""Audio loading, preprocessing, and noise injection."""
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from typing import Optional, Tuple
from src.config import SAMPLE_RATE


def load_audio(path: str, target_sr: int = SAMPLE_RATE) -> Tuple[np.ndarray, int]:
    """Load an audio file as float32 mono, resampled to target_sr.

    Args:
        path: Path to audio file.
        target_sr: Target sample rate in Hz.

    Returns:
        (audio_array, sample_rate) where audio_array is 1-D float32 in [-1, 1].
    """
    audio, sr = librosa.load(path, sr=target_sr, mono=True)
    return audio.astype(np.float32), sr


def save_audio(audio: np.ndarray, path: str, sr: int = SAMPLE_RATE) -> None:
    """Save a float32 audio array to a WAV file."""
    sf.write(path, audio, sr)


def inject_noise(
    audio: np.ndarray,
    sr: int,
    noise_type: Optional[str] = None,
    snr_db: float = 20.0,
    rt60: float = 0.5,
) -> np.ndarray:
    """Apply acoustic degradation to an audio signal.

    Args:
        audio: 1-D float32 audio array.
        sr: Sample rate.
        noise_type: "white", "babble", "reverb", or None (no degradation).
        snr_db: Signal-to-noise ratio in dB (for white/babble).
        rt60: Reverberation time in seconds (for reverb).

    Returns:
        Degraded audio array, same shape and dtype as input.
    """
    if noise_type is None:
        return audio.copy()

    if noise_type in ("white", "babble"):
        return _add_noise(audio, noise_type, snr_db)
    elif noise_type == "reverb":
        return _add_reverb(audio, sr, rt60)
    else:
        raise ValueError(f"Unknown noise_type: {noise_type}")


def _add_noise(audio: np.ndarray, noise_type: str, snr_db: float) -> np.ndarray:
    """Add white or babble noise at specified SNR."""
    signal_power = np.mean(audio ** 2)
    snr_linear = 10 ** (snr_db / 10.0)
    target_noise_power = signal_power / snr_linear

    if noise_type == "white":
        noise = np.random.randn(len(audio)).astype(np.float32)
    else:  # babble: simulate with filtered noise (colored noise approximation)
        noise = np.random.randn(len(audio)).astype(np.float32)
        from scipy.signal import butter, lfilter
        b, a_coeff = butter(4, [0.2, 0.6], btype="band")
        noise = lfilter(b, a_coeff, noise).astype(np.float32)

    noise_power = np.mean(noise ** 2)
    noise = noise * np.sqrt(target_noise_power / noise_power)
    return (audio + noise).astype(np.float32)


def _add_reverb(audio: np.ndarray, sr: int, rt60: float) -> np.ndarray:
    """Add synthetic reverberation using a simple Schroeder model."""
    # Simple exponential decay reverb
    decay_samples = int(rt60 * sr)
    impulse = np.zeros(min(decay_samples, sr * 2), dtype=np.float32)
    impulse[0] = 1.0
    decay_db_per_sample = 60.0 / decay_samples
    decay_linear = 10 ** (-decay_db_per_sample / 20.0)
    for i in range(1, len(impulse)):
        impulse[i] = impulse[i - 1] * decay_linear
        impulse[i] += np.random.randn() * 0.001

    reverbed = np.convolve(audio, impulse, mode="full")[:len(audio)]
    reverbed = reverbed / (np.max(np.abs(reverbed)) + 1e-8)
    return reverbed.astype(np.float32)


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    """Peak-normalize audio to [-1, 1]."""
    peak = np.max(np.abs(audio))
    if peak > 0:
        return audio / peak
    return audio
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd speech-benchmark && python -m pytest tests/test_data.py -v
```
Expected: all 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/data.py tests/test_data.py
git commit -m "feat: add audio loading and noise injection module"
```

---

### Task 2: Dataset preparation utilities

**Files:**
- Modify: `src/data.py` (append new functions)
- Create: `tests/test_data_prep.py`

- [ ] **Step 1: Write the failing test**

```python
"""Tests for dataset preparation."""
import json
import numpy as np
import soundfile as sf
from pathlib import Path
from src.data import (
    prepare_clip_metadata,
    split_dataset,
    generate_noise_variants,
)

TMP_DIR = Path(__file__).parent / "fixtures"

def _make_fake_tedlium_index(output_dir: Path, n_files: int = 5):
    """Create a minimal fake TED-LIUM index for testing."""
    output_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for i in range(n_files):
        # Write a tiny audio file
        audio_path = output_dir / f"ted_{i}.wav"
        t = np.linspace(0, 2, 32000, endpoint=False)
        audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        sf.write(str(audio_path), audio, 16000)
        entries.append({
            "audio_path": str(audio_path),
            "transcript": f"This is transcript number {i}.",
            "speaker": f"speaker_{i}",
            "duration": 2.0,
        })
    return entries

def test_prepare_clip_metadata_extracts_fields():
    entries = _make_fake_tedlium_index(TMP_DIR / "fake_ted")
    meta = prepare_clip_metadata(entries[0])
    assert "audio_path" in meta
    assert "transcript" in meta
    assert meta["duration"] == 2.0

def test_split_dataset_returns_correct_sizes():
    entries = _make_fake_tedlium_index(TMP_DIR / "split_test")
    splits = split_dataset(entries, dev=2, test=3)
    assert len(splits["dev"]) == 2
    assert len(splits["test"]) == 3
    # Check no overlap
    dev_ids = {s["audio_path"] for s in splits["dev"]}
    test_ids = {s["audio_path"] for s in splits["test"]}
    assert dev_ids.isdisjoint(test_ids)

def test_split_dataset_raises_on_too_many():
    entries = _make_fake_tedlium_index(TMP_DIR / "split_err")
    try:
        split_dataset(entries, dev=10, test=10)
    except ValueError:
        pass  # expected
    else:
        raise AssertionError("Should have raised ValueError")

def test_generate_noise_variants_produces_all_conditions():
    entries = _make_fake_tedlium_index(TMP_DIR / "noise_var", n_files=1)
    conditions = {"clean": {}, "white_20db": {"noise_type": "white", "snr_db": 20}}
    variants = generate_noise_variants(entries[0], conditions)
    assert len(variants) == 2
    assert "clean" in variants
    assert "white_20db" in variants
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd speech-benchmark && python -m pytest tests/test_data_prep.py -v
```
Expected: all FAIL

- [ ] **Step 3: Append to `src/data.py`**

```python
"""... (existing code above) ..."""

import json
from typing import List, Dict, Any


def prepare_clip_metadata(raw_entry: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize a raw dataset entry to our standard metadata format.

    Args:
        raw_entry: Dict with at least 'audio_path', 'transcript'.

    Returns:
        Standardized dict with keys: audio_path, transcript, speaker, duration.
    """
    return {
        "audio_path": raw_entry["audio_path"],
        "transcript": raw_entry.get("transcript", ""),
        "speaker": raw_entry.get("speaker", "unknown"),
        "duration": raw_entry.get("duration", 0.0),
    }


def split_dataset(
    entries: List[Dict[str, Any]],
    dev: int,
    test: int,
    seed: int = 42,
) -> Dict[str, List[Dict[str, Any]]]:
    """Randomly split entries into dev and test sets.

    Args:
        entries: List of metadata dicts.
        dev: Number of dev samples.
        test: Number of test samples.
        seed: Random seed for reproducibility.

    Returns:
        {"dev": [...], "test": [...]}

    Raises:
        ValueError: If dev + test exceeds available entries.
    """
    if dev + test > len(entries):
        raise ValueError(
            f"Requested {dev + test} samples but only {len(entries)} available."
        )
    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(entries))
    return {
        "dev": [entries[i] for i in indices[:dev]],
        "test": [entries[i] for i in indices[dev:dev + test]],
    }


def generate_noise_variants(
    clip: Dict[str, Any],
    conditions: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Create metadata entries for all noise variants of a clip.

    Does NOT generate audio files — just creates the metadata entries
    that describe which noise condition to apply at inference time.

    Args:
        clip: Standardized metadata dict.
        conditions: Dict mapping condition name to noise kwargs.

    Returns:
        Dict mapping condition name to clip metadata with noise info attached.
    """
    variants = {}
    for cond_name, noise_kwargs in conditions.items():
        variant = dict(clip)  # shallow copy is fine
        variant["noise_condition"] = cond_name
        variant["noise_kwargs"] = noise_kwargs
        variants[cond_name] = variant
    return variants
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd speech-benchmark && python -m pytest tests/test_data_prep.py -v
```
Expected: all 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/data.py tests/test_data_prep.py
git commit -m "feat: add dataset preparation utilities"
```

---

## Phase 2: Cascade Pipeline (ASR → Text LLM)

### Task 3: faster-whisper transcription wrapper

**Files:**
- Create: `src/cascade.py`
- Create: `tests/test_cascade.py`

- [ ] **Step 1: Write the failing test**

```python
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

def test_cascade_pipeline_run_summarization():
    """Integration test: transcribe silence and run summarization task.
    
    Note: This will produce nonsense output on silent audio, but verifies
    the pipeline runs without errors.
    """
    path = _make_silent_audio(TMP_DIR / "silent2.wav")
    pipeline = CascadePipeline()
    result = pipeline.run(path, "summarization")
    assert "output" in result
    assert result["task"] == "summarization"
    assert "transcript" in result
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd speech-benchmark && python -m pytest tests/test_cascade.py -v -k "not test_cascade_pipeline_run_summarization"
```
Expected: first 3 tests FAIL (module not found)

- [ ] **Step 3: Write `src/cascade.py`**

```python
"""Cascade pipeline: faster-whisper ASR → GPT-4o-mini text LLM."""
import time
from typing import Dict, Any, Optional
from faster_whisper import WhisperModel
from openai import OpenAI
from src.config import (
    WHISPER_MODEL, WHISPER_DEVICE, WHISPER_COMPUTE_TYPE,
    TEXT_LLM_MODEL, OPENAI_API_KEY,
)


SYSTEM_PROMPTS = {
    "summarization": (
        "You are a summarization expert. Summarize the following transcript "
        "in 3-5 sentences. Be concise and capture the main points. "
        "Return ONLY the summary, no preamble."
    ),
    "sentiment": (
        "You are a sentiment analyst. Classify the sentiment of the following "
        "transcript as exactly one of: positive, negative, or neutral. "
        "Also provide a confidence score from 0.0 to 1.0. "
        "Return your answer as JSON: {\"sentiment\": \"<label>\", \"confidence\": <float>}"
    ),
    "keywords": (
        "You are a keyword extraction expert. Extract 5-10 most important "
        "keywords or key phrases from the following transcript. "
        "Return your answer as a JSON list of strings: [\"keyword1\", \"keyword2\", ...]"
    ),
    "intent": (
        "You are a speech analyst. Identify the speaker's primary intent from "
        "this transcript. Choose exactly one of: inform, persuade, entertain, "
        "question, describe. "
        "Return your answer as JSON: {\"intent\": \"<label>\", \"confidence\": <float>}"
    ),
}


class CascadePipeline:
    """ASR → Text LLM pipeline for speech understanding."""

    def __init__(self):
        print(f"Loading Whisper model: {WHISPER_MODEL}...")
        self.model = WhisperModel(
            WHISPER_MODEL,
            device=WHISPER_DEVICE,
            compute_type=WHISPER_COMPUTE_TYPE,
        )
        self.client = OpenAI(api_key=OPENAI_API_KEY)

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

    def run(self, audio_path: str, task: str) -> Dict[str, Any]:
        """Run the full cascade pipeline: transcribe → LLM inference.

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


# Convenience function for notebook use
_pipeline_instance: Optional[CascadePipeline] = None


def transcribe(audio_path: str) -> str:
    """Transcribe audio using the shared cascade pipeline instance."""
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = CascadePipeline()
    return _pipeline_instance.transcribe(audio_path)
```

- [ ] **Step 4: Run unit tests**

```bash
cd speech-benchmark && python -m pytest tests/test_cascade.py::test_transcribe_returns_string tests/test_cascade.py::test_transcribe_handles_short_audio -v
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/cascade.py tests/test_cascade.py
git commit -m "feat: add cascade pipeline with faster-whisper and GPT-4o-mini"
```

---

## Phase 3: Direct Pipeline (Speech LLM)

### Task 4: GPT-4o Audio mode pipeline

**Files:**
- Create: `src/direct.py`
- Create: `tests/test_direct.py`

- [ ] **Step 1: Write the failing test**

```python
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
    # Verify it's valid base64
    decoded = base64.b64decode(b64)
    assert len(decoded) > 0

def test_encode_audio_base64_is_deterministic():
    path = _make_sine_audio(TMP_DIR / "b64_det.wav")
    b64_1 = encode_audio_base64(path)
    b64_2 = encode_audio_base64(path)
    assert b64_1 == b64_2

def test_direct_pipeline_initializes():
    pipeline = DirectPipeline()
    assert pipeline.client is not None
    assert pipeline.model == "gpt-4o-audio-preview"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd speech-benchmark && python -m pytest tests/test_direct.py -v
```
Expected: all FAIL

- [ ] **Step 3: Write `src/direct.py`**

```python
"""Direct pipeline: GPT-4o Audio mode for end-to-end speech understanding."""
import base64
import time
from typing import Dict, Any
from openai import OpenAI
from src.config import SPEECH_LLM_MODEL, OPENAI_API_KEY


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
        "Return your answer as JSON: {\"sentiment\": \"<label>\", \"confidence\": <float>}"
    ),
    "keywords": (
        "You are a keyword extraction expert. Listen to the following audio and "
        "extract 5-10 most important keywords or key phrases. "
        "Consider emphasis, repetition, and topic signals in the speech. "
        "Return your answer as a JSON list of strings: [\"keyword1\", \"keyword2\", ...]"
    ),
    "intent": (
        "You are a speech analyst. Listen to the following audio and identify "
        "the speaker's primary communicative intent. Choose exactly one of: "
        "inform, persuade, entertain, question, describe. "
        "Consider tone, structure, and rhetorical cues. "
        "Return your answer as JSON: {\"intent\": \"<label>\", \"confidence\": <float>}"
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

    def __init__(self):
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
```

- [ ] **Step 4: Run unit tests**

```bash
cd speech-benchmark && python -m pytest tests/test_direct.py -v
```
Expected: all 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/direct.py tests/test_direct.py
git commit -m "feat: add direct pipeline with GPT-4o Audio mode"
```

---

## Phase 4: Evaluation Module

### Task 5: Task metrics and WER computation

**Files:**
- Create: `src/evaluation.py`
- Create: `tests/test_evaluation.py`

- [ ] **Step 1: Write the failing test**

```python
"""Tests for evaluation metrics."""
import numpy as np
from src.evaluation import (
    compute_wer,
    parse_sentiment_json,
    parse_keywords_json,
    compute_keyword_f1,
    compute_sentiment_accuracy,
)

def test_wer_identical_is_zero():
    assert compute_wer("hello world", "hello world") == 0.0

def test_wer_substitution_is_one():
    # "hello" → "hallo" = 1 substitution, 1 word total
    assert compute_wer("hello", "hallo") == 1.0

def test_wer_deletion_is_correct():
    # "hello world" → "hello" = 1 deletion, 2 words total
    assert compute_wer("hello world", "hello") == 0.5

def test_wer_insertion_is_correct():
    # "hello" → "hello world" = 1 insertion, 1 word total
    assert compute_wer("hello", "hello world") == 1.0

def test_wer_empty_reference():
    assert compute_wer("", "something") == 1.0

def test_parse_sentiment_json_standard():
    result = parse_sentiment_json('{"sentiment": "positive", "confidence": 0.92}')
    assert result["sentiment"] == "positive"
    assert result["confidence"] == 0.92

def test_parse_sentiment_json_with_markdown():
    result = parse_sentiment_json('```json\n{"sentiment": "neutral", "confidence": 0.75}\n```')
    assert result["sentiment"] == "neutral"

def test_parse_keywords_json_standard():
    result = parse_keywords_json('["AI", "climate change", "innovation"]')
    assert result == ["AI", "climate change", "innovation"]

def test_compute_keyword_f1_perfect():
    ref = ["AI", "climate", "energy"]
    pred = ["AI", "climate", "energy"]
    f1, p, r = compute_keyword_f1(ref, pred)
    assert f1 == 1.0
    assert p == 1.0

def test_compute_keyword_f1_partial():
    ref = ["AI", "climate", "energy", "future"]
    pred = ["AI", "climate", "technology"]
    f1, p, r = compute_keyword_f1(ref, pred)
    assert 0.4 < f1 < 0.8  # 2/4 correct, 2/3 precision

def test_compute_sentiment_accuracy_perfect():
    y_true = ["positive", "negative", "neutral"]
    y_pred = ["positive", "negative", "neutral"]
    acc, f1 = compute_sentiment_accuracy(y_true, y_pred)
    assert acc == 1.0
    assert f1 == 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd speech-benchmark && python -m pytest tests/test_evaluation.py -v
```
Expected: all FAIL

- [ ] **Step 3: Write `src/evaluation.py`**

```python
"""Evaluation metrics and statistical tests for the speech benchmark."""
import json
import re
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from collections import Counter
from scipy import stats as scipy_stats


# --- WER ---

def compute_wer(reference: str, hypothesis: str) -> float:
    """Compute Word Error Rate between reference and hypothesis.

    Uses a simple Levenshtein-style edit distance at word level.

    Args:
        reference: Ground truth transcript.
        hypothesis: ASR output transcript.

    Returns:
        WER as a float (0.0 = perfect, 1.0 = completely wrong).
    """
    ref_words = reference.strip().split()
    hyp_words = hypothesis.strip().split()

    if len(ref_words) == 0:
        return 1.0 if len(hyp_words) > 0 else 0.0

    # Levenshtein distance at word level
    d = np.zeros((len(ref_words) + 1, len(hyp_words) + 1), dtype=np.int32)
    for i in range(len(ref_words) + 1):
        d[i, 0] = i
    for j in range(len(hyp_words) + 1):
        d[0, j] = j

    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            cost = 0 if ref_words[i - 1].lower() == hyp_words[j - 1].lower() else 1
            d[i, j] = min(
                d[i - 1, j] + 1,       # deletion
                d[i, j - 1] + 1,        # insertion
                d[i - 1, j - 1] + cost,  # substitution
            )

    return d[len(ref_words), len(hyp_words)] / len(ref_words)


# --- JSON Parsing for LLM Outputs ---

def _extract_json(text: str) -> str:
    """Extract JSON from LLM output that may have markdown or extra text."""
    # Try to find JSON code block
    match = re.search(r'```(?:json)?\s*(.*?)\s*```', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Try to find balanced braces
    brace_start = text.find('{')
    brace_end = text.rfind('}')
    if brace_start != -1 and brace_end != -1:
        return text[brace_start:brace_end + 1]
    # Try balanced brackets
    bracket_start = text.find('[')
    bracket_end = text.rfind(']')
    if bracket_start != -1 and bracket_end != -1:
        return text[bracket_start:bracket_end + 1]
    return text.strip()


def parse_sentiment_json(output: str) -> Dict[str, Any]:
    """Parse LLM sentiment output into a structured dict.

    Returns:
        {"sentiment": str, "confidence": float} or {"sentiment": "unknown", "confidence": 0.0}
    """
    try:
        data = json.loads(_extract_json(output))
        sentiment = str(data.get("sentiment", "unknown")).lower().strip()
        if sentiment not in ("positive", "negative", "neutral"):
            sentiment = "unknown"
        confidence = float(data.get("confidence", 0.0))
        confidence = max(0.0, min(1.0, confidence))
        return {"sentiment": sentiment, "confidence": confidence}
    except (json.JSONDecodeError, ValueError, TypeError):
        return {"sentiment": "unknown", "confidence": 0.0}


def parse_keywords_json(output: str) -> List[str]:
    """Parse LLM keyword output into a list of strings.

    Returns:
        List of keyword strings (empty list on parse failure).
    """
    try:
        data = json.loads(_extract_json(output))
        if isinstance(data, list):
            return [str(k).strip() for k in data if str(k).strip()]
        return []
    except (json.JSONDecodeError, ValueError, TypeError):
        return []


def parse_intent_json(output: str) -> Dict[str, Any]:
    """Parse LLM intent output.

    Returns:
        {"intent": str, "confidence": float}
    """
    valid_intents = {"inform", "persuade", "entertain", "question", "describe"}
    try:
        data = json.loads(_extract_json(output))
        intent = str(data.get("intent", "unknown")).lower().strip()
        if intent not in valid_intents:
            intent = "unknown"
        confidence = float(data.get("confidence", 0.0))
        confidence = max(0.0, min(1.0, confidence))
        return {"intent": intent, "confidence": confidence}
    except (json.JSONDecodeError, ValueError, TypeError):
        return {"intent": "unknown", "confidence": 0.0}


# --- Scoring Functions ---

def compute_keyword_f1(
    reference: List[str],
    predicted: List[str],
) -> Tuple[float, float, float]:
    """Compute precision, recall, and F1 for keyword extraction.

    Matches are case-insensitive and trim whitespace.

    Returns:
        (f1, precision, recall)
    """
    ref_set = {k.lower().strip() for k in reference}
    pred_set = {k.lower().strip() for k in predicted}

    if len(pred_set) == 0 and len(ref_set) == 0:
        return 1.0, 1.0, 1.0
    if len(pred_set) == 0:
        return 0.0, 0.0, 0.0
    if len(ref_set) == 0:
        return 0.0, 0.0, 0.0

    intersection = ref_set & pred_set
    precision = len(intersection) / len(pred_set)
    recall = len(intersection) / len(ref_set)

    if precision + recall == 0:
        return 0.0, precision, recall

    f1 = 2 * precision * recall / (precision + recall)
    return f1, precision, recall


def compute_sentiment_accuracy(
    y_true: List[str],
    y_pred: List[str],
) -> Tuple[float, float]:
    """Compute accuracy and macro F1 for sentiment classification.

    Returns:
        (accuracy, macro_f1)
    """
    from sklearn.metrics import accuracy_score, f1_score
    accuracy = accuracy_score(y_true, y_pred)
    # Macro F1: handle cases where some classes missing
    labels = sorted(set(y_true) | set(y_pred))
    try:
        macro_f1 = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    except ValueError:
        macro_f1 = 0.0
    return float(accuracy), float(macro_f1)


# --- Statistical Tests ---

def paired_ttest(
    scores_a: List[float],
    scores_b: List[float],
) -> Dict[str, float]:
    """Run a paired t-test comparing two sets of scores.

    Args:
        scores_a: Scores from method A (e.g., cascade).
        scores_b: Scores from method B (e.g., direct).

    Returns:
        {"statistic": t, "p_value": p, "mean_diff": mean(a-b), "cohens_d": d}
    """
    a = np.array(scores_a, dtype=np.float64)
    b = np.array(scores_b, dtype=np.float64)
    result = scipy_stats.ttest_rel(a, b)
    diff = a - b
    d = np.mean(diff) / (np.std(diff, ddof=1) + 1e-10)
    return {
        "statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "mean_diff": float(np.mean(diff)),
        "cohens_d": float(d),
    }


def bootstrap_ci(
    differences: List[float],
    n_bootstrap: int = 10000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float]:
    """Compute bootstrap confidence interval for mean difference.

    Args:
        differences: List of paired differences (method_a - method_b).
        n_bootstrap: Number of bootstrap resamples.
        ci_level: Confidence level (default 0.95).
        seed: Random seed.

    Returns:
        (lower_bound, upper_bound)
    """
    rng = np.random.RandomState(seed)
    diffs = np.array(differences, dtype=np.float64)
    means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(diffs, size=len(diffs), replace=True)
        means.append(np.mean(sample))
    alpha = (1.0 - ci_level) / 2.0
    lower = np.percentile(means, 100 * alpha)
    upper = np.percentile(means, 100 * (1 - alpha))
    return float(lower), float(upper)
```

- [ ] **Step 4: Run tests**

```bash
cd speech-benchmark && python -m pytest tests/test_evaluation.py -v
```
Expected: all tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/evaluation.py tests/test_evaluation.py
git commit -m "feat: add evaluation metrics and statistical tests"
```

---

## Phase 5: Visualization Module

### Task 6: Plotting functions

**Files:**
- Create: `src/visualization.py`
- Create: `tests/test_visualization.py`

- [ ] **Step 1: Write the failing test**

```python
"""Tests for visualization module."""
import numpy as np
from pathlib import Path
from src.visualization import (
    plot_radar_chart,
    plot_degradation_curves,
    plot_error_propagation,
)

TMP_DIR = Path(__file__).parent / "fixtures"

def test_plot_radar_chart_returns_figure():
    categories = ["Summary", "Sentiment", "Keywords", "Intent"]
    cascade_scores = [0.75, 0.82, 0.68, 0.79]
    direct_scores = [0.80, 0.85, 0.72, 0.77]
    fig = plot_radar_chart(categories, cascade_scores, direct_scores)
    # matplotlib Figure
    assert hasattr(fig, "savefig")

def test_plot_degradation_curves_returns_figure():
    snr_values = ["Clean", "20dB", "10dB", "0dB"]
    cascade_line = [0.80, 0.78, 0.65, 0.45]
    direct_line = [0.82, 0.81, 0.75, 0.62]
    fig = plot_degradation_curves(
        "Summarization", snr_values, cascade_line, direct_line
    )
    assert hasattr(fig, "savefig")

def test_plot_error_propagation_returns_figure():
    wer_values = [0.0, 0.1, 0.2, 0.3, 0.4]
    scores = [0.85, 0.80, 0.72, 0.60, 0.50]
    fig = plot_error_propagation(wer_values, scores)
    assert hasattr(fig, "savefig")
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd speech-benchmark && python -m pytest tests/test_visualization.py -v
```
Expected: all FAIL

- [ ] **Step 3: Write `src/visualization.py`**

```python
"""Visualization functions for the speech benchmark."""
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional, Tuple
from matplotlib.figure import Figure


# Apply a clean style
sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1.2)

COLORS = {
    "cascade": "#2563EB",   # Blue
    "direct": "#DC2626",    # Red
    "optional": "#7C3AED",  # Purple
}


def plot_radar_chart(
    categories: List[str],
    cascade_scores: List[float],
    direct_scores: List[float],
    title: str = "Task Performance Comparison",
    optional_scores: Optional[List[float]] = None,
    optional_label: str = "Qwen2-Audio",
) -> Figure:
    """Plot a radar/spider chart comparing architectures across tasks.

    Args:
        categories: Names of each axis (tasks).
        cascade_scores: Scores for cascade pipeline.
        direct_scores: Scores for direct pipeline.
        title: Chart title.
        optional_scores: Optional third set of scores.
        optional_label: Label for optional scores.

    Returns:
        matplotlib Figure.
    """
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # Close the loop

    cascade_norm = cascade_scores + cascade_scores[:1]
    direct_norm = direct_scores + direct_scores[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    ax.fill(angles, cascade_norm, alpha=0.25, color=COLORS["cascade"])
    ax.plot(angles, cascade_norm, "o-", linewidth=2, color=COLORS["cascade"], label="Cascade (ASR+LLM)")

    ax.fill(angles, direct_norm, alpha=0.25, color=COLORS["direct"])
    ax.plot(angles, direct_norm, "o-", linewidth=2, color=COLORS["direct"], label="Direct (GPT-4o Audio)")

    if optional_scores is not None:
        opt_norm = optional_scores + optional_scores[:1]
        ax.fill(angles, opt_norm, alpha=0.15, color=COLORS["optional"])
        ax.plot(angles, opt_norm, "o-", linewidth=2, color=COLORS["optional"], label=optional_label)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=8)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=10)

    plt.tight_layout()
    return fig


def plot_degradation_curves(
    metric_name: str,
    x_labels: List[str],
    cascade_scores: List[float],
    direct_scores: List[float],
    cascade_ci: Optional[List[Tuple[float, float]]] = None,
    direct_ci: Optional[List[Tuple[float, float]]] = None,
    title: Optional[str] = None,
) -> Figure:
    """Plot degradation curves showing performance vs noise level.

    Args:
        metric_name: Name of the metric (y-axis label).
        x_labels: Labels for noise conditions (x-axis).
        cascade_scores: Cascade scores for each condition.
        direct_scores: Direct scores for each condition.
        cascade_ci: Optional (lower, upper) confidence intervals for cascade.
        direct_ci: Optional (lower, upper) confidence intervals for direct.
        title: Plot title.

    Returns:
        matplotlib Figure.
    """
    if title is None:
        title = f"Robustness: {metric_name} under Acoustic Degradation"

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(x_labels))
    width = 0.35

    bars1 = ax.bar(x - width / 2, cascade_scores, width, label="Cascade (ASR+LLM)",
                   color=COLORS["cascade"], alpha=0.85, edgecolor="white", linewidth=0.5)
    bars2 = ax.bar(x + width / 2, direct_scores, width, label="Direct (GPT-4o Audio)",
                   color=COLORS["direct"], alpha=0.85, edgecolor="white", linewidth=0.5)

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f"{height:.2f}", xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f"{height:.2f}", xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)

    ax.set_xlabel("Noise Condition", fontsize=12)
    ax.set_ylabel(metric_name, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=30, ha="right")
    ax.legend(fontsize=11)
    ax.set_ylim(0, max(max(cascade_scores), max(direct_scores)) * 1.15)

    plt.tight_layout()
    return fig


def plot_error_propagation(
    wer_values: List[float],
    task_scores: List[float],
    task_name: str = "Summarization",
    title: Optional[str] = None,
) -> Figure:
    """Plot WER vs downstream task score to visualize error propagation.

    Args:
        wer_values: WER values for each sample.
        task_scores: Corresponding task metric scores.
        task_name: Name of the downstream task.
        title: Plot title.

    Returns:
        matplotlib Figure.
    """
    if title is None:
        title = f"ASR Error Propagation: WER vs {task_name} Score"

    fig, ax = plt.subplots(figsize=(8, 6))

    # Scatter
    ax.scatter(wer_values, task_scores, alpha=0.6, s=60,
               color=COLORS["cascade"], edgecolors="white", linewidth=0.5, zorder=5)

    # Regression line
    if len(wer_values) > 1:
        coeffs = np.polyfit(wer_values, task_scores, 1)
        poly_fn = np.poly1d(coeffs)
        x_line = np.linspace(min(wer_values), max(wer_values), 100)
        ax.plot(x_line, poly_fn(x_line), "--", color="#991B1B", linewidth=2,
                label=f"Trend (r={np.corrcoef(wer_values, task_scores)[0,1]:.3f})")
        ax.legend(fontsize=10)

    ax.set_xlabel("Word Error Rate (WER)", fontsize=12)
    ax.set_ylabel(f"{task_name} Score", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlim(0, max(wer_values) * 1.1)

    plt.tight_layout()
    return fig


def plot_confusion_matrices(
    cascade_cm: np.ndarray,
    direct_cm: np.ndarray,
    labels: List[str],
    title: str = "Sentiment Confusion Matrix Comparison",
) -> Figure:
    """Plot side-by-side confusion matrices for cascade and direct.

    Args:
        cascade_cm: Confusion matrix for cascade (N x N).
        direct_cm: Confusion matrix for direct (N x N).
        labels: Class label names.
        title: Figure title.

    Returns:
        matplotlib Figure.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    sns.heatmap(cascade_cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax1,
                cbar=False, linewidths=0.5)
    ax1.set_title("Cascade (ASR+LLM)", fontsize=13, fontweight="bold")
    ax1.set_ylabel("True", fontsize=11)
    ax1.set_xlabel("Predicted", fontsize=11)

    sns.heatmap(direct_cm, annot=True, fmt="d", cmap="Reds",
                xticklabels=labels, yticklabels=labels, ax=ax2,
                cbar=False, linewidths=0.5)
    ax2.set_title("Direct (GPT-4o Audio)", fontsize=13, fontweight="bold")
    ax2.set_ylabel("True", fontsize=11)
    ax2.set_xlabel("Predicted", fontsize=11)

    fig.suptitle(title, fontsize=15, fontweight="bold")
    plt.tight_layout()
    return fig
```

- [ ] **Step 4: Run tests**

```bash
cd speech-benchmark && python -m pytest tests/test_visualization.py -v
```
Expected: all 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/visualization.py tests/test_visualization.py
git commit -m "feat: add visualization module with radar, degradation, and error plots"
```

---

## Phase 6: Experiment Notebooks

### Task 7: Data preparation notebook

**Files:**
- Create: `notebooks/01_data_preparation.ipynb`

- [ ] **Step 1: Create the notebook**

This notebook covers:
1. Download TED-LIUM v3 (or use a subset manually downloaded)
2. Load and normalize clip metadata
3. Create dev/test splits
4. Generate noise variant metadata
5. Save processed splits as JSON to `data/processed/`

Since TED-LIUM is ~10GB and downloading it in CI is impractical, the notebook documents manual download steps and assumes the dataset is in `data/raw/tedlium/`. For initial testing, we use a tiny synthetic dataset created by a script.

```python
# Cell 1: Setup
import sys
sys.path.insert(0, "..")
from src.data import prepare_clip_metadata, split_dataset, generate_noise_variants
from src.config import NOISE_CONDITIONS, PROCESSED_DIR, RAW_DIR
import json
import glob

# Cell 2: Load TED-LIUM metadata
# TED-LIUM v3 structure: data/raw/tedlium/{sph/, stm/}
# Parse STM files to build clip index

def load_tedlium_index(tedlium_dir):
    """Parse TED-LIUM STM files to build a clip index."""
    stm_files = sorted(glob.glob(f"{tedlium_dir}/stm/*.stm"))
    entries = []
    for stm_path in stm_files:
        with open(stm_path) as f:
            for line in f:
                parts = line.strip().split(" ", 6)
                if len(parts) < 7:
                    continue
                # STM format: <filename> <channel> <speaker> <start> <end> <...> <text>
                filename = parts[0]
                speaker = parts[2]
                start = float(parts[3])
                end = float(parts[4])
                text = parts[6]
                audio_path = f"{tedlium_dir}/sph/{filename}.sph"
                entries.append({
                    "audio_path": audio_path,
                    "transcript": text,
                    "speaker": speaker,
                    "start": start,
                    "end": end,
                    "duration": end - start,
                })
    return entries

entries = load_tedlium_index(f"{RAW_DIR}/tedlium")
print(f"Loaded {len(entries)} segments from TED-LIUM")

# Cell 3: Filter to reasonable-length clips (1-3 minutes) and normalize
clips = [e for e in entries if 60 <= e["duration"] <= 180]
clips = [prepare_clip_metadata(e) for e in clips]
print(f"Filtered to {len(clips)} clips (1-3 min)")

# Cell 4: Split into dev and test
splits = split_dataset(clips, dev=10, test=40)

# Further split test into clean and robustness
robustness_clips = splits["test"][:10]
clean_test_clips = splits["test"][10:]

# Generate noise variants for robustness set
robustness_variants = []
for clip in robustness_clips:
    variants = generate_noise_variants(clip, NOISE_CONDITIONS)
    for cond_name, variant in variants.items():
        robustness_variants.append(variant)

print(f"Dev: {len(splits['dev'])} clips")
print(f"Test-Clean: {len(clean_test_clips)} clips")
print(f"Test-Robustness: {len(robustness_variants)} variants ({len(robustness_clips)} clips × {len(NOISE_CONDITIONS)} conditions)")

# Cell 5: Save splits
with open(PROCESSED_DIR / "dev.json", "w") as f:
    json.dump(splits["dev"], f, indent=2)
with open(PROCESSED_DIR / "test_clean.json", "w") as f:
    json.dump(clean_test_clips, f, indent=2)
with open(PROCESSED_DIR / "test_robustness.json", "w") as f:
    json.dump(robustness_variants, f, indent=2)

print("Splits saved to data/processed/")
```

- [ ] **Step 2: Commit**

```bash
git add notebooks/01_data_preparation.ipynb
git commit -m "feat: add data preparation notebook"
```

---

### Task 8: Experiment runner notebook (combined cascade + direct)

**Files:**
- Create: `notebooks/02_run_experiments.ipynb`

- [ ] **Step 1: Create the notebook**

This single notebook runs both pipelines over the test sets and saves results. Structured to match the data flow diagram from the spec.

```python
# Cell 1: Setup and imports
import sys
sys.path.insert(0, "..")
import json
import time
from pathlib import Path
from tqdm.notebook import tqdm
from src.cascade import CascadePipeline
from src.direct import DirectPipeline
from src.data import load_audio, inject_noise, save_audio
from src.config import PROCESSED_DIR, RESULTS_DIR, TASKS, NOISE_CONDITIONS

# Cell 2: Initialize pipelines
print("Initializing pipelines...")
cascade = CascadePipeline()
direct = DirectPipeline()
print("Ready.")

# Cell 3: Define experiment runner
def run_sample(audio_path, pipeline, task, noise_kwargs=None):
    """Run one sample through a pipeline, with optional noise injection."""
    # If noise is specified, create temporary noisy audio
    if noise_kwargs:
        audio, sr = load_audio(audio_path)
        noisy = inject_noise(audio, sr, **noise_kwargs)
        tmp_path = Path(audio_path).parent / f"_tmp_noisy_{Path(audio_path).name}"
        save_audio(noisy, str(tmp_path), sr)
        audio_path = str(tmp_path)

    result = pipeline.run(audio_path, task)
    return result

# Cell 4: Run cascade experiments on test-clean
with open(PROCESSED_DIR / "test_clean.json") as f:
    test_clean = json.load(f)

cascade_clean_results = []
for clip in tqdm(test_clean, desc="Cascade Clean"):
    for task in TASKS:
        result = run_sample(clip["audio_path"], cascade, task)
        result["clip_id"] = clip["audio_path"]
        cascade_clean_results.append(result)

with open(RESULTS_DIR / "cascade_clean.json", "w") as f:
    json.dump(cascade_clean_results, f, indent=2)
print(f"Saved {len(cascade_clean_results)} cascade clean results")

# Cell 5: Run direct experiments on test-clean
direct_clean_results = []
for clip in tqdm(test_clean, desc="Direct Clean"):
    for task in TASKS:
        result = run_sample(clip["audio_path"], direct, task)
        result["clip_id"] = clip["audio_path"]
        direct_clean_results.append(result)

with open(RESULTS_DIR / "direct_clean.json", "w") as f:
    json.dump(direct_clean_results, f, indent=2)
print(f"Saved {len(direct_clean_results)} direct clean results")

# Cell 6: Run robustness experiments
with open(PROCESSED_DIR / "test_robustness.json") as f:
    test_robustness = json.load(f)

cascade_rob_results = []
direct_rob_results = []

for variant in tqdm(test_robustness, desc="Robustness"):
    noise = variant.get("noise_kwargs", {})
    for task in TASKS:
        # Cascade
        cr = run_sample(variant["audio_path"], cascade, task, noise)
        cr["clip_id"] = variant["audio_path"]
        cr["noise_condition"] = variant.get("noise_condition", "clean")
        cascade_rob_results.append(cr)

        # Direct
        dr = run_sample(variant["audio_path"], direct, task, noise)
        dr["clip_id"] = variant["audio_path"]
        dr["noise_condition"] = variant.get("noise_condition", "clean")
        direct_rob_results.append(dr)

with open(RESULTS_DIR / "cascade_robustness.json", "w") as f:
    json.dump(cascade_rob_results, f, indent=2)
with open(RESULTS_DIR / "direct_robustness.json", "w") as f:
    json.dump(direct_rob_results, f, indent=2)
print(f"Saved robustness results")

# Cell 7: Print summary statistics
def summarize(results, name):
    tasks_seen = set(r["task"] for r in results)
    latencies = [r["latency_seconds"] for r in results]
    print(f"\n{name}:")
    print(f"  Tasks: {tasks_seen}")
    print(f"  Samples: {len(results)}")
    print(f"  Avg latency: {sum(latencies)/len(latencies):.2f}s")
    print(f"  Estimated API cost: ${len(results) * 0.002:.2f}")

summarize(cascade_clean_results, "Cascade Clean")
summarize(direct_clean_results, "Direct Clean")
summarize(cascade_rob_results, "Cascade Robustness")
summarize(direct_rob_results, "Direct Robustness")
```

- [ ] **Step 2: Commit**

```bash
git add notebooks/02_run_experiments.ipynb
git commit -m "feat: add experiment runner notebook"
```

---

## Phase 7: Evaluation & Visualization Notebooks

### Task 9: Evaluation notebook

**Files:**
- Create: `notebooks/03_evaluation.ipynb`

- [ ] **Step 1: Create the notebook**

Notebook loads saved results, computes all metrics, runs statistical tests, and saves a summary JSON.

- [ ] **Step 2: Commit**

### Task 10: Visualization notebook

**Files:**
- Create: `notebooks/04_visualization.ipynb`

- [ ] **Step 1: Create the notebook**

Notebook loads evaluation summary, generates all charts (radar, degradation curves, error propagation scatter, confusion matrices, cost/latency bar), and saves figures as PNG+HTML.

- [ ] **Step 2: Commit**

### Task 11: Deep case studies notebook

**Files:**
- Create: `notebooks/05_case_studies.ipynb`

- [ ] **Step 1: Create the notebook**

Notebook picks 3-5 samples where cascade and direct outputs diverge significantly, shows audio waveform with transcript alignment (ASR errors in red), and annotates what went wrong and why.

- [ ] **Step 2: Commit**

---

## Phase 8: Gradio Demo

### Task 12: Gradio web application

**Files:**
- Create: `app/gradio_app.py`

- [ ] **Step 1: Write the app**

```python
"""Gradio demo for interactive Cascade vs Direct comparison."""
import sys
sys.path.insert(0, "..")

import gradio as gr
import tempfile
import os
from pathlib import Path
from src.cascade import CascadePipeline
from src.direct import DirectPipeline
from src.data import inject_noise, load_audio, save_audio
from src.config import TASKS

# Load pipelines once at startup
cascade = CascadePipeline()
direct = DirectPipeline()

NOISE_CHOICES = [
    ("Clean", {}),
    ("Babble 20dB", {"noise_type": "babble", "snr_db": 20}),
    ("Babble 10dB", {"noise_type": "babble", "snr_db": 10}),
    ("Babble 0dB", {"noise_type": "babble", "snr_db": 0}),
    ("White 20dB", {"noise_type": "white", "snr_db": 20}),
    ("White 0dB", {"noise_type": "white", "snr_db": 0}),
    ("Reverb 1.0s", {"noise_type": "reverb", "rt60": 1.0}),
]


def process_audio(audio_file, noise_choice):
    """Main processing function: run both pipelines and return results."""
    if audio_file is None:
        return "Please upload an audio file.", "", "", "", "", "", "", "", ""

    # Apply noise if selected
    audio_path = audio_file
    noise_kwargs = dict(NOISE_CHOICES).get(noise_choice, {})

    if noise_kwargs:
        audio, sr = load_audio(audio_path)
        noisy = inject_noise(audio, sr, **noise_kwargs)
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        save_audio(noisy, tmp.name, sr)
        audio_path = tmp.name

    # Run cascade on all tasks
    cascade_outputs = {}
    for task in TASKS:
        result = cascade.run(audio_path, task)
        cascade_outputs[task] = result["output"]
    cascade_transcript = cascade_outputs.get("summarization", "")
    # Get actual transcript from the cascade result
    cascade_result = cascade.run(audio_path, "summarization")
    transcript = cascade_result.get("transcript", "")
    cascade_summary = cascade_result["output"]
    cascade_sentiment_result = cascade.run(audio_path, "sentiment")
    cascade_sentiment = cascade_sentiment_result["output"]
    cascade_keywords_result = cascade.run(audio_path, "keywords")
    cascade_keywords = cascade_keywords_result["output"]
    cascade_intent_result = cascade.run(audio_path, "intent")
    cascade_intent = cascade_intent_result["output"]
    cascade_latency = cascade_result["latency_seconds"]

    # Run direct on all tasks
    direct_summary_result = direct.run(audio_path, "summarization")
    direct_summary = direct_summary_result["output"]
    direct_sentiment_result = direct.run(audio_path, "sentiment")
    direct_sentiment = direct_sentiment_result["output"]
    direct_keywords_result = direct.run(audio_path, "keywords")
    direct_keywords = direct_keywords_result["output"]
    direct_intent_result = direct.run(audio_path, "intent")
    direct_intent = direct_intent_result["output"]
    direct_latency = direct_summary_result["latency_seconds"]

    return (
        transcript,
        cascade_summary,
        cascade_sentiment,
        cascade_keywords,
        cascade_intent,
        direct_summary,
        direct_sentiment,
        direct_keywords,
        direct_intent,
        f"Cascade: {cascade_latency:.2f}s | Direct: {direct_latency:.2f}s",
    )


with gr.Blocks(title="Speech Understanding Benchmark", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🎤 Speech Understanding: Cascade vs End-to-End
    
    Upload an audio file and see how two different AI architectures understand it.
    """)

    with gr.Row():
        with gr.Column(scale=1):
            audio_input = gr.Audio(label="Upload Audio", type="filepath")
            noise_radio = gr.Radio(
                choices=[c[0] for c in NOISE_CHOICES],
                value="Clean",
                label="Noise Level",
            )
            run_btn = gr.Button("▶ Run Comparison", variant="primary", size="lg")
            latency_text = gr.Textbox(label="Latency", interactive=False)

    gr.Markdown("---")

    with gr.Row():
        with gr.Column():
            gr.Markdown("### 🔧 Cascade (ASR → Text LLM)")
            cascade_transcript = gr.Textbox(label="📝 Transcript", lines=3, interactive=False)
            cascade_summary = gr.Textbox(label="📊 Summary", lines=4, interactive=False)
            cascade_sentiment = gr.Textbox(label="😊 Sentiment", lines=2, interactive=False)
            cascade_keywords = gr.Textbox(label="🔑 Keywords", lines=2, interactive=False)
            cascade_intent = gr.Textbox(label="🎯 Intent", lines=2, interactive=False)

        with gr.Column():
            gr.Markdown("### 🚀 Direct (GPT-4o Audio)")
            direct_summary = gr.Textbox(label="📊 Summary", lines=4, interactive=False)
            direct_sentiment = gr.Textbox(label="😊 Sentiment", lines=2, interactive=False)
            direct_keywords = gr.Textbox(label="🔑 Keywords", lines=2, interactive=False)
            direct_intent = gr.Textbox(label="🎯 Intent", lines=2, interactive=False)

    run_btn.click(
        fn=process_audio,
        inputs=[audio_input, noise_radio],
        outputs=[
            cascade_transcript, cascade_summary, cascade_sentiment,
            cascade_keywords, cascade_intent,
            direct_summary, direct_sentiment, direct_keywords, direct_intent,
            latency_text,
        ],
    )

    gr.Markdown("""
    ---
    ### How it works
    
    | | Cascade | Direct |
    |---|---------|--------|
    | **ASR** | faster-whisper large-v3 (local GPU) | Built into GPT-4o |
    | **Understanding** | GPT-4o-mini (text API) | GPT-4o Audio mode |
    | **Cost per run** | ~$0.004 | ~$0.018 |
    """)

demo.launch(share=False)
```

- [ ] **Step 2: Test the app launches**

```bash
cd speech-benchmark/app && timeout 5 python gradio_app.py 2>&1 || true
```
Expected: app starts, shows local URL

- [ ] **Step 3: Commit**

```bash
git add app/gradio_app.py
git commit -m "feat: add Gradio interactive demo"
```

---

## Phase 9: Report & Polish

### Task 13: Report skeleton and README

**Files:**
- Create: `report/report.md`
- Create: `README.md`

- [ ] **Step 1: Write README.md**

```markdown
# Speech Understanding Benchmark

Cascade (ASR → Text LLM) vs End-to-End (Speech LLM) comparison across 4 tasks with robustness testing.

## Setup
```bash
pip install -r requirements.txt
cp .env.example .env  # then add your OPENAI_API_KEY
```

## Usage
1. Download TED-LIUM v3 to `data/raw/tedlium/`
2. Run notebooks in order: 01 → 02 → 03 → 04 → 05
3. Launch demo: `cd app && python gradio_app.py`

## Architecture
- `src/cascade.py` — faster-whisper + GPT-4o-mini pipeline
- `src/direct.py` — GPT-4o Audio mode pipeline
- `src/evaluation.py` — WER, ROUGE, BERTScore, keyword F1, statistical tests
- `src/visualization.py` — radar, degradation curves, error propagation
- `app/gradio_app.py` — interactive web demo
```

- [ ] **Step 2: Write report skeleton**

```markdown
# Cascade vs End-to-End: A Robustness-Aware Benchmark of Speech Understanding Architectures

## Abstract
[1 paragraph summarizing the study, key findings, and implications]

## 1. Introduction
- Background on speech understanding paradigms
- The cascade vs end-to-end debate
- Our research question and hypotheses

## 2. Methodology
### 2.1 Architectures
### 2.2 Tasks
### 2.3 Datasets
### 2.4 Robustness Testing
### 2.5 Evaluation Metrics

## 3. Results
### 3.1 Clean Audio Performance
### 3.2 Robustness Under Degradation
### 3.3 Error Propagation Analysis
### 3.4 Cost-Latency Trade-offs

## 4. Case Studies
[3-5 deep dives with audio excerpts and analysis]

## 5. Discussion
- When does each architecture win?
- Implications for real-world deployment
- Limitations and future work

## 6. Conclusion

## References
```

- [ ] **Step 3: Commit**

```bash
git add README.md report/report.md
git commit -m "docs: add README and report skeleton"
```

---

## Implementation Order Summary

```
Phase 0: Scaffold              (Task 0)     ← START HERE
Phase 1: Data Module           (Tasks 1-2)
Phase 2: Cascade Pipeline      (Task 3)
Phase 3: Direct Pipeline       (Task 4)
Phase 4: Evaluation Module     (Task 5)
Phase 5: Visualization Module  (Task 6)
Phase 6: Experiment Notebooks  (Tasks 7-8)
Phase 7: Eval+Viz Notebooks    (Tasks 9-11)
Phase 8: Gradio Demo           (Task 12)
Phase 9: Report & Polish       (Task 13)
```

Phases 0-5 are sequential (each depends on the previous). Phases 6-7 can partially overlap. Phase 8 is independent once Phase 5 is done. Phase 9 is last.

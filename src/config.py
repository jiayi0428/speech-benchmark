"""Central configuration for the speech benchmark project."""
import os
from pathlib import Path
from dotenv import load_dotenv

# --- Paths ---
PROJECT_ROOT = Path(__file__).parent.parent

load_dotenv(PROJECT_ROOT / ".env")
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = DATA_DIR / "results"

# Ensure directories exist
for d in [RAW_DIR, PROCESSED_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# --- ASR Model ---
WHISPER_MODEL = "large-v3"
_WHISPER_DEVICE_RAW = os.getenv("WHISPER_DEVICE", "auto")
if _WHISPER_DEVICE_RAW == "auto":
    try:
        import torch

        WHISPER_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        WHISPER_DEVICE = "cpu"
else:
    WHISPER_DEVICE = _WHISPER_DEVICE_RAW
WHISPER_COMPUTE_TYPE = "float16" if WHISPER_DEVICE == "cuda" else "int8"

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
    import warnings
    warnings.warn(
        "OPENAI_API_KEY not set. LLM-based tasks will fail. "
        "Copy .env.example to .env and fill in your key."
    )

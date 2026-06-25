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

# --- API Provider Selection ---
# Supports OpenAI and DeepSeek (OpenAI-compatible API)
# Priority: OPENAI_API_KEY > DEEPSEEK_API_KEY
_OPENAI_KEY = os.getenv("OPENAI_API_KEY")
_DEEPSEEK_KEY = os.getenv("DEEPSEEK_API_KEY")

if _OPENAI_KEY:
    TEXT_LLM_PROVIDER = "openai"
    TEXT_LLM_MODEL = "gpt-4o-mini"
    TEXT_LLM_BASE_URL = "https://api.openai.com/v1"
    TEXT_LLM_API_KEY = _OPENAI_KEY
elif _DEEPSEEK_KEY:
    TEXT_LLM_PROVIDER = "deepseek"
    TEXT_LLM_MODEL = "deepseek-chat"
    TEXT_LLM_BASE_URL = "https://api.deepseek.com"
    TEXT_LLM_API_KEY = _DEEPSEEK_KEY
else:
    TEXT_LLM_PROVIDER = None
    TEXT_LLM_MODEL = "gpt-4o-mini"  # fallback
    TEXT_LLM_BASE_URL = "https://api.openai.com/v1"
    TEXT_LLM_API_KEY = None

# --- Speech LLM (Direct Pipeline) ---
# Supports: Gemini (free API), OpenAI GPT-4o Audio (paid), Qwen2-Audio (local)
# Priority: GEMINI_API_KEY > OPENAI_API_KEY > local Qwen2-Audio
_GEMINI_KEY = os.getenv("GEMINI_API_KEY")

if _GEMINI_KEY:
    SPEECH_LLM_PROVIDER = "gemini"
    SPEECH_LLM_MODEL = "gemini-2.5-flash"
    SPEECH_LLM_API_KEY = _GEMINI_KEY
elif _OPENAI_KEY:
    SPEECH_LLM_PROVIDER = "openai"
    SPEECH_LLM_MODEL = "gpt-4o-audio-preview"
    SPEECH_LLM_API_KEY = _OPENAI_KEY
else:
    # Fallback to local Qwen2-Audio (no API key required)
    SPEECH_LLM_PROVIDER = "qwen"
    SPEECH_LLM_MODEL = "Qwen/Qwen2-Audio-7B-Instruct"
    SPEECH_LLM_API_KEY = None

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

# --- API Key Validation ---
if not _OPENAI_KEY and not _DEEPSEEK_KEY and not _GEMINI_KEY:
    import warnings
    warnings.warn(
        "No API key found. Set DEEPSEEK_API_KEY, GEMINI_API_KEY, or OPENAI_API_KEY "
        "in .env. LLM and speech tasks will fail."
    )

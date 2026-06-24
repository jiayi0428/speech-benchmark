# 🎤 Speech Understanding Benchmark

**Cascade (ASR → Text LLM) vs End-to-End (Speech LLM) comparison with robustness testing.**

An undergraduate summer research project comparing two speech understanding architectures across 4 tasks under clean and degraded acoustic conditions.

## Architecture

| | Cascade Pipeline | Direct Pipeline |
|---|---|---|
| **Approach** | faster-whisper (ASR) → GPT-4o-mini (text) | GPT-4o Audio mode (end-to-end) |
| **Speech cues** | Lost in transcription | Captured (tone, prosody) |
| **Cost** | ~$0.004/task | ~$0.018/task |
| **Hardware** | Local GPU for ASR + API | API only |

## Tasks Evaluated

1. **Summarization** — ROUGE-L, BERTScore
2. **Sentiment Analysis** — Accuracy, F1, Confusion Matrix
3. **Keyword Extraction** — Precision, Recall, F1
4. **Intent Recognition** — Accuracy, Confidence Calibration

## Robustness Testing

3 noise types × 3 levels + clean baseline:
- Babble noise (20dB, 10dB, 0dB SNR)
- White noise (20dB, 10dB, 0dB SNR)  
- Reverberation (RT60: 0.5s, 1.0s, 1.5s)

## Project Structure

```
speech-benchmark/
├── src/
│   ├── config.py          # Central configuration
│   ├── data.py            # Audio I/O, noise injection, dataset prep
│   ├── cascade.py         # faster-whisper + GPT-4o-mini pipeline
│   ├── direct.py          # GPT-4o Audio mode pipeline
│   ├── evaluation.py      # Metrics and statistical tests
│   └── visualization.py   # Radar, degradation, error propagation plots
├── notebooks/
│   ├── 01_data_preparation.py   # Dataset loading and splitting
│   ├── 02_run_experiments.py    # Run both pipelines
│   ├── 03_evaluation.py         # Compute metrics and statistics
│   ├── 04_visualization.py      # Generate all figures
│   └── 05_case_studies.py       # Failure analysis
├── app/
│   └── gradio_app.py      # Interactive web demo
├── report/
│   ├── report.md           # Written report
│   └── figures/            # Generated plots
├── tests/
│   ├── test_data.py
│   ├── test_data_prep.py
│   ├── test_cascade.py
│   ├── test_direct.py
│   ├── test_evaluation.py
│   └── test_visualization.py
├── requirements.txt
└── .env.example
```

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set up OpenAI API key
cp .env.example .env
# Edit .env and add your key: OPENAI_API_KEY=sk-...

# 3. Run tests
python -m pytest tests/ -v

# 4. Run experiments (needs TED-LIUM or uses synthetic data)
cd notebooks
# Open 01_data_preparation.py in VSCode and run cells
# Open 02_run_experiments.py and run cells
# Open 03_evaluation.py and run cells
# Open 04_visualization.py and run cells

# 5. Launch demo
cd app
python gradio_app.py
```

## Requirements

- Python 3.10+
- NVIDIA GPU (optional, for local Whisper) — falls back to CPU
- OpenAI API key (for GPT-4o-mini and GPT-4o Audio)

## Reference

Based on the MSEB benchmark paradigm: *"Benchmarking LLMs on the Massive Sound Embedding Benchmark"* (Allauzen et al., 2025)

#  Speech Understanding Benchmark

[简体中文](README.zh-CN.md)

**Cascade (ASR Text LLM) vs End-to-End (Speech LLM) A Preliminary Comparison**

[![Tests](https://img.shields.io/badge/tests-31%20passed-brightgreen)](tests/)
[![Python](https://img.shields.io/badge/python-3.14-blue)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Repo](https://img.shields.io/badge/repo-github.com/jiayi0428/speech--benchmark-lightgrey)](https://github.com/jiayi0428/speech-benchmark)

> *Does removing the transcription bottleneck improve speech understanding*

An undergraduate summer research project (Jiayi Li, 2026) comparing two speech understanding architectures across 4 tasks with ground truth evaluation and noise robustness testing.

---

## Architecture

```
 Audio
     Cascade ("lego-block"):
   [faster-whisper large-v3] transcript [DeepSeek-chat API] output
   Local GPU +  API    ~$0.0005/task   ~16s
     Direct ("end-to-end"):
     [Qwen2-Audio-7B INT4] output
     Local GPU inference; structured tasks use text-API post-processing
     Original matched run: ~256s
```

---

## Key Findings (Preliminary, N=8)

| Dimension | Cascade | Direct | Winner |
|-----------|---------|--------|--------|
| **Speed (original matched run)** | ~16s | ~256s | Cascade (16x) |
| **Speech-model API cost** | ~$0.0005/task | Local inference | Direct |
| **Raw Structured Output** | **100% valid JSON** | 30% | Cascade |
| **Noise Robustness (0dB)** | Flatter change | Higher mean summary | No conclusive winner |
| **Emotion/Prosody** | Lost in transcription | Potentially available | Not directly tested |

>  **Pilot study with N=8 TTS samples.** See [Limitations](report/report.md#53-limitations) for transparency about scope.

---

## One-Click Reproduction

```bash
git clone <your-repo-url> speech-benchmark
cd speech-benchmark
pip install -r requirements.txt
cp .env.example .env  # Add your DEEPSEEK_API_KEY
python run_all.py     # Full pipeline: data cascade direct noise eval charts
```

Output: Experiment results in `data/results/`, 6 charts in `report/figures/`, report in `report/report.md`.

---

## Project Structure

```
speech-benchmark/
 run_all.py                   One-click reproduction
 src/                         Core pipeline (6 modules)
   cascade.py               # faster-whisper + DeepSeek-chat
   direct_qwen.py           # Qwen2-Audio-7B (local, INT4)
   data.py                  # Audio I/O, noise injection, dataset prep
   evaluation.py            # WER, ROUGE-L, F1, accuracy, t-test, bootstrap
   visualization.py         # Radar, degradation, error propagation plots
   config.py                # Auto-detects DeepSeek/OpenAI/Gemini/Qwen
 notebooks/                   Interactive exploration (5 notebooks)
 app/gradio_app.py            Live side-by-side comparison demo
 report/                      Paper + figures
   report.md                # Full research report (6 sections)
   figures/                 # Radar, latency, cost, degradation charts
 data/
   ground_truth.json        # Manually annotated reference labels
   results/                 # Experiment outputs
 tests/                       # 31 tests passing
```

---

## Tasks & Metrics

| Task | Metric | Description |
|------|--------|-------------|
| **Summarization** | ROUGE-L | Content overlap with ground truth summary |
| **Sentiment** | Accuracy | Positive / Negative / Neutral vs ground truth |
| **Keywords** | Precision, Recall, F1 | Overlap with ground truth key phrases |
| **Intent** | Accuracy | Inform / Persuade / Entertain / Question / Describe |

---

## Robustness Testing

| Noise Type | Levels | What It Simulates |
|-----------|--------|-------------------|
| White noise | 10dB, 0dB SNR | Baseline acoustic degradation |
| Babble noise | Framework ready | Crowded environments |
| Reverb | Framework ready | Large room acoustics |

---

## As-Recorded Human Speech Pilot

Cascade and Direct were additionally evaluated on 8 human-speech recordings
containing uncontrolled environmental and audience sounds.

| Task | Cascade | Direct |
|---|---:|---:|
| Summarization ROUGE-L | 0.2807 | 0.2388 |
| Sentiment accuracy | 75.0% | 62.5% |
| Keyword exact-phrase F1 | 0.4428 | 0.4167 |
| Intent accuracy | 62.5% | 25.0% |

Cascade had the higher mean on all four metrics, but every paired bootstrap
interval crossed zero at N=8. This is not a controlled noise experiment, and
latency came from different execution environments. See
[`experiments/HUMAN_SPEECH_V1.md`](experiments/HUMAN_SPEECH_V1.md) for the
workflow, paired results, API usage, and limitations.

---

## Evaluation Summary

| Metric | Cascade | Direct |
|--------|---------|--------|
| Sentiment Accuracy | 88% | 38% |
| Intent Accuracy | 88% | 62% |
| Keyword F1 | 0.36 | 0.29 |
| Summary ROUGE-L | 0.402 | 0.448 |

*See `data/results/final_summary.json` for complete metrics.*

---

## Quick Start (Demo Only)

```bash
cd app && python gradio_app.py
# Opens http://127.0.0.1:7860 upload audio, compare both pipelines live
```

---

## Limitations (Honest Scope)

1. **N=8 samples** Pilot study, not large-scale evaluation
2. **Main benchmark uses TTS** The supplemental human-speech pilot remains N=8 and has uncontrolled recording conditions
3. **Single model per paradigm** Qwen2-Audio-7B (INT4) and DeepSeek-chat only
4. **No human evaluation** Automated metrics against manually annotated ground truth
5. **Single noise type tested** Framework supports more (babble, reverb), not yet executed

Full limitations and future work in [report/report.md 5.3.4](report/report.md#53-limitations).

The paired human-speech results are reported separately in
[`report/human_speech_v1_report.md`](report/human_speech_v1_report.md).

---

## Author

**Jiayi Li** Undergraduate Summer Research, 2026

Built with Python 3.14, faster-whisper, DeepSeek API, Qwen2-Audio-7B, PyTorch, Gradio, and Matplotlib.

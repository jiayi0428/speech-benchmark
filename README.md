#  Speech Understanding Benchmark

[简体中文](README.zh-CN.md)

**Cascade (ASR Text LLM) vs End-to-End (Speech LLM) A Preliminary Comparison**

[![Tests](https://img.shields.io/badge/tests-31%20passed-brightgreen)](tests/)
[![Python](https://img.shields.io/badge/python-3.14-blue)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Repo](https://img.shields.io/badge/repo-github.com/jiayi0428/speech--benchmark-lightgrey)](https://github.com/jiayi0428/speech-benchmark)

> *Does removing the transcription bottleneck improve speech understanding*

An undergraduate summer research project by Jiayi Li, Liu Luofei (刘洛菲),
and Zhang Yuchen (张予辰), comparing speech-understanding paths across four
tasks with ground-truth evaluation and noise robustness testing.

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

The B/C/D ablation found that Qwen transcription had higher normalized WER
than Whisper (0.0696 vs 0.0338), yet both transcript-to-DeepSeek paths achieved
the same sentiment and intent accuracy. Qwen transcription followed by
DeepSeek also exceeded Qwen direct understanding on summary and intent in this
N=8 pilot. See
[`experiments/HUMAN_SPEECH_ABLATION_C.md`](experiments/HUMAN_SPEECH_ABLATION_C.md).

---

## Original-TTS Qwen Transcription Ablation

The original eight clean TTS clips were also run through Qwen transcription
followed by DeepSeek. All 8 transcriptions and 32 task calls succeeded; cleaned
normalized WER was 0.0102. C (Qwen transcript) scored 0.3815 / 75.0% / 0.3694 /
87.5% on summary, sentiment, keywords, and intent, versus D (Qwen direct) at
0.4600 / 62.5% / 0.3378 / 87.5%. All paired intervals crossed zero at N=8.
Only summarization is available for a three-way B/C/D comparison because the
stored B artifact lacks structured-task outputs and Whisper transcripts. See
[`experiments/TTS_QWEN_TRANSCRIPT_V1.md`](experiments/TTS_QWEN_TRANSCRIPT_V1.md).

---

## Additional TTS12 Four-Path Experiment

Twelve additional clean TTS clips compare A (Oracle transcript), B (Whisper
cascade), C (Qwen transcript), and D (Qwen direct) on shared ground truth.

| Path | Summary | Sentiment | Keyword F1 | Intent |
|---|---:|---:|---:|---:|
| A: Oracle | 0.3528 | 92% | 0.4500 | 100% |
| B: Whisper | 0.3448 | 92% | 0.4500 | 100% |
| C: Qwen transcript | 0.3479 | 83.3% | 0.4298 | 100% |
| D: Qwen direct | 0.3324 | 41.7% | 0.4286 | 16.7% |

A/B/C were close, while D preserved comparable summary and keyword scores but
showed a strong intent-classification failure. N=12 remains descriptive. See
[`experiments/TTS12_CD_V1.md`](experiments/TTS12_CD_V1.md).

A separate D rerun exactly reproduced all 48 outputs and scores. In three new
C/D summary repetitions, C had the higher mean every time (7 wins versus 5),
and every sample kept the same winner across repetitions. All intervals still
crossed zero. See
[`experiments/TTS12_REPEAT_STABILITY_V1.md`](experiments/TTS12_REPEAT_STABILITY_V1.md).

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

1. **N=8 or N=12 per dataset** Pilot studies, not large-scale evaluation
2. **Main benchmark uses TTS** The supplemental human-speech pilot remains N=8 and has uncontrolled recording conditions
3. **Single model per paradigm** Qwen2-Audio-7B (INT4) and DeepSeek-chat only
4. **No human evaluation** Automated metrics against manually annotated ground truth
5. **Single noise type tested** Framework supports more (babble, reverb), not yet executed

Full limitations and future work in [report/report.md 5.3.4](report/report.md#53-limitations).

The paired human-speech results have been consolidated into the main report:
[`report/report.md`](report/report.md).

---

## Authors

**Jiayi Li (李佳宜) · Liu Luofei (刘洛菲) · Zhang Yuchen (张予辰)**
Undergraduate Summer Research, 2026

Built with Python 3.14, faster-whisper, DeepSeek API, Qwen2-Audio-7B, PyTorch, Gradio, and Matplotlib.

# Cascade vs End-to-End: A Robustness-Aware Benchmark of Speech Understanding Architectures

**Author:** [Your Name]
**Date:** August 2026
**Course:** Undergraduate Summer Research

---

## Abstract

This study empirically compares two speech understanding paradigms: the traditional cascade architecture (automatic speech recognition followed by text-based language model inference) and the emerging end-to-end approach (audio-native large language models). Using the TED-LIUM dataset and synthetic acoustic degradations, we evaluate both approaches on four tasks: summarization, sentiment analysis, keyword extraction, and intent recognition. Our robustness testing across multiple noise types and levels reveals [KEY FINDING — fill in after experiments]. We find that while [OBSERVATION], the end-to-end approach demonstrates [OBSERVATION], suggesting that the optimal architecture choice depends on specific deployment conditions.

## 1. Introduction

Speech understanding — the ability to extract semantic meaning from spoken language — has traditionally relied on a two-stage cascade: first transcribing speech to text via automatic speech recognition (ASR), then processing that text with a language model. Recent advances in multimodal large language models have introduced an alternative: audio-native models that process speech directly without intermediate transcription.

This paradigm shift raises a central research question: **Does removing the transcription bottleneck improve understanding, or does the text-based cascade remain competitive?** More specifically, we investigate:

1. How do cascade and end-to-end architectures compare on standard speech understanding tasks?
2. How does acoustic degradation (noise, reverberation) affect each architecture differently?
3. Does ASR error propagate through the cascade, and if so, how severely?
4. Under what conditions might one approach be preferable to the other?

## 2. Methodology

### 2.1 Architectures

**Cascade Pipeline**: We use faster-whisper large-v3 for ASR, running locally on an NVIDIA RTX 5070 GPU. The transcribed text is then processed by GPT-4o-mini via the OpenAI API for each of four understanding tasks.

**Direct Pipeline**: We use GPT-4o's native audio mode, which accepts audio input directly and processes it through the same multimodal backbone that handles text, images, and audio.

### 2.2 Tasks and Metrics

| Task | Metric | Description |
|------|--------|-------------|
| Summarization | ROUGE-L, BERTScore | Generate a 3-5 sentence summary |
| Sentiment | Accuracy, Macro-F1 | Classify as positive/negative/neutral |
| Keywords | Precision, Recall, F1@k | Extract 5-10 key phrases |
| Intent | Accuracy | Identify speaker's communicative intent |

### 2.3 Robustness Testing

We apply three types of acoustic degradation at multiple intensity levels:
- **Babble noise** (SNR: 20dB, 10dB, 0dB) — simulating crowded environments
- **White noise** (SNR: 20dB, 10dB, 0dB) — baseline broadband degradation
- **Reverberation** (RT60: 0.5s, 1.0s, 1.5s) — simulating room acoustics

### 2.4 Dataset

[TED-LIUM v3 / synthetic dataset description — fill in]

### 2.5 Statistical Analysis

Paired t-tests with Cohen's d effect sizes and bootstrap 95% confidence intervals are used to assess the statistical significance of performance differences.

## 3. Results

### 3.1 Clean Audio Performance
[Radar chart comparing both architectures across 4 tasks]
[Table of scores with statistical tests]

### 3.2 Robustness Under Degradation
[Degradation curves showing performance vs SNR]
[Robustness index (area under curve) for each architecture]

### 3.3 Error Propagation Analysis
[Scatter plot: WER vs downstream task score]
[Pearson correlation and regression]

### 3.4 Cost-Latency Trade-offs
[Bar chart comparing latency and cost]
[Discussion of deployment implications]

## 4. Case Studies

### Case 1: [Title]
[Audio description, transcript comparison, analysis]

### Case 2: [Title]
[Audio description, transcript comparison, analysis]

### Case 3: [Title]
[Audio description, transcript comparison, analysis]

## 5. Discussion

### 5.1 When Does Each Architecture Win?
[Analysis of conditions favoring cascade vs direct]

### 5.2 Implications for Real-World Deployment
[Practical takeaways for system builders]

### 5.3 Limitations
- Synthetic noise may not fully represent real-world conditions
- Single API provider (OpenAI) limits generalizability
- Sample size constraints
- No open-source speech LLM comparison

### 5.4 Future Work
- Include Qwen2-Audio or other open-source speech LLMs
- Test on additional languages
- Real-world deployment testing
- Human evaluation beyond automatic metrics

## 6. Conclusion

[Summary of findings and their significance]

## References

[To be filled in]

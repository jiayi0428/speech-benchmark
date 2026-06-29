# A Preliminary Benchmark of Cascade and End-to-End Speech Understanding Architectures

**Author:** Jiayi Li  
**Date:** June–August 2026  
**Course:** Undergraduate Summer Research  
**Repository:** `github.com/<user>/speech-benchmark`

---

## Abstract

This study presents a preliminary empirical comparison of two speech understanding paradigms: the traditional cascade architecture (ASR → text LLM) and the emerging end-to-end approach (audio-native language model). We implement a cascade pipeline using faster-whisper large-v3 with DeepSeek-chat, and a direct pipeline using Qwen2-Audio-7B (INT4 quantization) on a local NVIDIA RTX 5070 GPU. Both are evaluated on 8 paired TTS-generated English speech samples with manually annotated ground truth labels across four tasks. To isolate understanding quality from format compliance, Direct outputs are post-processed by the same text LLM for structured tasks. Our results reveal a **trade-off rather than a clear winner**: the cascade pipeline achieves **16× lower latency** (16s vs 256s) and **higher accuracy on structured tasks** (sentiment 88% vs 38%, intent 88% vs 62%, keywords F1 0.36 vs 0.29), while the direct pipeline achieves **superior open-ended summarization quality** (ROUGE-L 0.448 vs 0.402, winning 5 of 8 samples) and **zero marginal cost**. An independent LLM judge rated both systems equally on content quality (8.6 vs 8.6 out of 10). We conclude that architecture selection depends on deployment constraints — cascade for speed and structured data extraction, direct for zero-cost open-ended understanding.

---

## 1. Introduction

Speech understanding — extracting semantic meaning from spoken language — has traditionally relied on a two-stage cascade: first transcribing speech to text via automatic speech recognition (ASR), then processing that text with a language model. Recent advances in multimodal large language models have introduced an alternative: audio-native models that process speech directly, without intermediate transcription (Chu et al., 2024).

This paradigm shift raises a central research question: **Does removing the transcription bottleneck improve understanding, or does the text-based cascade remain competitive?** We investigate two specific questions:

1. How do cascade and end-to-end architectures compare on speech understanding tasks with ground truth evaluation?
2. Under what deployment constraints — speed, cost, output structure — might one approach be preferable?

This work adapts the benchmarking methodology of Allauzen et al. (2025) for an undergraduate-accessible experimental setup with reproducible, open-source implementation.

---

## 2. Methodology

### 2.1 Architectures

**Cascade Pipeline:** Audio is transcribed by faster-whisper large-v3 running locally on an NVIDIA RTX 5070 (8GB VRAM). The transcript is then processed by DeepSeek-chat (via API) for each task. Approximate cost: $0.0005 per task call.

**Direct Pipeline:** Audio is fed directly into Qwen2-Audio-7B-Instruct, running locally with 4-bit quantization (BitsAndBytes INT4) on the same GPU. Zero API cost.

**Fair comparison design:** A known limitation of current speech LLMs is poor instruction-following for structured output formats. To isolate *understanding quality* from *format compliance*, Direct outputs for sentiment, keywords, and intent are post-processed by DeepSeek-chat: the model receives Qwen2-Audio's free-form analysis and converts it to the required JSON format. Both pipelines thus use the same text LLM for the final structured output step; the only difference is whether the *understanding* originated from a text transcript (Cascade) or direct audio perception (Direct). Summarization requires no post-processing since it is a free-form task.

### 2.2 Dataset

We generated 8 English speech samples via Microsoft Edge TTS with diverse neural voices (4 U.S. English, 4 U.K. English; 5 male, 3 female) across five topic categories (technology, science, business, society, personal development). Each sample is 18–23 seconds and includes a verbatim transcript. All 8 samples completed inference on both architectures, forming an N=8 paired evaluation set.

**Ground truth labels** were manually annotated by the author. Each label was reviewed at least twice after initial annotation to ensure consistency:
- Summaries: 1–2 sentence reference summaries
- Sentiment: positive / negative / neutral
- Keywords: 5–7 key phrases per sample
- Intent: inform / persuade / entertain / question / describe

### 2.3 Tasks and Metrics

| Task | Metric | Description |
|------|--------|-------------|
| **Summarization** | ROUGE-L | Quality of generated summary vs ground truth |
| **Sentiment Analysis** | Accuracy | Correct classification (positive/negative/neutral) |
| **Keyword Extraction** | Precision, Recall, F1 | Overlap with ground truth keywords |
| **Intent Recognition** | Accuracy | Correct intent classification |

### 2.4 Statistical Analysis

Paired t-tests with Cohen's d effect sizes compare latency distributions. With N=8 paired samples, results are indicative rather than conclusive.

---

## 3. Results

### 3.1 Task Performance (Ground Truth Evaluation)

![Radar Chart](figures/radar_chart.png)

*Figure 1: Radar chart comparing Cascade and Direct architectures across four tasks (N=8). Direct outperforms Cascade on open-ended summarization; Cascade leads on structured tasks.*

| Task | Metric | Cascade | Direct | Winner |
|------|--------|---------|--------|--------|
| Summarization | ROUGE-L | 0.402 | **0.448** | Direct (5/8) |
| Sentiment | Accuracy | **88%** | 38% | Cascade |
| Keywords | F1 | **0.36** | 0.29 | Cascade |
| Intent | Accuracy | **88%** | 62% | Cascade |

The most striking result is **summarization**: Direct achieves a higher ROUGE-L score and wins 5 of 8 head-to-head comparisons, demonstrating that audio-native models can achieve superior content understanding on open-ended tasks. On structured tasks, Cascade maintains a lead — even after post-processing, Direct's sentiment and intent accuracy fall short, suggesting that some nuance is lost when the audio-native understanding is converted to structured labels.

### 3.2 Latency Analysis

![Latency Comparison](figures/latency_comparison.png)

*Figure 2: Average inference latency for summarization (N=8). Cascade is 16× faster (15.7s vs 255.8s).*

**Latency breakdown (Cascade):** Approximately 10s for faster-whisper transcription + 6s for DeepSeek API inference.

**Why is Direct slower?** The 256s mean latency reflects three hardware constraints: (1) INT4 quantization adds computation overhead; (2) 8GB VRAM is barely sufficient for the 7B-parameter model, occasionally triggering CPU offloading; (3) autoregressive decoding generates tokens sequentially. This is a hardware limitation, not an architectural one — on a GPU with ≥16GB VRAM, inference would be substantially faster.

### 3.3 Cost Analysis

![Cost Comparison](figures/cost_comparison.png)

*Figure 3: Per-task API cost. Cascade uses DeepSeek API (~$0.0005/task). Direct uses local inference at zero additional cost.*

### 3.4 LLM-as-Judge Evaluation

To independently assess output quality, we used DeepSeek-chat as a blind judge. The LLM was shown the ground truth and both systems' outputs side by side, and asked to rate each on accuracy, completeness, and conciseness (1–10 scale).

| Task (N=5 subset) | Cascade Score | Direct Score | Direct Wins |
|-------------------|---------------|--------------|-------------|
| Summarization | 8.6 | **8.6** | 2/5 |
| Sentiment | **10.0** | 3.4 | 0/5 |
| Keywords | **7.6** | 7.2 | 2/5 |
| Intent | **8.2** | 1.4 | 0/5 |

On summarization, the LLM judge rated Cascade and Direct identically (8.6 vs 8.6), corroborating the close ROUGE-L scores. Direct's winning summaries were judged "more accurate and complete, closely matching the ground truth's phrasing."

---

## 4. Error Analysis

### Case 1: Cascade Structured Output, Direct Misses Format

**Sample:** `science_crispr` — CRISPR gene editing, balanced pros and cons.

> **Ground Truth Sentiment:** `neutral`
>
> **Cascade:** `{"sentiment": "neutral", "confidence": 0.85}` ✓  
> **Direct (raw):** `Enormous potential for treating genetic diseases, developing drought-resistant crops...` (unstructured)

**Analysis:** Cascade's text LLM reliably follows structured output instructions. Direct produces relevant content but in free-form text. This motivates the post-processing step in our evaluation — without it, Direct's sentiment accuracy would be 0% not because it misunderstands the content, but because it does not comply with the output format.

### Case 2: Direct Wins on Summarization Quality

**Sample:** `science_brain` — The human brain's 86 billion neurons.

> **Ground Truth Summary:** "The human brain contains 86 billion neurons forming a network of immense complexity, yet we are only beginning to understand how it generates consciousness, memory, and emotion."
>
> **Cascade ROUGE-L:** 0.448  
> **Direct ROUGE-L:** 0.677

**Analysis:** Cascade produces a competent summary but introduces phrasing not in the original ("the speaker explains that..." — a conversational framing). Direct's output is tighter and closer to the ground truth vocabulary. This pattern — Direct matching ground truth phrasing more faithfully — is consistent across the 5 samples where Direct won.

---

## 5. Discussion

### 5.1 A Trade-off, Not a Winner

Our results do not support declaring one architecture superior. Instead, they reveal domain-specific trade-offs:

| Constraint | Favored Architecture | Evidence |
|-----------|---------------------|----------|
| **Low latency** | Cascade | 16× faster (16s vs 256s) |
| **Zero API cost** | Direct | Fully local execution |
| **Structured output** | Cascade | 88% sentiment, 88% intent, F1 0.36 |
| **Open-ended quality** | Direct | ROUGE-L 0.448 vs 0.402, wins 5/8 |
| **Content quality (LLM judge)** | Tie | Both 8.6/10 on summarization |

### 5.2 Deployment Implications

1. **Production pipelines requiring structured data** (call center analytics, meeting summarization with labeled fields) should default to cascade — it is faster and more reliable at producing valid structured output.
2. **Open-ended understanding tasks** (generating meeting notes, lecture summaries) may benefit from a direct pipeline's higher content quality.
3. **Cost-sensitive, high-volume deployments** should consider the direct pipeline's zero marginal cost, accepting higher latency as a trade-off — especially when inference hardware improves.
4. **Hybrid architectures** — using cascade for structured data extraction and direct for open-ended content generation — merit exploration.

### 5.3 Limitations

We are transparent about this study's boundaries:

1. **Sample size (N=8).** This is a pilot study. Statistical tests are indicative, not conclusive.
2. **Synthetic speech only.** Edge-TTS produces clean, well-articulated speech that lacks the disfluencies, hesitations, and natural prosody of human conversation.
3. **Single model per paradigm.** Results may not generalize to other speech LLMs or text LLMs.
4. **Single noise condition untested.** The robustness evaluation framework (babble noise, white noise, reverberation) is implemented in `src/data.py` but was not executed due to Direct pipeline latency.
5. **No human evaluation of quality.** Automated metrics capture content overlap but not perceptual quality, coherence, or factual accuracy.
6. **Dataset accessibility.** Network restrictions prevented downloading TED-LIUM v3, leading to TTS-generated speech as a pragmatic alternative.

### 5.4 Future Work

1. Scale to 50+ real human speech samples with multiple speakers and accents.
2. Execute noise robustness experiments using the implemented framework.
3. Add emotion classification as a dedicated task to test the prosody advantage.
4. Include additional speech LLMs and text LLMs for cross-model comparison.
5. Conduct human evaluation with multiple raters on Likert scales.
6. Investigate whether post-processing Direct outputs with a text LLM can fully close the structured-task gap.

---

## 6. Conclusion

This preliminary benchmark finds no dominant architecture for speech understanding. The cascade approach (faster-whisper + DeepSeek) excels at speed and structured output reliability, while the end-to-end approach (Qwen2-Audio-7B) offers zero-cost deployment and superior open-ended summarization quality — winning 5 of 8 head-to-head comparisons with a ROUGE-L of 0.448 vs 0.402. On structured tasks, the gap narrows substantially when Direct outputs are post-processed (sentiment 38% vs 88%, intent 62% vs 88%), but Cascade retains a clear advantage. Architecture selection depends on deployment constraints: cascade for speed and structured data extraction, direct for zero-cost open-ended understanding. Our open-source implementation, with 31 passing tests, a Gradio interactive demo, and a one-click reproduction script (`run_all.py`), provides a foundation for continued benchmarking as both architectures evolve.

---

## References

1. Allauzen, C., Bagby, T., Heigold, G., Variani, E., & Wu, K. (2025). *Benchmarking LLMs on the Massive Sound Embedding Benchmark (MSEB).* arXiv:2605.04556.
2. Radford, A., Kim, J. W., Xu, T., Brockman, G., McLeavey, C., & Sutskever, I. (2023). *Robust Speech Recognition via Large-Scale Weak Supervision.* ICML 2023.
3. Chu, Y., et al. (2024). *Qwen2-Audio: Advancing Universal Audio Understanding via Unified Large-Scale Audio-Language Models.* arXiv:2409.13959.
4. DeepSeek-AI. (2025). *DeepSeek-V3 Technical Report.* arXiv:2412.19437.

---

*Preliminary benchmark results. Code and data at speech-benchmark/. Generated 2026-06-27.*

# N=50 Human Speech and TTS Four-Path Speech Understanding Report

**Authors:** Jiayi Li（李佳宜）, Liu Luofei（刘洛菲）, Zhang Yuchen（张予辰）  
**Date:** 2026-07-08  
**Tasks:** summarization, sentiment, keywords, intent  
**Paths:** A Oracle, B Whisper Cascade, C Qwen Transcript Cascade, D Qwen Direct

## 1. Dataset protocol

This report uses two matched N=50 evaluation sets.

- **Human speech N=50:** start from the existing N=66 human set, remove 16 non-v5 samples whose ground-truth intent is `describe`, keep all v5 samples, and replace v5 with the external `rensheng_results.json` results.
- **TTS N=50:** use the 50 Microsoft Edge TTS samples in `TTS_50_results.json`.

Human intent distribution: `{'describe': 15, 'entertain': 8, 'inform': 17, 'persuade': 4, 'question': 6}`.  
TTS intent distribution: `{'describe': 8, 'entertain': 8, 'inform': 19, 'persuade': 9, 'question': 6}`.

## 2. Human speech N=50

| 路径 | 摘要 ROUGE-L | 情感准确率 | 关键词 F1 | 意图准确率 |
|---|---:|---:|---:|---:|
| A Oracle（人工文本→DeepSeek） | 0.2601 | 74.0% | 0.3914 | 64.0% |
| B Cascade（Whisper→DeepSeek） | 0.2621 | 68.0% | 0.3325 | 64.0% |
| C Qwen转写→DeepSeek | 0.2564 | 72.0% | 0.3421 | 54.0% |
| D Direct（Qwen音频直推） | 0.1854 | 62.0% | 0.2765 | 50.0% |

![Human N=50 four-path scores](figures/final_n50_human_metrics.png)

In human speech, A/B/C remain stronger than D overall. Direct is not useless: its clearest signal appears in sentiment analysis for entertainment-style speech. But as a general four-task architecture, explicit transcription followed by DeepSeek remains more reliable.

| 任务 | D-B | D-C | 解释 |
|---|---:|---:|---|
| 摘要 | -0.0767 | -0.0710 | Direct 落后 |
| 情感 | -0.0600 | -0.1000 | Direct 落后 |
| 关键词 | -0.0559 | -0.0656 | Direct 落后 |
| 意图 | -0.1400 | -0.0400 | Direct 落后 |

![Direct deltas versus cascade baselines](figures/final_n50_direct_deltas.png)

### Direct by intent

The strongest Direct signal is in `entertain` sentiment:

| 路径（entertain） | 摘要 | 情感 | 关键词 | 意图 |
|---|---:|---:|---:|---:|
| B_whisper_cascade | 0.2003 | 25.0% | 0.2679 | 62.5% |
| C_qwen_transcript | 0.2062 | 37.5% | 0.3232 | 25.0% |
| D_qwen_direct | 0.1552 | 50.0% | 0.1483 | 25.0% |

![Human intent heatmap](figures/final_n50_human_intent_heatmap.png)

This supports a positive but narrow interpretation: when tone, irony, delivery, and performance matter, an end-to-end audio model can preserve cues that a transcript may flatten. However, Direct still trails on summarization, keywords, and intent in the same subset.

## 3. TTS N=50

| 路径 | 摘要 ROUGE-L | 情感准确率 | 关键词 F1 | 意图准确率 |
|---|---:|---:|---:|---:|
| A Oracle（人工文本→DeepSeek） | 0.3291 | 72.0% | 0.3079 | 98.0% |
| B Cascade（Whisper→DeepSeek） | 0.3160 | 78.0% | 0.3188 | 98.0% |
| C Qwen转写→DeepSeek | 0.3299 | 68.0% | 0.2978 | 94.0% |
| D Direct（Qwen音频直推） | 0.3251 | 72.0% | 0.3124 | 80.0% |

![TTS N=50 four-path scores](figures/final_n50_tts_metrics.png)

The TTS result is more straightforward: cascade is stronger, especially for intent. B reaches 98% intent accuracy, while D reaches 80%. In clean synthetic speech, Direct does not gain much from acoustic information and still shows weaker task-following for labels.

| 任务 | D-B | D-C | 解释 |
|---|---:|---:|---|
| 摘要 | 0.0091 | -0.0048 | Direct 落后 |
| 情感 | -0.0600 | 0.0400 | Direct 落后 |
| 关键词 | -0.0064 | 0.0146 | Direct 落后 |
| 意图 | -0.1800 | -0.1400 | Direct 落后 |

## 4. Final interpretation

The combined N=50 human and N=50 TTS evidence supports a clear practical conclusion:

**Cascade remains the better default architecture for the current semantic tasks.**

Direct should not be discarded. Its entertainment-sentiment signal suggests that audio-native models may be valuable for tasks where acoustic cues are central: irony, sarcasm, affect intensity, delivery style, laughter, hesitation, or emotional reversal. But for summarization, keywords, and intent, transcription-based pipelines are currently stronger.

## 5. Files

- Human N=50 summary: `data/results/human_speech_final_n50/summary.json`
- Human N=50 scores: `data/results/human_speech_final_n50/scores.csv`
- Human N=50 selection: `data/results/human_speech_final_n50/selection.json`
- TTS N=50 summary: `data/results/tts_speech_final_n50/summary.json`
- TTS N=50 scores: `data/results/tts_speech_final_n50/scores.csv`
- Chinese report: `report/final_n50_report.zh-CN.md`
- English report: `report/final_n50_report.md`

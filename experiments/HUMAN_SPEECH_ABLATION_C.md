# Human Speech Ablation C: Qwen Transcription to DeepSeek

## Goal

This ablation tests a cascade-like path that uses Qwen2-Audio as the
transcription model:

```text
Audio -> Qwen2-Audio transcription -> DeepSeek task output
```

It is compared with:

- B: Whisper transcription -> DeepSeek
- D: Audio -> Qwen2-Audio direct understanding -> DeepSeek formatting

## Stage 1: Local transcription

```powershell
.\.venv\Scripts\python.exe run_qwen_transcription.py `
  --model-path D:\path\to\Qwen2-Audio-7B-Instruct
.\.venv\Scripts\python.exe evaluate_qwen_transcription.py
```

Expected output: 8 successful transcripts plus Qwen/Whisper WER comparison.
This stage is local and does not call a paid API.

## Stage 2: Text-task inference

After explicit API cost approval, run the same four DeepSeek task prompts used
by the existing Cascade path on each Qwen transcript. Expected scope:
8 samples x 4 tasks = 32 API calls.

## Files

- `data/results/human_speech_v1/qwen_transcription_raw.jsonl`
- `data/results/human_speech_v1/qwen_transcription_scores.csv`
- `data/results/human_speech_v1/qwen_transcription_summary.json`
- `data/results/human_speech_v1/qwen_transcript_cascade_raw.jsonl`
- `data/results/human_speech_v1/qwen_transcript_cascade_call_audit.jsonl`
- `data/results/human_speech_v1/qwen_transcript_cascade_audit_summary.json`
- `data/results/human_speech_v1/bcd_ablation_scores.csv`
- `data/results/human_speech_v1/bcd_ablation_summary.json`

## Completed results

| Path | Summary ROUGE-L | Sentiment accuracy | Keyword F1 | Intent accuracy |
|---|---:|---:|---:|---:|
| B: Whisper transcript -> DeepSeek | 0.2807 | 75.0% | 0.4428 | 62.5% |
| C: Qwen transcript -> DeepSeek | 0.3064 | 75.0% | 0.3708 | 62.5% |
| D: Qwen direct understanding | 0.2388 | 62.5% | 0.4167 | 25.0% |

Qwen transcription had higher WER than Whisper. After deterministic removal of
fixed meta prefixes and punctuation normalization, mean WER was 0.0696 for
Qwen and 0.0338 for Whisper. Despite that gap, B and C produced identical
sentiment and intent correctness on all eight samples. C had higher mean
summary ROUGE-L but lower keyword F1 than B.

C beat D on 6/8 summarization samples. The paired bootstrap interval for
Direct minus C on summarization was [-0.1506, -0.0010]. This narrow interval
excluded zero in this resampling analysis, but N=8 and multiple comparisons
still preclude a broad statistical claim.

This is not a pure transcription-bottleneck test. C delegates semantic tasks
to DeepSeek after transcription, while D performs semantic understanding in
Qwen and uses DeepSeek only for structured formatting.

## API audit

The canonical C result contains 32 unique sample/task responses and 4,615
tokens. A timed-out process continued running while manual resumptions were
started, producing 20 duplicate calls. The complete 52-call, 7,541-token audit
is preserved. At the project's historical $0.0005-per-call assumption, the
estimated all-call cost is $0.026 rather than the planned $0.016.

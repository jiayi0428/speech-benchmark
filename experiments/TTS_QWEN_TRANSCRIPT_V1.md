# TTS Qwen Transcription to DeepSeek V1

## Goal

Run the original eight clean TTS clips through:

```text
Audio -> Qwen2-Audio transcription -> DeepSeek task output
```

This is the TTS counterpart of human-speech ablation C.

## Stage 1: Local transcription

```powershell
.\.venv\Scripts\python.exe run_tts_qwen_transcription.py `
  --model-path D:\path\to\Qwen2-Audio-7B-Instruct
.\.venv\Scripts\python.exe evaluate_tts_qwen_transcription.py
```

Expected output: 8 successful Qwen transcripts and WER against the original
TTS scripts. This stage does not call a paid API.

## Stage 2: DeepSeek tasks

After explicit cost approval, apply the existing Cascade prompts to every Qwen
transcript:

```text
8 samples x 4 tasks = 32 DeepSeek calls
```

Calls must run in synchronous batches of four. Do not launch a second batch
until the previous process has exited.

## Comparison boundary

- Full C versus D comparison is available for all four tasks using the
  white-noise experiment's `clean` Direct records.
- B/C/D comparison is available for summarization only.
- Stored Cascade TTS results do not contain Whisper transcripts or
  sample-level structured-task outputs.

## Completed results

All eight local transcriptions and all 32 DeepSeek task calls succeeded. Qwen
transcription achieved cleaned normalized WER 0.0102 against the source TTS
scripts (mean local transcription latency 4.394 seconds).

| Path | Summary ROUGE-L | Sentiment accuracy | Keyword F1 | Intent accuracy |
|---|---:|---:|---:|---:|
| B: Whisper cascade | 0.3971 | -- | -- | -- |
| C: Qwen transcript -> DeepSeek | 0.3815 | 75.0% | 0.3694 | 87.5% |
| D: Qwen direct | 0.4600 | 62.5% | 0.3378 | 87.5% |

For C minus D, the mean differences were -0.0785 for summarization, +0.125
for sentiment, +0.0316 for keywords, and 0 for intent. All paired bootstrap
95% intervals crossed zero. D won 5/8 summaries; C had more wins on sentiment
(3/8 versus 2/8, with three ties) and keywords (6/8 versus 2/8); intent had
one win per path and six ties.

The only task shared by all B/C/D stored outputs is summarization. B, C, and D
scored 0.3971, 0.3815, and 0.4600 respectively. The C-minus-B and D-minus-B
paired intervals also crossed zero.

The 32 successful DeepSeek calls used 3,328 prompt tokens and 972 completion
tokens (4,300 total). Using the project's historical per-call approximation,
the estimated cost is USD 0.016. No model was downloaded.

## Artifacts

- `data/results/tts_qwen_transcript_v1/qwen_transcription_raw.jsonl`
- `data/results/tts_qwen_transcript_v1/qwen_transcription_scores.csv`
- `data/results/tts_qwen_transcript_v1/qwen_transcription_summary.json`
- `data/results/tts_qwen_transcript_v1/qwen_transcript_cascade_raw.jsonl`
- `data/results/tts_qwen_transcript_v1/comparison_scores.csv`
- `data/results/tts_qwen_transcript_v1/comparison_summary.json`
- `report/figures/tts_qwen_transcript_comparison.png`

These N=8 results are descriptive only. B and D were reused from the earlier
white-noise experiment's clean condition, while C was run later. Latencies
must not be interpreted as architectural speed differences.

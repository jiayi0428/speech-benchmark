# TTS12 A/B/C/D V1

## Scope

Twelve clean, mono, 16 kHz TTS clips with supplied transcripts and task
ground truth. This experiment is kept separate from the original eight-sample
TTS benchmark.

- C: audio -> Qwen2-Audio transcription -> DeepSeek four tasks
- D: audio -> Qwen2-Audio direct task understanding -> DeepSeek formatting
  for sentiment, keywords, and intent

Supplied comparison results add:

- A: ground-truth transcript -> DeepSeek four tasks
- B: audio -> Whisper transcription -> DeepSeek four tasks

## Reproduction

```powershell
python prepare_tts12_cd.py --source D:\university\暑研\TTS\TTS
python run_tts12_cd.py --stage transcription --model-path D:\path\to\Qwen2-Audio-7B-Instruct
python run_tts12_cd.py --stage direct --model-path D:\path\to\Qwen2-Audio-7B-Instruct
python postprocess_tts12_cd.py --stage c
python postprocess_tts12_cd.py --stage d
```

The first three commands do not use a paid API. The final two require explicit
approval because they make 48 and 36 DeepSeek calls respectively.

## Comparison boundary

The same 12 audio files, ground truth, Qwen model, and task definitions are
used for C and D. C and D still differ in semantic task allocation: DeepSeek
performs C's tasks from text, while Qwen performs D's semantic understanding
and DeepSeek only formats its structured outputs. N=12 remains a pilot sample
and does not justify claims of statistical significance.

## Results

| Path | Summary ROUGE-L | Sentiment accuracy | Keyword F1 | Intent accuracy |
|---|---:|---:|---:|---:|
| A: Oracle transcript | 0.3528 | 92% | 0.4500 | 100% |
| B: Whisper cascade | 0.3448 | 92% | 0.4500 | 100% |
| C: Qwen transcript | 0.3479 | 83.3% | 0.4298 | 100% |
| D: Qwen direct | 0.3324 | 41.7% | 0.4286 | 16.7% |

All six pairwise summary bootstrap intervals crossed zero. A, B, and C were
close on every reported metric. D was also close on summary and keyword F1,
but substantially lower on sentiment and intent. In the raw Direct intent
outputs, 10 of 12 informational clips were labeled `persuade`.

The supplied A/B artifact contains sample-level summary scores but only
rounded aggregate values for the three structured tasks. Therefore,
sample-level structured comparisons involving A or B are unavailable. C and D
were recomputed from raw outputs using the project scoring code.

Qwen transcription mean normalized WER was 0.0087. The 84 successful DeepSeek
calls used 10,469 total tokens (C: 6,816; D: 3,653), with a historical
per-call cost estimate of USD 0.042. Twelve connection-error audit rows came
from a sandbox-blocked attempt and contain no API response.

## Artifacts

- `data/processed/tts12_cd_v1/index.json`
- `data/ground_truth_tts12_cd_v1.json`
- `data/results/tts12_cd_v1/ab_oracle_vs_cascade.txt`
- `data/results/tts12_cd_v1/qwen_transcription_raw.jsonl`
- `data/results/tts12_cd_v1/c_tasks_raw.jsonl`
- `data/results/tts12_cd_v1/direct_raw.jsonl`
- `data/results/tts12_cd_v1/direct_postprocessed.jsonl`
- `data/results/tts12_cd_v1/transcription_scores.csv`
- `data/results/tts12_cd_v1/scores.csv`
- `data/results/tts12_cd_v1/summary.json`
- `report/figures/tts12_abcd_comparison.png`

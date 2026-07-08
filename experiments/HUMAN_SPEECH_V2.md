# Human Speech V2: C/D Experiment

## Scope

Ten supplied as-recorded human English speech clips with transcript, summary,
sentiment, keyword, and intent annotations.

- C: audio -> Qwen2-Audio transcription -> DeepSeek four tasks
- D: audio -> Qwen2-Audio direct task understanding -> DeepSeek formatting
  for sentiment, keywords, and intent

The original 44.1 kHz stereo recordings are copied into the ignored raw-data
directory. Prepared experiment audio is 16 kHz mono, with no denoising and no
peak normalization.

## Cost boundary

Local Qwen stages do not use a paid API. The paid stages require 40 DeepSeek
calls for C and 30 calls for D, subject to explicit approval.

## Caveats

N=10 supports descriptive trends only. The clips are not content-matched to
the TTS datasets, so human-versus-TTS differences cannot be attributed solely
to voice type.

## Completed results

All 10 Qwen transcriptions, 40 C task calls, 40 D local task generations, and
30 D formatting calls succeeded.

| Path | Summary ROUGE-L | Sentiment accuracy | Keyword F1 | Intent accuracy |
|---|---:|---:|---:|---:|
| C: Qwen transcript -> DeepSeek | 0.1592 | 60% | 0.4011 | 60% |
| D: Qwen direct | 0.1726 | 20% | 0.3782 | 40% |

D had the higher summary mean by 0.0134, with five wins per path; the paired
interval was [-0.0598, 0.0321]. C exceeded D on sentiment by 40 percentage
points, keywords by 0.0229, and intent by 20 percentage points. Only the
sentiment interval did not cross zero in this bootstrap run, but N=10 and
label imbalance preclude a general significance claim.

Ground-truth sentiment contains seven neutral, two negative, and one positive
sample. C predicted six negative labels, while D predicted six positive
labels. Ground-truth intent contains five describe, four inform, and one
persuade sample. C predicted eight inform labels; D predicted six persuade
labels. The observed differences therefore include strong path-specific label
biases.

Qwen mean normalized WER was 0.0351. The 70 successful DeepSeek calls used
8,827 total tokens (C: 5,828; D: 2,999), historically estimated at USD 0.035.

## Artifacts

- `data/ground_truth_human_v2.json`
- `data/processed/human_speech_v2/audio_manifest.json`
- `data/results/human_speech_v2/qwen_transcription_raw.jsonl`
- `data/results/human_speech_v2/c_tasks_raw.jsonl`
- `data/results/human_speech_v2/direct_raw.jsonl`
- `data/results/human_speech_v2/direct_postprocessed.jsonl`
- `data/results/human_speech_v2/transcription_scores.csv`
- `data/results/human_speech_v2/scores.csv`
- `data/results/human_speech_v2/summary.json`
- `report/figures/human_speech_v2_cd.png`

# TTS12 D Rerun and C/D Summary Stability

## Goals

1. Re-run the complete D path on the same 12 TTS clips without modifying the
   original `tts12_cd_v1` outputs.
2. Run three new complete summary repetitions per sample for C and D.

## Isolation

- Complete D rerun: `data/results/tts12_d_rerun_v2/`
- Three-repeat summary experiment:
  `data/results/tts12_summary_stability_v1/`

The original `data/results/tts12_cd_v1/` directory is read-only input for
later comparisons.

## Generation policy

Qwen uses the project's unchanged deterministic decoding configuration:
`do_sample=False`, `temperature=None`, and `max_new_tokens=256`. Repetition
therefore tests operational reproducibility under the existing workflow, not
sensitivity to stochastic sampling. C regenerates the Qwen transcript in each
repetition and then uses DeepSeek at temperature zero for summarization. D
regenerates the summary directly from audio in each repetition.

## Planned paid calls

- Complete D rerun structured formatting: 36 DeepSeek calls.
- Three C summary repetitions: 36 DeepSeek calls.
- Total: 72 calls, historically estimated at USD 0.036.

No repeat output may overwrite the original TTS12 experiment.

## Completed results

### Complete D rerun

All 48 local task outputs and all 36 DeepSeek formatting calls succeeded.
The rerun exactly matched the original experiment on all 48 final output
strings and all 48 scores:

| Run | Summary | Sentiment | Keyword F1 | Intent |
|---|---:|---:|---:|---:|
| Original D | 0.3324 | 41.7% | 0.4286 | 16.7% |
| D rerun | 0.3324 | 41.7% | 0.4286 | 16.7% |

### Three-repeat C/D summaries

| Repetition | C mean | D mean | C - D | C wins | D wins | Paired bootstrap 95% interval |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 0.3497 | 0.3324 | +0.0173 | 7 | 5 | [-0.0405, 0.0740] |
| 2 | 0.3467 | 0.3324 | +0.0143 | 7 | 5 | [-0.0353, 0.0660] |
| 3 | 0.3566 | 0.3324 | +0.0242 | 7 | 5 | [-0.0265, 0.0745] |

C had the higher mean in all three repetitions. Every sample kept the same
C-versus-D winner across all three repetitions: C won the same seven samples
and D won the same five. All three paired intervals crossed zero, so this
supports stable direction under the current deterministic workflow, not a
general statistical-significance claim.

| Sample | C1 | D1 | C2 | D2 | C3 | D3 |
|---|---:|---:|---:|---:|---:|---:|
| ocean_currents | 0.4063 | 0.3333 | 0.3750 | 0.3333 | 0.3750 | 0.3333 |
| vaccine_development | 0.4231 | 0.2963 | 0.3438 | 0.2963 | 0.4231 | 0.2963 |
| remote_work | 0.4118 | 0.4242 | 0.4118 | 0.4242 | 0.4118 | 0.4242 |
| sleep_science | 0.2500 | 0.2759 | 0.2154 | 0.2759 | 0.2500 | 0.2759 |
| renewable_energy | 0.4348 | 0.4151 | 0.4348 | 0.4151 | 0.4348 | 0.4151 |
| memory_formation | 0.2857 | 0.2286 | 0.2857 | 0.2286 | 0.2857 | 0.2286 |
| urban_biodiversity | 0.3438 | 0.3733 | 0.3438 | 0.3733 | 0.3438 | 0.3733 |
| nuclear_fusion | 0.2667 | 0.2424 | 0.2813 | 0.2424 | 0.2813 | 0.2424 |
| microbiome | 0.3279 | 0.4848 | 0.3175 | 0.4848 | 0.3279 | 0.4848 |
| deep_sea | 0.3733 | 0.1600 | 0.3733 | 0.1600 | 0.3733 | 0.1600 |
| language_acquisition | 0.3279 | 0.4815 | 0.4333 | 0.4815 | 0.4333 | 0.4815 |
| plastic_pollution | 0.3448 | 0.2727 | 0.3448 | 0.2727 | 0.3390 | 0.2727 |

All three Qwen transcripts were identical for 12/12 samples. D's three
summaries were also identical for 12/12 samples. C's final DeepSeek summaries
were identical for only 5/12 samples, locating the observed output variation
in the text-LLM stage rather than Qwen transcription.

### API audit

The complete D rerun used 36 successful formatting calls and 3,653 tokens.
The C stability experiment required 36 canonical calls, but a checkpoint-key
bug caused 12 duplicate calls before it was detected. The full C audit
therefore contains 48 calls and 7,192 tokens; the canonical scored results
contain 36 calls and 5,327 tokens. Actual repeat-experiment API usage was 84
calls and 10,845 tokens, historically estimated at USD 0.042. Duplicate calls
are preserved in `c_summaries_call_audit.jsonl` and excluded from scoring.

## Result artifacts

- `data/results/tts12_d_rerun_v2/summary.json`
- `data/results/tts12_d_rerun_v2/scores.csv`
- `data/results/tts12_summary_stability_v1/summary.json`
- `data/results/tts12_summary_stability_v1/summary_scores.csv`
- `data/results/tts12_summary_stability_v1/c_summaries_call_audit.jsonl`
- `report/figures/tts12_summary_stability.png`

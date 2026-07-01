# White-Noise Experiment

## Scope

- Model: Qwen2-Audio-7B-Instruct, INT4, local inference
- Samples: 8
- Tasks: summarization, sentiment, keywords, intent
- Conditions: clean, white noise at 20 dB, 10 dB, and 0 dB SNR
- Total evaluated outputs: 128/128
- Random seed: 42
- Mean inference latency: 2.634 seconds per task on this machine
- Structured-task post-processing: DeepSeek-chat, matching the original report

The original Direct scores reported in `report/report.md` are included for
context. The new `clean` condition is the primary baseline for measuring the
effect of noise because it was generated in the same controlled run as all
three noisy conditions.

## Main results

| Task / metric | Clean baseline | White 20 dB | Change | White 10 dB | Change | White 0 dB | Change |
|---|---:|---:|---:|---:|---:|---:|---:|
| Summary ROUGE-L | 0.4600 | 0.4422 | -0.0178 | 0.4380 | -0.0220 | 0.4536 | -0.0065 |
| Sentiment accuracy | 62.5% | 50.0% | -12.5 pp | 62.5% | 0.0 pp | 87.5% | +25.0 pp |
| Keyword exact-phrase F1 | 0.3378 | 0.3650 | +0.0272 | 0.3757 | +0.0379 | 0.3223 | -0.0154 |
| Intent accuracy | 87.5% | 87.5% | 0.0 pp | 100.0% | +12.5 pp | 100.0% | +12.5 pp |

## Original-result comparison

| Metric | Original Direct Clean | New Same-Method Clean | White 20 dB | White 10 dB | White 0 dB |
|---|---:|---:|---:|---:|---:|
| Summary ROUGE-L | 0.448 | 0.460 | 0.442 | 0.438 | 0.454 |
| Sentiment accuracy | 38% | 62.5% | 50.0% | 62.5% | 87.5% |
| Keyword F1 | 0.29 | 0.338 | 0.365 | 0.376 | 0.322 |
| Intent accuracy | 62% | 87.5% | 87.5% | 100.0% | 100.0% |

All columns use the same Qwen-to-DeepSeek architecture. The original clean
column came from an earlier run, so the new same-method clean column remains the
correct causal baseline for noise comparisons.

## Interpretation

1. **Summarization is robust to white noise in this small test.** At 0 dB, where
   signal and noise have equal power, mean ROUGE-L decreased by only 0.0065.
   The 10 dB result was slightly worse than the 0 dB result, so the scores are
   not monotonic. With only eight samples, this should be treated as sampling
   variation rather than evidence that stronger noise helps.

2. **Sentiment is non-monotonic.** Accuracy fell by one sample at 20 dB,
   returned to baseline at 10 dB, and increased by two samples at 0 dB.

3. **Intent did not degrade in this run.** It matched the clean result at 20 dB
   and scored one sample higher at 10 dB and 0 dB.

4. **Keyword F1 changed little at 0 dB.** The decrease was 0.0154. This metric
   uses exact phrase matching, so semantically similar phrases receive no
   credit.

5. **No consistent degradation was detected.** Classification improvements at
   high noise must not be treated as evidence that noise helps: with N=8, one
   changed sample moves accuracy by 12.5 percentage points, and the DeepSeek
   conversion step adds another source of variation.

## Output formatting

Qwen's free-form structured-task analyses were converted by DeepSeek using the
same prompts and settings as the original project. Strict JSON validity after
post-processing was 100% for sentiment, keywords, and intent at every noise
level.

## Reproducibility artifacts

- Raw first-pass outputs: `data/results/white_noise_v1/direct_raw.jsonl`
- Corrected structured-task outputs:
  `data/results/white_noise_v1/direct_reprompted_raw.jsonl`
- Same-method DeepSeek outputs:
  `data/results/white_noise_v1/direct_postprocessed.jsonl`
- Per-sample scores: `data/results/white_noise_v1/scores.csv`
- Machine-readable summary: `data/results/white_noise_v1/summary.json`
- Main figure: `data/results/white_noise_v1/figures/task_scores_vs_snr.png`
- Horizontal comparison figure:
  `data/results/white_noise_v1/figures/original_vs_white_noise.png`

Because the dataset has only eight samples, these results are descriptive. They
should not be presented as statistically conclusive.

## Cascade vs Direct summarization

The supplied Cascade file contains summarization only, so this comparison does
not extend to sentiment, keywords, or intent.

| Condition | Cascade ROUGE-L | Direct ROUGE-L | Direct - Cascade | Direct Wins |
|---|---:|---:|---:|---:|
| Clean | 0.3971 | 0.4600 | +0.0629 | 5/8 |
| White 20 dB | 0.3947 | 0.4422 | +0.0475 | 5/8 |
| White 10 dB | 0.3923 | 0.4380 | +0.0457 | 5/8 |
| White 0 dB | 0.4032 | 0.4536 | +0.0503 | 5/8 |

Direct had the higher mean ROUGE-L under every condition, while Cascade changed
less relative to its own clean baseline. All paired bootstrap 95% confidence
intervals crossed zero, so the Direct advantage is not statistically
conclusive at N=8.

The recorded latency values came from separate runs without matched hardware
and timing-boundary metadata. They are retained for reproducibility but should
not be used to infer intrinsic architecture speed.

- Cascade input:
  `data/results/white_noise_v1/cascade_summary_raw.json`
- Path-comparison scores:
  `data/results/white_noise_v1/path_comparison_scores.csv`
- Path-comparison summary:
  `data/results/white_noise_v1/path_comparison_summary.json`
- Path-comparison figure:
  `data/results/white_noise_v1/figures/cascade_vs_direct_white_noise.png`

# Human Speech V1 Runbook

This experiment evaluates the existing Cascade and Direct paths on eight
as-recorded human-speech samples. Environmental noise, audience sounds, and
other recording conditions are retained. This is an in-the-wild paired pilot,
not a controlled noise experiment.

## 1. Source audio

Original WAV files are copied without modification to:

`data/raw/human_speech_v1/`

The source files are never overwritten by the preparation or inference steps.

## 2. Prepare normalized copies

```powershell
.\.venv\Scripts\python.exe prepare_human_speech.py
```

Preparation converts each source to mono 16 kHz floating-point WAV without
denoising or peak normalization. Expected output:

- 8 WAV files in `data/processed/human_speech_v1/as_recorded/`
- `data/processed/human_speech_v1/audio_manifest.json`

## 3. Preview the Direct scope

```powershell
.\.venv\Scripts\python.exe run_human_speech.py --dry-run
```

Expected scope: 8 samples x 4 tasks = 32 results.

## 4. Run local Direct inference

If the model is stored outside the default Hugging Face cache:

```powershell
.\.venv\Scripts\python.exe run_human_speech.py `
  --model-path D:\path\to\Qwen2-Audio-7B-Instruct
```

The runner forces offline Hugging Face mode and resumes from successful JSONL
records. It does not call a paid API.

## 5. Human annotation and evaluation

Fill `data/ground_truth_human_v1.json` using the supplied template. Ground
truth must be written or reviewed by a human; model outputs must not be used as
their own reference labels.

After explicit approval for paid DeepSeek calls:

```powershell
.\.venv\Scripts\python.exe postprocess_human_speech.py
.\.venv\Scripts\python.exe evaluate_human_speech.py
```

The post-processing command applies the same DeepSeek formatting step to all
24 structured-task outputs and resumes from successful JSONL records.

## 6. Import and compare supplied Cascade results

Place the supplied 8-sample Cascade file at:

`data/results/human_speech_v1/cascade_raw.json`

Then run:

```powershell
.\.venv\Scripts\python.exe compare_human_paths.py
```

The comparator validates all 32 paired sample/task keys and uses the same
ground truth and scoring functions for both paths.

## Outputs

- Raw Direct output: `data/results/human_speech_v1/direct_raw.jsonl`
- Structured Direct output:
  `data/results/human_speech_v1/direct_postprocessed.jsonl`
- Supplied Cascade output: `data/results/human_speech_v1/cascade_raw.json`
- Scores: `data/results/human_speech_v1/scores.csv`
- Summary: `data/results/human_speech_v1/summary.json`
- Paired scores:
  `data/results/human_speech_v1/path_comparison_scores.csv`
- Paired summary:
  `data/results/human_speech_v1/path_comparison_summary.json`
- Ground-truth template: `data/ground_truth_human_v1.template.json`
- Audio manifest: `data/processed/human_speech_v1/audio_manifest.json`
- Git-tracked inference audio:
  `data/processed/human_speech_v1/as_recorded/*.wav`

The original 44.1 kHz stereo files remain under `data/raw/human_speech_v1/`
locally and are intentionally not duplicated in Git.

## Completed run

The run completed on 2026-07-02 with 32/32 successful results for each path and
24/24 successful DeepSeek post-processing results for Direct. One earlier
connection failure is retained in the Direct JSONL audit trail and excluded
from evaluation.

| Task | Metric | Cascade | Direct | Direct - Cascade |
|---|---|---:|---:|---:|
| Summarization | ROUGE-L | 0.2807 | 0.2388 | -0.0419 |
| Sentiment | Accuracy | 75.0% | 62.5% | -12.5 pp |
| Keywords | Exact-phrase F1 | 0.4428 | 0.4167 | -0.0261 |
| Intent | Accuracy | 62.5% | 25.0% | -37.5 pp |

Cascade had the higher mean on all four metrics. It won 6/8 summarization
pairs, 2/8 sentiment pairs with 5 ties, 4/8 keyword pairs with 2 ties, and 4/8
intent pairs with 3 ties. Direct won the remaining 2, 1, 2, and 1 pairs,
respectively. Every paired bootstrap 95% interval for Direct minus Cascade
crossed zero, so none of these mean differences is statistically conclusive
with eight samples.

Cascade's mean ASR WER was 0.0813 under the project's simple whitespace-token
implementation.

Raw Qwen structured outputs used Python-style literals or other non-strict
structures, so strict JSON compliance was 0/24. Compliance was 24/24 after the
uniform DeepSeek formatting step.

Mean recorded Qwen task latency was 3.048 seconds (range 0.951-8.161 seconds).
The supplied Cascade task means ranged from 16.239 to 18.290 seconds. These
values came from different machines and timing boundaries, so their ratio is
not an architecture-level speed result.

The 24 successful DeepSeek calls used 1,832 prompt tokens and 478 completion
tokens (2,310 total). Applying the project's historical $0.0005-per-call
assumption gives an estimated cost of $0.012; this is not a billing receipt.

## Interpretation limits

- N=8 supports only descriptive observations, not statistical significance.
- All recordings retain uncontrolled environmental and audience sounds.
- There are no matched clean versions, so score differences cannot be
  attributed specifically to noise.
- The supplied Cascade file has no audio hashes, so exact byte identity with
  the Direct input cannot be independently verified.
- Cascade and Direct latency came from different execution environments.
- ROUGE-L and exact-phrase keyword F1 are sensitive to valid paraphrases.

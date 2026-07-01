# White Noise V1 Runbook

This experiment keeps the existing eight TTS samples and tests only white
noise at clean, 20 dB, 10 dB, and 0 dB SNR.

## 1. Prepare deterministic audio

```powershell
.\.venv\Scripts\python.exe prepare_white_noise.py
```

Expected result: 32 files and
`data/processed/white_noise_v1/audio_manifest.json`.

## 2. Preview the smoke-test scope

```powershell
.\.venv\Scripts\python.exe run_white_noise.py --pipeline cascade --smoke --dry-run
.\.venv\Scripts\python.exe run_white_noise.py --pipeline qwen --smoke --dry-run
```

Each command should report six pending results.

## 3. Run the 12-result smoke test

```powershell
.\.venv\Scripts\python.exe run_white_noise.py --pipeline cascade --smoke
.\.venv\Scripts\python.exe run_white_noise.py --pipeline qwen --smoke
.\.venv\Scripts\python.exe evaluate_white_noise.py --smoke
```

All commands resume automatically from successful JSONL records.
Smoke-test records are isolated in `cascade_smoke_raw.jsonl` and
`direct_smoke_raw.jsonl`; they are never reused as formal results.

## 4. Run the full 256-result experiment

```powershell
.\.venv\Scripts\python.exe run_white_noise.py --pipeline cascade
.\.venv\Scripts\python.exe run_white_noise.py --pipeline qwen
.\.venv\Scripts\python.exe postprocess_white_noise.py
.\.venv\Scripts\python.exe evaluate_white_noise.py
```

If Cascade summarization results are supplied in
`data/results/white_noise_v1/cascade_summary_raw.json`, compare the two paths
with:

```powershell
.\.venv\Scripts\python.exe compare_noise_paths.py
```

## Outputs

- Prepared audio: `data/processed/white_noise_v1/`
- Raw inference: `data/results/white_noise_v1/*_raw.jsonl`
- Structured Direct output: `data/results/white_noise_v1/direct_postprocessed.jsonl`
- Scores: `data/results/white_noise_v1/scores.csv`
- Summary: `data/results/white_noise_v1/summary.json`
- Path comparison:
  `data/results/white_noise_v1/path_comparison_summary.json`
- Figures: `data/results/white_noise_v1/figures/`

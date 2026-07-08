"""Import annotations and prepare the eight human-speech-v3 recordings."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import soundfile as sf


ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE = ROOT.parent / "yinpin" / "rensheng_v3"
CONFIG = ROOT / "experiments" / "human_speech_v3.json"
RAW_DIR = ROOT / "data" / "raw" / "human_speech_v3"
OUTPUT_DIR = ROOT / "data" / "processed" / "human_speech_v3"
GROUND_TRUTH = ROOT / "data" / "ground_truth_human_v3.json"
MANIFEST = OUTPUT_DIR / "audio_manifest.json"
LABELS = {
    "sentiment": {"positive", "negative", "neutral"},
    "intent": {"inform", "persuade", "entertain", "question", "describe"},
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_annotations(path: Path) -> dict[str, dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    pattern = re.compile(
        r"^##\s+([^\r\n]+)\r?\n\s*(\{.*?\})(?=\s*^##|\s*\Z)",
        re.MULTILINE | re.DOTALL,
    )
    annotations = {}
    for filename, payload in pattern.findall(text):
        sample = Path(filename.strip()).stem
        annotations[sample] = json.loads(payload)
    return annotations


def validate_annotation(sample: str, record: dict[str, Any]) -> None:
    expected = {"transcript", "summary", "sentiment", "keywords", "intent"}
    if set(record) != expected:
        raise ValueError(f"{sample}: expected fields {sorted(expected)}")
    if not record["transcript"].strip() or not record["summary"].strip():
        raise ValueError(f"{sample}: transcript and summary must be non-empty")
    if record["sentiment"] not in LABELS["sentiment"]:
        raise ValueError(f"{sample}: invalid sentiment")
    if record["intent"] not in LABELS["intent"]:
        raise ValueError(f"{sample}: invalid intent")
    if not isinstance(record["keywords"], list) or not record["keywords"]:
        raise ValueError(f"{sample}: keywords must be a non-empty list")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    args = parser.parse_args()
    source = args.source.resolve()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    annotations = parse_annotations(source / "ground_truth.txt")
    samples = list(config["samples"])
    audio_paths = {path.stem: path for path in source.glob("*.wav")}
    if set(annotations) != set(samples) or set(audio_paths) != set(samples):
        raise ValueError("Audio files, annotations, and config samples must match")
    for sample in samples:
        validate_annotation(sample, annotations[sample])

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    destination_dir = OUTPUT_DIR / config["condition"]
    destination_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for sample in samples:
        source_path = audio_paths[sample]
        raw_path = RAW_DIR / source_path.name
        shutil.copy2(source_path, raw_path)
        source_audio, source_sr = sf.read(
            raw_path, dtype="float32", always_2d=True
        )
        if source_audio.size == 0 or not np.isfinite(source_audio).all():
            raise ValueError(f"Invalid audio: {raw_path}")
        mono = source_audio.mean(axis=1, dtype=np.float32)
        prepared = librosa.resample(
            mono,
            orig_sr=source_sr,
            target_sr=config["sample_rate"],
            res_type="soxr_hq",
        ).astype(np.float32, copy=False)
        destination = destination_dir / source_path.name
        sf.write(destination, prepared, config["sample_rate"], subtype="FLOAT")
        records.append(
            {
                "experiment_id": config["experiment_id"],
                "sample": sample,
                "condition": config["condition"],
                "audio_path": destination.relative_to(ROOT).as_posix(),
                "source_path": raw_path.relative_to(ROOT).as_posix(),
                "source_sha256": sha256_file(raw_path),
                "output_sha256": sha256_file(destination),
                "source_sample_rate": int(source_sr),
                "source_channels": int(source_audio.shape[1]),
                "sample_rate": int(config["sample_rate"]),
                "channels": 1,
                "duration_seconds": round(
                    len(prepared) / config["sample_rate"], 6
                ),
                "denoising_applied": False,
                "normalization_applied": False,
            }
        )
        print(
            f"[OK] {sample}: {source_sr}Hz/{source_audio.shape[1]}ch "
            f"-> {config['sample_rate']}Hz/1ch"
        )
    GROUND_TRUTH.write_text(
        json.dumps(annotations, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    MANIFEST.write_text(
        json.dumps(
            {
                "experiment_id": config["experiment_id"],
                "source_annotation_file": str(
                    (source / "ground_truth.txt").resolve()
                ),
                "expected_records": len(records),
                "records": records,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Prepared {len(records)} recordings. Manifest: {MANIFEST}")


if __name__ == "__main__":
    main()

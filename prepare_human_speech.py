"""Prepare as-recorded human speech for the Direct benchmark.

The source recordings are preserved. Prepared copies are converted to mono
16 kHz WAV without denoising or peak normalization.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import soundfile as sf


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "experiments" / "human_speech_v1.json"
SOURCE_DIR = ROOT / "data" / "raw" / "human_speech_v1"
OUTPUT_DIR = ROOT / "data" / "processed" / "human_speech_v1"
MANIFEST = OUTPUT_DIR / "audio_manifest.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare(config_path: Path) -> list[dict[str, Any]]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    expected_sr = int(config["sample_rate"])
    condition = str(config["condition"])
    destination_dir = OUTPUT_DIR / condition
    destination_dir.mkdir(parents=True, exist_ok=True)

    expected_names = {f"{sample}.wav" for sample in config["samples"]}
    actual_names = {path.name for path in SOURCE_DIR.glob("*.wav")}
    missing = sorted(expected_names - actual_names)
    extra = sorted(actual_names - expected_names)
    if missing or extra:
        raise ValueError(f"Source-file mismatch; missing={missing}, extra={extra}")

    records: list[dict[str, Any]] = []
    for sample in config["samples"]:
        source = SOURCE_DIR / f"{sample}.wav"
        source_audio, source_sr = sf.read(
            source,
            dtype="float32",
            always_2d=True,
        )
        if source_audio.size == 0:
            raise ValueError(f"{source}: empty audio")
        if not np.isfinite(source_audio).all():
            raise ValueError(f"{source}: contains NaN or infinity")

        mono = source_audio.mean(axis=1, dtype=np.float32)
        if source_sr != expected_sr:
            prepared = librosa.resample(
                mono,
                orig_sr=source_sr,
                target_sr=expected_sr,
                res_type="soxr_hq",
            ).astype(np.float32, copy=False)
        else:
            prepared = mono.astype(np.float32, copy=False)

        peak = float(np.max(np.abs(prepared)))
        if peak > 1.0:
            raise ValueError(f"{source}: prepared peak exceeds 1.0 ({peak})")

        destination = destination_dir / source.name
        sf.write(destination, prepared, expected_sr, subtype="FLOAT")
        measured, measured_sr = sf.read(
            destination,
            dtype="float32",
            always_2d=False,
        )
        if measured_sr != expected_sr or measured.ndim != 1:
            raise ValueError(f"{destination}: invalid prepared format")

        record = {
            "experiment_id": config["experiment_id"],
            "pipeline_scope": "direct",
            "sample": sample,
            "condition": condition,
            "audio_path": destination.relative_to(ROOT).as_posix(),
            "source_path": source.relative_to(ROOT).as_posix(),
            "source_sha256": sha256_file(source),
            "output_sha256": sha256_file(destination),
            "source_sample_rate": int(source_sr),
            "source_channels": int(source_audio.shape[1]),
            "sample_rate": int(measured_sr),
            "channels": 1,
            "num_samples": int(measured.shape[0]),
            "duration_seconds": round(measured.shape[0] / measured_sr, 6),
            "peak": float(np.max(np.abs(measured))),
            "clipping_ratio": float(np.mean(np.abs(measured) >= 1.0)),
            "denoising_applied": False,
            "normalization_applied": False,
        }
        records.append(record)
        print(
            f"[OK] {sample:12s} {record['duration_seconds']:7.3f}s "
            f"{source_sr}Hz/{source_audio.shape[1]}ch -> "
            f"{measured_sr}Hz/1ch peak={record['peak']:.4f}"
        )

    manifest = {
        "experiment_id": config["experiment_id"],
        "description": config["description"],
        "config_path": config_path.relative_to(ROOT).as_posix(),
        "expected_records": len(config["samples"]),
        "records": records,
    }
    MANIFEST.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nPrepared {len(records)} files. Manifest: {MANIFEST}")
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    prepare(args.config.resolve())


if __name__ == "__main__":
    main()


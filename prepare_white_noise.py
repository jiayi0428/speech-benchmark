"""Generate and validate the fixed white_noise_v1 audio set.

Creates 8 clean files and 24 deterministic white-noise files.  The same
generated files are later consumed by both benchmark pipelines.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "experiments" / "white_noise_v1.json"
SOURCE_DIR = ROOT / "data" / "processed" / "tts_samples"
GROUND_TRUTH = ROOT / "data" / "ground_truth.json"
OUTPUT_DIR = ROOT / "data" / "processed" / "white_noise_v1"
MANIFEST = OUTPUT_DIR / "audio_manifest.json"
PEAK_LIMIT = 0.999


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_mono_float(path: Path, expected_sr: int) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1, dtype=np.float32)
    if sr != expected_sr:
        raise ValueError(f"{path.name}: expected {expected_sr}Hz, got {sr}Hz")
    if audio.size == 0:
        raise ValueError(f"{path.name}: empty audio")
    if not np.isfinite(audio).all():
        raise ValueError(f"{path.name}: contains NaN or infinity")
    return audio.astype(np.float32, copy=False), sr


def add_white_noise(
    clean: np.ndarray,
    snr_db: float,
    seed: int,
) -> tuple[np.ndarray, dict[str, float]]:
    signal_power = float(np.mean(np.square(clean, dtype=np.float64)))
    if signal_power <= 1e-12:
        raise ValueError("Cannot add calibrated noise to silent audio")

    rng = np.random.RandomState(seed)
    noise = rng.standard_normal(clean.shape[0]).astype(np.float32)
    noise -= np.mean(noise, dtype=np.float64)
    raw_noise_power = float(np.mean(np.square(noise, dtype=np.float64)))
    target_noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    noise *= math.sqrt(target_noise_power / raw_noise_power)

    mixed = clean + noise
    pre_scale_peak = float(np.max(np.abs(mixed)))
    gain = min(1.0, PEAK_LIMIT / max(pre_scale_peak, 1e-12))
    clean_scaled = clean * gain
    noise_scaled = noise * gain
    mixed = (clean_scaled + noise_scaled).astype(np.float32)

    measured_signal_power = float(
        np.mean(np.square(clean_scaled, dtype=np.float64))
    )
    measured_noise_power = float(
        np.mean(np.square(noise_scaled, dtype=np.float64))
    )
    measured_snr = 10.0 * math.log10(
        measured_signal_power / measured_noise_power
    )
    peak = float(np.max(np.abs(mixed)))
    clipping_ratio = float(np.mean(np.abs(mixed) >= 1.0))

    return mixed, {
        "requested_snr_db": float(snr_db),
        "measured_snr_db": measured_snr,
        "signal_power": measured_signal_power,
        "noise_power": measured_noise_power,
        "pre_scale_peak": pre_scale_peak,
        "gain": gain,
        "peak": peak,
        "clipping_ratio": clipping_ratio,
    }


def clean_metadata(audio: np.ndarray) -> dict[str, float | None]:
    return {
        "requested_snr_db": None,
        "measured_snr_db": None,
        "signal_power": float(np.mean(np.square(audio, dtype=np.float64))),
        "noise_power": 0.0,
        "pre_scale_peak": float(np.max(np.abs(audio))),
        "gain": 1.0,
        "peak": float(np.max(np.abs(audio))),
        "clipping_ratio": float(np.mean(np.abs(audio) >= 1.0)),
    }


def prepare(config_path: Path) -> list[dict[str, Any]]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    ground_truth = json.loads(GROUND_TRUTH.read_text(encoding="utf-8"))
    expected_sr = int(config["sample_rate"])
    seed = int(config["seed"])

    missing_labels = [name for name in config["samples"] if name not in ground_truth]
    if missing_labels:
        raise ValueError(f"Missing ground truth: {missing_labels}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []

    for sample in config["samples"]:
        source = SOURCE_DIR / f"{sample}.wav"
        if not source.exists():
            raise FileNotFoundError(f"Missing source audio: {source}")
        clean, sr = load_mono_float(source, expected_sr)
        source_hash = sha256_file(source)

        for condition in config["conditions"]:
            condition_name = condition["name"]
            destination_dir = OUTPUT_DIR / condition_name
            destination_dir.mkdir(parents=True, exist_ok=True)
            destination = destination_dir / f"{sample}.wav"

            if condition["snr_db"] is None:
                shutil.copy2(source, destination)
                metrics = clean_metadata(clean)
            else:
                noisy, metrics = add_white_noise(
                    clean,
                    float(condition["snr_db"]),
                    seed,
                )
                sf.write(destination, noisy, sr, subtype="FLOAT")

            measured, measured_sr = load_mono_float(destination, expected_sr)
            measured_peak = float(np.max(np.abs(measured)))
            if measured_peak > 1.0:
                raise ValueError(f"{destination}: peak exceeds 1.0")
            if metrics["measured_snr_db"] is not None:
                error = abs(
                    float(metrics["measured_snr_db"])
                    - float(metrics["requested_snr_db"])
                )
                if error > 0.2:
                    raise ValueError(f"{destination}: SNR error {error:.3f}dB")

            record = {
                "experiment_id": config["experiment_id"],
                "sample": sample,
                "condition": condition_name,
                "seed": seed,
                "audio_path": destination.relative_to(ROOT).as_posix(),
                "source_path": source.relative_to(ROOT).as_posix(),
                "source_sha256": source_hash,
                "output_sha256": sha256_file(destination),
                "sample_rate": measured_sr,
                "num_samples": int(measured.shape[0]),
                "duration_seconds": round(measured.shape[0] / measured_sr, 6),
                **metrics,
            }
            records.append(record)
            snr_label = (
                "clean"
                if metrics["measured_snr_db"] is None
                else f"{metrics['measured_snr_db']:.3f}dB"
            )
            print(
                f"[OK] {sample:24s} {condition_name:12s} "
                f"SNR={snr_label:9s} peak={record['peak']:.4f}"
            )

    manifest = {
        "experiment_id": config["experiment_id"],
        "config_path": config_path.relative_to(ROOT).as_posix(),
        "expected_records": len(config["samples"]) * len(config["conditions"]),
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

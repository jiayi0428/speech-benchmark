"""Parse annotations and prepare nine human-speech-v5 recordings."""
from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from prepare_human_speech_v3 import sha256_file, validate_annotation


ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE = ROOT.parent / "yinpin" / "rensheng_v5"
CONFIG = ROOT / "experiments" / "human_speech_v5.json"
RAW_DIR = ROOT / "data" / "raw" / "human_speech_v5"
OUTPUT_DIR = ROOT / "data" / "processed" / "human_speech_v5"
GROUND_TRUTH = ROOT / "data" / "ground_truth_human_v5.json"
MANIFEST = OUTPUT_DIR / "audio_manifest.json"


def parse_annotations(path: Path) -> dict[str, dict]:
    text = path.read_text(encoding="utf-8")
    marker = re.compile(r"(?m)^(?:##\s*)?([A-Za-z0-9_-]+(?:\.wav)?)\s*(?:\{)?\s*$")
    matches = list(marker.finditer(text))
    decoder = json.JSONDecoder()
    annotations = {}
    for index, match in enumerate(matches):
        raw_name = match.group(1)
        if raw_name.lower() == "json":
            continue
        block_end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        brace = text.find("{", match.start(), block_end)
        if brace < 0:
            continue
        block = text[brace:block_end].strip()
        try:
            payload, _ = decoder.raw_decode(block)
        except json.JSONDecodeError:
            # The provided v5 annotation has one human-readable quote phrase inside
            # a JSON string. Preserve the meaning while producing valid project JSON.
            fixed = block.replace(
                'accusations of "gay baiting" using',
                'accusations of \\"gay baiting\\" using',
            )
            payload = json.loads(fixed)
        sample = Path(raw_name).stem
        if {"transcript", "summary", "sentiment", "keywords", "intent"} <= set(payload):
            annotations[sample] = payload
    return annotations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    args = parser.parse_args()
    source = args.source.resolve()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    samples = list(config["samples"])
    annotations = parse_annotations(source / "ground_truth.txt")
    audio_paths = {path.stem: path for path in source.glob("*.wav")}
    if set(annotations) != set(samples):
        raise ValueError(
            f"Annotation mismatch: missing={sorted(set(samples)-set(annotations))}; "
            f"extra={sorted(set(annotations)-set(samples))}"
        )
    if set(audio_paths) != set(samples):
        raise ValueError(
            f"Audio mismatch: missing={sorted(set(samples)-set(audio_paths))}; "
            f"extra={sorted(set(audio_paths)-set(samples))}"
        )
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
        source_audio, source_sr = sf.read(raw_path, dtype="float32", always_2d=True)
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
                "duration_seconds": round(len(prepared) / config["sample_rate"], 6),
                "denoising_applied": False,
                "normalization_applied": False,
            }
        )
        print(f"[OK] {sample}: {source_sr}Hz/{source_audio.shape[1]}ch -> 16000Hz/1ch")
    GROUND_TRUTH.write_text(
        json.dumps(annotations, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    MANIFEST.write_text(
        json.dumps(
            {
                "experiment_id": config["experiment_id"],
                "source_annotation_file": str((source / "ground_truth.txt").resolve()),
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

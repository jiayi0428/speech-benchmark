"""Import and validate the 12-sample TTS C/D experiment."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import soundfile as sf


ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE = ROOT.parent / "TTS" / "TTS"
OUTPUT_DIR = ROOT / "data" / "processed" / "tts12_cd_v1"
GROUND_TRUTH_OUTPUT = ROOT / "data" / "ground_truth_tts12_cd_v1.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    args = parser.parse_args()
    source = args.source.resolve()
    truth_path = source / "ground_truth_new.json"
    truth = json.loads(truth_path.read_text(encoding="utf-8"))
    audio_paths = sorted(source.glob("*.wav"))
    audio_names = {path.stem for path in audio_paths}
    if len(audio_paths) != 12 or len(truth) != 12 or audio_names != set(truth):
        raise ValueError(
            "Expected exactly 12 WAV files with matching ground-truth keys"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    entries = []
    for source_audio in audio_paths:
        destination = OUTPUT_DIR / source_audio.name
        shutil.copy2(source_audio, destination)
        info = sf.info(str(destination))
        if info.samplerate != 16_000 or info.channels != 1:
            raise ValueError(f"{source_audio.name} must be mono 16 kHz")
        entries.append(
            {
                "audio_path": destination.relative_to(ROOT).as_posix(),
                "transcript": truth[source_audio.stem]["transcript"],
                "speaker": "unknown_tts_voice",
                "duration": info.duration,
                "topic": source_audio.stem,
                "audio_sha256": sha256_file(destination),
            }
        )

    (OUTPUT_DIR / "index.json").write_text(
        json.dumps(entries, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    GROUND_TRUTH_OUTPUT.write_text(
        json.dumps(truth, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Prepared {len(entries)} samples in {OUTPUT_DIR}")
    print(f"Ground truth: {GROUND_TRUTH_OUTPUT}")


if __name__ == "__main__":
    main()

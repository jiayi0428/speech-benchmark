"""Import the human-written Markdown annotations into benchmark JSON."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "experiments" / "human_speech_v1.json"
DEFAULT_OUTPUT = ROOT / "data" / "ground_truth_human_v1.json"
VALID_SENTIMENTS = {"positive", "negative", "neutral"}
VALID_INTENTS = {"inform", "persuade", "entertain", "question", "describe"}


def parse_markdown(path: Path) -> dict[str, dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    headings = list(re.finditer(r"^##\s+([A-Za-z0-9_-]+)\s*$", text, re.MULTILINE))
    annotations: dict[str, dict[str, Any]] = {}
    decoder = json.JSONDecoder()
    for index, heading in enumerate(headings):
        sample = heading.group(1)
        block_end = (
            headings[index + 1].start() if index + 1 < len(headings) else len(text)
        )
        block = text[heading.end() : block_end].lstrip()
        annotation, consumed = decoder.raw_decode(block)
        if block[consumed:].strip():
            raise ValueError(f"{sample}: unexpected text after JSON object")
        if not isinstance(annotation, dict):
            raise ValueError(f"{sample}: annotation must be a JSON object")
        if sample in annotations:
            raise ValueError(f"Duplicate annotation heading: {sample}")
        annotations[sample] = annotation
    return annotations


def validate(
    annotations: dict[str, dict[str, Any]],
    expected_samples: list[str],
) -> None:
    expected = set(expected_samples)
    actual = set(annotations)
    if expected != actual:
        raise ValueError(
            f"Annotation-key mismatch; missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )
    required = {"transcript", "summary", "sentiment", "keywords", "intent"}
    for sample in expected_samples:
        annotation = annotations[sample]
        missing_fields = required - set(annotation)
        if missing_fields:
            raise ValueError(f"{sample}: missing fields {sorted(missing_fields)}")
        for field in ["transcript", "summary"]:
            if not isinstance(annotation[field], str) or not annotation[field].strip():
                raise ValueError(f"{sample}: {field} must be non-empty text")
        if annotation["sentiment"] not in VALID_SENTIMENTS:
            raise ValueError(f"{sample}: invalid sentiment")
        if annotation["intent"] not in VALID_INTENTS:
            raise ValueError(f"{sample}: invalid intent")
        keywords = annotation["keywords"]
        if (
            not isinstance(keywords, list)
            or not 5 <= len(keywords) <= 7
            or not all(isinstance(item, str) and item.strip() for item in keywords)
        ):
            raise ValueError(f"{sample}: keywords must contain 5-7 non-empty strings")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("markdown", type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    annotations = parse_markdown(args.markdown.resolve())
    validate(annotations, list(config["samples"]))
    ordered = {sample: annotations[sample] for sample in config["samples"]}
    args.output.write_text(
        json.dumps(ordered, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Imported {len(ordered)} annotations into {args.output}")


if __name__ == "__main__":
    main()

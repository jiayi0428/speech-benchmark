"""Deterministic cleanup for model-generated transcription wrappers."""
from __future__ import annotations

import re


_META_PREFIX = re.compile(
    r"^(?:the speech transcribed verbatim is|"
    r"the original content of this audio is)\s*:\s*",
    re.IGNORECASE,
)


def strip_transcription_wrapper(text: str) -> str:
    """Remove only fixed meta prefixes and matching outer quotation marks."""
    cleaned = _META_PREFIX.sub("", text.strip(), count=1).strip()
    if len(cleaned) >= 3 and cleaned[0] in {"'", '"'}:
        if cleaned[-1] == cleaned[0]:
            cleaned = cleaned[1:-1]
        elif cleaned[-2:] == f"{cleaned[0]}.":
            cleaned = cleaned[1:-2] + "."
    return cleaned.strip()


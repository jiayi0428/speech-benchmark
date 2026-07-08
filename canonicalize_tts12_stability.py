"""Preserve the full C-summary call audit and keep the first success per key."""
from __future__ import annotations

import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RESULT_DIR = ROOT / "data" / "results" / "tts12_summary_stability_v1"
INPUT = RESULT_DIR / "c_summaries.jsonl"
AUDIT = RESULT_DIR / "c_summaries_call_audit.jsonl"


def main() -> None:
    records = [
        json.loads(line)
        for line in INPUT.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if AUDIT.exists():
        audit_records = [
            json.loads(line)
            for line in AUDIT.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        audited_keys = {
            (record["sample"], record["task"], record["repetition"])
            for record in audit_records
        }
        audit_records.extend(
            record
            for record in records
            if (record["sample"], record["task"], record["repetition"])
            not in audited_keys
        )
        AUDIT.write_text(
            "".join(
                json.dumps(record, ensure_ascii=False) + "\n"
                for record in audit_records
            ),
            encoding="utf-8",
        )
    else:
        shutil.copy2(INPUT, AUDIT)
        audit_records = records
    canonical = {}
    for record in records:
        if record.get("status") != "success":
            continue
        key = (record["sample"], record["task"], record["repetition"])
        canonical.setdefault(key, record)
    ordered = list(canonical.values())
    INPUT.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False) + "\n"
            for record in ordered
        ),
        encoding="utf-8",
    )
    print(
        f"Audit records={len(audit_records)}; canonical successes={len(ordered)}; "
        f"duplicate calls={len(audit_records) - len(ordered)}"
    )


if __name__ == "__main__":
    main()

"""Export all clean public Malaysian tender notices to JSONL.

Reads ONLY the public `tenders` table metadata columns. Never exports raw portal blobs,
private tables, email addresses, or excluded notices (matched by salted SHA-256 digest).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
from collections import defaultdict
from pathlib import Path

JUNK_TITLE_RE = re.compile(r"^(test|testing|dfhghf|asdf|xxx|dummy)\b", re.IGNORECASE)
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
COMPOUND_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[/-][a-z0-9]+)*")

GUARD_SALT = "tender-os-nda-guard:v1:"
EXCLUDED_TOKEN_HASHES = frozenset(
    {
        "02f6f40c3603acff",
        "25356cd999eb85aa",
        "413566d9a36d48f5",
        "6aafabd399b805ee",
        "b8cfc94ddacf3da3",
    }
)
EXCLUDED_PHRASE_HASHES = frozenset(
    {
        "207bb2799c8c7eb2",
        "4dcfdbb49acec6c5",
    }
)
ALLOW_PUBLIC_PLACE_ID_HASHES = frozenset(
    {
        "3f74e6126c82671d",
        "c24230f0dd3f60d3",
        "c35e5c973d3dfb3e",
    }
)


def _digest(value: str) -> str:
    norm = " ".join(value.lower().split())
    return hashlib.sha256((GUARD_SALT + norm).encode("utf-8")).hexdigest()[:16]


def is_excluded_notice(tender_id: str, text: str) -> bool:
    tokens = COMPOUND_TOKEN_RE.findall(text.lower())
    for a, b in zip(tokens, tokens[1:], strict=False):
        if _digest(f"{a} {b}") in EXCLUDED_PHRASE_HASHES:
            return True
    if _digest(tender_id) in ALLOW_PUBLIC_PLACE_ID_HASHES:
        return False
    for tok in tokens:
        if _digest(tok) in EXCLUDED_TOKEN_HASHES:
            return True
        for part in re.split(r"[/-]", tok):
            if part and _digest(part) in EXCLUDED_TOKEN_HASHES:
                return True
    return False


def clean_text(value: str | None) -> str | None:
    if value is None:
        return None
    text = " ".join(value.split()).strip()
    if not text:
        return None
    return EMAIL_RE.sub("[redacted-email]", text)


def export_all_public(source_db: Path, output_path: Path) -> dict[str, int]:
    conn = sqlite3.connect(f"file:{source_db}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT source, reference, title, agency, category, location,
                   closing_at, value_myr, eligibility, url, first_seen
            FROM tenders
            WHERE title IS NOT NULL AND length(trim(title)) >= 12
              AND url IS NOT NULL AND length(trim(url)) > 0
            ORDER BY source ASC, reference ASC
            """
        ).fetchall()
    finally:
        conn.close()

    records: list[dict[str, str | None]] = []
    seen_ids: set[str] = set()
    counts: dict[str, int] = defaultdict(int)

    for row in rows:
        title = clean_text(row["title"])
        source = clean_text(row["source"]) or ""
        reference = clean_text(row["reference"]) or ""
        if not title or not source or not reference:
            continue
        tender_id = f"{source}:{reference}"
        agency = clean_text(row["agency"]) or "Unknown agency"
        location = clean_text(row["location"])
        if JUNK_TITLE_RE.search(title):
            continue
        if is_excluded_notice(tender_id, f"{reference} {title} {agency} {location or ''}"):
            continue
        if tender_id in seen_ids:
            continue
        seen_ids.add(tender_id)
        records.append(
            {
                "id": tender_id,
                "source": source,
                "reference": reference,
                "title": title,
                "agency": agency,
                "category": clean_text(row["category"]),
                "location": location,
                "closing_at": clean_text(row["closing_at"]),
                "value_myr": clean_text(row["value_myr"]),
                "eligibility": clean_text(row["eligibility"]),
                "url": clean_text(row["url"]) or "",
                "published_at": clean_text(row["first_seen"]),
            }
        )
        counts[source] += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def main() -> None:
    parser = argparse.ArgumentParser(description="Export all clean public Malaysian tenders.")
    parser.add_argument("--source-db", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/tenders_public.jsonl"))
    args = parser.parse_args()

    counts = export_all_public(args.source_db, args.output)
    print(f"Exported {sum(counts.values())} public tenders to {args.output}")
    for src, n in counts.items():
        print(f"  {src:16s} {n}")


if __name__ == "__main__":
    main()

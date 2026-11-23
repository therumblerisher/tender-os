"""Structural CI guards: single ILMU router boundary and hashed NDA anonymity enforcement.

The NDA denylist is stored exclusively as salted SHA-256 prefixes so zero private strings
appear anywhere in the public repository, and ZERO source or data files are exempted.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ALLOWED_ILMU_FILES = {
    Path("src/tender_os/ilmu/router.py"),
    Path("src/tender_os/config.py"),
}

GUARD_SALT = "tender-os-nda-guard:v1:"
COMPOUND_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[/-][a-z0-9]+)*")
FORBIDDEN_TOKEN_HASHES = frozenset(
    {
        "02f6f40c3603acff",
        "25356cd999eb85aa",
        "413566d9a36d48f5",
        "6aafabd399b805ee",
        "b8cfc94ddacf3da3",
    }
)
FORBIDDEN_PHRASE_HASHES = frozenset(
    {
        "207bb2799c8c7eb2",
        "4dcfdbb49acec6c5",
    }
)
# Verified public Malaysian place-name notices in data/tenders_public.jsonl (hashed IDs only).
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


def _scan_text_for_hashes(text: str, *, allow_tokens: bool = False) -> str | None:
    tokens = COMPOUND_TOKEN_RE.findall(text.lower())
    for a, b in zip(tokens, tokens[1:], strict=False):
        h = _digest(f"{a} {b}")
        if h in FORBIDDEN_PHRASE_HASHES:
            return f"phrase-hash:{h}"
    if allow_tokens:
        return None
    for tok in tokens:
        h = _digest(tok)
        if h in FORBIDDEN_TOKEN_HASHES:
            return f"token-hash:{h}"
        for part in re.split(r"[/-]", tok):
            if part:
                ph = _digest(part)
                if ph in FORBIDDEN_TOKEN_HASHES:
                    return f"token-hash:{ph}"
    return None


def _iter_repo_files() -> list[Path]:
    files: list[Path] = []
    for folder in ("src", "tests", "evals", "scripts", "web", "data"):
        base = ROOT / folder
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if not path.is_file():
                continue
            if "__pycache__" in path.parts or path.suffix in {".pyc", ".f32"}:
                continue
            files.append(path)
    readme = ROOT / "README.md"
    if readme.is_file():
        files.append(readme)
    return sorted(files)


def test_single_ilmu_router_boundary() -> None:
    """Every direct reference to api.ilmu.ai must live in config.py or ilmu/router.py only."""
    violations: list[str] = []
    for path in _iter_repo_files():
        rel = path.relative_to(ROOT)
        if rel.parts[0] != "src":
            continue
        if rel in ALLOWED_ILMU_FILES:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if "api.ilmu.ai" in text:
            violations.append(str(rel))
    assert not violations, f"Direct api.ilmu.ai call outside router: {violations}"


def test_no_client_or_nda_strings_in_repo() -> None:
    """Fail CI if any hashed private client or excluded benchmark token enters any repo file."""
    violations: list[str] = []
    for path in _iter_repo_files():
        rel = path.relative_to(ROOT)
        if rel == Path("data/tenders_public.jsonl"):
            for lineno, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), start=1
            ):
                if not line.strip():
                    continue
                rec = json.loads(line)
                tid = str(rec.get("id", ""))
                allow_place_tok = _digest(tid) in ALLOW_PUBLIC_PLACE_ID_HASHES
                hit = _scan_text_for_hashes(line, allow_tokens=allow_place_tok)
                if hit:
                    violations.append(f"{rel}:{lineno} ({tid}): {hit}")
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        hit = _scan_text_for_hashes(text)
        if hit:
            violations.append(f"{rel}: {hit}")
    assert not violations, f"NDA/client hash matches detected: {violations}"

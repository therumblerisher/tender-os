"""Deterministic Malaysian MOF Kod Bidang and CIDB grade eligibility matcher.

Pure logic only. Never hides a tender silently; always classifies into
`eligible`, `check_code`, or `mismatch` with an explicit human-readable reason.
"""

from __future__ import annotations

import re

from tender_os.domain import EligibilityStatus, EligibilityVerdict, ProfileContext, TenderNotice

CURRENCY_AMOUNT_RE = re.compile(r"\b(?:RM|MYR)\s*[0-9][0-9,.]*", re.IGNORECASE)
MOF_CODE_RE = re.compile(r"\b([0-3][0-9]{5})\b")
CIDB_GRADE_RE = re.compile(r"\b(G[1-7])\b", re.IGNORECASE)
CIDB_UP_RE = re.compile(
    r"\b(G[1-7])\s*(?:\+|(?:dan\s+)?ke\s+atas|and\s+above)\b",
    re.IGNORECASE,
)
CIDB_RANGE_RE = re.compile(
    r"\b(G[1-7])\s*(?:-|hingga|to|sampai)\s*(G[1-7])\b",
    re.IGNORECASE,
)
PAREN_RE = re.compile(r"\([^)]*\)")
AND_JOIN_RE = re.compile(r"\b(DAN|AND)\b", re.IGNORECASE)
OR_JOIN_RE = re.compile(r"\b(ATAU|OR)\b", re.IGNORECASE)


def _strip_currency(text: str) -> str:
    return CURRENCY_AMOUNT_RE.sub(" ", text)


def extract_mof_codes(text: str) -> list[str]:
    return list(dict.fromkeys(MOF_CODE_RE.findall(_strip_currency(text))))


def extract_cidb_grades(text: str) -> list[str]:
    return list(dict.fromkeys(g.upper() for g in CIDB_GRADE_RE.findall(text)))


def _grade_num(grade: str) -> int:
    return int(grade.upper()[1])


def _allowed_cidb_grades(text: str, raw_grades: list[str]) -> set[str]:
    allowed: set[str] = set(raw_grades)
    for m in CIDB_UP_RE.finditer(text):
        floor = _grade_num(m.group(1))
        allowed.update(f"G{n}" for n in range(floor, 8))
    for m in CIDB_RANGE_RE.finditer(text):
        lo = _grade_num(m.group(1))
        hi = _grade_num(m.group(2))
        if lo <= hi:
            allowed.update(f"G{n}" for n in range(lo, hi + 1))
    return allowed


def _mof_requires_all(text: str) -> bool:
    """Return True only when DAN/AND explicitly joins two MOF codes outside parentheses."""
    stripped = PAREN_RE.sub(" ", _strip_currency(text))
    matches = list(MOF_CODE_RE.finditer(stripped))
    if len(matches) < 2:
        return False
    saw_and = False
    for prev, nxt in zip(matches, matches[1:], strict=False):
        between = stripped[prev.end() : nxt.start()]
        if OR_JOIN_RE.search(between):
            return False
        if AND_JOIN_RE.search(between):
            saw_and = True
    return saw_and


def evaluate_eligibility(profile: ProfileContext, tender: TenderNotice) -> EligibilityVerdict:
    """Deterministically evaluate whether a ProfileContext matches a tender's codes."""
    source_text = f"{tender.eligibility or ''} {tender.category or ''}".strip()
    req_mof = extract_mof_codes(source_text)
    req_cidb = extract_cidb_grades(source_text)

    if not req_mof and not req_cidb:
        return EligibilityVerdict(
            status=EligibilityStatus.CHECK_CODE,
            matched_codes=[],
            required_codes=[],
            reason="Notice lists no structured MOF or CIDB code; verify scope manually.",
        )

    prof_mof = {c.strip() for c in profile.mof_codes if c.strip()}
    prof_cidb = {g.strip().upper() for g in profile.cidb_grades if g.strip()}
    required: list[str] = [*req_mof, *req_cidb]

    if not prof_mof and not prof_cidb:
        return EligibilityVerdict(
            status=EligibilityStatus.CHECK_CODE,
            matched_codes=[],
            required_codes=required,
            reason=f"Profile lists no MOF or CIDB codes; notice requires {', '.join(required)}.",
        )

    matched: list[str] = []

    mof_ok = False
    mof_partial = False
    if req_mof:
        mof_hits = [c for c in req_mof if c in prof_mof]
        matched.extend(mof_hits)
        if _mof_requires_all(source_text):
            mof_ok = len(mof_hits) == len(req_mof)
            mof_partial = 0 < len(mof_hits) < len(req_mof)
        else:
            mof_ok = len(mof_hits) > 0

    cidb_ok = False
    if req_cidb:
        allowed_cidb = _allowed_cidb_grades(source_text, req_cidb)
        cidb_hits = sorted(g for g in prof_cidb if g in allowed_cidb)
        matched.extend(cidb_hits)
        cidb_ok = len(cidb_hits) > 0

    if req_mof and req_cidb:
        if mof_ok and cidb_ok:
            return EligibilityVerdict(
                status=EligibilityStatus.ELIGIBLE,
                matched_codes=matched,
                required_codes=required,
                reason=f"Matches required MOF and CIDB codes ({', '.join(matched)}).",
            )
        if matched:
            return EligibilityVerdict(
                status=EligibilityStatus.CHECK_CODE,
                matched_codes=matched,
                required_codes=required,
                reason=f"Partial match ({', '.join(matched)}); notice lists {', '.join(required)}.",
            )
    elif req_mof:
        if mof_ok:
            return EligibilityVerdict(
                status=EligibilityStatus.ELIGIBLE,
                matched_codes=matched,
                required_codes=required,
                reason=f"Matches MOF code {', '.join(matched)}.",
            )
        if mof_partial:
            return EligibilityVerdict(
                status=EligibilityStatus.CHECK_CODE,
                matched_codes=matched,
                required_codes=required,
                reason=f"Matches {', '.join(matched)}, but notice joins required codes with DAN ({', '.join(req_mof)}).",
            )
    elif req_cidb and cidb_ok:
        return EligibilityVerdict(
            status=EligibilityStatus.ELIGIBLE,
            matched_codes=matched,
            required_codes=required,
            reason=f"Matches CIDB grade {', '.join(matched)}.",
        )

    return EligibilityVerdict(
        status=EligibilityStatus.MISMATCH,
        matched_codes=[],
        required_codes=required,
        reason=f"Requires {', '.join(required)}; profile does not list a matching code.",
    )

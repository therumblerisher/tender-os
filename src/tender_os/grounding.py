"""Deterministic query routing and grounded citation/date verification."""

from __future__ import annotations

import re

from tender_os.domain import Citation, GroundingVerdict, RouteKind, TenderNotice

CURRENCY_AMOUNT_RE = re.compile(r"\b(?:RM|MYR)\s*[0-9][0-9,.]*", re.IGNORECASE)
EXACT_MOF_RE = re.compile(r"\b[0-3][0-9]{5}\b")
PORTAL_REF_RE = re.compile(r"\b[A-Z]{2,}[/\-.][A-Z0-9()/.\-]{3,}\b")
# Allows one level of inner brackets for portal IDs like [jkr:JKR/CJ/24961/2026 [ADENDUM]].
CITATION_RE = re.compile(r"\[([a-z][a-z0-9_]*:(?:[^\[\]]|\[[^\[\]]*\])+)\]")
ISO_DATE_RE = re.compile(r"\b(202[5-9]-[0-1][0-9]-[0-3][0-9])\b")
PUBLISHED_CUE_RE = re.compile(r"\b(publish(?:ed)?|diterbit(?:kan)?|first\s+seen)\b", re.IGNORECASE)

ABSTAIN_CUES = (
    "no matching",
    "could not find",
    "did not find",
    "no public tender",
    "no tenders found",
    "no open tenders were found",
    "not found in",
    "no relevant tender",
    "tiada tender",
)


def route_query(query: str) -> RouteKind:
    """Deterministically route exact MOF codes or portal references vs semantic queries."""
    stripped = CURRENCY_AMOUNT_RE.sub(" ", query.strip())
    if EXACT_MOF_RE.search(stripped) or PORTAL_REF_RE.search(stripped):
        return RouteKind.EXACT_CODE
    return RouteKind.SEMANTIC


def extract_citations(text: str) -> list[str]:
    return list(dict.fromkeys(m.group(1).strip() for m in CITATION_RE.finditer(text)))


def verify_grounding(
    answer: str,
    seen_tenders: dict[str, TenderNotice],
) -> GroundingVerdict:
    """Verify every cited [source:reference] ID and check inline ISO dates against DB rows."""
    cited_raw = extract_citations(answer)
    verified: list[Citation] = []
    unverified: list[str] = []
    date_mismatches: list[str] = []

    for cid in cited_raw:
        tender = seen_tenders.get(cid)
        if tender is None:
            unverified.append(cid)
            continue
        verified.append(
            Citation(
                tender_id=tender.id,
                title=tender.title,
                url=tender.url,
                closing_at=tender.closing_at,
            )
        )

    # Line-level ISO date check: if a line cites exactly one verified tender and states an ISO date,
    # require it to match closing_at (or published_at only when the line explicitly mentions publishing).
    for line in answer.splitlines():
        line_cids = extract_citations(line)
        if len(line_cids) != 1:
            continue
        tender = seen_tenders.get(line_cids[0])
        if tender is None:
            continue
        found_dates = ISO_DATE_RE.findall(line)
        if not found_dates:
            continue
        closing_day = tender.closing_at[:10] if tender.closing_at and len(tender.closing_at) >= 10 else None
        published_day = (
            tender.published_at[:10]
            if tender.published_at and len(tender.published_at) >= 10
            else None
        )
        allowed_dates: set[str] = set()
        if closing_day:
            allowed_dates.add(closing_day)
        if published_day and PUBLISHED_CUE_RE.search(line):
            allowed_dates.add(published_day)
        for found_date in found_dates:
            if found_date not in allowed_dates:
                expected = closing_day or "none recorded"
                date_mismatches.append(
                    f"{tender.id}: stated {found_date}, expected closing_at {expected}"
                )

    lowered = answer.lower()
    is_abstaining = any(cue in lowered for cue in ABSTAIN_CUES)

    if not verified and not unverified and (not seen_tenders or is_abstaining):
        return GroundingVerdict(
            status="abstained",
            abstain_reason="zero_hits" if not seen_tenders else "model_abstained",
            verified_citations=[],
            unverified_ids=[],
            date_mismatches=[],
        )
    if verified and not unverified and not date_mismatches:
        return GroundingVerdict(
            status="verified",
            verified_citations=verified,
            unverified_ids=[],
            date_mismatches=[],
        )
    if not seen_tenders and unverified:
        return GroundingVerdict(
            status="abstained",
            abstain_reason="zero_hits",
            verified_citations=[],
            unverified_ids=unverified,
            date_mismatches=[],
        )
    if not verified:
        return GroundingVerdict(
            status="unverified",
            verified_citations=[],
            unverified_ids=unverified,
            date_mismatches=date_mismatches,
        )
    return GroundingVerdict(
        status="partial",
        verified_citations=verified,
        unverified_ids=unverified,
        date_mismatches=date_mismatches,
    )

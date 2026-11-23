"""Property-based and behavioral tests for the pure Tender OS domain core."""

from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

from tender_os.domain import (
    EligibilityStatus,
    ProfileContext,
    RouteKind,
    TenderNotice,
    expand_location_terms,
)
from tender_os.eligibility import evaluate_eligibility, extract_cidb_grades, extract_mof_codes
from tender_os.fusion import RankedCandidate, fuse_rrf
from tender_os.grounding import route_query, verify_grounding


def _make_tender(
    tid: str = "dbkl:2026/A065 (ONLINE)",
    eligibility: str | None = "Kod Bidang: 221801 ATAU 221802",
    closing_at: str | None = "2026-12-10T12:00:00+08:00",
    published_at: str | None = "2026-11-01T08:00:00+08:00",
) -> TenderNotice:
    return TenderNotice(
        id=tid,
        source=tid.split(":")[0],
        reference=tid.split(":", 1)[1],
        title="PERKHIDMATAN PANEL SYARIKAT INSURANS DBKL",
        agency="Dewan Bandaraya Kuala Lumpur",
        location="Kuala Lumpur",
        closing_at=closing_at,
        published_at=published_at,
        eligibility=eligibility,
        url="https://example.gov.my/tender",
    )


@given(
    req_code=st.from_regex(r"[0-3][0-9]{5}", fullmatch=True),
    prof_code=st.from_regex(r"[0-3][0-9]{5}", fullmatch=True),
)
def test_eligibility_property_disjoint_codes_never_eligible(req_code: str, prof_code: str) -> None:
    tender = _make_tender(eligibility=f"Kod Bidang: {req_code}")
    verdict = evaluate_eligibility(ProfileContext(mof_codes=[prof_code]), tender)
    if req_code == prof_code:
        assert verdict.status == EligibilityStatus.ELIGIBLE
        assert req_code in verdict.matched_codes
    else:
        assert verdict.status == EligibilityStatus.MISMATCH
        assert verdict.matched_codes == []


def test_eligibility_handles_dan_atau_parentheses_cidb_ordering_and_empty_profile() -> None:
    # DAN between codes requires both
    both = _make_tender(eligibility="Kod Bidang: 210103 DAN 210104")
    assert (
        evaluate_eligibility(ProfileContext(mof_codes=["210103"]), both).status
        == EligibilityStatus.CHECK_CODE
    )
    assert (
        evaluate_eligibility(ProfileContext(mof_codes=["210103", "210104"]), both).status
        == EligibilityStatus.ELIGIBLE
    )

    # DAN inside parenthetical description must NOT turn ATAU into AND
    atau_with_dan_desc = _make_tender(
        eligibility="010302 (Peralatan Sistem Bunyi, Pembesar Suara Dan Projector) ATAU 210102"
    )
    assert (
        evaluate_eligibility(ProfileContext(mof_codes=["010302"]), atau_with_dan_desc).status
        == EligibilityStatus.ELIGIBLE
    )

    # CIDB grade exact and "dan ke atas" ordering
    cidb = _make_tender(eligibility="CIDB Gred G7 Kategori B")
    assert extract_cidb_grades(cidb.eligibility or "") == ["G7"]
    assert (
        evaluate_eligibility(ProfileContext(cidb_grades=["g7"]), cidb).status
        == EligibilityStatus.ELIGIBLE
    )
    assert (
        evaluate_eligibility(ProfileContext(cidb_grades=["G4"]), cidb).status
        == EligibilityStatus.MISMATCH
    )

    cidb_up = _make_tender(eligibility="CIDB Gred G4 dan ke atas Kategori B")
    assert (
        evaluate_eligibility(ProfileContext(cidb_grades=["G7"]), cidb_up).status
        == EligibilityStatus.ELIGIBLE
    )
    assert (
        evaluate_eligibility(ProfileContext(cidb_grades=["G3"]), cidb_up).status
        == EligibilityStatus.MISMATCH
    )

    # Empty profile -> CHECK_CODE, not MISMATCH
    assert evaluate_eligibility(ProfileContext(), cidb).status == EligibilityStatus.CHECK_CODE

    # Missing codes -> CHECK_CODE
    blank = _make_tender(eligibility=None)
    assert (
        evaluate_eligibility(ProfileContext(mof_codes=["210103"]), blank).status
        == EligibilityStatus.CHECK_CODE
    )
    assert extract_mof_codes("RM 250000 Kod 210103 DAN 210103") == ["210103"]


@given(n=st.integers(min_value=2, max_value=15))
def test_rrf_property_dual_list_hit_ranks_first(n: int) -> None:
    shared = _make_tender(tid="shared:000")
    fts_only = [_make_tender(tid=f"fts:{i:03d}") for i in range(n)]
    vec_only = [_make_tender(tid=f"vec:{i:03d}") for i in range(n)]

    fts_list = [RankedCandidate(fts_only[0], 10.0), RankedCandidate(shared, 9.0)] + [
        RankedCandidate(t, 5.0) for t in fts_only[1:]
    ]
    vec_list = [RankedCandidate(vec_only[0], 0.99), RankedCandidate(shared, 0.95)] + [
        RankedCandidate(t, 0.5) for t in vec_only[1:]
    ]
    fused = fuse_rrf(fts_list, vec_list, k=60)
    assert fused[0].tender.id == "shared:000"
    assert fused[0].breakdown.fts_rank == 2
    assert fused[0].breakdown.vector_rank == 2


def test_route_query_and_grounding_brackets_dates_and_uncited() -> None:
    assert route_query("Find tenders for 210110") == RouteKind.EXACT_CODE
    assert route_query("JKR/SEL/(P)/016(T)/2026") == RouteKind.EXACT_CODE
    assert route_query("hospital air conditioning maintenance under RM 250000") == RouteKind.SEMANTIC
    assert "pulau pinang" in expand_location_terms("Penang")

    t = _make_tender(closing_at="2026-12-10T12:00:00+08:00", published_at="2026-11-01T08:00:00+08:00")
    t_bracket = _make_tender(tid="jkr:JKR/CJ/24961/2026 [ADENDUM]", closing_at=None)
    seen = {t.id: t, t_bracket.id: t_bracket}

    ok = verify_grounding("DBKL opened [dbkl:2026/A065 (ONLINE)], closing 2026-12-10.", seen)
    assert ok.status == "verified"
    assert ok.date_mismatches == []

    # Bracketed IDs like [ADENDUM] must verify cleanly
    ok_bracket = verify_grounding("JKR issued [jkr:JKR/CJ/24961/2026 [ADENDUM]].", seen)
    assert ok_bracket.status == "verified"
    assert ok_bracket.verified_citations[0].tender_id == "jkr:JKR/CJ/24961/2026 [ADENDUM]"

    # Quoting published_at as closing date must be flagged
    pub_as_close = verify_grounding("DBKL opened [dbkl:2026/A065 (ONLINE)], closing 2026-11-01.", seen)
    assert pub_as_close.status == "partial"
    assert len(pub_as_close.date_mismatches) == 1

    # Fabricated date when closing_at is missing must be flagged
    no_close_fab = verify_grounding("JKR [jkr:JKR/CJ/24961/2026 [ADENDUM]] closes 2026-12-30.", seen)
    assert no_close_fab.status == "partial"
    assert len(no_close_fab.date_mismatches) == 1

    bad_id = verify_grounding("Found [dbkl:2026/A065 (ONLINE)] and [fake:999].", seen)
    assert bad_id.status == "partial"
    assert bad_id.unverified_ids == ["fake:999"]

    # Uncited non-abstaining answer fails as "unverified"
    uncited = verify_grounding("There is an insurance tender open right now.", seen)
    assert uncited.status == "unverified"

    abstain = verify_grounding("No matching public tenders were found.", {})
    assert abstain.status == "abstained"
    assert abstain.abstain_reason == "zero_hits"

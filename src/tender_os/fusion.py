"""Pure Reciprocal Rank Fusion (RRF) for merging full-text and dense vector lists."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from tender_os.domain import ScoreBreakdown, SearchHit, TenderNotice


@dataclass(frozen=True)
class RankedCandidate:
    tender: TenderNotice
    score: float


def fuse_rrf(
    fts_rows: Sequence[RankedCandidate],
    vec_rows: Sequence[RankedCandidate],
    *,
    k: int = 60,
) -> list[SearchHit]:
    """Deterministically merge FTS and vector rankings via 1/(k + rank)."""
    tenders: dict[str, TenderNotice] = {}
    scores: dict[str, float] = {}
    fts_info: dict[str, tuple[int, float]] = {}
    vec_info: dict[str, tuple[int, float]] = {}

    for rank, row in enumerate(fts_rows, start=1):
        did = row.tender.id
        tenders[did] = row.tender
        scores[did] = scores.get(did, 0.0) + 1.0 / (k + rank)
        fts_info[did] = (rank, row.score)

    for rank, row in enumerate(vec_rows, start=1):
        did = row.tender.id
        tenders[did] = row.tender
        scores[did] = scores.get(did, 0.0) + 1.0 / (k + rank)
        vec_info[did] = (rank, row.score)

    ordered = sorted(scores.keys(), key=lambda did: (-scores[did], did))
    hits: list[SearchHit] = []
    for did in ordered:
        f_rank, f_score = fts_info.get(did, (None, None))
        v_rank, v_score = vec_info.get(did, (None, None))
        hits.append(
            SearchHit(
                tender=tenders[did],
                score=scores[did],
                breakdown=ScoreBreakdown(
                    fts_rank=f_rank,
                    fts_score=f_score,
                    vector_rank=v_rank,
                    vector_score=v_score,
                    rrf_score=scores[did],
                ),
            )
        )
    return hits

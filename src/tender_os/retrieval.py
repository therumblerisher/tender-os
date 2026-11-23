"""Hybrid retrieval engine combining Postgres FTS, pgvector cosine search, RRF, and ILMU rerank."""

from __future__ import annotations

import time

from tender_os.config import Settings
from tender_os.domain import (
    AttemptRecord,
    ScoreBreakdown,
    SearchHit,
    SearchMode,
    SearchRequest,
    SearchResponse,
)
from tender_os.eligibility import evaluate_eligibility
from tender_os.fusion import RankedCandidate, fuse_rrf
from tender_os.ilmu.router import ILMUError, ILMURouter
from tender_os.ilmu.spend import SpendCapExceededError
from tender_os.storage.postgres import PostgresStore


def _attach_eligibility(hits: list[SearchHit], request: SearchRequest) -> list[SearchHit]:
    if request.profile is None:
        return hits
    return [
        hit.model_copy(
            update={"eligibility_verdict": evaluate_eligibility(request.profile, hit.tender)}
        )
        for hit in hits
    ]


def _fts_candidates_to_hits(rows: list[RankedCandidate], limit: int) -> list[SearchHit]:
    return [
        SearchHit(
            tender=row.tender,
            score=row.score,
            breakdown=ScoreBreakdown(fts_rank=rank, fts_score=row.score),
        )
        for rank, row in enumerate(rows[:limit], start=1)
    ]


def _vec_candidates_to_hits(rows: list[RankedCandidate], limit: int) -> list[SearchHit]:
    return [
        SearchHit(
            tender=row.tender,
            score=row.score,
            breakdown=ScoreBreakdown(vector_rank=rank, vector_score=row.score),
        )
        for rank, row in enumerate(rows[:limit], start=1)
    ]


class RetrievalEngine:
    """Executes `pg_fts`, `vector`, `hybrid`, and `hybrid_rerank` with explicit degradation."""

    def __init__(
        self,
        store: PostgresStore,
        router: ILMURouter | None,
        settings: Settings,
    ) -> None:
        self.store = store
        self.router = router
        self.settings = settings

    def search(self, request: SearchRequest) -> SearchResponse:
        t0 = time.monotonic()
        dismissed = request.profile.dismissed_ids if request.profile else []
        attempts: list[AttemptRecord] = []
        pool_size = max(self.settings.candidate_pool_size, request.limit)

        if request.mode == SearchMode.PG_FTS:
            fts_rows = self.store.search_fts(
                request.query,
                filters=request.filters,
                limit=pool_size,
                dismissed_ids=dismissed,
            )
            hits = _attach_eligibility(_fts_candidates_to_hits(fts_rows, request.limit), request)
            return SearchResponse(
                query=request.query,
                mode=request.mode,
                effective_mode=SearchMode.PG_FTS,
                degraded_mode=False,
                total_candidates=len(fts_rows),
                latency_ms=round((time.monotonic() - t0) * 1000, 2),
                hits=hits,
                attempts=attempts,
            )

        # Modes needing ILMU query embedding: degrade cleanly to pg_fts if key missing or cap/error hit.
        if self.router is None or not self.router.has_api_key:
            return self._degrade_to_fts(
                request,
                t0=t0,
                dismissed=dismissed,
                pool_size=pool_size,
                reason="missing_ilmu_api_key",
                attempts=attempts,
            )

        try:
            q_vec, embed_attempts = self.router.embed_query(request.query)
            attempts.extend(embed_attempts)
        except SpendCapExceededError as exc:
            attempts.extend(a for a in exc.attempts if isinstance(a, AttemptRecord))
            return self._degrade_to_fts(
                request,
                t0=t0,
                dismissed=dismissed,
                pool_size=pool_size,
                reason="spend_cap_exceeded",
                attempts=attempts,
            )
        except ILMUError as exc:
            attempts.extend(exc.attempts)
            return self._degrade_to_fts(
                request,
                t0=t0,
                dismissed=dismissed,
                pool_size=pool_size,
                reason="ilmu_embed_error",
                attempts=attempts,
            )

        vec_rows = self.store.search_vector(
            q_vec,
            filters=request.filters,
            limit=pool_size,
            dismissed_ids=dismissed,
        )

        if request.mode == SearchMode.VECTOR:
            hits = _attach_eligibility(_vec_candidates_to_hits(vec_rows, request.limit), request)
            return SearchResponse(
                query=request.query,
                mode=request.mode,
                effective_mode=SearchMode.VECTOR,
                degraded_mode=False,
                total_candidates=len(vec_rows),
                latency_ms=round((time.monotonic() - t0) * 1000, 2),
                hits=hits,
                attempts=attempts,
            )

        fts_rows = self.store.search_fts(
            request.query,
            filters=request.filters,
            limit=pool_size,
            dismissed_ids=dismissed,
        )
        fused = fuse_rrf(fts_rows, vec_rows, k=self.settings.rrf_k)

        if request.mode == SearchMode.HYBRID or not fused:
            hits = _attach_eligibility(fused[: request.limit], request)
            return SearchResponse(
                query=request.query,
                mode=request.mode,
                effective_mode=SearchMode.HYBRID,
                degraded_mode=False,
                total_candidates=len(fused),
                latency_ms=round((time.monotonic() - t0) * 1000, 2),
                hits=hits,
                attempts=attempts,
            )

        # HYBRID_RERANK: rerank top-K RRF pool via YTL ILMU bge-reranker, degrade to hybrid on failure.
        rerank_k = max(self.settings.rerank_pool_size, request.limit)
        rerank_pool = fused[:rerank_k]
        docs = [hit.tender.semantic_document() for hit in rerank_pool]

        try:
            ranked, rerank_attempts = self.router.rerank(request.query, docs)
            attempts.extend(rerank_attempts)
        except SpendCapExceededError as exc:
            attempts.extend(a for a in exc.attempts if isinstance(a, AttemptRecord))
            hits = _attach_eligibility(fused[: request.limit], request)
            return SearchResponse(
                query=request.query,
                mode=request.mode,
                effective_mode=SearchMode.HYBRID,
                degraded_mode=True,
                degraded_reason="rerank_spend_cap_exceeded",
                total_candidates=len(fused),
                latency_ms=round((time.monotonic() - t0) * 1000, 2),
                hits=hits,
                attempts=attempts,
            )
        except ILMUError as exc:
            attempts.extend(exc.attempts)
            hits = _attach_eligibility(fused[: request.limit], request)
            return SearchResponse(
                query=request.query,
                mode=request.mode,
                effective_mode=SearchMode.HYBRID,
                degraded_mode=True,
                degraded_reason="rerank_fallback_to_hybrid",
                total_candidates=len(fused),
                latency_ms=round((time.monotonic() - t0) * 1000, 2),
                hits=hits,
                attempts=attempts,
            )

        reranked_hits: list[SearchHit] = []
        seen_idx: set[int] = set()
        for idx, score in ranked:
            if idx in seen_idx or not (0 <= idx < len(rerank_pool)):
                continue
            seen_idx.add(idx)
            orig = rerank_pool[idx]
            reranked_hits.append(
                orig.model_copy(
                    update={
                        "score": score,
                        "breakdown": orig.breakdown.model_copy(update={"rerank_score": score}),
                    }
                )
            )
        # Preserve any candidates omitted by the reranker at the tail.
        for idx, orig in enumerate(rerank_pool):
            if idx not in seen_idx:
                reranked_hits.append(orig)

        hits = _attach_eligibility(reranked_hits[: request.limit], request)
        return SearchResponse(
            query=request.query,
            mode=request.mode,
            effective_mode=SearchMode.HYBRID_RERANK,
            degraded_mode=False,
            total_candidates=len(fused),
            latency_ms=round((time.monotonic() - t0) * 1000, 2),
            hits=hits,
            attempts=attempts,
        )

    def _degrade_to_fts(
        self,
        request: SearchRequest,
        *,
        t0: float,
        dismissed: list[str],
        pool_size: int,
        reason: str,
        attempts: list[AttemptRecord],
    ) -> SearchResponse:
        fts_rows = self.store.search_fts(
            request.query,
            filters=request.filters,
            limit=pool_size,
            dismissed_ids=dismissed,
        )
        hits = _attach_eligibility(_fts_candidates_to_hits(fts_rows, request.limit), request)
        return SearchResponse(
            query=request.query,
            mode=request.mode,
            effective_mode=SearchMode.PG_FTS,
            degraded_mode=True,
            degraded_reason=reason,
            total_candidates=len(fts_rows),
            latency_ms=round((time.monotonic() - t0) * 1000, 2),
            hits=hits,
            attempts=attempts,
        )

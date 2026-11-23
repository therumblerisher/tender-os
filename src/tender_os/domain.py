"""Pure domain types for Tender OS (zero I/O, zero network, zero database imports)."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal, NewType

from pydantic import BaseModel, Field

TenderId = NewType("TenderId", str)
MofCode = NewType("MofCode", str)
CidbGrade = NewType("CidbGrade", str)
MicroRm = NewType("MicroRm", int)

LOCATION_ALIASES: dict[str, tuple[str, ...]] = {
    "penang": ("penang", "pulau pinang"),
    "pulau pinang": ("pulau pinang", "penang"),
    "malacca": ("malacca", "melaka"),
    "melaka": ("melaka", "malacca"),
    "kl": ("kuala lumpur", "wp kl", "wilayah persekutuan"),
    "kuala lumpur": ("kuala lumpur", "wp kl"),
}


def expand_location_terms(location: str) -> tuple[str, ...]:
    key = location.strip().lower()
    if not key:
        return ()
    return LOCATION_ALIASES.get(key, (location.strip(),))


class SearchMode(StrEnum):
    PG_FTS = "pg_fts"
    VECTOR = "vector"
    HYBRID = "hybrid"
    HYBRID_RERANK = "hybrid_rerank"


class RouteKind(StrEnum):
    EXACT_CODE = "exact_code"
    SEMANTIC = "semantic"


class EligibilityStatus(StrEnum):
    ELIGIBLE = "eligible"
    CHECK_CODE = "check_code"
    MISMATCH = "mismatch"


class EligibilityVerdict(BaseModel):
    status: EligibilityStatus
    matched_codes: list[str] = Field(default_factory=list)
    required_codes: list[str] = Field(default_factory=list)
    reason: str


class ProfileContext(BaseModel):
    """Stateless per-request company criteria (presets in code; edits stay in browser localStorage)."""

    mof_codes: list[str] = Field(default_factory=list)
    cidb_grades: list[str] = Field(default_factory=list)
    preferred_states: list[str] = Field(default_factory=list)
    dismissed_ids: list[str] = Field(default_factory=list)


class TenderNotice(BaseModel):
    id: str = Field(..., min_length=3)
    source: str = Field(..., min_length=1)
    reference: str = Field(..., min_length=1)
    title: str = Field(..., min_length=1)
    agency: str = Field(default="Unknown agency")
    category: str | None = None
    location: str | None = None
    closing_at: str | None = None
    value_myr: str | None = None
    eligibility: str | None = None
    url: str = Field(default="")
    published_at: str | None = None

    def search_document(self) -> str:
        """Structured text used for full-text storage."""
        parts = [self.title, self.agency, self.source, self.reference]
        for field in (
            self.category,
            self.location,
            self.eligibility,
            self.value_myr,
            self.closing_at,
        ):
            if field:
                parts.append(field)
        return " | ".join(p.strip() for p in parts if p and p.strip())

    def semantic_document(self) -> str:
        """Lowercased, label-free text for neural embeddings and cross-encoder reranking.

        PR0 proved lowercasing ALL-CAPS Bahasa Melayu portal titles reduces bge-m3 token
        fragmentation by 20.5% and lifts English-to-Malay cross-lingual MRR@10 by +0.101.
        """
        parts = [self.title, self.agency]
        for field in (self.location, self.category, self.eligibility):
            if field:
                parts.append(field)
        parts.append(self.reference)
        return " | ".join(p.strip() for p in parts if p and p.strip()).lower()


class SearchFilters(BaseModel):
    source: str | None = None
    location: str | None = None
    agency: str | None = None
    eligibility_code: str | None = None
    open_after: str | None = None


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    mode: SearchMode = SearchMode.HYBRID_RERANK
    limit: int = Field(default=10, ge=1, le=50)
    filters: SearchFilters = Field(default_factory=SearchFilters)
    profile: ProfileContext | None = None


class ScoreBreakdown(BaseModel):
    fts_rank: int | None = None
    fts_score: float | None = None
    vector_rank: int | None = None
    vector_score: float | None = None
    rrf_score: float | None = None
    rerank_score: float | None = None


class SearchHit(BaseModel):
    tender: TenderNotice
    score: float
    breakdown: ScoreBreakdown
    eligibility_verdict: EligibilityVerdict | None = None


class AttemptRecord(BaseModel):
    stage: Literal["embed", "rerank", "chat"]
    model: str
    fallback_used: bool = False
    latency_ms: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_micro_rm: int = 0
    pricing_basis: Literal["published", "assumed_conservative"] = "published"
    outcome: Literal[
        "ok", "timeout", "rate_limited", "http_error", "schema_invalid", "cap_blocked"
    ] = "ok"


class SearchResponse(BaseModel):
    query: str
    mode: SearchMode
    effective_mode: SearchMode
    degraded_mode: bool = False
    degraded_reason: str | None = None
    total_candidates: int
    latency_ms: float
    hits: list[SearchHit]
    attempts: list[AttemptRecord] = Field(default_factory=list)


class Citation(BaseModel):
    tender_id: str
    title: str
    url: str
    closing_at: str | None = None


class GroundingVerdict(BaseModel):
    status: Literal["verified", "partial", "unverified", "abstained"]
    abstain_reason: (
        Literal["rerank_below_floor", "zero_hits", "model_abstained", "degraded_no_key"] | None
    ) = None
    verified_citations: list[Citation] = Field(default_factory=list)
    unverified_ids: list[str] = Field(default_factory=list)
    date_mismatches: list[str] = Field(default_factory=list)


class QueryTrace(BaseModel):
    route: RouteKind
    retrieval_mode: SearchMode
    effective_mode: SearchMode
    candidates_considered: int
    top_rerank_score: float | None = None
    rerank_floor: float | None = None
    db_ms: float = 0.0
    total_ms: float = 0.0
    total_cost_micro_rm: int = 0
    attempts: list[AttemptRecord] = Field(default_factory=list)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=2, max_length=1000)
    filters: SearchFilters = Field(default_factory=SearchFilters)
    profile: ProfileContext | None = None


class AskResponse(BaseModel):
    question: str
    answer: str
    route: RouteKind
    degraded_mode: bool = False
    grounding: GroundingVerdict
    hits: list[SearchHit] = Field(default_factory=list)
    trace: QueryTrace

"""FastAPI service for Tender OS with sync threadpool handlers and sliding-window rate limits."""

from __future__ import annotations

import logging
import math
import threading
import time
from collections import deque
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

from tender_os.agent import TenderAgent
from tender_os.config import Settings, get_settings
from tender_os.domain import AskRequest, AskResponse, SearchRequest, SearchResponse, TenderNotice
from tender_os.ilmu.router import ILMURouter
from tender_os.ilmu.spend import PostgresSpendLedger
from tender_os.retrieval import RetrievalEngine
from tender_os.storage.postgres import PostgresStore

logger = logging.getLogger(__name__)


class HealthResponse(BaseModel):
    status: str
    tenders: int
    embedded: int
    ilmu_configured: bool
    spend_today_myr: float
    daily_cap_myr: float


@dataclass
class _AppDeps:
    store: PostgresStore
    ledger: PostgresSpendLedger
    router: ILMURouter
    engine: RetrievalEngine
    agent: TenderAgent


class SlidingWindowRateLimiter:
    """Thread-safe per-key sliding-window rate limiter with bounded stale-key pruning."""

    def __init__(self, max_requests: int, window_seconds: float = 60.0) -> None:
        self.max_requests = max(1, max_requests)
        self.window_seconds = window_seconds
        self._lock = threading.Lock()
        self._hits: dict[str, deque[float]] = {}

    def allow(self, key: str) -> tuple[bool, int]:
        now = time.monotonic()
        cutoff = now - self.window_seconds
        with self._lock:
            if len(self._hits) > 4096:
                stale = [k for k, dq in self._hits.items() if not dq or dq[-1] < cutoff]
                for k in stale:
                    del self._hits[k]
            q = self._hits.setdefault(key, deque())
            while q and q[0] < cutoff:
                q.popleft()
            if len(q) >= self.max_requests:
                retry_after = max(1, int(math.ceil((q[0] + self.window_seconds) - now)))
                return False, retry_after
            q.append(now)
            return True, 0


def _client_key(request: Request) -> str:
    return request.client.host if request.client and request.client.host else "unknown"


def create_app(
    *,
    settings: Settings | None = None,
    store: PostgresStore | None = None,
    router: ILMURouter | None = None,
    engine: RetrievalEngine | None = None,
    agent: TenderAgent | None = None,
    spend_ledger: PostgresSpendLedger | None = None,
) -> FastAPI:
    """Build the FastAPI application cleanly without connecting on import."""
    resolved_settings = settings or get_settings()
    limiter = SlidingWindowRateLimiter(resolved_settings.rate_limit_rpm, window_seconds=60.0)
    holder: dict[str, _AppDeps] = {}

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        owned_store: PostgresStore | None = None
        owned_router: ILMURouter | None = None

        active_store = store
        if active_store is None:
            owned_store = PostgresStore(
                resolved_settings.database_url,
                min_size=resolved_settings.db_pool_min_size,
                max_size=resolved_settings.db_pool_max_size,
            )
            active_store = owned_store

        active_ledger = spend_ledger or PostgresSpendLedger(
            active_store.pool,
            cap_micro_rm=resolved_settings.daily_cap_micro_rm,
        )

        active_router = router
        if active_router is None:
            owned_router = ILMURouter(resolved_settings, spend_ledger=active_ledger)
            active_router = owned_router

        active_engine = engine or RetrievalEngine(active_store, active_router, resolved_settings)
        active_agent = agent or TenderAgent(
            active_store,
            active_engine,
            active_router,
            resolved_settings,
        )

        holder["deps"] = _AppDeps(
            store=active_store,
            ledger=active_ledger,
            router=active_router,
            engine=active_engine,
            agent=active_agent,
        )
        try:
            yield
        finally:
            if owned_router is not None:
                owned_router.close()
            if owned_store is not None:
                owned_store.close()

    app = FastAPI(
        title="Tender OS",
        version="0.2.0",
        description="Hybrid RAG search and grounded Q&A over public Malaysian government tenders.",
        lifespan=lifespan,
    )

    def _deps() -> _AppDeps:
        deps = holder.get("deps")
        if deps is None:
            raise HTTPException(status_code=503, detail="Service initializing.")
        return deps

    def _require_rate_limit(request: Request) -> None:
        allowed, retry_after = limiter.allow(_client_key(request))
        if not allowed:
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded; please retry shortly.",
                headers={"Retry-After": str(retry_after)},
            )

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        deps = _deps()
        try:
            total, embedded = deps.store.count_tenders()
            settled = deps.ledger.today_summary().settled_micro_rm
        except Exception as exc:
            logger.warning("Health check database failure: %s", exc)
            raise HTTPException(status_code=503, detail="Database unavailable.") from exc
        return HealthResponse(
            status="ok",
            tenders=total,
            embedded=embedded,
            ilmu_configured=deps.router.has_api_key,
            spend_today_myr=round(settled / 1_000_000, 6),
            daily_cap_myr=resolved_settings.daily_cap_myr,
        )

    @app.post("/v1/search", response_model=SearchResponse)
    def search(payload: SearchRequest, request: Request) -> SearchResponse:
        _require_rate_limit(request)
        deps = _deps()
        try:
            return deps.engine.search(payload)
        except Exception as exc:
            logger.exception("Unhandled error during /v1/search")
            raise HTTPException(
                status_code=502,
                detail="Search service temporarily unavailable.",
            ) from exc

    @app.get("/v1/tenders/{tender_id:path}", response_model=TenderNotice)
    def get_tender(tender_id: str, request: Request) -> TenderNotice:
        _require_rate_limit(request)
        deps = _deps()
        try:
            tender = deps.store.get_tender(tender_id)
        except Exception as exc:
            logger.exception("Unhandled error during /v1/tenders lookup")
            raise HTTPException(
                status_code=502,
                detail="Tender lookup temporarily unavailable.",
            ) from exc
        if tender is None:
            raise HTTPException(status_code=404, detail="Tender not found.")
        return tender

    @app.post("/v1/ask", response_model=AskResponse)
    def ask(payload: AskRequest, request: Request) -> AskResponse:
        _require_rate_limit(request)
        deps = _deps()
        try:
            return deps.agent.ask(payload)
        except Exception as exc:
            logger.exception("Unhandled error during /v1/ask")
            raise HTTPException(
                status_code=502,
                detail="Question answering service temporarily unavailable.",
            ) from exc

    return app

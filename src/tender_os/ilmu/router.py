"""Sole HTTP boundary to YTL ILMU (`api.ilmu.ai`).

Enforces:
- Single API key env var (`ILMU_API_KEY` via `Settings`)
- Lowercased inputs for `bge-m3` and `bge-reranker` (PR0 empirical winner)
- Atomic pre-call spend reservation that conservatively charges requests that reached the server
- Single wall-clock deadline across retries and tier-1 (`ilmu-mini-v3.3`) -> tier-2 (`ilmu-v3.1`) fallback
- Full `AttemptRecord` trail attached to every raised error so degraded callers retain observability
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import httpx

from tender_os.config import Settings
from tender_os.domain import AttemptRecord
from tender_os.ilmu.spend import (
    SpendCapExceededError,
    SpendLedger,
    SpendReservation,
    estimate_cost_micro_rm,
    myt_today,
)


class ILMUError(RuntimeError):
    """Base error for YTL ILMU router failures."""

    def __init__(
        self,
        message: str,
        *,
        attempts: list[AttemptRecord] | None = None,
    ) -> None:
        super().__init__(message)
        self.attempts: list[AttemptRecord] = list(attempts) if attempts else []


class ILMUUnavailableError(ILMUError):
    """Raised when ILMU_API_KEY is not configured."""


class ILMUTimeoutError(ILMUError):
    """Raised when an ILMU call exceeds its wall-clock deadline."""


class ILMUHTTPError(ILMUError):
    """Raised when ILMU returns a non-2xx HTTP response or invalid JSON payload."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        attempts: list[AttemptRecord] | None = None,
    ) -> None:
        super().__init__(message, attempts=attempts)
        self.status_code = status_code


@dataclass(frozen=True)
class ChatResult:
    content: str
    model: str
    fallback_used: bool
    attempts: list[AttemptRecord]


class _NoopSpendLedger:
    @contextmanager
    def reserve(self, estimated_micro_rm: int) -> Iterator[SpendReservation]:
        yield SpendReservation(day_myt=myt_today(), reserved_micro_rm=estimated_micro_rm)


def _approx_tokens(texts: Sequence[str]) -> int:
    total_chars = sum(len(t) for t in texts)
    return max(1, (total_chars + 2) // 3)


def _retry_after_seconds(resp: httpx.Response, cap: float) -> float:
    raw = resp.headers.get("Retry-After")
    if raw:
        try:
            return max(0.0, min(float(raw), cap))
        except ValueError:
            pass
    return min(0.5, cap)


def _bounded_timeout(remaining: float) -> httpx.Timeout:
    t = max(0.05, remaining)
    short = min(5.0, t)
    return httpx.Timeout(connect=short, write=short, read=t, pool=short)


class ILMURouter:
    """Single router for all YTL ILMU API calls (embeddings, rerank, chat)."""

    def __init__(
        self,
        settings: Settings,
        *,
        spend_ledger: SpendLedger | None = None,
        transport: httpx.BaseTransport | None = None,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        self.settings = settings
        self._ledger: Any = spend_ledger if spend_ledger is not None else _NoopSpendLedger()
        self._sleep = sleep_fn
        self._client = httpx.Client(
            base_url=settings.ilmu_base_url.rstrip("/"),
            transport=transport,
            timeout=_bounded_timeout(settings.ilmu_timeout_seconds),
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> ILMURouter:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    @property
    def has_api_key(self) -> bool:
        return bool(self.settings.ilmu_api_key and self.settings.ilmu_api_key.strip())

    def _headers(self) -> dict[str, str]:
        if not self.has_api_key:
            raise ILMUUnavailableError("ILMU_API_KEY is not configured.")
        return {
            "Authorization": f"Bearer {self.settings.ilmu_api_key}",
            "Content-Type": "application/json",
        }

    def _post_json(
        self,
        path: str,
        payload: dict[str, Any],
        *,
        timeout: float,
    ) -> httpx.Response:
        return self._client.post(
            path,
            headers=self._headers(),
            json=payload,
            timeout=_bounded_timeout(timeout),
        )

    def embed_documents(
        self,
        texts: Sequence[str],
        *,
        batch_size: int = 64,
    ) -> tuple[list[list[float]], list[AttemptRecord]]:
        """Embed documents in batches with PR0's lowercased `bge-m3` winner."""
        if not texts:
            return [], []
        if not self.has_api_key:
            raise ILMUUnavailableError("ILMU_API_KEY is not configured.")

        model = self.settings.ilmu_embed_model
        vectors: list[list[float]] = []
        attempts: list[AttemptRecord] = []

        for start in range(0, len(texts), batch_size):
            batch = [t.strip().lower() for t in texts[start : start + batch_size]]
            batch_vecs, batch_attempts = self._embed_batch(batch, model=model)
            vectors.extend(batch_vecs)
            attempts.extend(batch_attempts)

        return vectors, attempts

    def embed_query(self, query: str) -> tuple[list[float], list[AttemptRecord]]:
        vecs, attempts = self.embed_documents([query], batch_size=1)
        return vecs[0], attempts

    def _embed_batch(
        self,
        batch: list[str],
        *,
        model: str,
    ) -> tuple[list[list[float]], list[AttemptRecord]]:
        est_tokens = _approx_tokens(batch)
        est_micro, basis = estimate_cost_micro_rm(model, est_tokens, 0)
        attempts: list[AttemptRecord] = []
        deadline = time.monotonic() + self.settings.ilmu_timeout_seconds

        for attempt_idx in range(2):
            remaining = deadline - time.monotonic()
            if remaining <= 0.05:
                raise ILMUTimeoutError("ILMU embed call exceeded deadline.", attempts=attempts)
            t0 = time.monotonic()
            try:
                with self._ledger.reserve(est_micro) as reservation:
                    reservation.mark_in_flight()
                    resp = self._post_json(
                        "/embeddings",
                        {"model": model, "input": batch},
                        timeout=remaining,
                    )
                    latency_ms = int((time.monotonic() - t0) * 1000)

                    if resp.status_code == 429:
                        reservation.release()
                        attempts.append(
                            AttemptRecord(
                                stage="embed",
                                model=model,
                                latency_ms=latency_ms,
                                pricing_basis=basis,
                                outcome="rate_limited",
                            )
                        )
                        if attempt_idx == 0:
                            wait = _retry_after_seconds(resp, min(1.0, max(0.0, deadline - time.monotonic() - 0.2)))
                            if wait > 0:
                                self._sleep(wait)
                            continue
                        raise ILMUHTTPError(
                            "ILMU embeddings rate limited (429).",
                            status_code=429,
                            attempts=attempts,
                        )

                    if resp.status_code >= 400:
                        reservation.release()
                        attempts.append(
                            AttemptRecord(
                                stage="embed",
                                model=model,
                                latency_ms=latency_ms,
                                pricing_basis=basis,
                                outcome="http_error",
                            )
                        )
                        raise ILMUHTTPError(
                            f"ILMU embeddings HTTP {resp.status_code}.",
                            status_code=resp.status_code,
                            attempts=attempts,
                        )

                    try:
                        body = resp.json()
                        data = body["data"]
                        if not isinstance(data, list) or len(data) != len(batch):
                            raise ValueError("embedding data length mismatch")
                        by_idx: dict[int, list[float]] = {}
                        for item in data:
                            idx = int(item["index"])
                            vec = [float(x) for x in item["embedding"]]
                            if len(vec) != self.settings.ilmu_embed_dim:
                                raise ValueError(
                                    f"Expected {self.settings.ilmu_embed_dim}d, got {len(vec)}d"
                                )
                            by_idx[idx] = vec
                        if len(by_idx) != len(batch) or any(i not in by_idx for i in range(len(batch))):
                            raise ValueError("embedding indices missing or duplicated")
                        vecs = [by_idx[i] for i in range(len(batch))]
                    except (ValueError, KeyError, TypeError) as exc:
                        # Server returned HTTP 200 (likely billed): leave reservation in_flight -> settles est_micro.
                        attempts.append(
                            AttemptRecord(
                                stage="embed",
                                model=model,
                                latency_ms=latency_ms,
                                cost_micro_rm=est_micro,
                                pricing_basis=basis,
                                outcome="schema_invalid",
                            )
                        )
                        raise ILMUHTTPError(
                            f"ILMU embeddings returned invalid payload: {exc}",
                            status_code=resp.status_code,
                            attempts=attempts,
                        ) from exc

                    usage = body.get("usage") if isinstance(body, dict) else {}
                    usage = usage if isinstance(usage, dict) else {}
                    prompt_tokens = int(usage.get("prompt_tokens") or usage.get("total_tokens") or est_tokens)
                    actual_micro, actual_basis = estimate_cost_micro_rm(model, prompt_tokens, 0)
                    reservation.settle(actual_micro)
                    attempts.append(
                        AttemptRecord(
                            stage="embed",
                            model=model,
                            latency_ms=latency_ms,
                            prompt_tokens=prompt_tokens,
                            completion_tokens=0,
                            cost_micro_rm=actual_micro,
                            pricing_basis=actual_basis,
                            outcome="ok",
                        )
                    )
                    return vecs, attempts
            except SpendCapExceededError as exc:
                attempts.append(
                    AttemptRecord(
                        stage="embed",
                        model=model,
                        pricing_basis=basis,
                        outcome="cap_blocked",
                    )
                )
                exc.attempts = list(attempts)
                raise
            except httpx.TimeoutException as exc:
                latency_ms = int((time.monotonic() - t0) * 1000)
                attempts.append(
                    AttemptRecord(
                        stage="embed",
                        model=model,
                        latency_ms=latency_ms,
                        cost_micro_rm=est_micro,
                        pricing_basis=basis,
                        outcome="timeout",
                    )
                )
                raise ILMUTimeoutError("ILMU embeddings timed out.", attempts=attempts) from exc
            except httpx.HTTPError as exc:
                latency_ms = int((time.monotonic() - t0) * 1000)
                attempts.append(
                    AttemptRecord(
                        stage="embed",
                        model=model,
                        latency_ms=latency_ms,
                        cost_micro_rm=est_micro,
                        pricing_basis=basis,
                        outcome="http_error",
                    )
                )
                raise ILMUHTTPError(
                    f"ILMU embeddings transport error: {exc}",
                    attempts=attempts,
                ) from exc

        raise ILMUHTTPError("ILMU embeddings failed after retry.", attempts=attempts)

    def rerank(
        self,
        query: str,
        documents: Sequence[str],
    ) -> tuple[list[tuple[int, float]], list[AttemptRecord]]:
        """Rerank documents using PR0's lowercased `bge-reranker` winner."""
        if not documents:
            return [], []
        if not self.has_api_key:
            raise ILMUUnavailableError("ILMU_API_KEY is not configured.")

        model = self.settings.ilmu_rerank_model
        q_norm = query.strip().lower()
        docs_norm = [d.strip().lower() for d in documents]
        est_tokens = _approx_tokens([q_norm, *docs_norm])
        est_micro, basis = estimate_cost_micro_rm(model, est_tokens, 0)
        attempts: list[AttemptRecord] = []
        deadline = time.monotonic() + self.settings.ilmu_rerank_timeout_seconds

        for attempt_idx in range(2):
            remaining = deadline - time.monotonic()
            if remaining <= 0.05:
                raise ILMUTimeoutError("ILMU rerank exceeded deadline.", attempts=attempts)
            t0 = time.monotonic()
            try:
                with self._ledger.reserve(est_micro) as reservation:
                    reservation.mark_in_flight()
                    resp = self._post_json(
                        "/rerank",
                        {"model": model, "query": q_norm, "documents": docs_norm},
                        timeout=remaining,
                    )
                    latency_ms = int((time.monotonic() - t0) * 1000)

                    if resp.status_code == 429:
                        reservation.release()
                        attempts.append(
                            AttemptRecord(
                                stage="rerank",
                                model=model,
                                latency_ms=latency_ms,
                                pricing_basis=basis,
                                outcome="rate_limited",
                            )
                        )
                        if attempt_idx == 0:
                            wait = _retry_after_seconds(resp, min(1.0, max(0.0, deadline - time.monotonic() - 0.2)))
                            if wait > 0:
                                self._sleep(wait)
                            continue
                        raise ILMUHTTPError(
                            "ILMU rerank rate limited (429).",
                            status_code=429,
                            attempts=attempts,
                        )

                    if resp.status_code >= 400:
                        reservation.release()
                        attempts.append(
                            AttemptRecord(
                                stage="rerank",
                                model=model,
                                latency_ms=latency_ms,
                                pricing_basis=basis,
                                outcome="http_error",
                            )
                        )
                        raise ILMUHTTPError(
                            f"ILMU rerank HTTP {resp.status_code}.",
                            status_code=resp.status_code,
                            attempts=attempts,
                        )

                    try:
                        body = resp.json()
                        results = body["results"]
                        if not isinstance(results, list):
                            raise ValueError("rerank results is not a list")
                        best_by_idx: dict[int, float] = {}
                        for item in results:
                            idx = int(item["index"])
                            score = float(item["relevance_score"])
                            if 0 <= idx < len(documents) and (
                                idx not in best_by_idx or score > best_by_idx[idx]
                            ):
                                best_by_idx[idx] = score
                        ranked = sorted(best_by_idx.items(), key=lambda pair: (-pair[1], pair[0]))
                    except (ValueError, KeyError, TypeError) as exc:
                        attempts.append(
                            AttemptRecord(
                                stage="rerank",
                                model=model,
                                latency_ms=latency_ms,
                                cost_micro_rm=est_micro,
                                pricing_basis=basis,
                                outcome="schema_invalid",
                            )
                        )
                        raise ILMUHTTPError(
                            f"ILMU rerank returned invalid payload: {exc}",
                            status_code=resp.status_code,
                            attempts=attempts,
                        ) from exc

                    usage = body.get("usage") if isinstance(body, dict) else {}
                    usage = usage if isinstance(usage, dict) else {}
                    prompt_tokens = int(usage.get("prompt_tokens") or usage.get("total_tokens") or est_tokens)
                    actual_micro, actual_basis = estimate_cost_micro_rm(model, prompt_tokens, 0)
                    reservation.settle(actual_micro)
                    attempts.append(
                        AttemptRecord(
                            stage="rerank",
                            model=model,
                            latency_ms=latency_ms,
                            prompt_tokens=prompt_tokens,
                            completion_tokens=0,
                            cost_micro_rm=actual_micro,
                            pricing_basis=actual_basis,
                            outcome="ok",
                        )
                    )
                    return ranked, attempts
            except SpendCapExceededError as exc:
                attempts.append(
                    AttemptRecord(
                        stage="rerank",
                        model=model,
                        pricing_basis=basis,
                        outcome="cap_blocked",
                    )
                )
                exc.attempts = list(attempts)
                raise
            except httpx.TimeoutException as exc:
                latency_ms = int((time.monotonic() - t0) * 1000)
                attempts.append(
                    AttemptRecord(
                        stage="rerank",
                        model=model,
                        latency_ms=latency_ms,
                        cost_micro_rm=est_micro,
                        pricing_basis=basis,
                        outcome="timeout",
                    )
                )
                raise ILMUTimeoutError("ILMU rerank timed out.", attempts=attempts) from exc
            except httpx.HTTPError as exc:
                latency_ms = int((time.monotonic() - t0) * 1000)
                attempts.append(
                    AttemptRecord(
                        stage="rerank",
                        model=model,
                        latency_ms=latency_ms,
                        cost_micro_rm=est_micro,
                        pricing_basis=basis,
                        outcome="http_error",
                    )
                )
                raise ILMUHTTPError(
                    f"ILMU rerank transport error: {exc}",
                    attempts=attempts,
                ) from exc

        raise ILMUHTTPError("ILMU rerank failed after retry.", attempts=attempts)

    def chat_complete(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 600,
        temperature: float = 0.0,
    ) -> ChatResult:
        """Tier-1 (`ilmu-mini-v3.3`) -> Tier-2 (`ilmu-v3.1`) chat completion under 1 wall-clock deadline.

        - HTTP 429 retries the SAME tier once (capped wait) before escalating.
        - Timeouts, HTTP 5xx, or invalid 200 payloads escalate once to Tier 2 with remaining deadline.
        - SpendCapExceededError propagates immediately with attempts attached.
        """
        if not self.has_api_key:
            raise ILMUUnavailableError("ILMU_API_KEY is not configured.")

        tiers = [
            (self.settings.ilmu_chat_primary_model, False),
            (self.settings.ilmu_chat_fallback_model, True),
        ]
        attempts: list[AttemptRecord] = []
        deadline = time.monotonic() + self.settings.ilmu_timeout_seconds
        prompt_text = [m.get("content", "") for m in messages]
        est_prompt = _approx_tokens(prompt_text)
        last_error: ILMUError | None = None

        for model, is_fallback in tiers:
            est_micro, basis = estimate_cost_micro_rm(model, est_prompt, max_tokens)
            for attempt_idx in range(2):
                remaining = deadline - time.monotonic()
                if remaining <= 0.1:
                    last_error = ILMUTimeoutError(
                        f"Deadline exhausted before calling {model}.",
                        attempts=attempts,
                    )
                    break

                call_timeout = (
                    remaining
                    if is_fallback
                    else min(remaining, max(0.5, self.settings.ilmu_timeout_seconds * 0.55))
                )
                t0 = time.monotonic()
                try:
                    with self._ledger.reserve(est_micro) as reservation:
                        reservation.mark_in_flight()
                        resp = self._post_json(
                            "/chat/completions",
                            {
                                "model": model,
                                "messages": messages,
                                "max_tokens": max_tokens,
                                "temperature": temperature,
                            },
                            timeout=call_timeout,
                        )
                        latency_ms = int((time.monotonic() - t0) * 1000)

                        if resp.status_code == 429:
                            reservation.release()
                            attempts.append(
                                AttemptRecord(
                                    stage="chat",
                                    model=model,
                                    fallback_used=is_fallback,
                                    latency_ms=latency_ms,
                                    pricing_basis=basis,
                                    outcome="rate_limited",
                                )
                            )
                            last_error = ILMUHTTPError(
                                f"{model} rate limited (429).",
                                status_code=429,
                                attempts=attempts,
                            )
                            if attempt_idx == 0:
                                rem_after = max(0.0, deadline - time.monotonic())
                                wait = _retry_after_seconds(resp, min(1.0, rem_after * 0.25))
                                if wait > 0:
                                    self._sleep(wait)
                                continue
                            break

                        if resp.status_code >= 500:
                            reservation.release()
                            attempts.append(
                                AttemptRecord(
                                    stage="chat",
                                    model=model,
                                    fallback_used=is_fallback,
                                    latency_ms=latency_ms,
                                    pricing_basis=basis,
                                    outcome="http_error",
                                )
                            )
                            last_error = ILMUHTTPError(
                                f"{model} server error HTTP {resp.status_code}.",
                                status_code=resp.status_code,
                                attempts=attempts,
                            )
                            break

                        if resp.status_code >= 400:
                            reservation.release()
                            attempts.append(
                                AttemptRecord(
                                    stage="chat",
                                    model=model,
                                    fallback_used=is_fallback,
                                    latency_ms=latency_ms,
                                    pricing_basis=basis,
                                    outcome="http_error",
                                )
                            )
                            raise ILMUHTTPError(
                                f"{model} client error HTTP {resp.status_code}.",
                                status_code=resp.status_code,
                                attempts=attempts,
                            )

                        try:
                            body = resp.json()
                            choices = body["choices"]
                            if not isinstance(choices, list) or not choices:
                                raise ValueError("empty choices")
                            msg = choices[0]["message"]
                            if not isinstance(msg, dict) or "content" not in msg:
                                raise ValueError("missing message content")
                            content = str(msg["content"] or "").strip()
                        except (ValueError, KeyError, TypeError) as exc:
                            # HTTP 200 with bad shape may have been billed -> leave in_flight to settle est_micro.
                            attempts.append(
                                AttemptRecord(
                                    stage="chat",
                                    model=model,
                                    fallback_used=is_fallback,
                                    latency_ms=latency_ms,
                                    cost_micro_rm=est_micro,
                                    pricing_basis=basis,
                                    outcome="schema_invalid",
                                )
                            )
                            last_error = ILMUHTTPError(
                                f"{model} returned invalid payload: {exc}",
                                status_code=resp.status_code,
                                attempts=attempts,
                            )
                            break

                        usage = body.get("usage") if isinstance(body, dict) else {}
                        usage = usage if isinstance(usage, dict) else {}
                        p_tok = int(usage.get("prompt_tokens") or est_prompt)
                        c_tok = int(usage.get("completion_tokens") or _approx_tokens([content]))
                        actual_micro, actual_basis = estimate_cost_micro_rm(model, p_tok, c_tok)
                        reservation.settle(actual_micro)
                        attempts.append(
                            AttemptRecord(
                                stage="chat",
                                model=model,
                                fallback_used=is_fallback,
                                latency_ms=latency_ms,
                                prompt_tokens=p_tok,
                                completion_tokens=c_tok,
                                cost_micro_rm=actual_micro,
                                pricing_basis=actual_basis,
                                outcome="ok",
                            )
                        )
                        return ChatResult(
                            content=content,
                            model=model,
                            fallback_used=is_fallback,
                            attempts=attempts,
                        )
                except SpendCapExceededError as exc:
                    attempts.append(
                        AttemptRecord(
                            stage="chat",
                            model=model,
                            fallback_used=is_fallback,
                            pricing_basis=basis,
                            outcome="cap_blocked",
                        )
                    )
                    exc.attempts = list(attempts)
                    raise
                except httpx.TimeoutException as exc:
                    latency_ms = int((time.monotonic() - t0) * 1000)
                    attempts.append(
                        AttemptRecord(
                            stage="chat",
                            model=model,
                            fallback_used=is_fallback,
                            latency_ms=latency_ms,
                            cost_micro_rm=est_micro,
                            pricing_basis=basis,
                            outcome="timeout",
                        )
                    )
                    last_error = ILMUTimeoutError(f"{model} timed out.", attempts=attempts)
                    last_error.__cause__ = exc
                    break
                except httpx.HTTPError as exc:
                    latency_ms = int((time.monotonic() - t0) * 1000)
                    attempts.append(
                        AttemptRecord(
                            stage="chat",
                            model=model,
                            fallback_used=is_fallback,
                            latency_ms=latency_ms,
                            cost_micro_rm=est_micro,
                            pricing_basis=basis,
                            outcome="http_error",
                        )
                    )
                    last_error = ILMUHTTPError(f"{model} transport error: {exc}", attempts=attempts)
                    last_error.__cause__ = exc
                    break

        err = last_error or ILMUHTTPError("All ILMU chat tiers failed.", attempts=attempts)
        err.attempts = list(attempts)
        raise err

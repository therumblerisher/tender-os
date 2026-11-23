"""Atomic MYT daily spend cap backed by Postgres `spend_ledger`.

Uses a single atomic conditional UPDATE to reserve estimated micro-RM before every remote
ILMU call, paired with a context manager whose `finally` block:
- settles actual token cost when `settle(actual)` was called,
- settles the conservative estimate if the request was already sent (`mark_in_flight()`)
  and failed ambiguously (e.g. read timeout or HTTP 200 malformed/wrong-dim response),
- releases the reservation back to 0 if the call never reached the server or returned an
  unbilled rejection (`release()`).
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime
from typing import Literal, Protocol
from zoneinfo import ZoneInfo

from psycopg_pool import ConnectionPool

logger = logging.getLogger(__name__)
MYT_ZONE = ZoneInfo("Asia/Kuala_Lumpur")

# Published YTL ILMU per-1M-token rates in MYR (1 MYR = 1,000,000 micro-RM, so MYR/1M == micro-RM/token).
PUBLISHED_RATES_MYR_PER_1M: dict[str, tuple[float, float]] = {
    "ilmu-mini-v3.3": (0.20, 1.20),
    "ilmu-v3.1": (4.00, 16.00),
}
# Embed/rerank rates are not published on docs.ilmu.ai; assume a conservative RM 0.50 / 1M tokens.
ASSUMED_CONSERVATIVE_RATE_MYR_PER_1M = 0.50


class SpendCapExceededError(RuntimeError):
    """Raised when reserving an ILMU call would exceed the MYT daily RM cap."""

    def __init__(self, message: str, *, attempts: list[object] | None = None) -> None:
        super().__init__(message)
        self.attempts: list[object] = list(attempts) if attempts else []


def myt_today() -> date:
    return datetime.now(MYT_ZONE).date()


def estimate_cost_micro_rm(
    model: str,
    prompt_tokens: int,
    completion_tokens: int = 0,
) -> tuple[int, Literal["published", "assumed_conservative"]]:
    """Calculate exact integer micro-RM cost and disclose whether pricing is published or assumed."""
    p_tok = max(0, prompt_tokens)
    c_tok = max(0, completion_tokens)
    rates = PUBLISHED_RATES_MYR_PER_1M.get(model)
    if rates is not None:
        in_rate, out_rate = rates
        raw = p_tok * in_rate + c_tok * out_rate
        basis: Literal["published", "assumed_conservative"] = "published"
    else:
        raw = (p_tok + c_tok) * ASSUMED_CONSERVATIVE_RATE_MYR_PER_1M
        basis = "assumed_conservative"
    if raw <= 0:
        return 0, basis
    return max(1, int(math.ceil(raw))), basis


@dataclass(frozen=True)
class SpendSummary:
    day_myt: date
    reserved_micro_rm: int
    settled_micro_rm: int
    cap_micro_rm: int

    @property
    def remaining_micro_rm(self) -> int:
        return max(0, self.cap_micro_rm - self.reserved_micro_rm - self.settled_micro_rm)


class SpendReservation:
    """Mutable handle passed inside `with ledger.reserve(...)` to record actual settled cost."""

    def __init__(self, day_myt: date, reserved_micro_rm: int) -> None:
        self.day_myt = day_myt
        self.reserved_micro_rm = reserved_micro_rm
        self._settled_micro_rm: int | None = None
        self._in_flight: bool = False

    def mark_in_flight(self) -> None:
        """Mark that the HTTP request has been dispatched and may be billed by the provider."""
        self._in_flight = True

    def settle(self, actual_micro_rm: int) -> None:
        self._settled_micro_rm = max(0, actual_micro_rm)

    def release(self) -> None:
        """Explicitly release reservation (0 billed) for unbilled rejections such as HTTP 429/401."""
        self._settled_micro_rm = 0

    @property
    def final_settled_micro_rm(self) -> int:
        if self._settled_micro_rm is not None:
            return self._settled_micro_rm
        return self.reserved_micro_rm if self._in_flight else 0


class SpendLedger(Protocol):
    @contextmanager
    def reserve(self, estimated_micro_rm: int) -> Iterator[SpendReservation]: ...

    def today_summary(self) -> SpendSummary: ...


class PostgresSpendLedger:
    """Atomic daily MYT spend cap stored in Postgres `spend_ledger`."""

    def __init__(self, pool: ConnectionPool, cap_micro_rm: int) -> None:
        self._pool = pool
        self.cap_micro_rm = max(0, cap_micro_rm)

    def _try_reserve(self, day: date, estimated_micro_rm: int) -> bool:
        amount = max(0, estimated_micro_rm)
        with self._pool.connection() as conn, conn.transaction():
            conn.execute(
                """
                INSERT INTO spend_ledger (day_myt, reserved_micro_rm, settled_micro_rm)
                VALUES (%s, 0, 0)
                ON CONFLICT (day_myt) DO NOTHING
                """,
                (day,),
            )
            row = conn.execute(
                """
                UPDATE spend_ledger
                SET reserved_micro_rm = reserved_micro_rm + %s,
                    updated_at = NOW()
                WHERE day_myt = %s
                  AND reserved_micro_rm + settled_micro_rm + %s <= %s
                RETURNING day_myt
                """,
                (amount, day, amount, self.cap_micro_rm),
            ).fetchone()
            return row is not None

    def _finalize(self, day: date, reserved_micro_rm: int, settled_micro_rm: int) -> None:
        with self._pool.connection() as conn, conn.transaction():
            conn.execute(
                """
                UPDATE spend_ledger
                SET reserved_micro_rm = GREATEST(0, reserved_micro_rm - %s),
                    settled_micro_rm = settled_micro_rm + %s,
                    updated_at = NOW()
                WHERE day_myt = %s
                """,
                (max(0, reserved_micro_rm), max(0, settled_micro_rm), day),
            )

    @contextmanager
    def reserve(self, estimated_micro_rm: int) -> Iterator[SpendReservation]:
        day = myt_today()
        amount = max(0, estimated_micro_rm)
        if not self._try_reserve(day, amount):
            raise SpendCapExceededError(
                f"Daily MYT spend cap ({self.cap_micro_rm / 1_000_000:.2f} MYR) reached for {day}."
            )
        handle = SpendReservation(day_myt=day, reserved_micro_rm=amount)
        body_failed = False
        try:
            yield handle
        except BaseException:
            body_failed = True
            raise
        finally:
            try:
                self._finalize(day, amount, handle.final_settled_micro_rm)
            except Exception:
                if body_failed:
                    logger.warning("Failed to finalize spend reservation after caller error.", exc_info=True)
                else:
                    raise

    def today_summary(self) -> SpendSummary:
        day = myt_today()
        with self._pool.connection() as conn:
            row = conn.execute(
                """
                SELECT reserved_micro_rm, settled_micro_rm
                FROM spend_ledger
                WHERE day_myt = %s
                """,
                (day,),
            ).fetchone()
        reserved = int(row[0]) if row else 0
        settled = int(row[1]) if row else 0
        return SpendSummary(
            day_myt=day,
            reserved_micro_rm=reserved,
            settled_micro_rm=settled,
            cap_micro_rm=self.cap_micro_rm,
        )

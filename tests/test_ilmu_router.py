"""Recorded-fixture tests for ILMURouter and atomic PostgresSpendLedger concurrency."""

from __future__ import annotations

import json
import os
import threading
import uuid
from pathlib import Path
from urllib.parse import quote

import httpx
import psycopg
import pytest

from tender_os.config import Settings
from tender_os.ilmu.router import ILMUHTTPError, ILMURouter
from tender_os.ilmu.spend import PostgresSpendLedger, SpendCapExceededError, estimate_cost_micro_rm
from tender_os.storage.postgres import PostgresStore

SCHEMA_SQL = Path(__file__).resolve().parents[1] / "migrations" / "001_init.sql"
DEFAULT_DB_URL = os.environ.get(
    "TENDER_OS_DATABASE_URL",
    "postgresql://tenderos:tenderos@127.0.0.1:5439/tenderos",
)


def _isolated_schema_url(base_url: str, schema: str) -> str:
    sep = "&" if "?" in base_url else "?"
    opts = quote(f"-csearch_path={schema},public", safe="")
    return f"{base_url}{sep}options={opts}"


@pytest.fixture()
def isolated_ledger_store() -> PostgresStore:
    schema = f"test_spend_{uuid.uuid4().hex[:8]}"
    with psycopg.connect(DEFAULT_DB_URL, autocommit=True) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector SCHEMA public")
        conn.execute(f"CREATE SCHEMA {schema}")
    store = PostgresStore(_isolated_schema_url(DEFAULT_DB_URL, schema), min_size=2, max_size=6)
    store.apply_schema(SCHEMA_SQL)
    try:
        yield store
    finally:
        store.close()
        with psycopg.connect(DEFAULT_DB_URL, autocommit=True) as conn:
            conn.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")


def test_settings_reads_single_ilmu_api_key_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ILMU_API_KEY", "test-key-123")
    monkeypatch.delenv("TENDER_OS_ILMU_API_KEY", raising=False)
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    assert settings.ilmu_api_key == "test-key-123"


def test_chat_tier1_pass_and_published_pricing() -> None:
    seen_models: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode())
        seen_models.append(body["model"])
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "Found [dbkl:2026/A065]."}}],
                "usage": {"prompt_tokens": 1000, "completion_tokens": 200},
            },
        )

    settings = Settings(_env_file=None, ilmu_api_key="k")  # type: ignore[call-arg]
    with ILMURouter(settings, transport=httpx.MockTransport(handler), sleep_fn=lambda _: None) as router:
        res = router.chat_complete([{"role": "user", "content": "insurance"}])

    assert seen_models == ["ilmu-mini-v3.3"]
    assert res.model == "ilmu-mini-v3.3"
    assert res.fallback_used is False
    assert len(res.attempts) == 1
    assert res.attempts[0].outcome == "ok"
    assert res.attempts[0].pricing_basis == "published"
    expected_cost, _ = estimate_cost_micro_rm("ilmu-mini-v3.3", 1000, 200)
    assert res.attempts[0].cost_micro_rm == expected_cost


def test_chat_timeout_falls_back_to_tier2() -> None:
    seen_models: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode())
        seen_models.append(body["model"])
        if body["model"] == "ilmu-mini-v3.3":
            raise httpx.ReadTimeout("simulated tier-1 timeout")
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "Tier-2 answer [dbkl:2026/A065]."}}],
                "usage": {"prompt_tokens": 500, "completion_tokens": 100},
            },
        )

    settings = Settings(_env_file=None, ilmu_api_key="k")  # type: ignore[call-arg]
    with ILMURouter(settings, transport=httpx.MockTransport(handler), sleep_fn=lambda _: None) as router:
        res = router.chat_complete([{"role": "user", "content": "insurance"}])

    assert seen_models == ["ilmu-mini-v3.3", "ilmu-v3.1"]
    assert res.model == "ilmu-v3.1"
    assert res.fallback_used is True
    assert [a.outcome for a in res.attempts] == ["timeout", "ok"]
    assert res.attempts[1].fallback_used is True


def test_chat_429_retries_same_tier_once_before_escalating() -> None:
    seen_models: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode())
        seen_models.append(body["model"])
        if len(seen_models) == 1:
            return httpx.Response(429, headers={"Retry-After": "0"}, json={"error": "rate_limited"})
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "Recovered on tier-1 [dbkl:2026/A065]."}}],
                "usage": {"prompt_tokens": 400, "completion_tokens": 80},
            },
        )

    settings = Settings(_env_file=None, ilmu_api_key="k")  # type: ignore[call-arg]
    with ILMURouter(settings, transport=httpx.MockTransport(handler), sleep_fn=lambda _: None) as router:
        res = router.chat_complete([{"role": "user", "content": "insurance"}])

    assert seen_models == ["ilmu-mini-v3.3", "ilmu-mini-v3.3"]
    assert res.fallback_used is False
    assert [a.outcome for a in res.attempts] == ["rate_limited", "ok"]


def test_malformed_200_and_http_errors_attach_attempts_and_charge_ledger(
    isolated_ledger_store: PostgresStore,
) -> None:
    ledger = PostgresSpendLedger(isolated_ledger_store.pool, cap_micro_rm=1_000_000)
    settings = Settings(_env_file=None, ilmu_api_key="k")  # type: ignore[call-arg]

    # 1. HTML 200 on /embeddings -> raises ILMUHTTPError with schema_invalid attempt AND settles conservative est
    def html_handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html>gateway error</html>")

    with (
        ILMURouter(
            settings,
            spend_ledger=ledger,
            transport=httpx.MockTransport(html_handler),
            sleep_fn=lambda _: None,
        ) as router,
        pytest.raises(ILMUHTTPError) as exc_info,
    ):
        router.embed_query("hospital aircond")
    assert len(exc_info.value.attempts) == 1
    assert exc_info.value.attempts[0].outcome == "schema_invalid"
    summary_after_html = ledger.today_summary()
    assert summary_after_html.reserved_micro_rm == 0
    assert summary_after_html.settled_micro_rm > 0

    # 2. Wrong vector dim (768d instead of 1024d) -> schema_invalid AND charges conservative estimate
    settled_before = summary_after_html.settled_micro_rm

    def wrong_dim_handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [0.1] * 768}]})

    with (
        ILMURouter(
            settings,
            spend_ledger=ledger,
            transport=httpx.MockTransport(wrong_dim_handler),
            sleep_fn=lambda _: None,
        ) as router,
        pytest.raises(ILMUHTTPError) as exc_dim,
    ):
        router.embed_query("hospital aircond")
    assert exc_dim.value.attempts[0].outcome == "schema_invalid"
    assert ledger.today_summary().settled_micro_rm > settled_before

    # 3. Chat 400 -> raises ILMUHTTPError WITH attempts preserved and releases reservation (0 new spend)
    settled_before_400 = ledger.today_summary().settled_micro_rm

    def bad_req_handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": "bad_request"})

    with (
        ILMURouter(
            settings,
            spend_ledger=ledger,
            transport=httpx.MockTransport(bad_req_handler),
            sleep_fn=lambda _: None,
        ) as router,
        pytest.raises(ILMUHTTPError) as exc_400,
    ):
        router.chat_complete([{"role": "user", "content": "hi"}])
    assert len(exc_400.value.attempts) == 1
    assert exc_400.value.attempts[0].outcome == "http_error"
    assert ledger.today_summary().settled_micro_rm == settled_before_400

    # 4. Chat cap-hit -> raises SpendCapExceededError WITH cap_blocked attempt attached
    zero_ledger = PostgresSpendLedger(isolated_ledger_store.pool, cap_micro_rm=0)
    with (
        ILMURouter(
            settings,
            spend_ledger=zero_ledger,
            transport=httpx.MockTransport(bad_req_handler),
            sleep_fn=lambda _: None,
        ) as router,
        pytest.raises(SpendCapExceededError) as exc_cap,
    ):
        router.chat_complete([{"role": "user", "content": "hi"}])
    assert len(exc_cap.value.attempts) == 1


def test_embed_and_rerank_lowercase_inputs_and_mark_assumed_pricing() -> None:
    captured: list[tuple[str, dict]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode())
        captured.append((request.url.path, body))
        if request.url.path.endswith("/embeddings"):
            vec = [0.01] * 1024
            return httpx.Response(
                200,
                json={"data": [{"index": 0, "embedding": vec}], "usage": {"prompt_tokens": 25}},
            )
        return httpx.Response(
            200,
            json={
                "results": [
                    {"index": 1, "relevance_score": 0.91},
                    {"index": 0, "relevance_score": 0.32},
                ],
                "usage": {"total_tokens": 80},
            },
        )

    settings = Settings(_env_file=None, ilmu_api_key="k")  # type: ignore[call-arg]
    with ILMURouter(settings, transport=httpx.MockTransport(handler), sleep_fn=lambda _: None) as router:
        vec, emb_attempts = router.embed_query("HOSPITAL AIRCOND")
        ranked, rr_attempts = router.rerank("HOSPITAL AIRCOND", ["DOC A", "DOC B"])

    assert len(vec) == 1024
    assert captured[0][1]["input"] == ["hospital aircond"]
    assert emb_attempts[0].pricing_basis == "assumed_conservative"
    assert captured[1][1]["query"] == "hospital aircond"
    assert captured[1][1]["documents"] == ["doc a", "doc b"]
    assert ranked == [(1, 0.91), (0, 0.32)]
    assert rr_attempts[0].pricing_basis == "assumed_conservative"


def test_atomic_spend_cap_race_allows_exactly_one_thread(
    isolated_ledger_store: PostgresStore,
) -> None:
    ledger = PostgresSpendLedger(isolated_ledger_store.pool, cap_micro_rm=10_000)
    barrier = threading.Barrier(2)
    results: list[str] = []
    lock = threading.Lock()

    def worker() -> None:
        barrier.wait()
        try:
            with ledger.reserve(10_000) as handle:
                handle.settle(9_000)
            with lock:
                results.append("ok")
        except SpendCapExceededError:
            with lock:
                results.append("cap_blocked")

    t1 = threading.Thread(target=worker)
    t2 = threading.Thread(target=worker)
    t1.start()
    t2.start()
    t1.join(timeout=10)
    t2.join(timeout=10)

    assert sorted(results) == ["cap_blocked", "ok"]
    summary = ledger.today_summary()
    assert summary.reserved_micro_rm == 0
    assert summary.settled_micro_rm == 9_000
    assert summary.remaining_micro_rm == 1_000


def test_spend_reservation_released_on_pre_send_crash(isolated_ledger_store: PostgresStore) -> None:
    ledger = PostgresSpendLedger(isolated_ledger_store.pool, cap_micro_rm=50_000)

    with pytest.raises(RuntimeError, match="boom"), ledger.reserve(40_000):
        mid = ledger.today_summary()
        assert mid.reserved_micro_rm == 40_000
        raise RuntimeError("boom")

    after = ledger.today_summary()
    assert after.reserved_micro_rm == 0
    assert after.settled_micro_rm == 0
    assert after.remaining_micro_rm == 50_000

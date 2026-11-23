"""Live Postgres pgvector retrieval, filter, HNSW iterative_scan, degradation, and ingest tests."""

from __future__ import annotations

import json
import math
import os
import uuid
from pathlib import Path
from urllib.parse import quote

import httpx
import numpy as np
import psycopg
import pytest

from tender_os.config import Settings
from tender_os.domain import (
    EligibilityStatus,
    ProfileContext,
    SearchFilters,
    SearchMode,
    SearchRequest,
    TenderNotice,
)
from tender_os.ilmu.router import ILMURouter
from tender_os.ilmu.spend import PostgresSpendLedger
from tender_os.ingest import ingest_snapshot, sha256_file
from tender_os.retrieval import RetrievalEngine
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


def _unit_vec(active_dims: list[int], dim: int = 1024) -> list[float]:
    v = [0.0] * dim
    val = 1.0 / math.sqrt(len(active_dims))
    for d in active_dims:
        v[d] = val
    return v


SAMPLE_TENDERS = [
    TenderNotice(
        id="penang:PG/2026/001",
        source="penang",
        reference="PG/2026/001",
        title="PERKHIDMATAN PENYELENGGARAAN PENYAMAN UDARA HOSPITAL PULAU PINANG",
        agency="Jabatan Kesihatan Negeri Pulau Pinang",
        category="Mekanikal",
        location="Pulau Pinang",
        closing_at="2026-12-20T12:00:00+08:00",
        eligibility="Kod Bidang: 220301 ATAU 220302",
        url="https://ep.penang.gov.my/1",
    ),
    TenderNotice(
        id="jkr:JKR/PG/2026/002",
        source="jkr",
        reference="JKR/PG/2026/002",
        title="KERJA MENAIKTARAF SISTEM HAWA DINGIN KLINIK KESIHATAN PULAU PINANG",
        agency="Jabatan Kerja Raya Malaysia",
        category="Mekanikal",
        location=None,  # Federal portals leave location NULL and put state in title
        closing_at="2026-12-18T12:00:00+08:00",
        eligibility="CIDB Gred G5 dan ke atas",
        url="https://jkr.gov.my/2",
    ),
    TenderNotice(
        id="eperolehan:EP/2026/003",
        source="eperolehan",
        reference="EP/2026/003",
        title="PEROLEHAN PERKHIDMATAN KESELAMATAN SIBER DAN PERISIAN PENGURUSAN DATA",
        agency="Kementerian Kewangan Malaysia",
        category="ICT",
        location="Putrajaya",
        closing_at="2026-12-25T12:00:00+08:00",
        eligibility="Kod Bidang: 210103 DAN 210104",
        url="https://eperolehan.gov.my/3",
    ),
    TenderNotice(
        id="selangor:MBDK.JKEJ.600-7/1/390",
        source="selangor",
        reference="MBDK.JKEJ.600-7/1/390",
        title="KERJA MEMBAIKPULIH DEWAN ORANG RAMAI DAERAH KLANG",
        agency="Majlis Bandaraya Diraja Klang",
        category="Kerja",
        location="Selangor",
        closing_at="2026-11-01T12:00:00+08:00",
        eligibility=None,
        url="https://tender.selangor.my/4",
    ),
]

SAMPLE_EMBEDDINGS = [
    _unit_vec([0, 1]),  # aircond hospital Penang
    _unit_vec([0, 2]),  # aircond clinic Penang
    _unit_vec([10, 11]),  # cybersecurity IT
    _unit_vec([20, 21]),  # hall repair Selangor
]


@pytest.fixture()
def pg_store() -> PostgresStore:
    schema = f"test_pg_{uuid.uuid4().hex[:8]}"
    with psycopg.connect(DEFAULT_DB_URL, autocommit=True) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector SCHEMA public")
        conn.execute(f"CREATE SCHEMA {schema}")
    store = PostgresStore(_isolated_schema_url(DEFAULT_DB_URL, schema), min_size=1, max_size=4)
    store.apply_schema(SCHEMA_SQL)
    store.upsert_tenders(SAMPLE_TENDERS, SAMPLE_EMBEDDINGS)
    store.analyze()
    try:
        yield store
    finally:
        store.close()
        with psycopg.connect(DEFAULT_DB_URL, autocommit=True) as conn:
            conn.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")


def test_fts_exact_code_hyphen_ref_wildcards_hnsw_and_location_alias(pg_store: PostgresStore) -> None:
    assert pg_store.count_tenders() == (4, 4)
    assert pg_store.get_tender("eperolehan:EP/2026/003") is not None

    # Exact MOF code
    code_hits = pg_store.search_fts("210103")
    assert code_hits[0].tender.id == "eperolehan:EP/2026/003"

    # Hyphenated portal reference must not treat `-7` as negation
    ref_hits = pg_store.search_fts("MBDK.JKEJ.600-7/1/390")
    assert ref_hits[0].tender.id == "selangor:MBDK.JKEJ.600-7/1/390"

    # Literal '%' wildcard must NOT match everything
    assert pg_store.search_fts("%") == []

    # Location alias "Penang" matches both location="Pulau Pinang" and federal title fallback
    penang_hits = pg_store.search_fts(
        "pulau pinang",
        filters=SearchFilters(location="Penang", eligibility_code="   "),
    )
    assert {h.tender.id for h in penang_hits} == {"penang:PG/2026/001", "jkr:JKR/PG/2026/002"}

    # Forced HNSW path with Penang filter returns both notices (iterative_scan prevents truncation)
    hnsw_penang = pg_store.search_vector(
        _unit_vec([0, 1]),
        filters=SearchFilters(location="Penang"),
        limit=10,
        force_index=True,
    )
    assert {h.tender.id for h in hnsw_penang} == {"penang:PG/2026/001", "jkr:JKR/PG/2026/002"}

    # Open-after filter drops closed Selangor notice and retains open JKR notice
    open_hits = pg_store.search_fts(
        "kerja",
        filters=SearchFilters(open_after="2026-12-01"),
    )
    assert len(open_hits) == 1
    assert open_hits[0].tender.id == "jkr:JKR/PG/2026/002"


def test_hybrid_rerank_eligibility_dismissed_and_degradations(pg_store: PostgresStore) -> None:
    def ok_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/embeddings"):
            return httpx.Response(
                200,
                json={"data": [{"index": 0, "embedding": _unit_vec([0, 2])}], "usage": {"prompt_tokens": 10}},
            )
        body = json.loads(request.content.decode())
        docs: list[str] = body["documents"]
        results = []
        for idx, doc in enumerate(docs):
            score = 0.95 if "jkr/pg/2026/002" in doc else 0.40
            results.append({"index": idx, "relevance_score": score})
        return httpx.Response(200, json={"results": results, "usage": {"total_tokens": 40}})

    settings = Settings(_env_file=None, ilmu_api_key="k")  # type: ignore[call-arg]
    with ILMURouter(settings, transport=httpx.MockTransport(ok_handler), sleep_fn=lambda _: None) as router:
        engine = RetrievalEngine(pg_store, router, settings)
        profile = ProfileContext(
            mof_codes=["220301"],
            cidb_grades=["G7"],
            dismissed_ids=["selangor:MBDK.JKEJ.600-7/1/390"],
        )
        resp = engine.search(
            SearchRequest(
                query="air conditioning maintenance Penang",
                mode=SearchMode.HYBRID_RERANK,
                limit=5,
                filters=SearchFilters(location="Penang"),
                profile=profile,
            )
        )

    assert resp.effective_mode == SearchMode.HYBRID_RERANK
    assert resp.degraded_mode is False
    assert resp.hits[0].tender.id == "jkr:JKR/PG/2026/002"
    assert resp.hits[0].breakdown.rerank_score == 0.95
    assert all(h.eligibility_verdict is not None for h in resp.hits)
    assert resp.hits[0].eligibility_verdict is not None
    assert resp.hits[0].eligibility_verdict.status == EligibilityStatus.ELIGIBLE
    assert "selangor:MBDK.JKEJ.600-7/1/390" not in {h.tender.id for h in resp.hits}

    # Malformed HTML 200 on /embeddings -> degrades cleanly to PG_FTS WITH schema_invalid attempt preserved
    def html_embed_handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html>bad gateway</html>")

    with ILMURouter(
        settings, transport=httpx.MockTransport(html_embed_handler), sleep_fn=lambda _: None
    ) as router_bad:
        resp_bad = RetrievalEngine(pg_store, router_bad, settings).search(
            SearchRequest(query="keselamatan siber", mode=SearchMode.HYBRID_RERANK)
        )
    assert resp_bad.effective_mode == SearchMode.PG_FTS
    assert resp_bad.degraded_mode is True
    assert any(a.stage == "embed" and a.outcome == "schema_invalid" for a in resp_bad.attempts)

    # Rerank timeout -> degrades cleanly to HYBRID WITH timeout attempt preserved
    def rerank_timeout_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/embeddings"):
            return httpx.Response(
                200,
                json={"data": [{"index": 0, "embedding": _unit_vec([0, 1])}], "usage": {"prompt_tokens": 10}},
            )
        raise httpx.ReadTimeout("rerank timed out")

    with ILMURouter(
        settings, transport=httpx.MockTransport(rerank_timeout_handler), sleep_fn=lambda _: None
    ) as router_t:
        resp_t = RetrievalEngine(pg_store, router_t, settings).search(
            SearchRequest(query="penyaman udara", mode=SearchMode.HYBRID_RERANK)
        )
    assert resp_t.effective_mode == SearchMode.HYBRID
    assert resp_t.degraded_mode is True
    assert resp_t.degraded_reason == "rerank_fallback_to_hybrid"
    assert any(a.stage == "rerank" and a.outcome == "timeout" for a in resp_t.attempts)

    # Missing ILMU_API_KEY -> degrades cleanly to PG_FTS
    no_key_settings = Settings(_env_file=None, ilmu_api_key=None)  # type: ignore[call-arg]
    with ILMURouter(no_key_settings) as router_nk:
        resp_nk = RetrievalEngine(pg_store, router_nk, no_key_settings).search(
            SearchRequest(query="keselamatan siber", mode=SearchMode.HYBRID_RERANK)
        )
    assert resp_nk.effective_mode == SearchMode.PG_FTS
    assert resp_nk.degraded_mode is True
    assert resp_nk.degraded_reason == "missing_ilmu_api_key"
    assert resp_nk.hits[0].tender.id == "eperolehan:EP/2026/003"

    # Exhausted daily spend cap -> degrades cleanly to PG_FTS WITH cap_blocked attempt preserved
    zero_ledger = PostgresSpendLedger(pg_store.pool, cap_micro_rm=0)
    with ILMURouter(
        settings,
        spend_ledger=zero_ledger,
        transport=httpx.MockTransport(ok_handler),
        sleep_fn=lambda _: None,
    ) as router_cap:
        resp_cap = RetrievalEngine(pg_store, router_cap, settings).search(
            SearchRequest(query="keselamatan siber", mode=SearchMode.HYBRID_RERANK)
        )
    assert resp_cap.effective_mode == SearchMode.PG_FTS
    assert resp_cap.degraded_mode is True
    assert resp_cap.degraded_reason == "spend_cap_exceeded"
    assert any(a.outcome == "cap_blocked" for a in resp_cap.attempts)


def test_ingest_snapshot_roundtrip_with_float32_artifact(tmp_path: Path) -> None:
    snap = tmp_path / "snap.jsonl"
    snap.write_text(
        "\n".join(t.model_dump_json() for t in SAMPLE_TENDERS) + "\n",
        encoding="utf-8",
    )
    emb_path = tmp_path / "embeddings.f32"
    np.asarray(SAMPLE_EMBEDDINGS, dtype="<f4").tofile(emb_path)
    meta_path = tmp_path / "embeddings.meta.json"
    meta_path.write_text(
        json.dumps(
            {
                "model": "bge-m3",
                "lowercased": True,
                "dim": 1024,
                "count": len(SAMPLE_TENDERS),
                "snapshot_sha256": sha256_file(snap),
                "embeddings_sha256": sha256_file(emb_path),
                "ids": [t.id for t in SAMPLE_TENDERS],
            }
        ),
        encoding="utf-8",
    )

    schema = f"test_ingest_{uuid.uuid4().hex[:8]}"
    with psycopg.connect(DEFAULT_DB_URL, autocommit=True) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector SCHEMA public")
        conn.execute(f"CREATE SCHEMA {schema}")
    store = PostgresStore(_isolated_schema_url(DEFAULT_DB_URL, schema), min_size=1, max_size=2)
    try:
        report = ingest_snapshot(
            store,
            snap,
            embeddings_path=emb_path,
            meta_path=meta_path,
            schema_path=SCHEMA_SQL,
        )
        assert report.rows_loaded == 4
        assert report.embeddings_loaded == 4
        assert report.total_rows_in_db == 4
        assert report.embedded_rows_in_db == 4
    finally:
        store.close()
        with psycopg.connect(DEFAULT_DB_URL, autocommit=True) as conn:
            conn.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")

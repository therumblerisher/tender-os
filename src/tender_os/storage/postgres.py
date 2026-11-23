"""Pooled PostgreSQL + pgvector store for Tender OS."""

from __future__ import annotations

import re
from collections.abc import Sequence
from contextlib import suppress
from pathlib import Path
from typing import Any

import psycopg
from pgvector import Vector
from pgvector.psycopg import register_vector
from psycopg import Connection
from psycopg_pool import ConnectionPool

from tender_os.domain import RouteKind, SearchFilters, TenderNotice, expand_location_terms
from tender_os.fusion import RankedCandidate
from tender_os.grounding import route_query

TS_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")

TENDER_SELECT_COLS = """
    id, source, reference, title, agency, category, location,
    closing_at, value_myr, eligibility, url, published_at
"""


def _escape_like(value: str) -> str:
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _configure_connection(conn: Connection[Any]) -> None:
    # Fresh database before `CREATE EXTENSION vector` runs via apply_schema().
    with suppress(ValueError):
        register_vector(conn)


def _row_to_tender(row: tuple[Any, ...]) -> TenderNotice:
    return TenderNotice(
        id=str(row[0]),
        source=str(row[1]),
        reference=str(row[2]),
        title=str(row[3]),
        agency=str(row[4]),
        category=row[5],
        location=row[6],
        closing_at=row[7],
        value_myr=row[8],
        eligibility=row[9],
        url=str(row[10]),
        published_at=row[11],
    )


def _build_filter_clauses(
    filters: SearchFilters,
    dismissed_ids: Sequence[str],
) -> tuple[list[str], list[Any]]:
    clauses: list[str] = []
    params: list[Any] = []

    if filters.source and filters.source.strip():
        clauses.append("LOWER(source) = LOWER(%s)")
        params.append(filters.source.strip())

    if filters.agency and filters.agency.strip():
        clauses.append("agency ILIKE %s ESCAPE '\\'")
        params.append(f"%{_escape_like(filters.agency.strip())}%")

    if filters.location and filters.location.strip():
        terms = expand_location_terms(filters.location)
        if terms:
            loc_parts: list[str] = []
            for term in terms:
                like = f"%{_escape_like(term)}%"
                loc_parts.append(
                    "(COALESCE(location, '') ILIKE %s ESCAPE '\\' OR title ILIKE %s ESCAPE '\\' OR agency ILIKE %s ESCAPE '\\')"
                )
                params.extend([like, like, like])
            clauses.append("(" + " OR ".join(loc_parts) + ")")

    if filters.eligibility_code and filters.eligibility_code.strip():
        code_like = f"%{_escape_like(filters.eligibility_code.strip())}%"
        clauses.append(
            "(COALESCE(eligibility, '') ILIKE %s ESCAPE '\\' OR COALESCE(category, '') ILIKE %s ESCAPE '\\')"
        )
        params.extend([code_like, code_like])

    if filters.open_after and filters.open_after.strip():
        clauses.append("closing_at IS NOT NULL AND closing_at >= %s")
        params.append(filters.open_after.strip())

    cleaned_dismissed = [d.strip() for d in dismissed_ids if d and d.strip()]
    if cleaned_dismissed:
        clauses.append("NOT (id = ANY(%s))")
        params.append(cleaned_dismissed)

    return clauses, params


class PostgresStore:
    """Single storage owner backed by PostgreSQL + pgvector."""

    def __init__(
        self,
        database_url: str,
        *,
        min_size: int = 1,
        max_size: int = 10,
    ) -> None:
        self.database_url = database_url
        self.pool = ConnectionPool(
            conninfo=database_url,
            min_size=min_size,
            max_size=max_size,
            configure=_configure_connection,
            open=True,
        )

    def close(self) -> None:
        self.pool.close()

    def __enter__(self) -> PostgresStore:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def apply_schema(self, schema_path: Path) -> None:
        sql = schema_path.read_text(encoding="utf-8")
        # Use a dedicated non-pooled autocommit connection so pooled connections stay transactional.
        with psycopg.connect(self.database_url, autocommit=True) as conn:
            conn.execute(sql)
        # Ensure existing idle pool connections have vector type info registered.
        with self.pool.connection() as conn:
            register_vector(conn)

    def analyze(self) -> None:
        with psycopg.connect(self.database_url, autocommit=True) as conn:
            conn.execute("ANALYZE tenders")

    def health_check(self) -> bool:
        with self.pool.connection() as conn:
            row = conn.execute("SELECT 1").fetchone()
            return bool(row and row[0] == 1)

    def count_tenders(self) -> tuple[int, int]:
        with self.pool.connection() as conn:
            row = conn.execute("SELECT COUNT(*), COUNT(embedding) FROM tenders").fetchone()
        if not row:
            return 0, 0
        return int(row[0]), int(row[1])

    def get_tender(self, tender_id: str) -> TenderNotice | None:
        with self.pool.connection() as conn:
            row = conn.execute(
                f"SELECT {TENDER_SELECT_COLS} FROM tenders WHERE id = %s",
                (tender_id,),
            ).fetchone()
        return _row_to_tender(row) if row else None

    def upsert_tenders(
        self,
        tenders: Sequence[TenderNotice],
        embeddings: Sequence[Sequence[float] | None] | None = None,
        *,
        batch_size: int = 500,
    ) -> int:
        if not tenders:
            return 0
        if embeddings is not None and len(embeddings) != len(tenders):
            raise ValueError("embeddings length must match tenders length")

        sql = """
            INSERT INTO tenders (
                id, source, reference, title, agency, category, location,
                closing_at, value_myr, eligibility, url, published_at,
                search_text, embedding, updated_at
            )
            VALUES (
                %s, %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s,
                %s, %s, NOW()
            )
            ON CONFLICT (id) DO UPDATE SET
                source = EXCLUDED.source,
                reference = EXCLUDED.reference,
                title = EXCLUDED.title,
                agency = EXCLUDED.agency,
                category = EXCLUDED.category,
                location = EXCLUDED.location,
                closing_at = EXCLUDED.closing_at,
                value_myr = EXCLUDED.value_myr,
                eligibility = EXCLUDED.eligibility,
                url = EXCLUDED.url,
                published_at = EXCLUDED.published_at,
                search_text = EXCLUDED.search_text,
                embedding = CASE
                    WHEN EXCLUDED.embedding IS NOT NULL THEN EXCLUDED.embedding
                    WHEN tenders.search_text <> EXCLUDED.search_text THEN NULL
                    ELSE tenders.embedding
                END,
                updated_at = NOW()
        """

        total = 0
        with self.pool.connection() as conn:
            for start in range(0, len(tenders), batch_size):
                chunk = tenders[start : start + batch_size]
                emb_chunk = (
                    embeddings[start : start + batch_size]
                    if embeddings is not None
                    else [None] * len(chunk)
                )
                rows = [
                    (
                        t.id,
                        t.source,
                        t.reference,
                        t.title,
                        t.agency,
                        t.category,
                        t.location,
                        t.closing_at,
                        t.value_myr,
                        t.eligibility,
                        t.url,
                        t.published_at,
                        t.search_document(),
                        Vector(list(emb)) if emb is not None else None,
                    )
                    for t, emb in zip(chunk, emb_chunk, strict=True)
                ]
                with conn.transaction(), conn.cursor() as cur:
                    cur.executemany(sql, rows)
                total += len(chunk)
        return total

    def search_fts(
        self,
        query: str,
        *,
        filters: SearchFilters | None = None,
        limit: int = 50,
        dismissed_ids: Sequence[str] = (),
    ) -> list[RankedCandidate]:
        cleaned = query.strip()
        if not cleaned:
            return []

        flt = filters or SearchFilters()
        filter_clauses, filter_params = _build_filter_clauses(flt, dismissed_ids)

        # Keep GIN index usable: run plainto_tsquery('simple') directly on search_tsv without an OR ILIKE.
        and_where = ["search_tsv @@ plainto_tsquery('simple', %s)", *filter_clauses]
        sql_and = f"""
            SELECT {TENDER_SELECT_COLS},
                   ts_rank_cd(search_tsv, plainto_tsquery('simple', %s)) AS fts_score
            FROM tenders
            WHERE {' AND '.join(and_where)}
            ORDER BY fts_score DESC, id ASC
            LIMIT %s
        """
        params_and: list[Any] = [cleaned, cleaned, *filter_params, limit]

        with self.pool.connection() as conn:
            fts_rows = conn.execute(sql_and, params_and).fetchall()

            # For exact portal reference / ID queries (or when FTS AND returned nothing), check escaped reference match.
            exact_rows: list[tuple[Any, ...]] = []
            if route_query(cleaned) == RouteKind.EXACT_CODE or ":" in cleaned:
                exact_like = f"%{_escape_like(cleaned)}%"
                exact_where = [
                    "(reference ILIKE %s ESCAPE '\\' OR id ILIKE %s ESCAPE '\\')",
                    *filter_clauses,
                ]
                sql_exact = f"""
                    SELECT {TENDER_SELECT_COLS},
                           CASE WHEN LOWER(reference) = LOWER(%s) OR LOWER(id) = LOWER(%s)
                                THEN 5.0 ELSE 2.0 END AS fts_score
                    FROM tenders
                    WHERE {' AND '.join(exact_where)}
                    ORDER BY fts_score DESC, id ASC
                    LIMIT %s
                """
                exact_rows = conn.execute(
                    sql_exact,
                    [cleaned, cleaned, exact_like, exact_like, *filter_params, limit],
                ).fetchall()

        if exact_rows or fts_rows:
            merged: dict[str, RankedCandidate] = {}
            for r in fts_rows:
                t = _row_to_tender(r[:12])
                merged[t.id] = RankedCandidate(tender=t, score=float(r[12]))
            for r in exact_rows:
                t = _row_to_tender(r[:12])
                prev = merged.get(t.id)
                merged[t.id] = RankedCandidate(
                    tender=t,
                    score=(prev.score if prev else 0.0) + float(r[12]),
                )
            ordered = sorted(merged.values(), key=lambda c: (-c.score, c.tender.id))
            return ordered[:limit]

        # Pass 2: OR fallback over alphanumeric tokens so multi-word English/BM queries still recall.
        tokens = list(dict.fromkeys(t.lower() for t in TS_TOKEN_RE.findall(cleaned) if len(t) >= 2))
        if len(tokens) < 2:
            return []
        or_expr = " | ".join(tokens[:16])
        or_where = ["search_tsv @@ to_tsquery('simple', %s)", *filter_clauses]
        sql_or = f"""
            SELECT {TENDER_SELECT_COLS},
                   ts_rank_cd(search_tsv, to_tsquery('simple', %s)) AS fts_score
            FROM tenders
            WHERE {' AND '.join(or_where)}
            ORDER BY fts_score DESC, id ASC
            LIMIT %s
        """
        params_or: list[Any] = [or_expr, or_expr, *filter_params, limit]
        with self.pool.connection() as conn:
            or_rows = conn.execute(sql_or, params_or).fetchall()
        return [RankedCandidate(tender=_row_to_tender(r[:12]), score=float(r[12])) for r in or_rows]

    def search_vector(
        self,
        query_embedding: Sequence[float],
        *,
        filters: SearchFilters | None = None,
        limit: int = 50,
        dismissed_ids: Sequence[str] = (),
        force_index: bool = False,
    ) -> list[RankedCandidate]:
        """Cosine similarity search with HNSW iterative_scan and ef_search safeguards."""
        flt = filters or SearchFilters()
        filter_clauses, filter_params = _build_filter_clauses(flt, dismissed_ids)
        where_clauses = ["embedding IS NOT NULL", *filter_clauses]
        vec = Vector(list(query_embedding))
        ef_search = max(limit * 2, 64)
        sql = f"""
            SELECT {TENDER_SELECT_COLS},
                   (1.0 - (embedding <=> %s)) AS vec_score
            FROM tenders
            WHERE {' AND '.join(where_clauses)}
            ORDER BY embedding <=> %s ASC
            LIMIT %s
        """
        params: list[Any] = [vec, *filter_params, vec, limit]
        with self.pool.connection() as conn, conn.transaction():
            conn.execute(f"SET LOCAL hnsw.ef_search = {int(ef_search)}")
            conn.execute("SET LOCAL hnsw.iterative_scan = relaxed_order")
            if force_index:
                conn.execute("SET LOCAL enable_seqscan = off")
            rows = conn.execute(sql, params).fetchall()
        return [RankedCandidate(tender=_row_to_tender(r[:12]), score=float(r[12])) for r in rows]

    def explain_vector_query(
        self,
        query_embedding: Sequence[float],
        *,
        limit: int = 50,
    ) -> list[str]:
        """Return EXPLAIN (ANALYZE, BUFFERS) lines for an unfiltered cosine search."""
        vec = Vector(list(query_embedding))
        ef_search = max(limit * 2, 64)
        sql = """
            EXPLAIN (ANALYZE, BUFFERS)
            SELECT id, (1.0 - (embedding <=> %s)) AS vec_score
            FROM tenders
            WHERE embedding IS NOT NULL
            ORDER BY embedding <=> %s ASC
            LIMIT %s
        """
        with self.pool.connection() as conn, conn.transaction():
            conn.execute(f"SET LOCAL hnsw.ef_search = {int(ef_search)}")
            conn.execute("SET LOCAL hnsw.iterative_scan = relaxed_order")
            rows = conn.execute(sql, (vec, vec, limit)).fetchall()
        return [str(r[0]) for r in rows]

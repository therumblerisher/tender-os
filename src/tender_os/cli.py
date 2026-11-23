"""Command-line interface for Tender OS database setup, ingestion, search, Q&A, and API serving."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import uvicorn

from tender_os.agent import TenderAgent
from tender_os.config import get_settings
from tender_os.domain import AskRequest, SearchFilters, SearchMode, SearchRequest
from tender_os.ilmu.router import ILMURouter
from tender_os.ilmu.spend import PostgresSpendLedger
from tender_os.ingest import ingest_snapshot
from tender_os.retrieval import RetrievalEngine
from tender_os.storage.postgres import PostgresStore


def _cmd_init_db(args: argparse.Namespace) -> int:
    settings = get_settings()
    with PostgresStore(settings.database_url) as store:
        store.apply_schema(args.schema)
    print(f"Applied schema from {args.schema}")
    return 0


def _cmd_ingest(args: argparse.Namespace) -> int:
    settings = get_settings()
    snapshot_path: Path = args.snapshot or Path(settings.snapshot_path)
    requested_emb: Path = args.embeddings or Path(settings.embeddings_path)
    emb_path: Path | None = requested_emb if requested_emb.is_file() else None
    meta_path: Path | None = args.meta
    if meta_path is None and emb_path is not None:
        meta_path = emb_path.with_suffix(".meta.json")
    if emb_path is not None and (meta_path is None or not meta_path.is_file()):
        emb_path = None
    if emb_path is None:
        print(
            f"Warning: {requested_emb} or its .meta.json not found; ingesting text only (vector search inactive).",
            file=sys.stderr,
        )

    with PostgresStore(settings.database_url) as store:
        if args.skip_if_loaded:
            expected_lines = sum(
                1 for line in snapshot_path.read_text(encoding="utf-8").splitlines() if line.strip()
            )
            total, embedded = store.count_tenders()
            needs_emb = emb_path is not None
            if total >= expected_lines and (not needs_emb or embedded >= expected_lines):
                print(
                    f"Skipping ingest: database already has {total} rows ({embedded} embedded)."
                )
                return 0

        report = ingest_snapshot(
            store,
            snapshot_path,
            embeddings_path=emb_path,
            meta_path=meta_path,
            schema_path=args.schema if args.schema.is_file() else None,
        )
    print(
        json.dumps(
            {
                "rows_loaded": report.rows_loaded,
                "embeddings_loaded": report.embeddings_loaded,
                "total_rows_in_db": report.total_rows_in_db,
                "embedded_rows_in_db": report.embedded_rows_in_db,
            },
            indent=2,
        )
    )
    return 0


def _cmd_search(args: argparse.Namespace) -> int:
    settings = get_settings()
    with PostgresStore(settings.database_url) as store:
        ledger = PostgresSpendLedger(store.pool, cap_micro_rm=settings.daily_cap_micro_rm)
        with ILMURouter(settings, spend_ledger=ledger) as router:
            engine = RetrievalEngine(store, router, settings)
            req = SearchRequest(
                query=args.query,
                mode=SearchMode(args.mode),
                limit=args.limit,
                filters=SearchFilters(location=args.location, source=args.source),
            )
            resp = engine.search(req)
    print(resp.model_dump_json(indent=2))
    return 0


def _cmd_ask(args: argparse.Namespace) -> int:
    settings = get_settings()
    with PostgresStore(settings.database_url) as store:
        ledger = PostgresSpendLedger(store.pool, cap_micro_rm=settings.daily_cap_micro_rm)
        with ILMURouter(settings, spend_ledger=ledger) as router:
            engine = RetrievalEngine(store, router, settings)
            agent = TenderAgent(store, engine, router, settings)
            req = AskRequest(
                question=args.question,
                filters=SearchFilters(location=args.location, source=args.source),
            )
            resp = agent.ask(req)
    print(resp.model_dump_json(indent=2))
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    uvicorn.run(
        "tender_os.api:create_app",
        factory=True,
        host=args.host,
        port=args.port,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tender-os", description="Tender OS CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init-db", help="Apply Postgres pgvector schema")
    p_init.add_argument("--schema", type=Path, default=Path("migrations/001_init.sql"))

    p_ingest = sub.add_parser("ingest", help="Load public notices and precomputed embeddings")
    p_ingest.add_argument("--snapshot", type=Path, default=None)
    p_ingest.add_argument("--embeddings", type=Path, default=None)
    p_ingest.add_argument("--meta", type=Path, default=None)
    p_ingest.add_argument("--schema", type=Path, default=Path("migrations/001_init.sql"))
    p_ingest.add_argument("--skip-if-loaded", action="store_true")

    p_search = sub.add_parser("search", help="Run hybrid tender search")
    p_search.add_argument("query", type=str)
    p_search.add_argument("--mode", choices=[m.value for m in SearchMode], default="hybrid_rerank")
    p_search.add_argument("--limit", type=int, default=5)
    p_search.add_argument("--location", type=str, default=None)
    p_search.add_argument("--source", type=str, default=None)

    p_ask = sub.add_parser("ask", help="Ask a grounded question over Malaysian tenders")
    p_ask.add_argument("question", type=str)
    p_ask.add_argument("--location", type=str, default=None)
    p_ask.add_argument("--source", type=str, default=None)

    p_serve = sub.add_parser("serve", help="Start FastAPI server")
    p_serve.add_argument("--host", type=str, default="0.0.0.0")
    p_serve.add_argument("--port", type=int, default=8000)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "init-db":
        return _cmd_init_db(args)
    if args.command == "ingest":
        return _cmd_ingest(args)
    if args.command == "search":
        return _cmd_search(args)
    if args.command == "ask":
        return _cmd_ask(args)
    if args.command == "serve":
        return _cmd_serve(args)
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())

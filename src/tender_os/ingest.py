"""Snapshot and precomputed float32 embedding ingestion into PostgreSQL."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tender_os.domain import TenderNotice
from tender_os.storage.postgres import PostgresStore


@dataclass(frozen=True)
class IngestReport:
    rows_loaded: int
    embeddings_loaded: int
    total_rows_in_db: int
    embedded_rows_in_db: int


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_snapshot(snapshot_path: Path) -> list[TenderNotice]:
    tenders: list[TenderNotice] = []
    for line in snapshot_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        tenders.append(TenderNotice.model_validate_json(line))
    return tenders


def load_embeddings_artifact(
    embeddings_path: Path,
    meta_path: Path,
    tenders: list[TenderNotice],
    *,
    snapshot_path: Path | None = None,
    expected_dim: int = 1024,
) -> list[list[float]]:
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    dim = int(meta["dim"])
    count = int(meta["count"])
    ids = list(meta["ids"])

    if dim != expected_dim:
        raise ValueError(f"Embedding dim mismatch: expected {expected_dim}, got {dim}")
    if meta.get("lowercased") is False:
        raise ValueError("Embedding artifact must be built with lowercased=True (PR0 requirement)")
    if len(ids) != count:
        raise ValueError(f"Metadata count mismatch: count={count}, len(ids)={len(ids)}")

    expected_snap_sha = meta.get("snapshot_sha256")
    if expected_snap_sha and snapshot_path is not None and sha256_file(snapshot_path) != expected_snap_sha:
        raise ValueError("tenders snapshot SHA-256 does not match embeddings.meta.json")

    expected_emb_sha = meta.get("embeddings_sha256")
    if expected_emb_sha and sha256_file(embeddings_path) != expected_emb_sha:
        raise ValueError("embeddings.f32 SHA-256 does not match embeddings.meta.json")

    raw = np.fromfile(embeddings_path, dtype="<f4")
    if raw.size != count * dim:
        raise ValueError(f"Expected {count * dim} float32 values, got {raw.size}")
    matrix = raw.reshape((count, dim))

    by_id = {tid: idx for idx, tid in enumerate(ids)}
    ordered: list[list[float]] = []
    missing = [t.id for t in tenders if t.id not in by_id]
    if missing:
        raise ValueError(f"{len(missing)} snapshot IDs missing from embeddings.meta.json (e.g. {missing[0]})")

    for tender in tenders:
        ordered.append(matrix[by_id[tender.id]].tolist())
    return ordered


def ingest_snapshot(
    store: PostgresStore,
    snapshot_path: Path,
    *,
    embeddings_path: Path | None = None,
    meta_path: Path | None = None,
    schema_path: Path | None = None,
    batch_size: int = 500,
) -> IngestReport:
    if schema_path is not None:
        store.apply_schema(schema_path)

    tenders = load_snapshot(snapshot_path)
    embeddings: list[list[float]] | None = None
    if embeddings_path is not None:
        resolved_meta = meta_path or embeddings_path.with_suffix(".meta.json")
        embeddings = load_embeddings_artifact(
            embeddings_path,
            resolved_meta,
            tenders,
            snapshot_path=snapshot_path,
        )

    loaded = store.upsert_tenders(tenders, embeddings, batch_size=batch_size)
    store.analyze()
    total_rows, embedded_rows = store.count_tenders()
    return IngestReport(
        rows_loaded=loaded,
        embeddings_loaded=len(embeddings) if embeddings is not None else 0,
        total_rows_in_db=total_rows,
        embedded_rows_in_db=embedded_rows,
    )

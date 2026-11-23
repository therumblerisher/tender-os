"""Precompute float32 corpus embeddings via YTL ILMU `bge-m3` and save a verifiable artifact."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from tender_os.config import Settings
from tender_os.ilmu.router import ILMURouter
from tender_os.ingest import load_snapshot, sha256_file


def build_embeddings(
    snapshot_path: Path,
    output_path: Path,
    meta_path: Path,
    *,
    batch_size: int = 64,
) -> dict[str, object]:
    settings = Settings()
    tenders = load_snapshot(snapshot_path)
    docs = [t.semantic_document() for t in tenders]

    t0 = time.monotonic()
    with ILMURouter(settings) as router:
        vectors, attempts = router.embed_documents(docs, batch_size=batch_size)
    elapsed_s = round(time.monotonic() - t0, 2)

    matrix = np.asarray(vectors, dtype="<f4")
    if matrix.shape != (len(tenders), settings.ilmu_embed_dim):
        raise RuntimeError(f"Unexpected embedding matrix shape: {matrix.shape}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    matrix.tofile(output_path)

    total_tokens = sum(a.prompt_tokens for a in attempts)
    total_micro_rm = sum(a.cost_micro_rm for a in attempts)
    meta = {
        "model": settings.ilmu_embed_model,
        "lowercased": True,
        "dim": settings.ilmu_embed_dim,
        "count": len(tenders),
        "snapshot_sha256": sha256_file(snapshot_path),
        "embeddings_sha256": sha256_file(output_path),
        "prompt_tokens": total_tokens,
        "assumed_cost_myr": round(total_micro_rm / 1_000_000, 6),
        "pricing_basis": "assumed_conservative",
        "elapsed_s": elapsed_s,
        "ids": [t.id for t in tenders],
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "count": len(tenders),
        "dim": settings.ilmu_embed_dim,
        "batches": len(attempts),
        "prompt_tokens": total_tokens,
        "assumed_cost_myr": meta["assumed_cost_myr"],
        "elapsed_s": elapsed_s,
        "bytes": output_path.stat().st_size,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build float32 corpus embeddings via YTL ILMU.")
    parser.add_argument("--snapshot", type=Path, default=Path("data/tenders_public.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/embeddings.f32"))
    parser.add_argument("--meta", type=Path, default=Path("data/embeddings.meta.json"))
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    summary = build_embeddings(
        args.snapshot,
        args.output,
        args.meta,
        batch_size=args.batch_size,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

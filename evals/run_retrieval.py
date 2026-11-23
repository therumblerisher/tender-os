"""Run split retrieval evaluation across dev_30 and untouched holdout_30 against live Postgres + ILMU."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from tender_os.config import Settings
from tender_os.domain import SearchMode, SearchRequest
from tender_os.ilmu.router import ILMURouter
from tender_os.ilmu.spend import PostgresSpendLedger
from tender_os.retrieval import RetrievalEngine
from tender_os.storage.postgres import PostgresStore

MODES: tuple[SearchMode, ...] = (
    SearchMode.PG_FTS,
    SearchMode.VECTOR,
    SearchMode.HYBRID,
    SearchMode.HYBRID_RERANK,
)
BUCKETS: tuple[str, ...] = ("exact_code_ref", "malay_keyword", "english_cross_lingual")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _score_query(retrieved_ids: list[str], relevant: set[str]) -> tuple[float, float, float]:
    mrr10 = 0.0
    for rank, tid in enumerate(retrieved_ids[:10], start=1):
        if tid in relevant:
            mrr10 = 1.0 / rank
            break
    r10 = len(set(retrieved_ids[:10]) & relevant) / len(relevant)
    r50 = len(set(retrieved_ids[:50]) & relevant) / len(relevant)
    return mrr10, r10, r50


def _round_metrics(rows: list[tuple[float, float, float]]) -> dict[str, Any]:
    if not rows:
        return {"count": 0, "mrr_at_10": 0.0, "recall_at_10": 0.0, "recall_at_50": 0.0}
    n = len(rows)
    return {
        "count": n,
        "mrr_at_10": round(sum(r[0] for r in rows) / n, 4),
        "recall_at_10": round(sum(r[1] for r in rows) / n, 4),
        "recall_at_50": round(sum(r[2] for r in rows) / n, 4),
    }


def eval_split(
    engine: RetrievalEngine,
    queries: list[dict[str, Any]],
    *,
    budget_limit_micro_rm: int,
    cumulative_micro_rm: list[int],
    cost_by_basis: dict[str, int],
) -> dict[str, Any]:
    """Run 1 deterministic quality pass (limit=50) over all 4 retrieval modes on one dataset split."""
    results_by_mode: dict[str, Any] = {}
    missed_hybrid_rerank: list[dict[str, Any]] = []

    for mode in MODES:
        overall: list[tuple[float, float, float]] = []
        per_bucket: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
        degraded_count = 0

        for q in queries:
            if cumulative_micro_rm[0] >= budget_limit_micro_rm:
                raise RuntimeError(
                    f"Eval aborted at hard budget cap ({cumulative_micro_rm[0] / 1_000_000:.4f} MYR)"
                )
            resp = engine.search(SearchRequest(query=q["query"], mode=mode, limit=50))
            for att in resp.attempts:
                cumulative_micro_rm[0] += att.cost_micro_rm
                cost_by_basis[att.pricing_basis] = (
                    cost_by_basis.get(att.pricing_basis, 0) + att.cost_micro_rm
                )
            if resp.degraded_mode:
                degraded_count += 1

            retrieved_ids = [h.tender.id for h in resp.hits]
            relevant = set(q["relevant_ids"])
            scores = _score_query(retrieved_ids, relevant)
            overall.append(scores)
            per_bucket[q["bucket"]].append(scores)

            if mode == SearchMode.HYBRID_RERANK and scores[1] == 0.0:
                missed_hybrid_rerank.append(
                    {
                        "id": q["id"],
                        "bucket": q["bucket"],
                        "query": q["query"],
                        "relevant_ids": q["relevant_ids"],
                        "top5_ids": retrieved_ids[:5],
                    }
                )

        summary = _round_metrics(overall)
        summary["degraded_queries"] = degraded_count
        summary["buckets"] = {b: _round_metrics(per_bucket[b]) for b in BUCKETS}
        results_by_mode[mode.value] = summary

    return {
        "query_count": len(queries),
        "modes": results_by_mode,
        "missed_hybrid_rerank_top10": missed_hybrid_rerank,
    }


def benchmark_latency(
    engine: RetrievalEngine,
    dev_queries: list[dict[str, Any]],
    *,
    trials: int = 5,
    budget_limit_micro_rm: int,
    cumulative_micro_rm: list[int],
    cost_by_basis: dict[str, int],
) -> dict[str, Any]:
    """Run 5 interleaved latency trials over 6 stratified dev queries (2 per bucket, limit=10)."""
    sample: list[dict[str, Any]] = []
    for bucket in BUCKETS:
        sample.extend([q for q in dev_queries if q["bucket"] == bucket][:2])

    by_mode: dict[str, list[float]] = {m.value: [] for m in MODES}
    for _trial in range(trials):
        for q in sample:
            for mode in MODES:
                if cumulative_micro_rm[0] >= budget_limit_micro_rm:
                    raise RuntimeError("Eval aborted during latency benchmark at budget cap.")
                resp = engine.search(SearchRequest(query=q["query"], mode=mode, limit=10))
                for att in resp.attempts:
                    cumulative_micro_rm[0] += att.cost_micro_rm
                    cost_by_basis[att.pricing_basis] = (
                        cost_by_basis.get(att.pricing_basis, 0) + att.cost_micro_rm
                    )
                by_mode[mode.value].append(resp.latency_ms)

    stats_out: dict[str, Any] = {}
    for mode_name, vals in by_mode.items():
        ordered = sorted(vals)
        p95_idx = min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))
        stats_out[mode_name] = {
            "samples": len(vals),
            "p50_ms": round(statistics.median(vals), 1),
            "p95_ms": round(ordered[p95_idx], 1),
            "min_ms": round(ordered[0], 1),
            "max_ms": round(ordered[-1], 1),
        }
    return {
        "trials_per_query": trials,
        "queries_sampled": len(sample),
        "limit": 10,
        "modes": stats_out,
    }


def render_report_md(report: dict[str, Any]) -> str:
    """Render concise plain-Markdown summary from evals/report.json."""
    lines: list[str] = ["# Tender OS Evaluation Report", ""]
    corpus = report.get("corpus", {})
    lines.append(
        f"Evaluated on {corpus.get('total_rows', 0)} cleaned public notices "
        f"({corpus.get('embedded_rows', 0)} embedded with YTL `bge-m3`, 1,024 dims) across "
        f"{corpus.get('portals', 15)} Malaysian procurement portals."
    )
    lines.append("")

    ret = report.get("retrieval", {})
    if ret:
        for split_key, title in (
            ("holdout_30", "Untouched Holdout Retrieval (holdout_30, n=30)"),
            ("dev_30", "Development Retrieval (dev_30, n=30)"),
        ):
            split = ret.get(split_key, {})
            if not split:
                continue
            lines.append(f"## {title}")
            lines.append("")
            lines.append(
                "| Mode | Overall MRR@10 | Overall R@10 | Overall R@50 | Exact MRR@10 | Malay MRR@10 | Cross-Lingual MRR@10 |"
            )
            lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
            for mode in ("pg_fts", "vector", "hybrid", "hybrid_rerank"):
                m = split["modes"][mode]
                b = m["buckets"]
                lines.append(
                    f"| `{mode}` | {m['mrr_at_10']:.4f} | {m['recall_at_10']:.4f} | "
                    f"{m['recall_at_50']:.4f} | {b['exact_code_ref']['mrr_at_10']:.4f} | "
                    f"{b['malay_keyword']['mrr_at_10']:.4f} | "
                    f"{b['english_cross_lingual']['mrr_at_10']:.4f} |"
                )
            lines.append("")
            misses = split.get("missed_hybrid_rerank_top10", [])
            lines.append(f"Missed in `hybrid_rerank` top-10 on `{split_key}`: {len(misses)}")
            for miss in misses:
                lines.append(
                    f"- `{miss['id']}` ({miss['bucket']}): `{miss['query']}` (top hit: `{miss['top5_ids'][:1]}`)"
                )
            lines.append("")

        lat = ret.get("latency", {})
        if lat:
            lines.append(
                f"## Search Latency ({lat['trials_per_query']} interleaved trials x {lat['queries_sampled']} dev queries, limit={lat['limit']})"
            )
            lines.append("")
            lines.append("| Mode | Samples | p50 (ms) | p95 (ms) | Min (ms) | Max (ms) |")
            lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
            for mode in ("pg_fts", "vector", "hybrid", "hybrid_rerank"):
                s = lat["modes"][mode]
                lines.append(
                    f"| `{mode}` | {s['samples']} | {s['p50_ms']:.1f} | {s['p95_ms']:.1f} | {s['min_ms']:.1f} | {s['max_ms']:.1f} |"
                )
            lines.append("")

        plan = ret.get("explain_analyze_vector_50", [])
        if plan:
            lines.append("## Postgres Query Plan (`EXPLAIN ANALYZE`, limit=50)")
            lines.append("")
            lines.append("```text")
            for pl in plan[:8]:
                lines.append(pl[:220] + ("..." if len(pl) > 220 else ""))
            lines.append("```")
            lines.append("")

    agent = report.get("agent", {})
    if agent:
        lines.append("## End-to-End Agent Evaluation (dev vs untouched holdout)")
        lines.append("")
        lines.append(
            "| Split | Answerable Hits | OOS Abstentions | Pre-Gen OOS (0 chat calls) | Verified Cites | Unverified IDs | Date Mismatches | Fallbacks | p50 (ms) | Spend (MYR) |"
        )
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for key, label in (
            ("holdout_10", "Untouched holdout (n=10)"),
            ("dev_10", "Dev (n=10)"),
            ("combined_20", "Combined (n=20)"),
        ):
            s = agent[key]
            lines.append(
                f"| {label} | {s['answerable_hit_count']}/{s['answerable_total']} | "
                f"{s['abstain_correct_count']}/{s['abstain_total']} | "
                f"{s['pre_gen_abstain_count']}/{s['abstain_total']} | "
                f"{s['verified_citations_total']} | {s['unverified_ids_total']} | "
                f"{s['date_mismatches_total']} | {s['fallback_count']} | "
                f"{s['p50_ms']:.1f} | RM {s['cost_myr']:.6f} |"
            )
        lines.append("")

    total_cost = report.get("total_eval_cost_myr")
    if total_cost is not None:
        lines.append(f"Total recorded eval spend: RM {total_cost:.6f}")
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run split retrieval evaluation against live Postgres + ILMU.")
    parser.add_argument("--dev-file", type=Path, default=Path("evals/dev_30.jsonl"))
    parser.add_argument("--holdout-file", type=Path, default=Path("evals/holdout_30.jsonl"))
    parser.add_argument("--out-json", type=Path, default=Path("evals/report.json"))
    parser.add_argument("--out-md", type=Path, default=Path("evals/REPORT.md"))
    parser.add_argument("--budget-myr", type=float, default=2.0)
    parser.add_argument(
        "--pr0-summary",
        type=Path,
        default=Path("evals/pr0/summary.json"),
    )
    args = parser.parse_args()

    settings = Settings()
    if not settings.ilmu_api_key:
        raise SystemExit("ILMU_API_KEY must be set to run live retrieval evaluation.")

    dev_qs = _load_jsonl(args.dev_file)
    holdout_qs = _load_jsonl(args.holdout_file)

    existing: dict[str, Any] = (
        json.loads(args.out_json.read_text(encoding="utf-8")) if args.out_json.is_file() else {}
    )
    prior_agent_micro = int(existing.get("agent", {}).get("cost_micro_rm", 0))
    budget_micro = int(round(args.budget_myr * 1_000_000))
    cumulative_micro = [prior_agent_micro]
    cost_by_basis: dict[str, int] = {}

    t0 = time.monotonic()
    with PostgresStore(settings.database_url) as store:
        total_rows, embedded_rows = store.count_tenders()
        ledger = PostgresSpendLedger(store.pool, cap_micro_rm=settings.daily_cap_micro_rm)
        with ILMURouter(settings, spend_ledger=ledger) as router:
            engine = RetrievalEngine(store, router, settings)
            dev_res = eval_split(
                engine,
                dev_qs,
                budget_limit_micro_rm=budget_micro,
                cumulative_micro_rm=cumulative_micro,
                cost_by_basis=cost_by_basis,
            )
            holdout_res = eval_split(
                engine,
                holdout_qs,
                budget_limit_micro_rm=budget_micro,
                cumulative_micro_rm=cumulative_micro,
                cost_by_basis=cost_by_basis,
            )
            latency_res = benchmark_latency(
                engine,
                dev_qs,
                trials=5,
                budget_limit_micro_rm=budget_micro,
                cumulative_micro_rm=cumulative_micro,
                cost_by_basis=cost_by_basis,
            )
            q_vec, q_att = router.embed_query(dev_qs[0]["query"])
            for att in q_att:
                cumulative_micro[0] += att.cost_micro_rm
                cost_by_basis[att.pricing_basis] = (
                    cost_by_basis.get(att.pricing_basis, 0) + att.cost_micro_rm
                )
            explain_lines = store.explain_vector_query(q_vec, limit=50)

    retrieval_micro = cumulative_micro[0] - prior_agent_micro
    pr0_data = (
        json.loads(args.pr0_summary.read_text(encoding="utf-8"))
        if args.pr0_summary.is_file()
        else None
    )

    portals_count = len(
        {
            json.loads(line)["source"]
            for line in Path(settings.snapshot_path).read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
    )

    report = dict(existing)
    report["corpus"] = {
        "total_rows": total_rows,
        "embedded_rows": embedded_rows,
        "portals": portals_count,
        "embed_model": settings.ilmu_embed_model,
        "embed_dim": settings.ilmu_embed_dim,
        "rerank_model": settings.ilmu_rerank_model,
    }
    if pr0_data is not None:
        report["pr0_bakeoff"] = pr0_data
    report["retrieval"] = {
        "run_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "elapsed_s": round(time.monotonic() - t0, 2),
        "dev_30": dev_res,
        "holdout_30": holdout_res,
        "latency": latency_res,
        "explain_analyze_vector_50": explain_lines,
        "cost_micro_rm": retrieval_micro,
        "cost_myr": round(retrieval_micro / 1_000_000, 6),
        "cost_by_pricing_basis_micro_rm": cost_by_basis,
    }
    report["total_eval_cost_myr"] = round(
        (retrieval_micro + prior_agent_micro) / 1_000_000,
        6,
    )

    args.out_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    args.out_md.write_text(render_report_md(report), encoding="utf-8")
    os.sync()
    print(
        json.dumps(
            {
                "dev_hybrid_rerank": dev_res["modes"]["hybrid_rerank"],
                "holdout_hybrid_rerank": holdout_res["modes"]["hybrid_rerank"],
                "latency": latency_res["modes"],
                "retrieval_cost_myr": report["retrieval"]["cost_myr"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

"""Verify that every metric row and headline number in README.md matches evals/report.json verbatim."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


def _must_contain(readme: str, snippet: str, label: str, errors: list[str]) -> None:
    if snippet not in readme:
        errors.append(f"Missing {label}: expected {snippet!r} in README.md")


def check_readme(readme_path: Path, report_path: Path) -> list[str]:
    """Return a list of discrepancies between README.md and evals/report.json (empty when valid)."""
    readme = readme_path.read_text(encoding="utf-8")
    report: dict[str, Any] = json.loads(report_path.read_text(encoding="utf-8"))
    errors: list[str] = []

    if "\u2014" in readme or "\u2013" in readme:
        errors.append("README.md must not contain em/en dashes.")

    corpus = report["corpus"]
    total_rows = int(corpus["total_rows"])
    portals = int(corpus["portals"])
    _must_contain(readme, f"**{total_rows:,}** cleaned public Malaysian", "corpus row headline", errors)
    _must_contain(readme, f"**{portals} portals**", "portal count headline", errors)

    # Full Markdown table row checks for both retrieval splits
    ret = report["retrieval"]
    for split_key, label in (("holdout_30", "holdout"), ("dev_30", "dev")):
        split = ret[split_key]
        for mode in ("pg_fts", "vector", "hybrid", "hybrid_rerank"):
            m = split["modes"][mode]
            b = m["buckets"]
            row = (
                f"| `{mode}` | {m['mrr_at_10']:.4f} | {m['recall_at_10']:.4f} | "
                f"{m['recall_at_50']:.4f} | {b['exact_code_ref']['mrr_at_10']:.4f} | "
                f"{b['malay_keyword']['mrr_at_10']:.4f} | "
                f"{b['english_cross_lingual']['mrr_at_10']:.4f} |"
            )
            _must_contain(readme, row, f"{label} retrieval row ({mode})", errors)

        # Check bucket query counts (8/7/15)
        h_buckets = split["modes"]["hybrid_rerank"]["buckets"]
        _must_contain(
            readme,
            f"`{h_buckets['exact_code_ref']['count']}` exact MOF code / portal reference, "
            f"`{h_buckets['malay_keyword']['count']}` Malay keyword, "
            f"`{h_buckets['english_cross_lingual']['count']}` English-to-Malay cross-lingual",
            "bucket count breakdown",
            errors,
        ) if split_key == "holdout_30" else None

    misses = ret["dev_30"].get("missed_hybrid_rerank_top10", [])
    miss_ids = ", ".join(f"`{m['id']}`" for m in misses)
    _must_contain(
        readme,
        f"`{len(misses)}/30` queries ({miss_ids})",
        "dev miss count and IDs",
        errors,
    )

    # Full latency rows (Samples, p50, p95, Min, Max)
    lat = ret["latency"]["modes"]
    for mode in ("pg_fts", "vector", "hybrid", "hybrid_rerank"):
        s = lat[mode]
        row = (
            f"| `{mode}` | {s['samples']} | {s['p50_ms']:.1f} | {s['p95_ms']:.1f} | "
            f"{s['min_ms']:.1f} | {s['max_ms']:.1f} |"
        )
        _must_contain(readme, row, f"latency row ({mode})", errors)

    # EXPLAIN ANALYZE execution time check parsed dynamically from report.json
    plan_text = "\n".join(ret.get("explain_analyze_vector_50", []))
    exec_match = re.search(r"actual time=[0-9.]+\.\.([0-9.]+)", plan_text)
    if exec_match:
        _must_contain(
            readme,
            f"**{float(exec_match.group(1)):.1f} ms**",
            "EXPLAIN ANALYZE execution time",
            errors,
        )

    # Full agent table rows
    agent = report["agent"]
    for split_key, label in (
        ("holdout_10", "Untouched holdout (n=10)"),
        ("dev_10", "Dev (n=10)"),
        ("combined_20", "Combined (n=20)"),
    ):
        s = agent[split_key]
        row = (
            f"| {label} | {s['answerable_hit_count']}/{s['answerable_total']} | "
            f"{s['abstain_correct_count']}/{s['abstain_total']} | "
            f"{s['pre_gen_abstain_count']}/{s['abstain_total']} | "
            f"{s['verified_citations_total']} | {s['unverified_ids_total']} | "
            f"{s['date_mismatches_total']} | {s['fallback_count']} | "
            f"{s['p50_ms']:.1f} | RM {s['cost_myr']:.6f} |"
        )
        _must_contain(readme, row, f"agent row ({split_key})", errors)

    # Check cited per-case rerank scores from agent report.json
    by_case = {
        c["id"]: c
        for c in agent.get("dev_cases", []) + agent.get("holdout_cases", [])
    }
    for cid in ("ad06", "ad07", "ah08"):
        if cid in by_case and by_case[cid]["top_rerank_score"] is not None:
            score_str = f"`{by_case[cid]['top_rerank_score']:.4f}`"
            _must_contain(readme, score_str, f"agent case {cid} top_rerank_score", errors)

    total_cost = f"RM {report['total_eval_cost_myr']:.6f}"
    _must_contain(readme, total_cost, "total eval spend", errors)

    # Require PR0 bakeoff in report.json and check every row verbatim
    pr0 = report.get("pr0_bakeoff")
    if not pr0:
        errors.append("Missing required `pr0_bakeoff` block in evals/report.json")
        return errors

    emb = pr0["embedders"]
    rr = pr0["rerankers"]
    pr0_checks = [
        (
            "PR0 bge-m3_lower",
            f"| YTL `bge-m3` lowercased (`vector(1024)`) | "
            f"{emb['bge-m3_lower']['scores']['overall']['r50']:.4f} | "
            f"{emb['bge-m3_lower']['scores']['overall']['mrr10']:.4f} | "
            f"{emb['bge-m3_lower']['scores']['english_cross_lingual']['r50']:.4f} | "
            f"{emb['bge-m3_lower']['scores']['english_cross_lingual']['mrr10']:.4f} | "
            f"{emb['bge-m3_lower']['single_query_p50_ms']:.1f} ms p50 (1-query) |",
        ),
        (
            "PR0 bge-m3_raw",
            f"| YTL `bge-m3` raw ALL-CAPS | "
            f"{emb['bge-m3_raw']['scores']['overall']['r50']:.4f} | "
            f"{emb['bge-m3_raw']['scores']['overall']['mrr10']:.4f} | "
            f"{emb['bge-m3_raw']['scores']['english_cross_lingual']['r50']:.4f} | "
            f"{emb['bge-m3_raw']['scores']['english_cross_lingual']['mrr10']:.4f} | "
            f"{emb['bge-m3_raw']['single_query_p50_ms']:.1f} ms p50 (1-query) |",
        ),
        (
            "PR0 ilmu-embedding-v1_lower",
            f"| YTL `ilmu-embedding-v1` lowercased (4096d) | "
            f"{emb['ilmu-embedding-v1_lower']['scores']['overall']['r50']:.4f} | "
            f"{emb['ilmu-embedding-v1_lower']['scores']['overall']['mrr10']:.4f} | "
            f"{emb['ilmu-embedding-v1_lower']['scores']['english_cross_lingual']['r50']:.4f} | "
            f"{emb['ilmu-embedding-v1_lower']['scores']['english_cross_lingual']['mrr10']:.4f} | "
            f"{emb['ilmu-embedding-v1_lower']['single_query_p50_ms']:.1f} ms p50 (1-query) |",
        ),
        (
            "PR0 minilm_lower",
            f"| Local `MiniLM-L12-v2` lowercased (384d) | "
            f"{emb['minilm_lower']['scores']['overall']['r50']:.4f} | "
            f"{emb['minilm_lower']['scores']['overall']['mrr10']:.4f} | "
            f"{emb['minilm_lower']['scores']['english_cross_lingual']['r50']:.4f} | "
            f"{emb['minilm_lower']['scores']['english_cross_lingual']['mrr10']:.4f} | "
            f"{emb['minilm_lower']['query_avg_ms']:.2f} ms avg |",
        ),
        (
            "PR0 ilmu_bge_reranker_lower",
            f"| YTL `bge-reranker` lowercased (top-20 pool) | "
            f"{rr['ilmu_bge_reranker_lower']['scores']['overall']['r50']:.4f} | "
            f"{rr['ilmu_bge_reranker_lower']['scores']['overall']['mrr10']:.4f} | "
            f"{rr['ilmu_bge_reranker_lower']['scores']['english_cross_lingual']['r50']:.4f} | "
            f"{rr['ilmu_bge_reranker_lower']['scores']['english_cross_lingual']['mrr10']:.4f} | "
            f"{rr['ilmu_bge_reranker_lower']['p50_ms']:.1f} ms p50 / {rr['ilmu_bge_reranker_lower']['p95_ms']:.1f} ms p95 |",
        ),
        (
            "PR0 local_bge_reranker_int8_lower",
            f"| Local `bge-reranker-v2-m3-int8` CPU | "
            f"{rr['local_bge_reranker_int8_lower']['scores']['overall']['r50']:.4f} | "
            f"{rr['local_bge_reranker_int8_lower']['scores']['overall']['mrr10']:.4f} | "
            f"{rr['local_bge_reranker_int8_lower']['scores']['english_cross_lingual']['r50']:.4f} | "
            f"{rr['local_bge_reranker_int8_lower']['scores']['english_cross_lingual']['mrr10']:.4f} | "
            f"{rr['local_bge_reranker_int8_lower']['p50_ms']:.1f} ms p50 / {rr['local_bge_reranker_int8_lower']['p95_ms']:.1f} ms p95 |",
        ),
        (
            "PR0 token count lowercased",
            f"`{emb['bge-m3_lower']['tokens']:,}` vs `{emb['bge-m3_raw']['tokens']:,}` tokens",
        ),
    ]
    for lbl, snip in pr0_checks:
        _must_contain(readme, snip, lbl, errors)

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate README.md numbers against evals/report.json.")
    parser.add_argument("--readme", type=Path, default=Path("README.md"))
    parser.add_argument("--report", type=Path, default=Path("evals/report.json"))
    args = parser.parse_args()

    errors = check_readme(args.readme, args.report)
    if errors:
        for err in errors:
            print(f"ERROR: {err}", file=sys.stderr)
        return 1
    print(f"README metrics verified cleanly against {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

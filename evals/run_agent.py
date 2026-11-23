"""Run split end-to-end TenderAgent evaluation across agent_dev_10 and untouched agent_holdout_10."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

from evals.run_retrieval import render_report_md
from tender_os.agent import TenderAgent
from tender_os.config import Settings
from tender_os.domain import AskRequest
from tender_os.ilmu.router import ILMURouter
from tender_os.ilmu.spend import PostgresSpendLedger
from tender_os.retrieval import RetrievalEngine
from tender_os.storage.postgres import PostgresStore


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _run_split(
    agent: TenderAgent,
    cases: list[dict[str, Any]],
    *,
    budget_limit_micro_rm: int,
    cumulative_micro_rm: list[int],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    case_records: list[dict[str, Any]] = []
    answerable_total = 0
    answerable_hit_count = 0
    abstain_total = 0
    abstain_correct_count = 0
    pre_gen_abstain_count = 0
    verified_citations_total = 0
    unverified_ids_total = 0
    date_mismatches_total = 0
    uncited_answerable_count = 0
    degraded_count = 0
    fallback_count = 0
    latencies: list[float] = []
    split_micro_rm = 0
    cost_by_basis: dict[str, int] = {}

    for case in cases:
        if cumulative_micro_rm[0] >= budget_limit_micro_rm:
            raise RuntimeError(
                f"Agent eval aborted at hard budget cap ({cumulative_micro_rm[0] / 1_000_000:.4f} MYR)"
            )
        t0 = time.monotonic()
        resp = agent.ask(AskRequest(question=case["question"]))
        elapsed_ms = round((time.monotonic() - t0) * 1000, 1)
        latencies.append(elapsed_ms)

        case_cost = resp.trace.total_cost_micro_rm
        split_micro_rm += case_cost
        cumulative_micro_rm[0] += case_cost
        for att in resp.trace.attempts:
            cost_by_basis[att.pricing_basis] = cost_by_basis.get(att.pricing_basis, 0) + att.cost_micro_rm
            if att.fallback_used:
                fallback_count += 1
        if resp.degraded_mode:
            degraded_count += 1

        cited_ids = [c.tender_id for c in resp.grounding.verified_citations]
        verified_citations_total += len(cited_ids)
        unverified_ids_total += len(resp.grounding.unverified_ids)
        date_mismatches_total += len(resp.grounding.date_mismatches)
        chat_calls = sum(1 for a in resp.trace.attempts if a.stage == "chat")

        expected = set(case["expected_ids"])
        if case["expect_abstain"]:
            abstain_total += 1
            if resp.grounding.status == "abstained":
                abstain_correct_count += 1
                if chat_calls == 0:
                    pre_gen_abstain_count += 1
            hit_ok = resp.grounding.status == "abstained"
        else:
            answerable_total += 1
            hit_ok = bool(set(cited_ids) & expected)
            if hit_ok:
                answerable_hit_count += 1
            if resp.grounding.status == "unverified":
                uncited_answerable_count += 1

        case_records.append(
            {
                "id": case["id"],
                "expect_abstain": case["expect_abstain"],
                "passed": hit_ok,
                "route": resp.route.value,
                "grounding_status": resp.grounding.status,
                "abstain_reason": resp.grounding.abstain_reason,
                "top_rerank_score": resp.trace.top_rerank_score,
                "chat_calls": chat_calls,
                "cited_ids": cited_ids,
                "expected_ids": case["expected_ids"],
                "unverified_ids": resp.grounding.unverified_ids,
                "date_mismatches": resp.grounding.date_mismatches,
                "latency_ms": elapsed_ms,
                "cost_micro_rm": case_cost,
            }
        )

    ordered = sorted(latencies)
    p95_idx = min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1)))) if ordered else 0
    summary = {
        "total_cases": len(cases),
        "answerable_total": answerable_total,
        "answerable_hit_count": answerable_hit_count,
        "abstain_total": abstain_total,
        "abstain_correct_count": abstain_correct_count,
        "pre_gen_abstain_count": pre_gen_abstain_count,
        "verified_citations_total": verified_citations_total,
        "unverified_ids_total": unverified_ids_total,
        "date_mismatches_total": date_mismatches_total,
        "uncited_answerable_count": uncited_answerable_count,
        "degraded_count": degraded_count,
        "fallback_count": fallback_count,
        "p50_ms": round(statistics.median(latencies), 1) if latencies else 0.0,
        "p95_ms": round(ordered[p95_idx], 1) if latencies else 0.0,
        "cost_micro_rm": split_micro_rm,
        "cost_myr": round(split_micro_rm / 1_000_000, 6),
        "cost_by_pricing_basis_micro_rm": cost_by_basis,
        "_latencies": latencies,
    }
    return summary, case_records


def _combine(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    lat = sorted(a["_latencies"] + b["_latencies"])
    p95_idx = min(len(lat) - 1, int(round(0.95 * (len(lat) - 1)))) if lat else 0
    combined_micro = a["cost_micro_rm"] + b["cost_micro_rm"]
    basis: dict[str, int] = {}
    for src in (a["cost_by_pricing_basis_micro_rm"], b["cost_by_pricing_basis_micro_rm"]):
        for k, v in src.items():
            basis[k] = basis.get(k, 0) + v
    return {
        "total_cases": a["total_cases"] + b["total_cases"],
        "answerable_total": a["answerable_total"] + b["answerable_total"],
        "answerable_hit_count": a["answerable_hit_count"] + b["answerable_hit_count"],
        "abstain_total": a["abstain_total"] + b["abstain_total"],
        "abstain_correct_count": a["abstain_correct_count"] + b["abstain_correct_count"],
        "pre_gen_abstain_count": a["pre_gen_abstain_count"] + b["pre_gen_abstain_count"],
        "verified_citations_total": a["verified_citations_total"] + b["verified_citations_total"],
        "unverified_ids_total": a["unverified_ids_total"] + b["unverified_ids_total"],
        "date_mismatches_total": a["date_mismatches_total"] + b["date_mismatches_total"],
        "uncited_answerable_count": a["uncited_answerable_count"] + b["uncited_answerable_count"],
        "degraded_count": a["degraded_count"] + b["degraded_count"],
        "fallback_count": a["fallback_count"] + b["fallback_count"],
        "p50_ms": round(statistics.median(lat), 1) if lat else 0.0,
        "p95_ms": round(lat[p95_idx], 1) if lat else 0.0,
        "cost_micro_rm": combined_micro,
        "cost_myr": round(combined_micro / 1_000_000, 6),
        "cost_by_pricing_basis_micro_rm": basis,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 20-case split agent eval against live Postgres + ILMU.")
    parser.add_argument("--dev-file", type=Path, default=Path("evals/agent_dev_10.jsonl"))
    parser.add_argument("--holdout-file", type=Path, default=Path("evals/agent_holdout_10.jsonl"))
    parser.add_argument("--out-json", type=Path, default=Path("evals/report.json"))
    parser.add_argument("--out-md", type=Path, default=Path("evals/REPORT.md"))
    parser.add_argument("--budget-myr", type=float, default=2.0)
    args = parser.parse_args()

    settings = Settings()
    if not settings.ilmu_api_key:
        raise SystemExit("ILMU_API_KEY must be set to run live agent evaluation.")

    dev_cases = _load_jsonl(args.dev_file)
    holdout_cases = _load_jsonl(args.holdout_file)

    existing: dict[str, Any] = (
        json.loads(args.out_json.read_text(encoding="utf-8")) if args.out_json.is_file() else {}
    )
    prior_retrieval_micro = int(existing.get("retrieval", {}).get("cost_micro_rm", 0))
    budget_micro = int(round(args.budget_myr * 1_000_000))
    cumulative_micro = [prior_retrieval_micro]

    with PostgresStore(settings.database_url) as store:
        ledger = PostgresSpendLedger(store.pool, cap_micro_rm=settings.daily_cap_micro_rm)
        with ILMURouter(settings, spend_ledger=ledger) as router:
            engine = RetrievalEngine(store, router, settings)
            agent = TenderAgent(store, engine, router, settings)
            dev_summary, dev_records = _run_split(
                agent,
                dev_cases,
                budget_limit_micro_rm=budget_micro,
                cumulative_micro_rm=cumulative_micro,
            )
            holdout_summary, holdout_records = _run_split(
                agent,
                holdout_cases,
                budget_limit_micro_rm=budget_micro,
                cumulative_micro_rm=cumulative_micro,
            )

    combined = _combine(dev_summary, holdout_summary)
    dev_summary.pop("_latencies", None)
    holdout_summary.pop("_latencies", None)

    report = dict(existing)
    report["agent"] = {
        "run_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "dev_10": dev_summary,
        "holdout_10": holdout_summary,
        "combined_20": combined,
        "cost_micro_rm": combined["cost_micro_rm"],
        "cost_myr": combined["cost_myr"],
        "dev_cases": dev_records,
        "holdout_cases": holdout_records,
    }
    report["total_eval_cost_myr"] = round(
        (prior_retrieval_micro + combined["cost_micro_rm"]) / 1_000_000,
        6,
    )

    args.out_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    args.out_md.write_text(render_report_md(report), encoding="utf-8")
    print(json.dumps({"dev_10": dev_summary, "holdout_10": holdout_summary, "total_eval_cost_myr": report["total_eval_cost_myr"]}, indent=2))


if __name__ == "__main__":
    main()

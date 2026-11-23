"""Integrity and strict zero-overlap tests for dev, PR0 tuning, and untouched holdout eval datasets."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "data" / "tenders_public.jsonl"
PR0_45 = ROOT / "evals" / "pr0" / "queries_45.jsonl"
DEV_30 = ROOT / "evals" / "dev_30.jsonl"
HOLDOUT_30 = ROOT / "evals" / "holdout_30.jsonl"
AGENT_DEV_10 = ROOT / "evals" / "agent_dev_10.jsonl"
AGENT_HOLDOUT_10 = ROOT / "evals" / "agent_holdout_10.jsonl"


def _load(path: Path) -> list[dict]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _check_retrieval_split(path: Path, by_id: dict[str, dict]) -> tuple[set[str], set[str]]:
    qs = _load(path)
    assert len(qs) == 30
    assert len({q["id"] for q in qs}) == 30
    assert len({q["query"].strip().lower() for q in qs}) == 30

    buckets = Counter(q["bucket"] for q in qs)
    assert buckets == {
        "exact_code_ref": 8,
        "malay_keyword": 7,
        "english_cross_lingual": 15,
    }

    portals: set[str] = set()
    all_rids: set[str] = set()
    for q in qs:
        assert q["relevant_ids"]
        assert len(q["relevant_ids"]) == len(set(q["relevant_ids"])), (
            f"{q['id']} has duplicate relevant_ids"
        )
        for rid in q["relevant_ids"]:
            assert rid in by_id, f"{q['id']} references missing tender id: {rid}"
            portals.add(by_id[rid]["source"])
            all_rids.add(rid)
    assert len(portals) >= 8
    return {q["query"].strip().lower() for q in qs}, all_rids


def _check_agent_split(path: Path, valid_ids: set[str]) -> tuple[set[str], set[str]]:
    qs = _load(path)
    assert len(qs) == 10
    assert len({q["id"] for q in qs}) == 10
    assert len({q["question"].strip().lower() for q in qs}) == 10

    answerable = [q for q in qs if not q["expect_abstain"]]
    abstain = [q for q in qs if q["expect_abstain"]]
    assert len(answerable) == 7
    assert len(abstain) == 3

    all_eids: set[str] = set()
    for q in answerable:
        assert q["expected_ids"]
        for eid in q["expected_ids"]:
            assert eid in valid_ids, f"{q['id']} references missing tender id: {eid}"
            all_eids.add(eid)
    for q in abstain:
        assert q["expected_ids"] == []
    return {q["question"].strip().lower() for q in qs}, all_eids


def test_dev_and_holdout_splits_integrity_and_zero_overlap() -> None:
    by_id = {r["id"]: r for r in _load(SNAPSHOT)}
    assert len(by_id) >= 6000
    valid_ids = set(by_id)

    holdout_qs, holdout_rids = _check_retrieval_split(HOLDOUT_30, by_id)
    dev_qs, dev_rids = _check_retrieval_split(DEV_30, by_id)
    ah_qs, ah_eids = _check_agent_split(AGENT_HOLDOUT_10, valid_ids)
    ad_qs, ad_eids = _check_agent_split(AGENT_DEV_10, valid_ids)

    pr0_rows = _load(PR0_45)
    pr0_qs = {q["query"].strip().lower() for q in pr0_rows}
    pr0_rids = {rid for q in pr0_rows for rid in q["relevant_ids"]}

    # Verify zero query and tender ID overlap across every dev/PR0 vs holdout boundary
    assert not (holdout_qs & dev_qs), "Retrieval dev and holdout share query strings!"
    assert not (holdout_rids & dev_rids), "Retrieval dev and holdout share relevant tender IDs!"
    assert not (ah_qs & ad_qs), "Agent dev and holdout share questions!"
    assert not (ah_eids & ad_eids), "Agent dev and holdout share expected tender IDs!"
    assert not (holdout_rids & ad_eids), "holdout_30 and agent_dev_10 share tender IDs!"
    assert not (dev_rids & ah_eids), "dev_30 and agent_holdout_10 share tender IDs!"
    assert not (holdout_qs & pr0_qs), "holdout_30 shares query strings with PR0!"
    assert not (holdout_rids & pr0_rids), "holdout_30 shares tender IDs with PR0!"
    assert not (ah_eids & pr0_rids), "agent_holdout_10 shares tender IDs with PR0!"

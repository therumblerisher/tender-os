"""Integrity tests for the frozen PR1 untouched holdout datasets."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "data" / "tenders_public.jsonl"
HOLDOUT_30 = ROOT / "evals" / "holdout_30.jsonl"
AGENT_HOLDOUT_10 = ROOT / "evals" / "agent_holdout_10.jsonl"


def _load(path: Path) -> list[dict]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def test_frozen_holdout_30_integrity() -> None:
    by_id = {r["id"]: r for r in _load(SNAPSHOT)}
    assert len(by_id) >= 6000

    qs = _load(HOLDOUT_30)
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
    for q in qs:
        assert q["relevant_ids"]
        assert len(q["relevant_ids"]) == len(set(q["relevant_ids"])), (
            f"{q['id']} has duplicate relevant_ids"
        )
        for rid in q["relevant_ids"]:
            assert rid in by_id, f"{q['id']} references missing tender id: {rid}"
            portals.add(by_id[rid]["source"])
    assert len(portals) >= 8


def test_frozen_agent_holdout_10_integrity() -> None:
    valid = {r["id"] for r in _load(SNAPSHOT)}
    qs = _load(AGENT_HOLDOUT_10)
    assert len(qs) == 10
    answerable = [q for q in qs if not q["expect_abstain"]]
    abstain = [q for q in qs if q["expect_abstain"]]
    assert len(answerable) == 7
    assert len(abstain) == 3

    for q in answerable:
        assert q["expected_ids"]
        for eid in q["expected_ids"]:
            assert eid in valid, f"{q['id']} references missing tender id: {eid}"
    for q in abstain:
        assert q["expected_ids"] == []

"""Tests ensuring README.md headline metrics match evals/report.json and drift fails CI."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.check_readme_numbers import check_readme

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
REPORT = ROOT / "evals" / "report.json"


def test_readme_numbers_match_eval_report() -> None:
    assert check_readme(README, REPORT) == []


def test_readme_numbers_detects_seeded_mismatch(tmp_path: Path) -> None:
    drifted = tmp_path / "README.md"
    drifted.write_text(
        README.read_text(encoding="utf-8").replace("0.9500", "0.9999", 1),
        encoding="utf-8",
    )
    errors = check_readme(drifted, REPORT)
    assert errors
    assert any("holdout retrieval row (hybrid_rerank)" in e for e in errors)

    # Also fail loudly if exact_code_ref column drifts or pr0_bakeoff is missing
    drifted_exact = tmp_path / "README_exact.md"
    drifted_exact.write_text(
        README.read_text(encoding="utf-8").replace("0.8750", "0.8700", 1),
        encoding="utf-8",
    )
    assert any("holdout retrieval row (hybrid_rerank)" in e for e in check_readme(drifted_exact, REPORT))

    no_pr0 = tmp_path / "report_no_pr0.json"
    data = json.loads(REPORT.read_text(encoding="utf-8"))
    data.pop("pr0_bakeoff", None)
    no_pr0.write_text(json.dumps(data), encoding="utf-8")
    assert any("pr0_bakeoff" in e for e in check_readme(README, no_pr0))

"""Verify evaluation reporting and honest separation from model quality."""

import json
from pathlib import Path

import pytest

from evals.run import evaluate


def test_offline_fixture_replay() -> None:
    """All checked-in pipeline fixtures pass with clearly labeled provenance."""
    report = evaluate()
    assert report["purpose"] == "pipeline_contracts"
    assert report["model"] == "scripted_fixture_replay"
    assert report["passed"] == report["total"] == 7
    assert len(report["dataset_sha256"]) == 64


def test_evaluator_reports_failure(tmp_path: Path) -> None:
    """Incorrect expected outcomes remain failures rather than fabricated passes."""
    dataset = tmp_path / "cases.json"
    dataset.write_text(json.dumps([{"id": "wrong-expectation", "documents": ["launch Monday"],
                                  "question": "bananas", "expected_abstained": False}]))
    report = evaluate(dataset=dataset)
    assert report["passed"] == 0
    assert report["total"] == 1
    assert report["cases"][0]["passed"] is False


def test_evaluator_validates_mode() -> None:
    """Unknown modes cannot accidentally trigger a live provider call."""
    with pytest.raises(ValueError):
        evaluate("invalid")


def test_live_provider_errors_never_score_as_abstention(monkeypatch: pytest.MonkeyPatch) -> None:
    """Live-mode outages cannot inflate the abstention success count."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    report = evaluate("live")
    assert report["purpose"] == "live_fixture_checks"
    assert report["total"] == 5
    assert report["passed"] == 2  # Two no-evidence cases do not require a provider.
    errors = [row for row in report["cases"] if row["reason"] == "model_unavailable"]
    assert len(errors) == 3
    assert all(not row["passed"] for row in errors)

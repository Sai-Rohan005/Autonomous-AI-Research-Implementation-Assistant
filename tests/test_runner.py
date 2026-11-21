"""Offline end-to-end structured routing without replacing the actual manager."""

import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from core.runner import run_agents


@pytest.mark.parametrize("query", ["", None, "Explain AI"])
def test_structured_runner(query: str | None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Real manager validation integrates with safety, final output, and trace writes."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("core.runner.generate_text", Mock(return_value="SAFE"))
    monkeypatch.setattr("tools.manager_tool.generate_structured", Mock(return_value=json.dumps({
        "tool": "final_answer", "arguments": {"answer": "answer"}, "rationale": "Enough information"})))
    result = run_agents(query)
    assert result["final_answer"] == "answer" and result["stop_reason"] == "final_answer"
    assert (tmp_path / "traces" / f'{result["trace"]["run_id"]}.json').exists()

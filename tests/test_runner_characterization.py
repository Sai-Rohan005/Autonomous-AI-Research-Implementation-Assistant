"""Fifteen runner contract tests, updated for the intentional structured migration."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from core.models import ToolCall


def call(tool: str, **arguments: str) -> ToolCall:
    """Build a validated router decision for a test."""
    return ToolCall(tool=tool, arguments=arguments, rationale="test")


@pytest.fixture
def harness(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> SimpleNamespace:
    """Load the runner while replacing all network and worker entry points."""
    path = Path(__file__).resolve().parents[1] / "core" / "runner.py"
    spec = importlib.util.spec_from_file_location("characterized_runner", path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    llm = Mock(side_effect=["SAFE", "final answer"])
    manager = Mock()
    manager._run.return_value = call("final_answer")
    execute = Mock(return_value={"status": "ok", "results": "tool result"})
    monkeypatch.setattr(runner, "generate_text", llm)
    monkeypatch.setattr(runner, "ManagerTool", Mock(return_value=manager))
    monkeypatch.setattr(runner, "execute_tool", execute)
    monkeypatch.chdir(tmp_path)
    return SimpleNamespace(runner=runner, llm=llm, manager=manager, execute=execute)


def test_01_non_safe_responses_block_without_planning(harness: SimpleNamespace) -> None:
    """Only the normalized SAFE verdict permits execution."""
    for verdict in ["UNSAFE", "", "SAFE explanation"]:
        harness.llm.side_effect = [verdict]
        result = harness.runner.run_agents("query")
        assert result["stop_reason"] == "blocked" and result["history"] == []
        assert result["final_answer"] == "❌ This query cannot be processed due to safety policies."
    harness.manager._run.assert_not_called()


def test_02_safe_normalization_and_final_synthesis(harness: SimpleNamespace) -> None:
    """An empty final-answer payload requests the existing final generation."""
    harness.llm.side_effect = ["  safe\n", "final answer"]
    result = harness.runner.run_agents("query")
    assert result["final_answer"] == "final answer" and result["history"] == []
    assert all(result[key] is None for key in ["summary", "code", "comparison", "report", "search"])
    assert harness.llm.call_count == 2


def test_03_unstructured_decision_clarifies(harness: SimpleNamespace) -> None:
    """Text labels are rejected rather than silently treated as finish."""
    harness.manager._run.return_value = "Action: code"
    assert harness.runner.run_agents("query")["stop_reason"] == "clarify"
    harness.execute.assert_not_called()


def test_04_unknown_tool_clarifies(harness: SimpleNamespace) -> None:
    """A stub cannot bypass the runner's second validation boundary."""
    harness.manager._run.return_value = {"tool": "shell", "arguments": {}, "rationale": "test"}
    assert harness.runner.run_agents("query")["stop_reason"] == "clarify"
    harness.execute.assert_not_called()


def test_05_web_search_receives_validated_arguments(harness: SimpleNamespace) -> None:
    """Search uses router arguments and retains the UI's search result field."""
    harness.manager._run.side_effect = [call("web_search", query="refined"), call("final_answer")]
    result = harness.runner.run_agents("query")
    assert harness.execute.call_args.args[1] == {"query": "refined"}
    assert result["search"] == {"status": "ok", "results": "tool result"}
    assert result["history"][0]["action"] == "web_search"


def test_06_summarize_receives_explicit_text(harness: SimpleNamespace) -> None:
    """Summarization uses the text in its validated argument contract."""
    harness.manager._run.side_effect = [call("summarize", text="source"), call("final_answer")]
    result = harness.runner.run_agents("query")
    assert harness.execute.call_args.args[1] == {"text": "source"}
    assert result["summary"]["status"] == "ok"


def test_07_code_receives_validated_query(harness: SimpleNamespace) -> None:
    """Code generates text through the named registry entry."""
    harness.manager._run.side_effect = [call("code", query="task"), call("final_answer")]
    assert harness.runner.run_agents("query")["code"]["status"] == "ok"
    assert harness.execute.call_args.args[1] == {"query": "task"}


def test_08_compare_retains_ui_field(harness: SimpleNamespace) -> None:
    """Comparison still uses the comparison response key."""
    harness.manager._run.side_effect = [call("compare", query="A vs B"), call("final_answer")]
    assert harness.runner.run_agents("query")["comparison"]["status"] == "ok"


def test_09_report_is_not_a_routable_action() -> None:
    """The requested tool enum excludes the old report action."""
    with pytest.raises(ValueError):
        call("report")


def test_10_nonconsecutive_repeated_call_stops(harness: SimpleNamespace) -> None:
    """A full-run seen set catches repeated calls even after another tool."""
    harness.manager._run.side_effect = [call("web_search", query="q"), call("code", query="q"),
                                      call("web_search", query="q")]
    result = harness.runner.run_agents("query")
    assert result["stop_reason"] == "loop_detected"
    assert harness.execute.call_count == 2
    assert harness.llm.call_count == 1


def test_11_default_five_steps_and_three_item_history(harness: SimpleNamespace) -> None:
    """Distinct arguments allow repeated tools; default exhaustion is explicit."""
    harness.manager._run.side_effect = [call("code", query=str(index)) for index in range(5)]
    result = harness.runner.run_agents("query")
    assert result["stop_reason"] == "max_steps" and harness.execute.call_count == 5
    assert [item["step"] for item in result["history"]] == [2, 3, 4]
    assert harness.llm.call_count == 1


def test_12_failed_observation_reflects_and_truncates(harness: SimpleNamespace) -> None:
    """Returned error results still enter history and inform the next decision."""
    harness.execute.return_value = {"status": "error", "results": "x" * 400}
    harness.manager._run.side_effect = [call("code", query="q"), call("final_answer")]
    result = harness.runner.run_agents("query")
    assert result["history"][0]["observation"] == "x" * 300
    context, previous = harness.manager._run.call_args.args
    assert context["reflection"] == "Previous step failed"
    assert previous == result["history"][0]


def test_13_manager_exception_clarifies(harness: SimpleNamespace) -> None:
    """Router outages do not trigger ungrounded final generation."""
    harness.manager._run.side_effect = RuntimeError("SECRET")
    result = harness.runner.run_agents("query")
    assert result["stop_reason"] == "clarify" and harness.llm.call_count == 1
    assert "SECRET" not in result["final_answer"]


def test_14_direct_final_answer_skips_extra_generation(harness: SimpleNamespace) -> None:
    """A supplied final answer returns directly after safety and routing."""
    harness.manager._run.return_value = call("final_answer", answer="answer")
    assert harness.runner.run_agents("query")["final_answer"] == "answer"
    assert harness.llm.call_count == 1


def test_15_empty_and_none_queries_are_forwarded(harness: SimpleNamespace) -> None:
    """The unchanged safety filter still receives empty and None queries."""
    for query in ["", None]:
        harness.llm.side_effect = ["SAFE", "final answer"]
        assert harness.runner.run_agents(query)["final_answer"] == "final answer"
        assert f"Query: {query}" in harness.llm.call_args_list[-2].args[0]

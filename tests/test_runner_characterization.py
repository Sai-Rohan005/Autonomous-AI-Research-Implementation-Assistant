"""Fifteen offline characterizations of the existing runner decision behavior."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.fixture
def harness(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> SimpleNamespace:
    """Import the runner with isolated LLM, planner, and agent dependencies."""
    llm = Mock(side_effect=["SAFE", "final answer"])
    manager = Mock()
    manager._run.return_value = "Action: finish"
    agents = {}
    dependencies = {"utils.llm": {"generate_text": llm},
                    "tools.manager_tool": {"ManagerTool": Mock(return_value=manager)}}
    for module, symbol in [("search", "Search_Agent"), ("summarization", "Summarization_Agent"),
                           ("code", "Code_Agent"), ("comparision", "Comparison_Agent"),
                           ("report", "Report_Agent")]:
        agent = Mock()
        agent.return_value.run.return_value = {"status": "ok", "results": module}
        agents[module] = agent
        dependencies[f"agents.{module}_agent"] = {symbol: agent}
    for name, values in dependencies.items():
        stub = ModuleType(name)
        stub.__dict__.update(values)
        monkeypatch.setitem(sys.modules, name, stub)
    path = Path(__file__).resolve().parents[1] / "core" / "runner.py"
    spec = importlib.util.spec_from_file_location("characterized_runner", path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    monkeypatch.chdir(tmp_path)
    return SimpleNamespace(runner=runner, llm=llm, manager=manager, agents=agents)


def test_01_non_safe_responses_block_without_planning(harness: SimpleNamespace) -> None:
    """Only the exact normalized SAFE verdict permits execution."""
    for verdict in ["UNSAFE", "", "SAFE explanation"]:
        harness.llm.side_effect = [verdict]
        result = harness.runner.run_agents("query")
        assert result["final_answer"] == "❌ This query cannot be processed due to safety policies."
        assert result["history"] == []
    harness.manager._run.assert_not_called()


def test_02_safe_normalization_and_finish(harness: SimpleNamespace) -> None:
    """Whitespace and casing normalize; finish still generates a final answer."""
    harness.llm.side_effect = ["  safe\n", "final answer"]
    result = harness.runner.run_agents("query")
    assert result["final_answer"] == "final answer"
    assert result["history"] == []
    assert all(result[key] is None for key in ["summary", "code", "comparison", "report", "search"])
    assert harness.llm.call_count == 2


def test_03_missing_action_defaults_to_finish(harness: SimpleNamespace) -> None:
    """Unparseable manager output terminates without a tool call."""
    harness.manager._run.return_value = "No structured action"
    assert harness.runner.run_agents("query")["history"] == []
    assert harness.manager._run.call_count == 1


def test_04_unknown_action_stops(harness: SimpleNamespace) -> None:
    """Unknown actions stop the loop and still produce the final response."""
    harness.manager._run.return_value = "Action: unknown"
    result = harness.runner.run_agents("query")
    assert result["history"] == []
    assert result["final_answer"] == "final answer"


def test_05_search_case_insensitive_and_query_forwarding(harness: SimpleNamespace) -> None:
    """Search receives the original query, and its full result is returned."""
    harness.manager._run.side_effect = ["Thought: next\naCtIoN : SEARCH", "Action: finish"]
    result = harness.runner.run_agents("query")
    harness.agents["search"].return_value.run.assert_called_once_with("query")
    assert result["search"] == {"status": "ok", "results": "search"}
    assert result["history"] == [{"step": 0, "action": "search", "observation": "search", "status": "ok"}]


def test_06_summarize_receives_results_as_string(harness: SimpleNamespace) -> None:
    """Summarization uses accumulated results and returns the summary field."""
    harness.manager._run.side_effect = ["Action: search", "Action: summarize", "Action: finish"]
    result = harness.runner.run_agents("query")
    harness.agents["summarization"].return_value.run.assert_called_once_with(
        "{'search': {'status': 'ok', 'results': 'search'}}")
    assert result["summary"] == {"status": "ok", "results": "summarization"}


def test_07_code_receives_query(harness: SimpleNamespace) -> None:
    """Code generation receives the original query."""
    harness.manager._run.side_effect = ["Action: code", "Action: finish"]
    result = harness.runner.run_agents("query")
    harness.agents["code"].return_value.run.assert_called_once_with("query")
    assert result["code"] == {"status": "ok", "results": "code"}


def test_08_compare_receives_query(harness: SimpleNamespace) -> None:
    """Comparison uses the query and the comparison response key."""
    harness.manager._run.side_effect = ["Action: compare", "Action: finish"]
    result = harness.runner.run_agents("query")
    harness.agents["comparision"].return_value.run.assert_called_once_with("query")
    assert result["comparison"] == {"status": "ok", "results": "comparision"}


def test_09_report_receives_results_dictionary(harness: SimpleNamespace) -> None:
    """Reporting receives the accumulated dictionary, rather than the query."""
    harness.manager._run.side_effect = ["Action: search", "Action: report", "Action: finish"]
    result = harness.runner.run_agents("query")
    passed = harness.agents["report"].return_value.run.call_args.args[0]
    assert passed["search"] == result["search"]
    assert result["report"] == {"status": "ok", "results": "report"}


def test_10_consecutive_repeated_action_stops(harness: SimpleNamespace) -> None:
    """A repeated action stops before executing the same tool twice."""
    harness.manager._run.return_value = "Action: search"
    result = harness.runner.run_agents("query")
    assert len(result["history"]) == 1
    assert harness.manager._run.call_count == 2
    assert harness.agents["search"].return_value.run.call_count == 1


def test_11_five_step_limit_and_three_item_history(harness: SimpleNamespace) -> None:
    """Nonconsecutive repeats are allowed; history retains only the last three."""
    harness.manager._run.side_effect = [f"Action: {action}" for action in
                                      ["search", "code", "search", "code", "search"]]
    result = harness.runner.run_agents("query")
    assert harness.manager._run.call_count == 5
    assert [item["step"] for item in result["history"]] == [2, 3, 4]
    assert harness.agents["search"].return_value.run.call_count == 3
    assert harness.agents["code"].return_value.run.call_count == 2


def test_12_failed_observation_reflects_and_truncates(harness: SimpleNamespace) -> None:
    """Tool failures enter history with a 300-character observation and reflection."""
    harness.agents["search"].return_value.run.return_value = {"status": "error", "results": "x" * 400}
    harness.manager._run.side_effect = ["Action: search", "Action: finish"]
    result = harness.runner.run_agents("query")
    assert result["history"][0]["observation"] == "x" * 300
    context, previous = harness.manager._run.call_args.args
    assert context["reflection"] == "Previous step failed"
    assert previous == result["history"][0]


def test_13_manager_exception_still_generates_final(harness: SimpleNamespace) -> None:
    """A loop exception ends planning but does not suppress final generation."""
    harness.manager._run.side_effect = RuntimeError("planner failed")
    result = harness.runner.run_agents("query")
    assert result["final_answer"] == "final answer"
    assert result["history"] == []
    assert "'status': 'error'" in harness.llm.call_args.args[0]


def test_14_malformed_observation_preserves_partial_result(harness: SimpleNamespace) -> None:
    """Missing status ends the loop after the result has already been stored."""
    harness.manager._run.return_value = "Action: search"
    harness.agents["search"].return_value.run.return_value = {"results": "partial"}
    result = harness.runner.run_agents("query")
    assert result["search"] == {"results": "partial"}
    assert result["history"] == []
    assert harness.manager._run.call_count == 1


def test_15_empty_and_none_queries_are_forwarded(harness: SimpleNamespace) -> None:
    """The existing runner does not validate empty or None queries locally."""
    for query in ["", None]:
        harness.llm.side_effect = ["SAFE", "final answer"]
        assert harness.runner.run_agents(query)["final_answer"] == "final answer"
        assert f"Query: {query}" in harness.llm.call_args_list[-2].args[0]

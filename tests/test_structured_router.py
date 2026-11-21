"""Router schema, repair policy, argument validation, and loop termination."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from core.execution import ToolExecutionError, ToolSpec
from core.models import ToolCall
from test_runner_characterization import call, harness
from tools.manager_tool import ManagerTool


def decision(tool: str = "final_answer", arguments: dict | None = None) -> str:
    """Serialize a complete model routing decision."""
    return json.dumps({"tool": tool, "arguments": {} if arguments is None else arguments, "rationale": "test"})


@pytest.mark.parametrize("bad", ["", "not json", "Action: code", "{}", decision("shell"),
                                 decision("code", {"query": 42}), decision("code", {"query": "x", "command": "SECRET"})])
def test_malformed_router_output_repairs_once(bad: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Malformed data triggers exactly one schema-constrained repair with an error."""
    generate = Mock(side_effect=[bad, decision()])
    monkeypatch.setattr("tools.manager_tool.generate_structured", generate)
    result = ManagerTool()._run({"query": "query"})
    assert result.tool == "final_answer" and generate.call_count == 2
    assert "Validation error:" in generate.call_args_list[1].args[0]
    assert "SECRET" not in generate.call_args_list[1].args[0]
    assert generate.call_args_list[0].args[1] == ToolCall.model_json_schema()


def test_two_invalid_outputs_fall_back_to_clarify(monkeypatch: pytest.MonkeyPatch) -> None:
    """Repeated schema failure cannot execute a tool or cause an unbounded retry."""
    generate = Mock(return_value="Action: web_search")
    monkeypatch.setattr("tools.manager_tool.generate_structured", generate)
    assert ManagerTool()._run({}).tool == "clarify"
    assert generate.call_count == 2


def test_router_outage_does_not_retry_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Transport failure has no invalid JSON to repair and clarifies immediately."""
    generate = Mock(side_effect=RuntimeError("SECRET"))
    monkeypatch.setattr("tools.manager_tool.generate_structured", generate)
    assert ManagerTool()._run({}).tool == "clarify"
    assert generate.call_count == 1


@pytest.mark.parametrize("tool,arguments", [("code", {"query": " "}), ("summarize", {"query": "x"}),
                                          ("web_search", {"query": "x", "url": "x"}),
                                          ("search_document", {"query": "x", "document_ids": []}),
                                          ("clarify", {}), ("final_answer", {"answer": 42})])
def test_tool_specific_argument_validation(tool: str, arguments: dict) -> None:
    """A valid enum name cannot smuggle invalid or additional arguments."""
    with pytest.raises(ValueError):
        ToolCall.model_validate_json(decision(tool, arguments))


def test_same_tool_with_different_arguments_can_continue(harness: SimpleNamespace) -> None:
    """Loop detection compares canonical arguments, not just tool names."""
    harness.manager._run.side_effect = [call("code", query="first"), call("code", query="second"), call("final_answer")]
    assert harness.runner.run_agents("query")["stop_reason"] == "final_answer"
    assert harness.execute.call_count == 2


def test_loop_key_ignores_argument_order_and_rationale(harness: SimpleNamespace) -> None:
    """Equivalent argument mappings are one call regardless of ordering or rationale."""
    first = ToolCall(tool="search_document", arguments={"query": "q", "document_ids": ["id"]}, rationale="one")
    second = ToolCall(tool="search_document", arguments={"document_ids": ["id"], "query": "q"}, rationale="two")
    harness.manager._run.side_effect = [first, second]
    registry = {"search_document": ToolSpec("search_document", Mock(), idempotent=True)}
    assert harness.runner.run_agents("query", tools=registry)["stop_reason"] == "loop_detected"
    assert harness.execute.call_count == 1


def test_configurable_max_steps_exhaustion(harness: SimpleNamespace) -> None:
    """The configured decision bound ends without an extra final LLM call."""
    harness.manager._run.side_effect = [call("code", query="first"), call("code", query="second")]
    result = harness.runner.run_agents("query", max_steps=2)
    assert result["stop_reason"] == "max_steps"
    assert harness.manager._run.call_count == harness.execute.call_count == 2
    assert harness.llm.call_count == 1


@pytest.mark.parametrize("max_steps", [0, -1, 1.5, True, 101])
def test_invalid_max_steps_rejected(max_steps: int, harness: SimpleNamespace) -> None:
    """Invalid execution budgets cannot silently disable the bound."""
    with pytest.raises(ValueError):
        harness.runner.run_agents("query", max_steps=max_steps)
    harness.llm.assert_not_called()


def test_timeout_is_explicit_and_does_not_generate_final(harness: SimpleNamespace) -> None:
    """A timed-out tool stops the run and exposes no raw exception text."""
    harness.manager._run.return_value = call("code", query="q")
    harness.execute.side_effect = TimeoutError("SECRET")
    result = harness.runner.run_agents("query")
    assert result["stop_reason"] == "timeout" and harness.llm.call_count == 1
    assert "SECRET" not in result["final_answer"]


def test_permanent_tool_error_stops_safely(harness: SimpleNamespace) -> None:
    """Permanent execution failures stop without another generation or raw error text."""
    harness.manager._run.return_value = call("code", query="q")
    harness.execute.side_effect = ToolExecutionError("SECRET")
    result = harness.runner.run_agents("query")
    assert result["stop_reason"] == "tool_error" and harness.llm.call_count == 1
    assert "SECRET" not in result["final_answer"]


def test_clarify_is_terminal(harness: SimpleNamespace) -> None:
    """Clarification returns the validated question without calling a tool."""
    harness.manager._run.return_value = call("clarify", question="Which document?")
    result = harness.runner.run_agents("query")
    assert result["final_answer"] == "Which document?" and result["stop_reason"] == "clarify"
    harness.execute.assert_not_called()


def test_unconfigured_document_search_clarifies(harness: SimpleNamespace) -> None:
    """Document search is unavailable until a trusted handler is configured."""
    harness.manager._run.return_value = ToolCall(tool="search_document", rationale="test",
                                               arguments={"query": "q", "document_ids": ["id"]})
    assert harness.runner.run_agents("query")["stop_reason"] == "clarify"
    harness.execute.assert_not_called()

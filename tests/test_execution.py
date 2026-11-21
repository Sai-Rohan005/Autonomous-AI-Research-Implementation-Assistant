"""Real worker deadlines plus Tenacity retry and trace integration tests."""

import multiprocessing
import time
from pathlib import Path
from unittest.mock import Mock

import pytest

from core.execution import ToolExecutionError, ToolSpec, TransientToolError, execute_once, execute_tool
from utils.trace import Trace, active_trace, operation


def slow_tool(arguments: dict) -> dict:
    """Create a delayed side effect only if the worker is not terminated."""
    time.sleep(3)
    Path(arguments["path"]).write_text("late side effect")
    return {"status": "ok", "results": "late"}


def traced_tool(arguments: dict) -> dict:
    """Return a child-process token event and a normal tool result."""
    with operation("llm:test", "input") as step:
        step.tokens_in, step.tokens_out = 3, 2
    return {"status": "ok", "results": arguments["text"]}


def invalid_tool(arguments: dict) -> dict:
    """Simulate a malformed adapter result."""
    return {"results": "missing status"}


def test_worker_timeout_terminates_process(tmp_path: Path) -> None:
    """A deadline returns promptly with no live worker left to perform side effects."""
    path = tmp_path / "late.txt"
    before = {child.pid for child in multiprocessing.active_children()}
    started = time.monotonic()
    with pytest.raises(TimeoutError):
        execute_once(ToolSpec("slow", slow_tool, timeout_seconds=0.4), {"path": str(path)})
    assert time.monotonic() - started < 2
    assert not path.exists()
    assert {child.pid for child in multiprocessing.active_children()} <= before


def test_worker_result_and_usage_join_parent_trace() -> None:
    """Spawned tool results and LLM accounting reach the parent's run trace."""
    trace = Trace()
    token = active_trace.set(trace)
    try:
        assert execute_tool(ToolSpec("test", traced_tool, 5), {"text": "result"}) == {
            "status": "ok", "results": "result"}
    finally:
        active_trace.reset(token)
    assert [step.tool for step in trace.steps] == ["test", "llm:test"]
    assert (trace.steps[1].tokens_in, trace.steps[1].tokens_out) == (3, 2)
    assert [step.step for step in trace.steps] == [1, 2]


@pytest.mark.parametrize("failure", [TimeoutError, TransientToolError])
@pytest.mark.parametrize("idempotent,attempts", [(True, 2), (False, 1)])
def test_retry_requires_explicit_idempotency(failure: type[Exception], idempotent: bool,
                                           attempts: int, monkeypatch: pytest.MonkeyPatch) -> None:
    """Timeout and transport errors retry once only for idempotent registrations."""
    invoke = Mock(side_effect=failure("SECRET"))
    monkeypatch.setattr("core.execution.execute_once", invoke)
    with pytest.raises(failure):
        execute_tool(ToolSpec("test", traced_tool, idempotent=idempotent), {})
    assert invoke.call_count == attempts


def test_permanent_errors_never_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    """A permanent adapter failure is not made transient by idempotency."""
    invoke = Mock(side_effect=ToolExecutionError("failure"))
    monkeypatch.setattr("core.execution.execute_once", invoke)
    with pytest.raises(ToolExecutionError):
        execute_tool(ToolSpec("test", traced_tool, idempotent=True), {})
    assert invoke.call_count == 1


def test_idempotent_transport_retry_can_recover(monkeypatch: pytest.MonkeyPatch) -> None:
    """A successful second attempt returns its actual result."""
    result = {"status": "ok", "results": "recovered"}
    invoke = Mock(side_effect=[TransientToolError("failure"), result])
    monkeypatch.setattr("core.execution.execute_once", invoke)
    assert execute_tool(ToolSpec("test", traced_tool, idempotent=True), {}) is result
    assert invoke.call_count == 2


def test_invalid_result_fails_closed() -> None:
    """A real worker cannot return an observation without a status contract."""
    with pytest.raises(ToolExecutionError):
        execute_once(ToolSpec("test", invalid_tool, 5), {})


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan")])
def test_invalid_tool_deadlines(timeout: float) -> None:
    """Unbounded or nonpositive per-tool execution budgets are rejected."""
    with pytest.raises(ValueError):
        ToolSpec("test", traced_tool, timeout)


def test_idempotency_must_be_explicit_boolean() -> None:
    """A truthy configuration string cannot accidentally enable retries."""
    with pytest.raises(ValueError):
        ToolSpec("test", traced_tool, idempotent="false")

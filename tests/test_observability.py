"""Offline checks for trace persistence, secret-safe logs, and Gemini accounting."""

import importlib.util
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from test_runner_characterization import harness  # Shared isolated runner fixture.
from utils.logger import JsonFormatter
from utils.trace import Trace, active_trace, operation, traced_run


@pytest.fixture
def llm_module(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Load the real Gemini wrapper with a mocked provider client."""
    path = Path(__file__).resolve().parents[1] / "utils" / "llm.py"
    spec = importlib.util.spec_from_file_location("observed_llm", path)
    module = importlib.util.module_from_spec(spec)
    # The backend test environment intentionally omits dotenv; isolate that import.
    import sys
    from types import ModuleType

    dotenv = ModuleType("dotenv")
    dotenv.load_dotenv = Mock()
    monkeypatch.setitem(sys.modules, "dotenv", dotenv)
    spec.loader.exec_module(module)
    module.client = Mock()
    return module


def response(text: str = "answer", incoming: int = 11, outgoing: int = 5, thoughts: int = 2) -> SimpleNamespace:
    """Create a provider-shaped response with explicit, non-fabricated test usage."""
    return SimpleNamespace(text=text, usage_metadata=SimpleNamespace(
        prompt_token_count=incoming, candidates_token_count=outgoing, thoughts_token_count=thoughts))


def test_trace_return_matches_persisted_file(harness: SimpleNamespace, tmp_path: Path) -> None:
    """A completed run returns the exact trace JSON written under its run ID."""
    result = harness.runner.run_agents("SECRET-QUERY")
    trace = result["trace"]
    path = tmp_path / "traces" / f'{trace["run_id"]}.json'
    assert json.loads(path.read_text()) == trace
    assert [step["tool"] for step in trace["steps"]] == ["safety", "manager", "final_answer"]
    expected = {"step", "tool", "input_summary", "output_summary", "tokens_in", "tokens_out", "latency_ms", "status", "error"}
    assert all(set(step) == expected for step in trace["steps"])
    assert all(step["latency_ms"] >= 0 and step["status"] == "ok" for step in trace["steps"])
    assert "SECRET-QUERY" not in path.read_text()
    assert path.stat().st_mode & 0o077 == 0
    assert active_trace.get() is None


def test_blocked_run_is_persisted(harness: SimpleNamespace, tmp_path: Path) -> None:
    """Safety refusals also have run IDs, a blocked step, and a file."""
    harness.llm.side_effect = ["UNSAFE"]
    result = harness.runner.run_agents("query")
    assert result["trace"]["steps"][0]["status"] == "blocked"
    assert len(list((tmp_path / "traces").glob("*.json"))) == 1


def test_loop_failure_trace_excludes_exception_text(harness: SimpleNamespace) -> None:
    """Planner failure remains visible without persisting raw error content."""
    harness.manager._run.side_effect = RuntimeError("SECRET-ERROR")
    trace = harness.runner.run_agents("SECRET-QUERY")["trace"]
    assert trace["steps"][1]["status"] == "error"
    assert trace["steps"][1]["error"] == "OperationFailed"
    assert "SECRET" not in json.dumps(trace)


def test_uncaught_exception_still_persists(harness: SimpleNamespace, tmp_path: Path) -> None:
    """An original safety exception is re-raised after trace persistence."""
    harness.llm.side_effect = RuntimeError("SECRET")
    with pytest.raises(RuntimeError, match="SECRET"):
        harness.runner.run_agents("query")
    path = next((tmp_path / "traces").glob("*.json"))
    assert json.loads(path.read_text())["steps"][0]["status"] == "error"
    assert "SECRET" not in path.read_text()
    assert active_trace.get() is None


def test_disk_failure_does_not_replace_answer(harness: SimpleNamespace, monkeypatch: pytest.MonkeyPatch) -> None:
    """Unavailable trace storage is logged without changing the original answer."""
    monkeypatch.setattr(Trace, "persist", Mock(side_effect=OSError("SECRET")))
    log = Mock()
    monkeypatch.setattr("utils.trace.log_error", log)
    assert harness.runner.run_agents("query")["final_answer"] == "final answer"
    assert log.call_args.args == ("trace_write_failed",)
    assert active_trace.get() is None


def test_traces_are_isolated_between_threads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Concurrent runs have independent IDs, steps, and files."""
    monkeypatch.chdir(tmp_path)

    @traced_run
    def run(value: str) -> dict:
        """Exercise context-local tracing without external dependencies."""
        with operation("search", value):
            return {"final_answer": value}

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(run, ["first", "second"]))
    assert len({result["trace"]["run_id"] for result in results}) == 2
    assert all(len(result["trace"]["steps"]) == 1 for result in results)
    assert all(result["trace"]["steps"][0]["status"] == "ok" for result in results)
    assert len(list((tmp_path / "traces").glob("*.json"))) == 2


def test_formatter_drops_untrusted_messages_and_exceptions() -> None:
    """JSON formatting never emits raw messages, args, extras, or tracebacks."""
    record = logging.LogRecord("test", logging.ERROR, "", 1, "SECRET %s", ("TOKEN",),
                               (RuntimeError, RuntimeError("PASSWORD"), None))
    record.run_id = "SECRET-ID"
    record.prompt = "SECRET-PROMPT"
    record.tokens_in = 12
    encoded = JsonFormatter().format(record)
    assert json.loads(encoded) == {"level": "ERROR", "event": "application_event", "tokens_in": 12}
    assert "SECRET" not in encoded


def test_usage_metadata_is_recorded_once(llm_module: SimpleNamespace) -> None:
    """Successful Gemini calls record prompt, candidate, and thinking tokens."""
    llm_module.client.models.generate_content.return_value = response()
    trace = Trace()
    token = active_trace.set(trace)
    try:
        assert llm_module.generate_text("SECRET") == "answer"
    finally:
        active_trace.reset(token)
    step = trace.steps[0]
    assert (step.tokens_in, step.tokens_out) == (11, 7)
    assert step.tool == "llm:gemini-flash-latest" and step.status == "ok"
    llm_module.client.models.count_tokens.assert_not_called()
    assert "SECRET" not in json.dumps(trace.to_dict())


def test_legacy_client_creation_is_lazy(llm_module: SimpleNamespace, monkeypatch: pytest.MonkeyPatch) -> None:
    """Provider client construction is deferred and then reused across calls."""
    client = Mock()
    factory = Mock(return_value=client)
    monkeypatch.setattr(llm_module.genai, "Client", factory)
    llm_module.client = None
    factory.assert_not_called()
    assert llm_module._get_client() is client
    assert llm_module._get_client() is client
    assert factory.call_count == 1


def test_primary_failure_and_fallback_have_separate_usage(llm_module: SimpleNamespace) -> None:
    """A failed attempt is retained when fallback succeeds."""
    llm_module.client.models.generate_content.side_effect = [RuntimeError("SECRET"), response()]
    llm_module.client.models.count_tokens.return_value.total_tokens = 9
    trace = Trace()
    token = active_trace.set(trace)
    try:
        assert llm_module.generate_text("query") == "answer"
    finally:
        active_trace.reset(token)
    assert [step.status for step in trace.steps] == ["error", "ok"]
    assert trace.steps[0].tokens_in == 9 and trace.steps[0].tokens_out is None
    assert (trace.steps[1].tokens_in, trace.steps[1].tokens_out) == (11, 7)
    assert "SECRET" not in json.dumps(trace.to_dict())


def test_both_models_fail_preserves_empty_string(llm_module: SimpleNamespace) -> None:
    """Two provider failures keep the previous empty-string fallback contract."""
    llm_module.client.models.generate_content.side_effect = RuntimeError("SECRET")
    llm_module.client.models.count_tokens.side_effect = RuntimeError("SECRET")
    trace = Trace()
    token = active_trace.set(trace)
    try:
        assert llm_module.generate_text("query") == ""
    finally:
        active_trace.reset(token)
    assert len(trace.steps) == 2
    assert all(step.status == "error" and step.tokens_in is None for step in trace.steps)


def test_missing_metadata_uses_tokenizer(llm_module: SimpleNamespace) -> None:
    """Absent usage metadata triggers explicit prompt and visible-output counts."""
    llm_module.client.models.generate_content.return_value = SimpleNamespace(text="answer", usage_metadata=None)
    llm_module.client.models.count_tokens.side_effect = [SimpleNamespace(total_tokens=13), SimpleNamespace(total_tokens=3)]
    trace = Trace()
    token = active_trace.set(trace)
    try:
        assert llm_module.generate_text("query") == "answer"
    finally:
        active_trace.reset(token)
    assert (trace.steps[0].tokens_in, trace.steps[0].tokens_out) == (13, 3)


def test_counting_failure_keeps_successful_answer(llm_module: SimpleNamespace) -> None:
    """Telemetry failures produce null usage rather than fabricated counts."""
    llm_module.client.models.generate_content.return_value = SimpleNamespace(text="answer", usage_metadata=None)
    llm_module.client.models.count_tokens.side_effect = RuntimeError("SECRET")
    trace = Trace()
    token = active_trace.set(trace)
    try:
        assert llm_module.generate_text("query") == "answer"
    finally:
        active_trace.reset(token)
    assert trace.steps[0].tokens_in is None and trace.steps[0].tokens_out is None
    assert trace.steps[0].error == "TokenUsageUnavailable"
    assert trace.steps[0].status == "ok"


def test_empty_primary_response_still_falls_back(llm_module: SimpleNamespace) -> None:
    """Empty primary text keeps its usage record before fallback executes."""
    llm_module.client.models.generate_content.side_effect = [response(text="", outgoing=0), response()]
    trace = Trace()
    token = active_trace.set(trace)
    try:
        assert llm_module.generate_text("query") == "answer"
    finally:
        active_trace.reset(token)
    assert [step.status for step in trace.steps] == ["error", "ok"]
    assert trace.steps[0].tokens_out == 2


def test_runner_contains_all_nested_llm_attempts(harness: SimpleNamespace, llm_module: SimpleNamespace,
                                               tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Safety, planning, tool generation, and final usage belong to the same run."""
    llm_module.client.models.generate_content.side_effect = [response("SAFE"), response("Action: search"),
                                                            response("tool result"), response("Action: finish"),
                                                            response("final")]
    monkeypatch.setattr(harness.runner, "generate_text", llm_module.generate_text)
    harness.manager._run.side_effect = lambda *_: llm_module.generate_text("planner")
    harness.agents["search"].return_value.run.side_effect = lambda _: {
        "status": "ok", "results": llm_module.generate_text("tool")}
    result = harness.runner.run_agents("query")
    steps = result["trace"]["steps"]
    calls = [step for step in steps if step["tool"].startswith("llm:")]
    assert len(calls) == 5
    assert sum(step["tokens_in"] for step in calls) == 55
    assert sum(step["tokens_out"] for step in calls) == 35
    assert [step["step"] for step in steps] == list(range(1, len(steps) + 1))
    assert all(step["status"] != "running" for step in steps)
    path = tmp_path / "traces" / f'{result["trace"]["run_id"]}.json'
    assert json.loads(path.read_text()) == result["trace"]


def test_prototype_adapter_records_usage() -> None:
    """The separate backend's direct Gemini generation also records usage."""
    from backend.gemini import GeminiModel

    model = GeminiModel()
    model._client = Mock()
    model._client.models.generate_content.return_value = response()
    trace = Trace()
    token = active_trace.set(trace)
    try:
        assert model.generate("SECRET", 100) == "answer"
    finally:
        active_trace.reset(token)
    assert (trace.steps[0].tokens_in, trace.steps[0].tokens_out) == (11, 7)
    assert "SECRET" not in json.dumps(trace.to_dict())


@pytest.mark.parametrize("name,symbol", [("code", "CodeTool"), ("compare", "CompareTool"),
                                        ("report", "ReportTool"), ("summarize", "SummarizeTool")])
def test_tool_logging_preserves_success_and_fallback(name: str, symbol: str,
                                                    monkeypatch: pytest.MonkeyPatch,
                                                    capsys: pytest.CaptureFixture[str]) -> None:
    """Replacing error prints does not alter each tool's return and fallback paths."""
    import sys
    from types import ModuleType

    base = ModuleType("crewai.tools")
    base.BaseTool = object
    monkeypatch.setitem(sys.modules, "crewai.tools", base)
    llm = ModuleType("utils.llm")
    llm.generate_text = Mock(return_value="primary")
    llm.generate_text_fallback = Mock(return_value="fallback")
    monkeypatch.setitem(sys.modules, "utils.llm", llm)
    path = Path(__file__).resolve().parents[1] / "tools" / f"{name}_tool.py"
    spec = importlib.util.spec_from_file_location(f"observed_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    tool = getattr(module, symbol)()
    value = {} if name == "report" else "SECRET-INPUT"
    assert tool._run(value) == {"status": "ok", "results": "primary"}
    llm.generate_text.side_effect = RuntimeError("SECRET-EXCEPTION")
    assert tool._run(value) == {"status": "ok", "results": "fallback"}
    llm.generate_text_fallback.side_effect = RuntimeError("SECRET-FALLBACK")
    assert tool._run(value) == {"status": "error", "results": ""}
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "SECRET" not in captured.err


def test_search_logging_preserves_retry_and_fallback(monkeypatch: pytest.MonkeyPatch,
                                                   capsys: pytest.CaptureFixture[str]) -> None:
    """Search retains its existing delayed retry and LLM fallback without prints."""
    import sys
    from types import ModuleType

    base = ModuleType("crewai.tools")
    base.BaseTool = object
    ddg = ModuleType("duckduckgo_search")
    ddg.DDGS = Mock(side_effect=RuntimeError("SECRET-SEARCH"))
    llm = ModuleType("utils.llm")
    llm.generate_text = Mock(return_value="SECRET-RESPONSE")
    for name, stub in [("crewai.tools", base), ("duckduckgo_search", ddg), ("utils.llm", llm)]:
        monkeypatch.setitem(sys.modules, name, stub)
    path = Path(__file__).resolve().parents[1] / "tools" / "search_tool.py"
    spec = importlib.util.spec_from_file_location("observed_search", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module.time, "sleep", Mock())
    assert module.SearchTool()._run("SECRET-INPUT") == {"status": "ok", "results": "SECRET-RESPONSE"}
    assert ddg.DDGS.call_count == 2
    module.time.sleep.assert_called_once_with(2)
    llm.generate_text.side_effect = RuntimeError("SECRET-LLM")
    assert module.SearchTool()._run("query") == {"status": "error", "results": ""}
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "SECRET" not in captured.err

"""Per-run, context-local traces with content-free summaries and atomic writes."""

import json
import os
import tempfile
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from functools import wraps
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Iterator
from uuid import uuid4

from utils.logger import log_error, log_info


def summarize(value: Any) -> str:
    """Describe size and type only; never retain user or model content."""
    if isinstance(value, str):
        return f"text(chars={len(value)})"
    if isinstance(value, (list, dict, tuple)):
        return f"collection(items={len(value)})"
    return "none" if value is None else "value"


@dataclass
class TraceStep:
    """One operation; null token counts mean unavailable, never estimated zero."""

    step: int
    tool: str
    input_summary: str
    output_summary: str = "none"
    tokens_in: int | None = None
    tokens_out: int | None = None
    latency_ms: float = 0.0
    status: str = "running"
    error: str | None = None


@dataclass
class Trace:
    """Ordered operations for a single run, including nested LLM attempts."""

    run_id: str = field(default_factory=lambda: uuid4().hex)
    steps: list[TraceStep] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Produce the same JSON-compatible object returned and persisted."""
        return asdict(self)

    def persist(self, directory: Path = Path("traces")) -> None:
        """Atomically write a private trace file; cleanup on write failure."""
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        temporary: str | None = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=directory, delete=False) as output:
                temporary = output.name
                json.dump(self.to_dict(), output, ensure_ascii=False, indent=2)
            os.replace(temporary, directory / f"{self.run_id}.json")
        finally:
            if temporary and os.path.exists(temporary):
                os.unlink(temporary)


active_trace: ContextVar[Trace | None] = ContextVar("active_trace", default=None)


@contextmanager
def operation(tool: str, value: Any) -> Iterator[TraceStep]:
    """Record an operation, its elapsed time, and sanitized failure status."""
    trace = active_trace.get()
    step = TraceStep(step=len(trace.steps) + 1 if trace else 1, tool=tool, input_summary=summarize(value))
    if trace is not None:
        trace.steps.append(step)
    started = perf_counter()
    try:
        yield step
        if step.status == "running":
            step.status = "ok"
    except BaseException:
        step.status, step.error = "error", "OperationFailed"
        raise
    finally:
        step.latency_ms = round((perf_counter() - started) * 1000, 3)
        log_info("step_finished", run_id=trace.run_id if trace else None,
                 step=step.step, tokens_in=step.tokens_in, tokens_out=step.tokens_out,
                 latency_ms=step.latency_ms, status=step.status)


def traced_run(function: Callable[..., dict[str, Any]]) -> Callable[..., dict[str, Any]]:
    """Wrap a runner without changing its exception or answer behavior."""
    @wraps(function)
    def wrapped(*args: Any, **kwargs: Any) -> dict[str, Any]:
        """Attach and persist traces even when execution raises."""
        trace = Trace()
        token = active_trace.set(trace)
        log_info("run_started", run_id=trace.run_id)
        try:
            result = function(*args, **kwargs)
            result["trace"] = trace.to_dict()
            return result
        finally:
            try:
                trace.persist()
            except OSError:
                log_error("trace_write_failed", run_id=trace.run_id)
            finally:
                active_trace.reset(token)
                log_info("run_finished", run_id=trace.run_id)

    return wrapped

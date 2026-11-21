"""Terminating per-attempt deadlines and explicit idempotency-based retries."""

import multiprocessing
from dataclasses import dataclass
from math import isfinite
from time import monotonic
from typing import Any, Callable

from tenacity import Retrying, retry_if_exception_type, stop_after_attempt, wait_fixed

from utils.trace import Trace, active_trace, operation, summarize


class ToolExecutionError(RuntimeError):
    """A permanent or invalid tool failure; never automatically retried."""


class TransientToolError(ToolExecutionError):
    """A transport failure eligible for retry only on an idempotent tool."""


@dataclass(frozen=True)
class ToolSpec:
    """A trusted registered callable and its per-attempt execution policy."""

    name: str
    handler: Callable[[dict[str, Any]], dict[str, Any]]
    timeout_seconds: float = 30.0
    idempotent: bool = False

    def __post_init__(self) -> None:
        """Reject missing, infinite, or nonpositive deadlines."""
        if type(self.idempotent) is not bool or not callable(self.handler):
            raise ValueError("Tool idempotency must be boolean and handler must be callable")
        if not isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("Tool timeout must be finite and positive")


def _worker(connection: Any, handler: Callable[..., Any], arguments: dict[str, Any], run_id: str) -> None:
    """Execute in a fresh process; transmit sanitized trace metadata and results."""
    trace = Trace(run_id=run_id)
    token = active_trace.set(trace)
    try:
        try:
            result = handler(arguments)
            if not isinstance(result, dict) or result.get("status") not in {"ok", "error"}:
                raise ToolExecutionError("Invalid tool result")
            payload = ("ok", result)
        except (ConnectionError, TimeoutError):
            payload = ("transient", None)
        except Exception:
            payload = ("error", None)
        connection.send((payload, trace.steps))
    finally:
        active_trace.reset(token)
        connection.close()


def execute_once(spec: ToolSpec, arguments: dict[str, Any]) -> dict[str, Any]:
    """Terminate the worker before returning or retrying after a deadline."""
    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    trace = active_trace.get()
    process = context.Process(target=_worker, args=(sender, spec.handler, arguments,
                                                  trace.run_id if trace else ""), daemon=True)
    deadline = monotonic() + spec.timeout_seconds
    try:
        process.start()
        sender.close()
        if not receiver.poll(max(0.0, deadline - monotonic())):
            raise TimeoutError("Tool deadline exceeded")
        try:
            (status, result), child_steps = receiver.recv()
        except EOFError:
            raise ToolExecutionError("Tool worker exited without a result") from None
        if trace is not None:
            for child_step in child_steps:
                child_step.step = len(trace.steps) + 1
                trace.steps.append(child_step)
        if status == "transient":
            raise TransientToolError("Tool transport failed")
        if status != "ok":
            raise ToolExecutionError("Tool failed")
        return result
    except (TimeoutError, ToolExecutionError):
        raise
    except Exception:
        raise ToolExecutionError("Tool worker could not execute") from None
    finally:
        if process.pid is not None:
            if process.is_alive():
                process.terminate()
            process.join(timeout=0.5)
            if process.is_alive():
                process.kill()
                process.join(timeout=0.5)
            process.close()
        receiver.close()
        sender.close()


def execute_tool(spec: ToolSpec, arguments: dict[str, Any]) -> dict[str, Any]:
    """Use Tenacity for at most two attempts on explicitly idempotent tools."""
    retrying = Retrying(stop=stop_after_attempt(2 if spec.idempotent else 1),
                        retry=retry_if_exception_type((TimeoutError, TransientToolError)),
                        wait=wait_fixed(0.1), reraise=True)
    for attempt in retrying:
        with attempt:
            with operation(spec.name, arguments) as step:
                result = execute_once(spec, arguments)
                step.output_summary = summarize(result)
                if result["status"] != "ok":
                    step.status, step.error = "error", "ToolReportedFailure"
                return result
    raise ToolExecutionError("Tool attempts exhausted")

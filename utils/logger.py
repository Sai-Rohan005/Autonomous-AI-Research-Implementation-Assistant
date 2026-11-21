"""Content-free JSON logging for application-owned events."""

import json
import logging
import re
from typing import Any

EVENTS = frozenset({"run_started", "run_finished", "step_finished", "trace_write_failed",
                    "agent_execution_failed", "llm_failed", "llm_fallback", "tool_failed",
                    "search_retry", "search_fallback"})


class JsonFormatter(logging.Formatter):
    """Allowlist metadata; never serialize messages, exceptions, or arbitrary extras."""

    def format(self, record: logging.LogRecord) -> str:
        """Return one JSON object without untrusted content."""
        event = record.msg if isinstance(record.msg, str) and record.msg in EVENTS else "application_event"
        payload = {"level": record.levelname, "event": event}
        run_id = getattr(record, "run_id", "")
        if isinstance(run_id, str) and re.fullmatch(r"[0-9a-f]{32}", run_id):
            payload["run_id"] = run_id
        for name in ("step", "tokens_in", "tokens_out", "latency_ms"):
            value = getattr(record, name, None)
            if isinstance(value, (int, float)) and 0 <= value < float("inf"):
                payload[name] = value
        status = getattr(record, "status", None)
        if status in {"ok", "error", "blocked", "running"}:
            payload["status"] = status
        return json.dumps(payload)


logger = logging.getLogger("research_assistant")
logger.setLevel(logging.INFO)
logger.propagate = False
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)


def log_info(event: str, *, run_id: str | None = None, **metadata: Any) -> None:
    """Emit an allowlisted informational event."""
    logger.info(event, extra={"run_id": run_id, **metadata})


def log_error(event: str, *, run_id: str | None = None) -> None:
    """Emit an allowlisted error event without raw exception details."""
    logger.error(event, extra={"run_id": run_id})

"""Shared Gemini usage accounting without recording prompt or response content."""

from typing import Any, Callable

from utils.trace import TraceStep, summarize


def record_usage(response: Any, prompt: str, step: TraceStep,
                 count: Callable[[str], int | None]) -> str:
    """Use provider counts, retokenizing visible text only when metadata is absent."""
    usage = getattr(response, "usage_metadata", None)
    incoming = getattr(usage, "prompt_token_count", None)
    outgoing = getattr(usage, "candidates_token_count", None)
    thoughts = getattr(usage, "thoughts_token_count", None)
    step.tokens_in = incoming if isinstance(incoming, int) and incoming >= 0 else count(prompt)
    text = response.text or ""
    step.tokens_out = outgoing if isinstance(outgoing, int) and outgoing >= 0 else count(text)
    if isinstance(thoughts, int) and thoughts >= 0 and step.tokens_out is not None:
        step.tokens_out += thoughts
    step.output_summary = summarize(text)
    if step.tokens_in is None or step.tokens_out is None:
        step.error = "TokenUsageUnavailable"
    return text

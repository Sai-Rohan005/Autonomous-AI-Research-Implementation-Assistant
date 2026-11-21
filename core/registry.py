"""Trusted tool registry; router output cannot supply executable code."""

from typing import Any

from core.execution import ToolSpec


def web_search(arguments: dict[str, Any]) -> dict[str, Any]:
    """Call the read-only web search adapter."""
    from agents.search_agent import Search_Agent

    return Search_Agent().run(arguments["query"])


def code(arguments: dict[str, Any]) -> dict[str, Any]:
    """Generate text containing code; never execute generated code."""
    from agents.code_agent import Code_Agent

    return Code_Agent().run(arguments["query"])


def compare(arguments: dict[str, Any]) -> dict[str, Any]:
    """Call the comparison prompt adapter."""
    from agents.comparision_agent import Comparison_Agent

    return Comparison_Agent().run(arguments["query"])


def summarize(arguments: dict[str, Any]) -> dict[str, Any]:
    """Summarize the validated explicit text argument."""
    from agents.summarization_agent import Summarization_Agent

    return Summarization_Agent().run(arguments["text"])


def default_tools() -> dict[str, ToolSpec]:
    """Retry only read-only web search; prompt generation defaults to one attempt."""
    return {"web_search": ToolSpec("web_search", web_search, 15, idempotent=True),
            "code": ToolSpec("code", code, 30), "compare": ToolSpec("compare", compare, 30),
            "summarize": ToolSpec("summarize", summarize, 30)}

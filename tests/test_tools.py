"""Plain prompt and web tools return the established result shape offline."""

import sys
from types import ModuleType
from unittest.mock import MagicMock

import pytest

from tools.search_tool import SearchTool
from tools.summarize_tool import SummarizeTool


def test_search_tool(monkeypatch: pytest.MonkeyPatch) -> None:
    """Web results retain real titles, snippets, and URLs."""
    module = ModuleType("duckduckgo_search")
    module.DDGS = MagicMock()
    module.DDGS.return_value.__enter__.return_value.text.return_value = [
        {"title": "Title", "body": "Text", "href": "https://example.com"}]
    monkeypatch.setitem(sys.modules, "duckduckgo_search", module)
    assert SearchTool()._run("query") == {"status": "ok", "results": [
        {"title": "Title", "snippet": "Text", "link": "https://example.com"}]}


def test_summarize_tool(monkeypatch: pytest.MonkeyPatch) -> None:
    """A prompt tool needs no CrewAI base class or network for unit testing."""
    monkeypatch.setattr("tools.summarize_tool.generate_text", lambda _: "summary")
    assert SummarizeTool()._run("text") == {"status": "ok", "results": "summary"}

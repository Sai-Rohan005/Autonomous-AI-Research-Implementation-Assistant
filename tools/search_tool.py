"""Read-only web search; retry policy belongs to the tool executor."""

from typing import Any


class SearchTool:
    """Return real web results without substituting generated text as sources."""

    def _run(self, query: str) -> dict[str, Any]:
        """Perform one search, exposing transport failures to the bounded executor."""
        from duckduckgo_search import DDGS

        try:
            with DDGS() as ddgs:
                rows = list(ddgs.text(f"{query} explanation implementation example", max_results=5))
        except Exception:
            raise ConnectionError("Web search failed") from None
        return {"status": "ok", "results": [
            {"title": row.get("title", ""), "snippet": row.get("body", ""), "link": row.get("href", "")}
            for row in rows]}

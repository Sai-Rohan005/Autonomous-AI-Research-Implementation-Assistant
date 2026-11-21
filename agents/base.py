"""Small agent adapter preserving the existing run and _run interfaces."""

from typing import Any, ClassVar


class PromptAgent:
    """Delegate a single request to one plain Python tool."""

    tool_class: ClassVar[type[Any]]

    def run(self, data: Any) -> dict[str, Any]:
        """Run the wrapped tool once without an orchestration framework."""
        return self.tool_class()._run(data)

    def _run(self, data: Any) -> dict[str, Any]:
        """Retain the tool-style interface used by existing callers."""
        return self.run(data)

"""Plain Python adapter for the search tool."""

from agents.base import PromptAgent
from tools.search_tool import SearchTool


class Search_Agent(PromptAgent):
    """Preserve the public agent interface without CrewAI orchestration."""

    tool_class = SearchTool

"""Plain Python adapter for the compare tool."""

from agents.base import PromptAgent
from tools.compare_tool import CompareTool


class Comparison_Agent(PromptAgent):
    """Preserve the public agent interface without CrewAI orchestration."""

    tool_class = CompareTool

"""Plain Python adapter for the summarize tool."""

from agents.base import PromptAgent
from tools.summarize_tool import SummarizeTool


class Summarization_Agent(PromptAgent):
    """Preserve the public agent interface without CrewAI orchestration."""

    tool_class = SummarizeTool

"""Plain Python adapter for the code tool."""

from agents.base import PromptAgent
from tools.code_tool import CodeTool


class Code_Agent(PromptAgent):
    """Preserve the public agent interface without CrewAI orchestration."""

    tool_class = CodeTool

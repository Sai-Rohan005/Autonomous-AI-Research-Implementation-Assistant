"""Plain Python adapter for the report tool."""

from agents.base import PromptAgent
from tools.report_tool import ReportTool


class Report_Agent(PromptAgent):
    """Preserve the public agent interface without CrewAI orchestration."""

    tool_class = ReportTool

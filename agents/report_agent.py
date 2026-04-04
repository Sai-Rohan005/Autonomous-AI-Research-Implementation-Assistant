from crewai import Agent,Task
# from tools.tool_registry import report_tool_wrapper
from tools.report_tool import ReportTool

Report_Agent = Agent(
    role="Final Report Generator",
    goal=(
        "Combine all outputs including summary, generated code, and comparisons "
        "into a well-structured, clear, and professional report."
    ),
    backstory="A professional report writer who compiles structured, clear, and comprehensive reports from multiple sources.",
    # tools=[report_tool_wrapper]
    tools=[ReportTool()],
    llm="gemini-flash-latest"
)
Report_Task = Task(
    description=(
        "Generate the final report using all available information. "
        "Ensure the output is clear, structured, and easy to understand."
    ),
    agent=Report_Agent,
    expected_output="Final structured report combining all insights"

)

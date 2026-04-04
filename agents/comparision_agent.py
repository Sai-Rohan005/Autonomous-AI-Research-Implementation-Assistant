from crewai import Agent,Task
# from tools.tool_registry import compare_tool_wrapper
from tools.compare_tool import CompareTool

Comparison_Agent = Agent(
    role="Information Comparison Specialist",
    goal=(
        "Compare different techniques, models, or concepts in a structured format. "
        "Highlight pros, cons, and provide a clear summary."
    ),
    backstory="An analytical expert who evaluates different approaches and provides clear comparisons with pros and cons.",
    # tools=[compare_tool_wrapper]
    tools=[CompareTool()],
    llm="gemini-flash-latest"
)

Comparison_Task = Task(
    description=(
        "Compare the given concepts or models. "
        "Return structured output including pros, cons, and a summary."
    ),
    agent=Comparison_Agent,
    expected_output="Structured comparison with pros and cons"

)
from crewai import Agent,Task
# from tools.tool_registry import code_tool_wrapper
from tools.code_tool import CodeTool




Code_Agent = Agent(
    role="Code Generation Specialist",
    goal=(
        "Generate clean, readable, and efficient code implementations based on the query"
    ),
    backstory="A senior software engineer who writes clean, efficient, and well-documented code following best practices.",
    # tools=[code_tool_wrapper]
    tools=[CodeTool()],
    llm="gemini-flash-latest"
)

Code_Task = Task(
    description=(
        "Generate high-quality code for the given query. "
        "Ensure the code is well-structured, commented, and easy to understand."
    ),
    agent=Code_Agent,
    expected_output="Well-structured, readable code with comments"

)

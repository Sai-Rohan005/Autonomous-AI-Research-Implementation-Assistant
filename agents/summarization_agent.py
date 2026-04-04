from crewai import Agent, Task
from tools.summarize_tool import SummarizeTool

Summarization_Agent = Agent(
    role="Information Summarization Specialist",
    goal="Convert information into concise explanation",
    backstory="Expert in simplifying complex information",
    tools=[SummarizeTool()], 
    llm="gemini-flash-latest"
)

Summarization_Task = Task(
    description=(
        "Summarize the retrieved information into a simple and precise explanation. "
        "Focus on clarity and key concepts."
    ),
    agent=Summarization_Agent,
    expected_output="A concise and clear summary of the input text"

)
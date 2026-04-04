from crewai import Agent, Task
from tools.search_tool import SearchTool

Search_Agent = Agent(
    role="Information Retrieval Specialist",
    goal="Search and retrieve relevant information",
    backstory="Expert in retrieving accurate information",
    tools=[SearchTool()],
    llm="gemini-flash-latest"
)

Search_Task = Task(
    description="Search for relevant information based on the query",
    agent=Search_Agent,
    expected_output="A list of relevant search results with titles and summaries"

)
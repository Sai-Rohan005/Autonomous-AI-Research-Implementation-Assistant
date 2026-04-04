from crewai.tools import BaseTool
from utils.llm import generate_text
import json

class ManagerTool(BaseTool):
    name: str = "Manager Tool"
    description: str = "Decides which agents to call based on user query"

    def _run(self, query: str):
        prompt = f"""
You are an intelligent AI manager.

Your job is to decide which agents are required to answer the query.

Available agents:
- search → for fetching information
- summarize → for summarizing
- code → for generating code
- compare → for comparisons
- report → for final formatting

Rules:
- Return ONLY a JSON list
- Do NOT explain anything

Examples:
Query: "gold price today"
Output: ["search"]

Query: "compare transformers and rnn"
Output: ["search", "summarize", "compare"]

Query: "write quicksort code"
Output: ["code"]

Now decide for:
Query: {query}
"""

        response = generate_text(prompt)

        try:
            return json.loads(response)
        except:
            return ["search"]  # fallback
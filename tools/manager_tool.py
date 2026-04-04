from crewai.tools import BaseTool
from utils.llm import generate_text
import json

class ManagerTool(BaseTool):
    name: str = "Manager Tool"
    description: str = "Decides which agents to call based on user query"

    def _run(self, query: str):
        prompt = f"""
You are a precise AI workflow router. Your job is to decide the minimal set of agents needed. 

When the query is clear and the main LLM can handle it well using its built-in knowledge, return []. 
When the query is unclear, ambiguous, confusing, or the router cannot confidently understand what the user wants, return ["search"] as a safe fallback.

### Available Agents and When to Use Them:

- **search**    → Use for:
  - Any real-time, current, or up-to-date information (prices, news, weather, rankings, etc.)
  - Queries that require external or latest facts
  - When the query is ambiguous, unclear, or you cannot confidently interpret what the user is asking → use this as fallback

- **summarize** → Use ONLY if the user explicitly asks to summarize, condense, or make a summary of some content.

- **code**      → Use ONLY if the user asks to write, generate, implement, debug, or show code.

- **compare**   → Use ONLY if the query explicitly asks to compare two or more things (contains "compare", "vs", "versus", "difference between", "which is better", etc.).
                 Never use for explaining a single concept.

- **report**    → Use ONLY if the user asks for a "report", "detailed report", "structured report", or final polished output combining multiple steps.

### Strict Decision Rules:
- If the query is a clear, single-topic explanation ("Explain...", "What is...", "Tell me about...") → return [] 
- If the query is ambiguous, vague, poorly worded, or you are unsure what the user wants → return ["search"]
- Be minimal when the intent is obvious. Use "search" as fallback when interpretation fails.
- Return agents in logical order.

### Examples:

Query: "Compare Transformer and RNN"
Output: ["compare"]

Query: "What is the current gold price in Hyderabad?"
Output: ["search"]

Query: "Write quicksort code in Python"
Output: ["code"]

Query: "blablabla what is this thing??"
Output: ["search"]     # ambiguous → fallback to search

Query: "latest news about AI"
Output: ["search"]

Query: "Compare RNN and LSTM and make a report"
Output: ["compare", "report"]

Query: "idk what you mean by this"
Output: ["search"]

Now analyze the query below. Return **ONLY** a valid JSON array. No explanations, no extra text.

Query: {query}
"""
        response = generate_text(prompt)

        try:
            return json.loads(response)
        except:
            return ["search"]  # fallback
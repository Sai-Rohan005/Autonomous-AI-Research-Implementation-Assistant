from crewai.tools import BaseTool
from utils.llm import generate_text

class ManagerTool(BaseTool):
    name: str = "Manager Tool"
    description: str = "Decides next action using ReAct reasoning"

    def _run(self, context: str):
        """
        context = {
            "query": "...",
            "history": [...],
            "result": {...}
        }
        """

        prompt = f"""
You are an intelligent AI agent using ReAct reasoning.

Your task is to decide the NEXT BEST ACTION step-by-step.

⚠️ IMPORTANT:
- You MUST choose ONLY ONE action at a time
- You are NOT allowed to plan multiple steps
- You must act based on the USER INTENT

---

### 🔍 First: Understand the USER INTENT

Classify the query into ONE of these:

1. EXPLANATION → (e.g., "What is...", "Explain...", "Tell me about...")
2. REAL-TIME / FACTUAL → (e.g., price, news, current data)
3. CODE REQUEST → (e.g., "write code", "implement", "program")
4. COMPARISON → (e.g., "compare", "difference between", "vs")
5. SUMMARIZATION → (e.g., "summarize this text")
6. REPORT → (e.g., "generate report")

---

### 🎯 Action Mapping Rules

- If EXPLANATION → finish  ✅ (LLM can answer directly)
- If REAL-TIME → search
- If CODE REQUEST → code
- If COMPARISON → compare
- If SUMMARIZATION → summarize
- If REPORT → report

---

### 🚫 STRICT RESTRICTIONS

- DO NOT use "code" unless user explicitly asks for code
- DO NOT use "search" for general knowledge (LLM already knows)
- DO NOT overuse tools
- DO NOT repeat actions
- If enough information is available → finish

---

### 🧠 Context:
{context}

---

### ✅ Output Format:
Return ONLY ONE word from:
search / summarize / code / compare / report / finish

NO explanation.
"""

        response = generate_text(prompt)

        # Clean output
        action = response.strip().lower()

        valid_actions = ["search", "summarize", "code", "compare", "report", "finish"]

        if action in valid_actions:
            return action

        return "finish"  # safe fallback
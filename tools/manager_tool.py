from crewai.tools import BaseTool
from utils.llm import generate_text
import json
import re

class ManagerTool(BaseTool):
    name: str = "Manager Tool"
    description: str = "Decides next action using ReAct reasoning"

    def _run(self, context: dict, last_obs: str):

        prompt_context = json.dumps(context, indent=2)

        prompt = f"""
        You are an advanced AI agent using ReAct reasoning.

        Your task is to decide the NEXT BEST ACTION step-by-step.

        You MUST:
        - Think carefully before acting
        - Use previous observations
        - Choose ONLY ONE action at a time

        ---

        ## 🧠 Step 1: Understand USER INTENT

        Classify the query into ONE category:

        1. EXPLANATION → general knowledge ("What is...", "Explain...")
        2. REAL-TIME → current info (price, news, live data)
        3. CODE → programming tasks ("write code", "implement")
        4. COMPARISON → ("compare", "difference", "vs")
        5. SUMMARIZATION → ("summarize this")
        6. REPORT → ("generate report")

        ---

        ## 🎯 Step 2: Decide Action

        Follow STRICT rules:

        - EXPLANATION → finish (LLM can answer directly)
        - REAL-TIME → search
        - CODE → code
        - COMPARISON → compare
        - SUMMARIZATION → summarize
        - REPORT → report

        ---

        ## ⚠️ CRITICAL RULES

        - DO NOT use "search" for basic knowledge
        - DO NOT use "code" unless explicitly asked
        - DO NOT repeat the same action
        - If enough information is already available → finish
        - If previous step failed → try a different action or finish
        - Be minimal (avoid unnecessary steps)

        ---

        ## 🧠 Context:
        {prompt_context}

        ---

        ## 📊 Previous Observation:
        {last_obs}

        ---

        ## ✅ Output Format (STRICT)

        Thought: <short reasoning>
        Action: <one action>

        ---

        ## Allowed Actions ONLY:
        search / summarize / code / compare / report / finish

        NO explanation. ONLY the format above.
        """

        response = generate_text(prompt).strip()

        return response
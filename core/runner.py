from tools.manager_tool import ManagerTool
from tools.search_tool import SearchTool
from tools.summarize_tool import SummarizeTool
from tools.code_tool import CodeTool
from tools.compare_tool import CompareTool
from tools.report_tool import ReportTool
from utils.llm import generate_text
import re
def run_agents(query):

    manager = ManagerTool()

    context = {
        "query": query,
        "history": [],
        "result": {},
        "status": "ok"
    }

    max_steps = 5

    for step in range(max_steps):

        try:
            last_obs = context["history"][-1] if context["history"] else None
            context["step"] = step

            # -------- THINK --------
            response = manager._run(context, last_obs)

            match = re.search(r"action\s*:\s*(\w+)", response, re.IGNORECASE)
            action = match.group(1).lower() if match else "finish"

            thought_match = re.search(r"thought\s*:\s*(.*)", response, re.IGNORECASE)
            thought = thought_match.group(1) if thought_match else ""

            print(f"\nStep {step+1} Thought:", thought)

            # -------- STOP --------
            if action == "finish":
                break

            if len(context["history"]) > 0 and action == context["history"][-1]["action"]:
                break

            # -------- ACTION --------
            if action == "search":
                observation = SearchTool()._run(query)
                context["result"]["search"] = observation

            elif action == "summarize":
                observation = SummarizeTool()._run(str(context["result"]))
                context["result"]["summary"] = observation

            elif action == "code":
                observation = CodeTool()._run(query)
                context["result"]["code"] = observation

            elif action == "compare":
                observation = CompareTool()._run(query)
                context["result"]["comparison"] = observation

            elif action == "report":
                observation = ReportTool()._run(context["result"])
                context["result"]["report"] = observation

            else:
                break

            # -------- OBSERVATION --------
            if observation["status"] != "ok":
                context["reflection"] = "Previous step failed"

            context["history"].append({
                "step": step,
                "action": action,
                "observation": str(observation.get("results", ""))[:300],
                "status": observation["status"]
            })

            # -------- LIMIT HISTORY --------
            if len(context["history"]) > 3:
                context["history"] = context["history"][-3:]
            # print(context)

        except Exception as e:
            context["status"] = "error"
            context["error"] = str(e)
            break

    # -------- FINAL ANSWER --------
    final_answer = generate_text(f"""
Answer the user query clearly.

Use:
- Summary if available
- Code if relevant
- Comparison if useful

Be structured and clear.

Context:
{context}
""")

    return {
    "final_answer": final_answer,
    "history": context["history"],
    "summary": context["result"].get("summary", None),
    "code": context["result"].get("code", None),
    "comparison": context["result"].get("comparison", None),
    "report": context["result"].get("report", None),
    "search": context["result"].get("search", None)
}
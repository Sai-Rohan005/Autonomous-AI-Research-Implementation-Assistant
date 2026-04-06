from tools.manager_tool import ManagerTool
from tools.search_tool import SearchTool
from tools.summarize_tool import SummarizeTool
from tools.code_tool import CodeTool
from tools.compare_tool import CompareTool
from tools.report_tool import ReportTool

def run_agents(query):

    manager = ManagerTool()

    context = {
        "query": query,
        "history": [],
        "result": {},
        "status":"ok"
    }

    max_steps = 5

    for step in range(max_steps):
        last_obs = context["history"][-1] if context["history"] else None

        # -------- THINK --------
        thought = manager._run(str(context),str(last_obs))

        print(f"\nStep {step+1} Thought:", thought)

        # -------- STOP CONDITION --------
        if "finish" in thought:
            break
        if len(context["history"]) > 0 and thought == context["history"][-1]["action"]:
            break

        # -------- ACTION --------
        action = None

        if "search" in thought:
            observation = SearchTool()._run(query)
            context["result"]["search"] = observation
            action = "search"

        elif "summarize" in thought:
            observation = SummarizeTool()._run(str(context["result"]))
            context["result"]["summary"] = observation
            action = "summarize"

        elif "code" in thought:
            observation = CodeTool()._run(query)
            context["result"]["code"] = observation
            action = "code"

        elif "compare" in thought:
            observation = CompareTool()._run(query)
            context["result"]["comparison"] = observation
            action = "compare"

        elif "report" in thought:
            observation = ReportTool()._run(context["result"])
            context["result"]["report"] = observation
            action = "report"

        else:
            print("No valid action → stopping")
            break

        # -------- OBSERVATION --------
        context["history"].append({
            "step": step,
            "action": action,
            "observation": observation,
            "status":observation["status"]
        })

    return context["result"]
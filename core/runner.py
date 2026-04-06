from tools.manager_tool import ManagerTool
from tools.search_tool import SearchTool
from tools.summarize_tool import SummarizeTool
from tools.code_tool import CodeTool
from tools.compare_tool import CompareTool
from tools.report_tool import ReportTool
from utils.llm import generate_text

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
        response = manager._run(str(context),str(last_obs))
        thought = response.split("Thought:")[1].split("Action:")[0].strip()
        action = response.split("Action:")[1].strip()   

        print(f"\nStep {step+1} Thought:", thought)

        # -------- STOP CONDITION --------
        if "enough information" in thought.lower():
            break
        if "finish" in action:
            break
        if len(context["history"]) > 0 and action == context["history"][-1]["action"]:
            break


        if "search" in action:
            observation = SearchTool()._run(query)
            context["result"]["search"] = observation
            action = "search"

        elif "summarize" in action:
            observation = SummarizeTool()._run(str(context["result"]))
            context["result"]["summary"] = observation
            action = "summarize"

        elif "code" in action:
            observation = CodeTool()._run(query)
            context["result"]["code"] = observation
            action = "code"

        elif "compare" in action:
            observation = CompareTool()._run(query)
            context["result"]["comparison"] = observation
            action = "compare"

        elif "report" in action:
            observation = ReportTool()._run(context["result"])
            context["result"]["report"] = observation
            action = "report"

        else:
            print("No valid action → stopping")
            break

        # -------- OBSERVATION --------
        if observation.results is None or not observation.results:
            context["reflection"] = "Previous step failed"
        context["history"].append({
            "step": step,
            "action": action,
            "observation": observation,
            "status":observation["status"]
        })
        if len(context["history"]) > 1:
            if context["history"][-1]["action"] == context["history"][-2]["action"]:
                break

    return  generate_text(f"Answer the user query using this context:\n{context}")
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
        "status":"ok"
    }

    max_steps = 5

    for step in range(max_steps):
        
        try:
        
            last_obs = context["history"][-1] if context["history"] else None
            context["step"] = step
            # -------- THINK --------
            response = manager._run(context,last_obs)
            match = re.search(r"action\s*:\s*(\w+)", response, re.IGNORECASE)
            action = match.group(1).lower() if match else "finish"

            thought_match = re.search(r"thought\s*:\s*(.*)", response, re.IGNORECASE)
            thought = thought_match.group(1) if thought_match else ""

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

            elif "summarize" in action:
                observation = SummarizeTool()._run(str(context["result"]))
                context["result"]["summary"] = observation


            elif "code" in action:
                observation = CodeTool()._run(query)
                context["result"]["code"] = observation


            elif "compare" in action:
                observation = CompareTool()._run(query)
                context["result"]["comparison"] = observation

            elif "report" in action:
                observation = ReportTool()._run(context["result"])
                context["result"]["report"] = observation

            else:
                print("No valid action → stopping")
                break

            # -------- OBSERVATION --------
            if observation["status"] != "ok":
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
            if len(context["history"]) > 3:
                context["history"] = context["history"][-3:]
        except Exception as e:
            print("Error in ReAct Loop: ",e)

    return  generate_text(f"Answer the user query using this context:\n{context}")
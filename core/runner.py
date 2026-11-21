from tools.manager_tool import ManagerTool
from agents.code_agent import Code_Agent
from agents.comparision_agent import Comparison_Agent
from agents.report_agent import Report_Agent
from agents.search_agent import Search_Agent
from agents.summarization_agent import Summarization_Agent
from utils.logger import log_error
from utils.llm import generate_text
from utils.trace import operation, summarize, traced_run
from typing import Any
import re

def safety_check(query: str | None) -> bool:
    """Preserve the existing LLM SAFE/UNSAFE verdict interpretation."""
    prompt = f"""
    You are a strict AI safety filter.

    Check if the following query is SAFE.

    Block if it involves:
    - illegal activities
    - hacking
    - violence
    - harmful instructions
    - unethical behavior

    Query: {query}

    Return ONLY:
    SAFE or UNSAFE
    """

    result = generate_text(prompt).strip().upper()

    return result == "SAFE"



def _run_tool(agent: Any, tool: str, value: Any) -> Any:
    """Trace an agent invocation without changing its arguments or result."""
    with operation(tool, value) as step:
        result = agent().run(value)
        step.output_summary = summarize(result)
        if isinstance(result, dict) and result.get("status") != "ok":
            step.status, step.error = "error", "ToolReportedFailure"
        return result


@traced_run
def run_agents(query: str | None) -> dict[str, Any]:
    """Run the legacy decision loop, returning its answer plus a persisted trace."""

    with operation("safety", query) as safety_step:
        safe = safety_check(query)
        safety_step.status = "ok" if safe else "blocked"
        safety_step.output_summary = "safe" if safe else "blocked"
    if not safe:
        return {
            "final_answer": "❌ This query cannot be processed due to safety policies.",
            "history": [],
        }

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
            with operation("manager", context) as manager_step:
                response = manager._run(context, last_obs)
                manager_step.output_summary = summarize(response)

            match = re.search(r"action\s*:\s*(\w+)", response, re.IGNORECASE)
            action = match.group(1).lower() if match else "finish"

            # -------- STOP --------
            if action == "finish":
                break

            if len(context["history"]) > 0 and action == context["history"][-1]["action"]:
                break

            # -------- ACTION --------
            if action == "search":
                observation = _run_tool(Search_Agent, "search", query)
                context["result"]["search"] = observation

            elif action == "summarize":
                observation = _run_tool(Summarization_Agent, "summarize", str(context["result"]))
                context["result"]["summary"] = observation

            elif action == "code":
                observation = _run_tool(Code_Agent, "code", query)
                context["result"]["code"] = observation

            elif action == "compare":
                observation = _run_tool(Comparison_Agent, "compare", query)
                context["result"]["comparison"] = observation

            elif action == "report":
                observation = _run_tool(Report_Agent, "report", context["result"])
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

        except Exception as e:
            log_error("agent_execution_failed")
            context["status"] = "error"
            context["error_details"] = str(e)
            break

    # -------- FINAL ANSWER --------
    with operation("final_answer", context) as final_step:
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
        final_step.output_summary = summarize(final_answer)

    return {
    "final_answer": final_answer,
    "history": context["history"],
    "summary": context["result"].get("summary", None),
    "code": context["result"].get("code", None),
    "comparison": context["result"].get("comparison", None),
    "report": context["result"].get("report", None),
    "search": context["result"].get("search", None)
}

"""Bounded structured routing over an explicit, trusted tool registry."""

import json
from collections.abc import Mapping
from typing import Any

from core.execution import ToolExecutionError, ToolSpec, execute_tool
from core.models import ToolCall
from core.registry import default_tools
from tools.manager_tool import ManagerTool, clarification
from utils.llm import generate_text
from utils.logger import log_error
from utils.trace import operation, summarize, traced_run


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
    return generate_text(prompt).strip().upper() == "SAFE"


def _response(answer: str, history: list[dict[str, Any]], results: dict[str, Any],
              reason: str) -> dict[str, Any]:
    """Keep the UI response fields while exposing an explicit stopping reason."""
    return {"final_answer": answer, "history": history, "stop_reason": reason,
            **{key: results.get(key) for key in ("summary", "code", "comparison", "report", "search", "documents")}}


@traced_run
def run_agents(query: str | None, *, max_steps: int = 5,
               tools: Mapping[str, ToolSpec] | None = None) -> dict[str, Any]:
    """Route validated calls, stop repeated calls, and enforce bounded tool policies."""
    if type(max_steps) is not int or not 1 <= max_steps <= 100:
        raise ValueError("max_steps must be an integer between 1 and 100")
    registry = dict(default_tools() if tools is None else tools)
    if any(name != spec.name or name not in {"search_document", "web_search", "code", "compare", "summarize"}
           for name, spec in registry.items()):
        raise ValueError("Tool registry contains an invalid name")
    with operation("safety", query) as step:
        safe = safety_check(query)
        step.status, step.output_summary = ("ok", "safe") if safe else ("blocked", "blocked")
    if not safe:
        return _response("❌ This query cannot be processed due to safety policies.", [], {}, "blocked")

    manager = ManagerTool()
    context: dict[str, Any] = {"query": query, "history": [], "result": {}, "status": "ok",
                               "available_tools": sorted(registry)}
    seen: set[str] = set()
    keys = {"search_document": "documents", "web_search": "search", "code": "code",
            "compare": "comparison", "summarize": "summary"}
    for index in range(max_steps):
        context["step"] = index
        previous = context["history"][-1] if context["history"] else None
        with operation("manager", context) as step:
            try:
                call = ToolCall.model_validate(manager._run(context, previous))
                step.output_summary = "validated tool call"
            except Exception:
                log_error("agent_execution_failed")
                step.status, step.error = "error", "RouterUnavailable"
                call = clarification()
        if call.tool == "clarify":
            return _response(call.arguments["question"], context["history"], context["result"], "clarify")
        if call.tool == "final_answer":
            answer = call.arguments["answer"]
            if not answer:
                with operation("final_answer", context) as step:
                    answer = generate_text(f"Answer the user query using the available results. Context:\n{context}")
                    step.output_summary = summarize(answer)
            return _response(answer, context["history"], context["result"], "final_answer")
        if call.tool not in registry:
            return _response("Please select documents and configure document search before using this tool.",
                             context["history"], context["result"], "clarify")
        identity = json.dumps([call.tool, call.arguments], sort_keys=True, ensure_ascii=False, allow_nan=False)
        if identity in seen:
            return _response("I stopped because the same tool and arguments were requested again.",
                             context["history"], context["result"], "loop_detected")
        seen.add(identity)
        try:
            observation = execute_tool(registry[call.tool], call.arguments)
        except TimeoutError:
            return _response("The tool exceeded its time limit. Please narrow the request or try again.",
                             context["history"], context["result"], "timeout")
        except ToolExecutionError:
            return _response("The tool could not complete the request.", context["history"], context["result"], "tool_error")
        context["result"][keys[call.tool]] = observation
        if observation["status"] != "ok":
            context["reflection"] = "Previous step failed"
        context["history"].append({"step": index, "action": call.tool,
                                   "observation": str(observation.get("results", ""))[:300],
                                   "status": observation["status"]})
        context["history"] = context["history"][-3:]
    return _response("I reached the step limit before completing the request.", context["history"],
                     context["result"], "max_steps")

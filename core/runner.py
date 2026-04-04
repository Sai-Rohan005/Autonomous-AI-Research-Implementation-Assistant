from tools.manager_tool import ManagerTool
from tools.search_tool import SearchTool
from tools.summarize_tool import SummarizeTool
from tools.code_tool import CodeTool
from tools.compare_tool import CompareTool
from tools.report_tool import ReportTool


def run_agents(query):

    manager = ManagerTool()
    flow = manager._run(query)
    print(flow)
    result = {}

    if "search" in flow:
        result["search"] = SearchTool()._run(query)

    if "summarize" in flow:
        result["summary"] = SummarizeTool()._run(str(result))

    if "code" in flow:
        result["code"] = CodeTool()._run(query)

    if "compare" in flow:
        result["comparison"] = CompareTool()._run(query)

    if "report" in flow:
        result["report"] = ReportTool()._run(result)

    return result
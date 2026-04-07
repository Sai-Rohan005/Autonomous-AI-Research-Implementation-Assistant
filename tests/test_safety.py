from tools.search_tool import SearchTool
from tools.summarize_tool import SummarizeTool

def test_search_tool():
    tool = SearchTool()
    result = tool._run("AI trends")

    assert isinstance(result, list) or isinstance(result, dict)


def test_summarize_tool():
    tool = SummarizeTool()
    result = tool._run("Artificial Intelligence is a field...")

    assert result["status"] == "ok"
    assert "results" in result
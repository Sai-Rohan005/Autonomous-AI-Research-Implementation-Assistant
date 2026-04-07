from agents.search_agent import Search_Agent
from agents.code_agent import Code_Agent
from agents.comparision_agent import Comparison_Agent
from agents.summarization_agent import Summarization_Agent

def test_search_agent(sample_query):
    agent = Search_Agent()
    result = agent._run(sample_query)

    assert isinstance(result, dict)
    assert "status" in result
    assert "results" in result


def test_code_agent():
    agent = Code_Agent()
    result = agent._run("write quicksort in python")

    assert result["status"] == "ok"
    assert len(result["results"]) > 0


def test_compare_agent(sample_query):
    agent = Comparison_Agent()
    result = agent._run(sample_query)

    assert result["status"] == "ok"
    assert "results" in result


def test_summarization_agent():
    agent = Summarization_Agent()
    data = "Transformers are deep learning models..."

    result = agent._run(data)

    assert result["status"] == "ok"
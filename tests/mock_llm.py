# tests/mock_llm.py
from unittest.mock import patch

def mock_generate_text(prompt):
    return "Mock response"


@patch("utils.llm.generate_text", return_value="Mock response")
def test_runner_mocked(mock_llm):
    from core.runner import run_agents

    result = run_agents("Explain AI")

    assert result["final_answer"] == "Mock response"
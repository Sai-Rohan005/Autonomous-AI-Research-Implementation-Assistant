"""Plain agent interfaces delegate to their single tool without live services."""

from unittest.mock import Mock

import pytest

from agents.code_agent import Code_Agent
from agents.comparision_agent import Comparison_Agent
from agents.report_agent import Report_Agent
from agents.search_agent import Search_Agent
from agents.summarization_agent import Summarization_Agent


@pytest.mark.parametrize("agent_type", [Code_Agent, Comparison_Agent, Report_Agent, Search_Agent, Summarization_Agent])
def test_plain_agent_interfaces(agent_type: type, monkeypatch: pytest.MonkeyPatch) -> None:
    """Both interfaces pass data unchanged and return the underlying tool result."""
    result = {"status": "ok", "results": "answer"}
    execute = Mock(return_value=result)
    monkeypatch.setattr(agent_type.tool_class, "_run", execute)
    agent = agent_type()
    assert agent.run("data") is result
    assert agent._run("data") is result
    assert execute.call_count == 2
    execute.assert_called_with("data")

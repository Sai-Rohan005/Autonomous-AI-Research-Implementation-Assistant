from core.runner import run_agents
def test_empty_query():
    result = run_agents("")

    assert result is not None


def test_invalid_input():
    result = run_agents(None)

    assert result is not None
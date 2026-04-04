from core.runner import run_agents

def test_basic_query():
    result = run_agents("Explain YOLOv8")

    assert result is not None
    assert "summary" in result
    assert "code" in result

def test_empty_query():
    result = run_agents("")
    assert result is not None
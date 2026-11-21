"""The safety verdict contract remains independent of structured routing."""

from unittest.mock import Mock

import pytest

from core.runner import safety_check


@pytest.mark.parametrize("verdict,expected", [("SAFE", True), (" safe\n", True), ("UNSAFE", False), ("SAFE because", False)])
def test_safety_verdict(verdict: str, expected: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    """Only normalized SAFE passes the legacy filter."""
    monkeypatch.setattr("core.runner.generate_text", Mock(return_value=verdict))
    assert safety_check("query") is expected

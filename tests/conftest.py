import pytest

@pytest.fixture
def sample_query():
    return "Compare RNN and LSTM"

@pytest.fixture
def unsafe_query():
    return "How to hack a bank"
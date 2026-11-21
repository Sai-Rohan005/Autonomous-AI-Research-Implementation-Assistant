"""Gemini calls with per-attempt usage accounting and content-free logging."""

import os
from typing import Any

from dotenv import load_dotenv
from google import genai

from utils.logger import log_error, log_info
from utils.trace import operation
from utils.token_usage import record_usage

load_dotenv()
client: Any = None


def _get_client() -> Any:
    """Initialize lazily so offline imports do not require an API key."""
    global client
    if client is None:
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    return client


def _count(model: str, text: str) -> int | None:
    """Count through Gemini when usage metadata is unavailable; never guess."""
    try:
        count = _get_client().models.count_tokens(model=model, contents=text).total_tokens
        return count if isinstance(count, int) and count >= 0 else None
    except Exception:
        return None


def _generate(prompt: str, model: str) -> str:
    """Record every generation attempt, including failures and empty responses."""
    with operation(f"llm:{model}", prompt) as step:
        try:
            response = _get_client().models.generate_content(model=model, contents=prompt)
        except Exception:
            step.tokens_in = _count(model, prompt)
            raise
        text = record_usage(response, prompt, step, lambda text: _count(model, text))
        if not text:
            raise RuntimeError("Empty model response")
        return text


def generate_text(prompt: str) -> str:
    """Preserve the existing primary-then-fallback text return contract."""
    try:
        return _generate(prompt, "gemini-flash-latest")
    except Exception:
        log_error("llm_failed")
        return generate_text_fallback(prompt)


def generate_text_fallback(prompt: str) -> str:
    """Preserve the existing fallback behavior, returning empty text on failure."""
    log_info("llm_fallback")
    try:
        return _generate(prompt, "gemini-2.5-flash-lite")
    except Exception:
        log_error("llm_failed")
        return ""

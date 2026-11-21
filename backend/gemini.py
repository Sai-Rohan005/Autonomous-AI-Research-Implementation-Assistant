"""Lazy Gemini adapter; importing the backend never needs credentials."""

import os
from typing import Any

class GeminiModel:
    """Use the same provider model for token counting and generation."""

    def __init__(self) -> None:
        """Read configuration without creating a network client."""
        self.model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        self._client: Any = None

    def _get_client(self) -> Any:
        """Initialize only on first use; never print provider exceptions."""
        if self._client is None:
            from google import genai

            key = os.getenv("GEMINI_API_KEY")
            if not key:
                raise RuntimeError("GEMINI_API_KEY is not configured")
            self._client = genai.Client(api_key=key, http_options={"timeout": 30_000})
        return self._client

    def count_tokens(self, prompt: str) -> int:
        """Ask the selected model's tokenizer for the entire input count."""
        result = self._get_client().models.count_tokens(model=self.model, contents=prompt)
        if result.total_tokens is None:
            raise RuntimeError("Provider did not return a token count")
        return result.total_tokens

    def generate(self, prompt: str, max_output_tokens: int) -> str:
        """Request JSON within the output cap; the counted prompt carries schema."""
        result = self._get_client().models.generate_content(
            model=self.model, contents=prompt,
            config={"temperature": 0, "max_output_tokens": max_output_tokens,
                    "response_mime_type": "application/json"},
        )
        if not result.text:
            raise RuntimeError("Provider returned no text")
        return result.text

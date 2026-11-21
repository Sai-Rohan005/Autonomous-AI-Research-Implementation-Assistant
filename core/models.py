"""Validated router output and tool-specific argument contracts."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator


class Arguments(BaseModel):
    """Reject unknown argument names and implicit type coercion."""

    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)


class QueryArguments(Arguments):
    """Arguments for query-based tools."""

    query: str = Field(min_length=1, max_length=4000)


class DocumentArguments(QueryArguments):
    """Search only the explicitly selected documents."""

    document_ids: list[str] = Field(min_length=1, max_length=20)


class TextArguments(Arguments):
    """Arguments for summarizing explicit source text."""

    text: str = Field(min_length=1, max_length=20000)


class ClarifyArguments(Arguments):
    """A question to return directly to the user."""

    question: str = Field(min_length=1, max_length=2000)


class FinalArguments(Arguments):
    """An optional final answer; empty arguments request final synthesis."""

    answer: str = Field(default="", max_length=20000)


ARGUMENT_MODELS = {"search_document": DocumentArguments, "web_search": QueryArguments,
                   "code": QueryArguments, "compare": QueryArguments, "summarize": TextArguments,
                   "clarify": ClarifyArguments, "final_answer": FinalArguments}


class ToolCall(Arguments):
    """The sole accepted structured routing decision."""

    tool: Literal["search_document", "web_search", "code", "compare", "summarize", "clarify", "final_answer"]
    arguments: dict[str, Any] = Field(json_schema_extra={
        "additionalProperties": False,
        "properties": {"query": {"type": "string"}, "text": {"type": "string"},
                       "question": {"type": "string"}, "answer": {"type": "string"},
                       "document_ids": {"type": "array", "items": {"type": "string"}}},
    })
    rationale: str = Field(max_length=2000)

    @model_validator(mode="after")
    def validate_arguments(self) -> "ToolCall":
        """Validate and normalize arguments according to the selected tool."""
        self.arguments = ARGUMENT_MODELS[self.tool].model_validate(self.arguments).model_dump()
        return self


def validation_feedback(error: ValidationError) -> str:
    """Explain validation errors without echoing raw model content or secrets."""
    fields = {"tool", "arguments", "rationale", "query", "text", "document_ids", "question", "answer"}
    return "; ".join(
        f"{'.'.join(str(part) if part in fields else 'field' for part in item['loc']) or 'router'}: {item['type']}"
        for item in error.errors(include_input=False))

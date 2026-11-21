"""Strict request, tool, evidence, and response contracts."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

Nonempty = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class StrictModel(BaseModel):
    """Reject unknown fields rather than silently accepting tool arguments."""

    model_config = ConfigDict(extra="forbid", strict=True)


class Question(StrictModel):
    """Ask only within an explicit set of uploaded documents."""

    question: Annotated[Nonempty, Field(max_length=2000)]
    document_ids: list[Nonempty] = Field(min_length=1, max_length=20)


class RetrievalCall(StrictModel):
    """The sole allowed tool; documents cannot add capabilities."""

    tool: Literal["retrieve"]
    arguments: Question


class Chunk(StrictModel):
    """A stable source span with zero-based Unicode character offsets."""

    chunk_id: str
    document_id: str
    filename: str
    start: int
    end: int
    text: str


class Selection(StrictModel):
    """A model-selected exact quote from one supplied chunk."""

    chunk_id: Nonempty
    quote: Nonempty


class GroundedOutput(StrictModel):
    """An empty selection means the available evidence cannot answer."""

    citations: list[Selection] = Field(max_length=5)


class Citation(Selection):
    """A validated quotation and its original document location."""

    document_id: str
    filename: str
    start: int
    end: int


class Trace(StrictModel):
    """Operational events only; excludes prompts and hidden reasoning."""

    step: int
    action: str
    status: str
    detail: str


class Answer(StrictModel):
    """A source-excerpt answer or an explicit abstention, with its trace."""

    answer: str
    abstained: bool
    reason: str | None = None
    citations: list[Citation] = Field(default_factory=list)
    trace: list[Trace] = Field(default_factory=list)

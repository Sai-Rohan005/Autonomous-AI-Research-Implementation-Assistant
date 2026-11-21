"""FastAPI entry point for a local, single-user document-grounded assistant."""

from fastapi import FastAPI, HTTPException, Query, Request

from backend.engine import LanguageModel, ResearchEngine
from backend.gemini import GeminiModel
from backend.models import Answer, Question
from backend.store import DocumentStore


def create_app(model: LanguageModel | None = None, store: DocumentStore | None = None) -> FastAPI:
    """Build isolated application state; tests can supply an offline model."""
    application = FastAPI(title="Document-grounded Research Assistant", version="0.1.0")
    documents = store if store is not None else DocumentStore()
    engine = ResearchEngine(documents, model if model is not None else GeminiModel())

    @application.post("/documents", status_code=201)
    async def upload(request: Request, filename: str = Query(min_length=1, max_length=200)) -> dict[str, str | int]:
        """Accept a bounded raw UTF-8 .txt/.md body (not multipart or PDF)."""
        if request.headers.get("content-type", "").split(";", 1)[0].strip() not in {"text/plain", "text/markdown"}:
            raise HTTPException(415, "Use text/plain or text/markdown")
        if "/" in filename or "\\" in filename:
            raise HTTPException(422, "Filename must not contain a path")
        content = bytearray()
        async for part in request.stream():
            if len(content) + len(part) > documents.max_document_bytes:
                raise HTTPException(413, "Document exceeds upload limit")
            content.extend(part)
        try:
            return documents.upload(filename, bytes(content))
        except OverflowError as exc:
            raise HTTPException(413, str(exc)) from None
        except ValueError:
            raise HTTPException(422, "Supply nonempty UTF-8 text in a .txt or .md file") from None

    @application.post("/questions", response_model=Answer)
    def answer(question: Question) -> Answer:
        """Answer using only the explicitly requested document IDs."""
        try:
            return engine.answer(question)
        except KeyError:
            raise HTTPException(404, "Unknown document ID") from None

    return application


app = create_app()

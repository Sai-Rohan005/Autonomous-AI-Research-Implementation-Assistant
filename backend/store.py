"""Bounded, in-memory text ingestion and simple lexical retrieval."""

import re
from threading import RLock
from uuid import uuid4

from backend.models import Chunk, RetrievalCall

STOPWORDS = frozenset("a an and are as at be by do does for from how i in is it of on or the this to was what when where which who why with".split())


def terms(text: str) -> set[str]:
    """Extract case-insensitive terms, excluding common question words."""
    return set(re.findall(r"\w+", text.casefold())) - STOPWORDS


class DocumentStore:
    """Keep a small single-process corpus; restart discards all documents."""

    def __init__(self, max_document_bytes: int = 1_000_000, max_total_bytes: int = 10_000_000,
                 max_documents: int = 100) -> None:
        """Set per-document, corpus-byte, and metadata-count limits."""
        self.max_document_bytes = max_document_bytes
        self.max_total_bytes = max_total_bytes
        self.max_documents = max_documents
        self._documents: dict[str, list[Chunk]] = {}
        self._total_bytes = 0
        self._lock = RLock()

    def upload(self, filename: str, content: bytes) -> dict[str, str | int]:
        """Decode UTF-8 text and chunk it without executing document content."""
        if not filename.lower().endswith((".txt", ".md")):
            raise ValueError("Only .txt and .md documents are supported")
        if len(content) > self.max_document_bytes:
            raise OverflowError("Document exceeds upload limit")
        text = content.decode("utf-8", errors="strict")
        if not text.strip() or "\x00" in text:
            raise ValueError("Document must contain nonempty UTF-8 text without NUL bytes")
        document_id = uuid4().hex
        chunks = [Chunk(chunk_id=f"{document_id}:{start}", document_id=document_id,
                        filename=filename, start=start, end=min(start + 1000, len(text)),
                        text=text[start:start + 1000])
                  for start in range(0, len(text), 900) if text[start:start + 1000].strip()]
        with self._lock:
            if self._total_bytes + len(content) > self.max_total_bytes or len(self._documents) >= self.max_documents:
                raise OverflowError("Corpus capacity reached; restart to clear documents")
            self._documents[document_id] = chunks
            self._total_bytes += len(content)
        return {"document_id": document_id, "filename": filename, "chunks": len(chunks)}

    def retrieve(self, call: RetrievalCall) -> list[Chunk]:
        """Dispatch a validated tool call, ranking term overlap deterministically."""
        with self._lock:
            ids = list(dict.fromkeys(call.arguments.document_ids))
            if any(document_id not in self._documents for document_id in ids):
                raise KeyError("Unknown document ID")
            chunks = [chunk for document_id in ids for chunk in self._documents[document_id]]
        query_terms = terms(call.arguments.question)
        ranked = [(len(query_terms & terms(chunk.text)), index, chunk)
                  for index, chunk in enumerate(chunks)]
        ranked.sort(key=lambda row: (-row[0], row[1]))
        return [chunk for score, _, chunk in ranked if score > 0][:5]

"""Core data models for the AI Document Intelligence Platform.

Two model families live here, chosen to match where each model is used
(R17.1, R17.2 — everything is typed and documented):

* **Internal dataclasses** (``Chunk``, ``DocumentRecord``, ``RetrievedChunk``,
  ``EvalQuestion``) model data that flows *inside* the system — between the
  ingestion, storage, retrieval, and evaluation components. They are plain
  ``dataclasses`` because they never cross an HTTP boundary and need no request
  validation.
* **Pydantic v2 ``BaseModel`` API-boundary models** (``Citation``,
  ``QueryRequest``, ``QueryResponse``, ``UploadResponse``,
  ``DocumentListResponse``, ``ReembedResponse``, ``ErrorResponse``) model the
  request/response shapes exposed by the FastAPI backend. Pydantic gives us
  validation and JSON (de)serialization at the edge.

Per the design's Error Handling section, request-level validation such as
rejecting an empty question is enforced at the API/service layer, **not** on the
Pydantic models themselves — so ``QueryRequest`` intentionally accepts any
string for ``question``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from pydantic import BaseModel, ConfigDict


# ---------------------------------------------------------------------------
# Internal dataclasses (system-internal data flow; not HTTP-facing)
# ---------------------------------------------------------------------------


@dataclass
class Chunk:
    """A single text chunk extracted from one page of a source document.

    Chunks never straddle a page boundary (a deliberate design tradeoff for
    citation accuracy), so every chunk maps to exactly one page.

    Attributes:
        document_name: Filename of the source PDF the chunk came from.
        page_number: 1-based page number the chunk originated from.
        section: Section identifier if one is available, otherwise ``None``
            (R4.4).
        chunk_text: The chunk's text content.
    """

    document_name: str
    page_number: int
    section: str | None
    chunk_text: str


@dataclass
class DocumentRecord:
    """A per-document summary record stored alongside its chunks.

    ``document_id`` and ``document_hash`` are **two distinct fields with two
    distinct jobs**, and one is *never* derived from the other:

    * ``document_id`` is the document's **stable internal identity** — a UUID
      (in string form) generated once when a new, non-duplicate document is
      first ingested. It is the handle used to *address* a document (for
      example, the ``document_id`` path parameter of
      ``POST /documents/{document_id}/reembed`` and the document-scoped vector
      store operations). It is deliberately **not** equal to ``document_hash``.
    * ``document_hash`` is the **SHA-256 of the file bytes**, used **only** for
      duplicate detection (R11.6). It answers "have I already ingested these
      exact bytes?" and is never used to address a document for re-embedding.

    Attributes:
        document_id: Stable UUID (string form) generated at ingestion; the
            addressing handle for the document (distinct from
            ``document_hash``).
        document_name: Source PDF filename.
        document_hash: SHA-256 of the file bytes, used only for dedupe (R11.6).
        page_count: Number of pages in the source document.
        chunk_count: Number of chunks indexed for the document.
        embedding_provider: Provider that produced the embeddings (provenance,
            R5.5), e.g. ``'sentence-transformers'``.
        embedding_model: Embedding model used (provenance, R5.5), e.g.
            ``'all-MiniLM-L6-v2'``.
        ingested_at: Timestamp when the document was ingested.
    """

    document_id: str
    document_name: str
    document_hash: str
    page_count: int
    chunk_count: int
    embedding_provider: str
    embedding_model: str
    ingested_at: datetime


@dataclass
class RetrievedChunk:
    """A chunk returned from a similarity search, with its retrieval metadata.

    Attributes:
        chunk: The retrieved :class:`Chunk`.
        score: Similarity score for this chunk against the query.
        embedding_provider: Provider recorded for the chunk, used for
            embedding-space compatibility verification (R15.6).
        embedding_model: Embedding model recorded for the chunk, used for
            compatibility verification (R15.6).
    """

    chunk: Chunk
    score: float
    embedding_provider: str
    embedding_model: str


@dataclass
class EvalQuestion:
    """A single question in the version-controlled evaluation set (R2.3, R14).

    Attributes:
        id: Stable identifier for the question.
        question: The natural-language question text.
        essential_facts: Predefined key facts a correct answer must cover
            (R2.3, R14.3).
        source_document: Filename of the document where the answer lives
            (R2.3).
        source_page: 1-based page number where the answer lives.
    """

    id: str
    question: str
    essential_facts: list[str]
    source_document: str
    source_page: int


# ---------------------------------------------------------------------------
# Pydantic v2 API-boundary models (request/response shapes)
# ---------------------------------------------------------------------------


class Citation(BaseModel):
    """A citation backing an answer, derived from an in-context chunk (R9.2).

    Attributes:
        document: ``document_name`` of the source chunk.
        page: Page number, or a section identifier string when a page number
            is unavailable (hence ``int | str``).
        excerpt: Supporting text drawn from the in-context chunk.
    """

    document: str
    page: int | str
    excerpt: str


class QueryRequest(BaseModel):
    """Request body for ``POST /query`` (R10).

    Note: empty/whitespace-question rejection is enforced at the API/service
    layer per the design's Error Handling section, **not** here — so this model
    accepts any string for ``question``.

    Attributes:
        question: The user's natural-language question.
        top_k: Optional per-query override for the number of chunks to
            retrieve; ``None`` means fall back to the configured ``TOP_K``.
    """

    question: str
    top_k: int | None = None


class QueryResponse(BaseModel):
    """Response body for ``POST /query`` (R10.1, R10.2).

    Attributes:
        answer: The grounded answer text.
        citations: Citations backing the answer; empty when the context does
            not support an answer (``No_Answer_Response``, R9.3).
    """

    answer: str
    citations: list[Citation]


class UploadResponse(BaseModel):
    """Response body for ``POST /documents/upload`` (R11.1, R11.2).

    Attributes:
        document_id: Stable id of the ingested (or already-existing duplicate)
            document, so a client can later address it (e.g. for re-embed).
        document_name: Source PDF filename.
        status: Either ``'ingested'`` (new document) or ``'duplicate'``.
        page_count: Number of pages in the document.
        chunk_count: Number of chunks indexed for the document.
        message: Human-readable status message.
    """

    document_id: str
    document_name: str
    status: str
    page_count: int
    chunk_count: int
    message: str


class DocumentListResponse(BaseModel):
    """Response body for ``GET /documents`` (R12.1, R12.2).

    ``DocumentRecord`` is an internal dataclass nested inside this Pydantic
    model. ``ConfigDict(from_attributes=True)`` lets Pydantic read the
    dataclass instances by attribute, and Pydantic v2 serializes each nested
    record cleanly — including ``ingested_at`` as an ISO-8601 string — via
    ``model_dump``/``model_dump_json``.

    Attributes:
        documents: The ingested documents; an empty list when none are
            ingested (R12.2).
    """

    model_config = ConfigDict(from_attributes=True)

    documents: list[DocumentRecord]


class ReembedResponse(BaseModel):
    """Response body for ``POST /documents/{document_id}/reembed`` (R15.7).

    Attributes:
        document_id: The re-embedded document's stable id (echoed back).
        status: Always ``'reembedded'`` on success.
        embedding_provider: Updated provenance — the active provider (R5.5).
        embedding_model: Updated provenance — the active model (R5.5).
        chunk_count: Number of chunks that were re-embedded.
    """

    document_id: str
    status: str
    embedding_provider: str
    embedding_model: str
    chunk_count: int


class ErrorResponse(BaseModel):
    """Standard error body returned by the API (R16).

    Attributes:
        error: Machine-readable error code, e.g. ``'EMPTY_QUESTION'``.
        detail: Human-readable, credential-safe message (never leaks secrets).
    """

    error: str
    detail: str

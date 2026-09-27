"""Unit tests for :mod:`src.models`.

Covers construction and typing of the internal dataclasses and the validation
and JSON-serialization behavior of the Pydantic v2 API-boundary models.

Import of ``src`` relies on the workspace-root ``conftest.py`` (which puts the
root on ``sys.path``); no test-local conftest is added.
"""

from __future__ import annotations

from datetime import datetime

import pytest
from pydantic import ValidationError

from src.models import (
    Chunk,
    Citation,
    DocumentListResponse,
    DocumentRecord,
    ErrorResponse,
    EvalQuestion,
    QueryRequest,
    QueryResponse,
    ReembedResponse,
    RetrievedChunk,
    UploadResponse,
)


# ---------------------------------------------------------------------------
# Internal dataclasses
# ---------------------------------------------------------------------------


def test_chunk_construction_with_section_string() -> None:
    """A ``Chunk`` builds with a string ``section`` and preserves field values/types."""
    chunk = Chunk(
        document_name="report.pdf",
        page_number=3,
        section="1.2 Scope",
        chunk_text="Some chunk text.",
    )

    assert chunk.document_name == "report.pdf"
    assert isinstance(chunk.document_name, str)
    assert chunk.page_number == 3
    assert isinstance(chunk.page_number, int)
    assert chunk.section == "1.2 Scope"
    assert chunk.chunk_text == "Some chunk text."


def test_chunk_section_accepts_none() -> None:
    """``Chunk.section`` accepts ``None`` when no section is available (R4.4)."""
    chunk = Chunk(
        document_name="report.pdf",
        page_number=1,
        section=None,
        chunk_text="Body text.",
    )

    assert chunk.section is None


def test_document_record_construction_field_values_and_types() -> None:
    """A ``DocumentRecord`` builds with correct field values and types."""
    ingested = datetime(2024, 1, 2, 3, 4, 5)
    record = DocumentRecord(
        document_id="11111111-1111-1111-1111-111111111111",
        document_name="report.pdf",
        document_hash="a" * 64,
        page_count=10,
        chunk_count=42,
        embedding_provider="sentence-transformers",
        embedding_model="all-MiniLM-L6-v2",
        ingested_at=ingested,
    )

    assert record.document_id == "11111111-1111-1111-1111-111111111111"
    assert record.document_name == "report.pdf"
    assert record.document_hash == "a" * 64
    assert record.page_count == 10
    assert record.chunk_count == 42
    assert record.embedding_provider == "sentence-transformers"
    assert record.embedding_model == "all-MiniLM-L6-v2"
    assert record.ingested_at == ingested
    assert isinstance(record.ingested_at, datetime)


def test_document_record_id_and_hash_are_independent_fields() -> None:
    """``document_id`` and ``document_hash`` are distinct, independent fields.

    They are set to clearly different values and both must persist; the model
    must not force them equal or derive one from the other.
    """
    document_id = "22222222-2222-2222-2222-222222222222"
    document_hash = "f" * 64  # deliberately unrelated to the id

    record = DocumentRecord(
        document_id=document_id,
        document_name="doc.pdf",
        document_hash=document_hash,
        page_count=1,
        chunk_count=1,
        embedding_provider="sentence-transformers",
        embedding_model="all-MiniLM-L6-v2",
        ingested_at=datetime(2024, 5, 6, 7, 8, 9),
    )

    # Both persist exactly as provided ...
    assert record.document_id == document_id
    assert record.document_hash == document_hash
    # ... and they are NOT forced to be equal.
    assert record.document_id != record.document_hash


def test_retrieved_chunk_construction() -> None:
    """A ``RetrievedChunk`` wraps a ``Chunk`` with score and provenance."""
    chunk = Chunk(
        document_name="report.pdf",
        page_number=2,
        section=None,
        chunk_text="Relevant passage.",
    )
    retrieved = RetrievedChunk(
        chunk=chunk,
        score=0.87,
        embedding_provider="sentence-transformers",
        embedding_model="all-MiniLM-L6-v2",
    )

    assert retrieved.chunk is chunk
    assert retrieved.score == pytest.approx(0.87)
    assert isinstance(retrieved.score, float)
    assert retrieved.embedding_provider == "sentence-transformers"
    assert retrieved.embedding_model == "all-MiniLM-L6-v2"


def test_eval_question_construction() -> None:
    """An ``EvalQuestion`` builds with correct field values and types."""
    question = EvalQuestion(
        id="q1",
        question="What is the reporting threshold?",
        essential_facts=["threshold is $10,000", "reported within 15 days"],
        source_document="regulation.pdf",
        source_page=12,
    )

    assert question.id == "q1"
    assert question.question == "What is the reporting threshold?"
    assert question.essential_facts == [
        "threshold is $10,000",
        "reported within 15 days",
    ]
    assert isinstance(question.essential_facts, list)
    assert question.source_document == "regulation.pdf"
    assert question.source_page == 12
    assert isinstance(question.source_page, int)


# ---------------------------------------------------------------------------
# Citation (Pydantic)
# ---------------------------------------------------------------------------


def test_citation_page_accepts_int() -> None:
    """``Citation.page`` accepts an integer page number."""
    citation = Citation(document="report.pdf", page=7, excerpt="supporting text")

    assert citation.page == 7
    assert isinstance(citation.page, int)


def test_citation_page_accepts_str() -> None:
    """``Citation.page`` accepts a string section identifier."""
    citation = Citation(
        document="report.pdf", page="Appendix B", excerpt="supporting text"
    )

    assert citation.page == "Appendix B"
    assert isinstance(citation.page, str)


# ---------------------------------------------------------------------------
# QueryRequest / QueryResponse (Pydantic)
# ---------------------------------------------------------------------------


def test_query_request_top_k_defaults_to_none() -> None:
    """``QueryRequest.top_k`` defaults to ``None`` when omitted."""
    request = QueryRequest(question="What is the limit?")

    assert request.question == "What is the limit?"
    assert request.top_k is None


def test_query_request_requires_question() -> None:
    """Constructing ``QueryRequest`` without ``question`` raises ValidationError."""
    with pytest.raises(ValidationError):
        QueryRequest()  # type: ignore[call-arg]


def test_query_response_allows_empty_citations() -> None:
    """``QueryResponse`` with an empty citations list is valid (No_Answer_Response)."""
    response = QueryResponse(answer="No answer found in the documents.", citations=[])

    assert response.answer == "No answer found in the documents."
    assert response.citations == []


# ---------------------------------------------------------------------------
# DocumentListResponse round-trip serialization
# ---------------------------------------------------------------------------


def test_document_list_response_serializes_nested_dataclass_with_iso_datetime() -> None:
    """A nested ``DocumentRecord`` serializes cleanly; datetime becomes ISO-8601."""
    record = DocumentRecord(
        document_id="33333333-3333-3333-3333-333333333333",
        document_name="report.pdf",
        document_hash="b" * 64,
        page_count=5,
        chunk_count=20,
        embedding_provider="sentence-transformers",
        embedding_model="all-MiniLM-L6-v2",
        ingested_at=datetime(2024, 3, 4, 5, 6, 7),
    )
    payload = DocumentListResponse(documents=[record])

    # JSON round-trip works without error ...
    json_text = payload.model_dump_json()
    assert "33333333-3333-3333-3333-333333333333" in json_text
    # ... and the datetime is present as an ISO-8601 string.
    assert "2024-03-04T05:06:07" in json_text


def test_document_list_response_allows_empty_list() -> None:
    """An empty ``documents`` list is valid (R12.2) and serializes cleanly."""
    payload = DocumentListResponse(documents=[])

    assert payload.documents == []
    assert payload.model_dump_json() == '{"documents":[]}'


# ---------------------------------------------------------------------------
# UploadResponse / ReembedResponse / ErrorResponse (Pydantic)
# ---------------------------------------------------------------------------


def test_upload_response_construction_and_required_fields() -> None:
    """``UploadResponse`` builds with all required fields; missing fields fail."""
    response = UploadResponse(
        document_id="44444444-4444-4444-4444-444444444444",
        document_name="report.pdf",
        status="ingested",
        page_count=5,
        chunk_count=20,
        message="Ingested successfully.",
    )
    assert response.document_id == "44444444-4444-4444-4444-444444444444"
    assert response.status == "ingested"

    with pytest.raises(ValidationError):
        UploadResponse(document_name="report.pdf")  # type: ignore[call-arg]


def test_reembed_response_construction_and_required_fields() -> None:
    """``ReembedResponse`` builds with all required fields; missing fields fail."""
    response = ReembedResponse(
        document_id="55555555-5555-5555-5555-555555555555",
        status="reembedded",
        embedding_provider="sentence-transformers",
        embedding_model="all-MiniLM-L6-v2",
        chunk_count=20,
    )
    assert response.document_id == "55555555-5555-5555-5555-555555555555"
    assert response.status == "reembedded"

    with pytest.raises(ValidationError):
        ReembedResponse(document_id="x")  # type: ignore[call-arg]


def test_error_response_construction_and_required_fields() -> None:
    """``ErrorResponse`` builds with required fields; missing fields fail."""
    error = ErrorResponse(error="EMPTY_QUESTION", detail="The question was empty.")
    assert error.error == "EMPTY_QUESTION"
    assert error.detail == "The question was empty."

    with pytest.raises(ValidationError):
        ErrorResponse(error="EMPTY_QUESTION")  # type: ignore[call-arg]

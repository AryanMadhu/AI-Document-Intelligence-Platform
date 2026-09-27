"""Tests for :class:`src.ingestion.Ingestion_Service` (Task 9).

Test isolation and speed
------------------------
* **Vector store isolation.** Every store is created under a pytest
  ``tmp_path`` / ``tmp_path_factory`` temporary directory — never the real
  ``chroma_db/``. A fresh temp store is used per Hypothesis example.
* **Stub embeddings.** ``_StubEmbeddingProvider`` implements the full
  :class:`~src.embeddings.base.EmbeddingProvider` interface with deterministic
  8-dim vectors, so no real model loads and there is no network access.
  ``_FailingEmbeddingProvider`` raises :class:`EmbeddingError` to exercise the
  embed-then-store atomicity guarantee (decision C).
* **Chunking.** Most tests inject ``_FakeChunkingModule`` (one deterministic
  :class:`Chunk` per non-empty page) so they never load the HuggingFace
  tokenizer — keeping unit and property tests fast and deterministic. **One**
  integration-style test (``test_integration_with_real_chunking_module``) uses
  the *real* :class:`Chunking_Module` with a tiny chunk size to prove the
  wiring; the tokenizer is cached locally from Task 8.
* **Real PDFs.** Tiny valid PDFs are built in-memory with ``fitz`` (as in
  Task 7); invalid input uses raw non-PDF bytes.

Coverage:
    * 9.1 unit: ``compute_document_hash``; size limit; page limit; invalid PDF.
    * 9.2 unit: happy path; embedding-failure atomicity (C); empty-document (D2).
    * 9.3 Property 6 (Validates R5.5): provenance on every chunk + record.
    * 9.4 Property 7 (Validates R11.6-8): idempotent duplicate ingestion.
    * 9.5 Property 8 (Validates R12.1-2): listing reflects the ingested set.
    * 9.6 boundary unit: just over size / just over pages rejected, nothing
      stored.
"""

from __future__ import annotations

import dataclasses

import fitz  # PyMuPDF — used only to construct real PDFs for the tests
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from src.config import Config
from src.embeddings.base import EmbeddingError, EmbeddingProvider
from src.ingestion import (
    Chunking_Module,
    EmptyDocumentError,
    FileTooLargeError,
    Ingestion_Service,
    PDFParseError,
    TooManyPagesError,
    compute_document_hash,
)
from src.ingestion.pdf_parser import PageText
from src.models import Chunk
from src.vectorstore import VectorStore

_DIM = 8


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _StubEmbeddingProvider(EmbeddingProvider):
    """Deterministic, network-free embedding provider for tests.

    Implements the full ABC. Vectors are derived from text length so they are
    deterministic and cheap; the exact values do not matter for these tests.
    """

    def __init__(
        self,
        provider_name: str = "stub-provider",
        model_name: str = "stub-model",
    ) -> None:
        self._provider_name = provider_name
        self._model_name = model_name

    @property
    def provider_name(self) -> str:
        return self._provider_name

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return _DIM

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)

    @staticmethod
    def _vector(text: str) -> list[float]:
        base = float(len(text))
        return [(base + offset) % 7 for offset in range(_DIM)]


class _FailingEmbeddingProvider(_StubEmbeddingProvider):
    """Stub provider whose ``embed_texts`` always raises ``EmbeddingError``."""

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        raise EmbeddingError("stub embedding failure")


class _FakeChunkingModule:
    """Fast, tokenizer-free chunker: one Chunk per non-empty page.

    Mirrors the ``chunk_pages(pages, document_name=...)`` signature the service
    calls. Blank pages (empty text) yield no chunk, so an all-blank PDF yields
    zero chunks (exercising decision D2 without the real tokenizer).
    """

    def chunk_pages(
        self, pages: list[PageText], document_name: str
    ) -> list[Chunk]:
        chunks: list[Chunk] = []
        for page in pages:
            if page.text and page.text.strip():
                chunks.append(
                    Chunk(
                        document_name=document_name,
                        page_number=page.page_number,
                        section=None,
                        chunk_text=page.text,
                    )
                )
        return chunks


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pdf(pages: list[str | None]) -> bytes:
    """Build valid PDF bytes; a ``None`` entry produces a blank page."""
    doc = fitz.open()
    try:
        for content in pages:
            page = doc.new_page()
            if content is not None:
                page.insert_text((72, 72), content)
        return doc.tobytes()
    finally:
        doc.close()


def _make_store(tmp_dir) -> VectorStore:
    """Create an isolated VectorStore under ``tmp_dir``."""
    return VectorStore(persist_path=str(tmp_dir), collection_name="documents")


def _make_service(
    tmp_dir,
    *,
    embedding_provider: EmbeddingProvider | None = None,
    chunking_module=None,
    config: Config | None = None,
) -> tuple[Ingestion_Service, VectorStore]:
    """Build an Ingestion_Service + its store, wired with fast test doubles.

    Defaults: a stub embedding provider and the fake (tokenizer-free) chunker.
    """
    store = _make_store(tmp_dir)
    service = Ingestion_Service(
        config=config if config is not None else Config(),
        vector_store=store,
        embedding_provider=embedding_provider
        if embedding_provider is not None
        else _StubEmbeddingProvider(),
        chunking_module=chunking_module
        if chunking_module is not None
        else _FakeChunkingModule(),
    )
    return service, store


# ---------------------------------------------------------------------------
# 9.1 — compute_document_hash unit tests
# ---------------------------------------------------------------------------


def test_compute_document_hash_matches_known_value() -> None:
    """A known byte string hashes to its known SHA-256 hex digest."""
    # SHA-256 of b"" (well-known empty-input digest).
    assert compute_document_hash(b"") == (
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    )


def test_compute_document_hash_is_deterministic() -> None:
    """Identical bytes hash identically; different bytes hash differently."""
    assert compute_document_hash(b"hello") == compute_document_hash(b"hello")
    assert compute_document_hash(b"hello") != compute_document_hash(b"world")


# ---------------------------------------------------------------------------
# 9.1 — size / page / invalid-PDF unit tests
# ---------------------------------------------------------------------------


def test_size_limit_rejects_and_stores_nothing(tmp_path) -> None:
    """Bytes over MAX_PDF_SIZE_MB raise FileTooLargeError; nothing stored.

    Uses a tiny config override (1 MB) so no large allocation is needed.
    """
    config = dataclasses.replace(Config(), max_pdf_size_mb=1)
    service, store = _make_service(tmp_path, config=config)

    oversized = b"x" * (1 * 1024 * 1024 + 1)  # 1 byte over the 1 MB limit
    with pytest.raises(FileTooLargeError) as exc_info:
        service.ingest(oversized, "big.pdf")

    err = exc_info.value
    assert err.file == "big.pdf"
    assert err.limit_mb == 1
    assert "big.pdf" in str(err)
    assert "1 MB" in str(err)
    # Nothing stored.
    assert store.count() == 0
    assert store.list_documents() == []


def test_page_limit_rejects_and_stores_nothing(tmp_path) -> None:
    """A PDF with more pages than MAX_PDF_PAGES raises TooManyPagesError."""
    config = dataclasses.replace(Config(), max_pdf_pages=2)
    service, store = _make_service(tmp_path, config=config)

    three_page = _make_pdf(["page one", "page two", "page three"])
    with pytest.raises(TooManyPagesError) as exc_info:
        service.ingest(three_page, "toolong.pdf")

    err = exc_info.value
    assert err.file == "toolong.pdf"
    assert err.page_count == 3
    assert err.limit_pages == 2
    assert "toolong.pdf" in str(err)
    assert "2 pages" in str(err)
    assert store.count() == 0
    assert store.list_documents() == []


def test_invalid_pdf_propagates_and_prior_docs_intact(tmp_path) -> None:
    """Invalid PDF -> PDFParseError; a previously-ingested doc stays intact."""
    service, store = _make_service(tmp_path)

    # Ingest a good doc first.
    good = _make_pdf(["real content here"])
    first = service.ingest(good, "good.pdf")
    assert first.status == "ingested"
    count_after_good = store.count()
    assert count_after_good >= 1

    # Now an invalid (non-PDF) upload must surface PDFParseError.
    with pytest.raises(PDFParseError) as exc_info:
        service.ingest(b"not a pdf at all", "bad.pdf")
    assert exc_info.value.file == "bad.pdf"

    # The previously-ingested document is untouched.
    assert store.count() == count_after_good
    docs = store.list_documents()
    assert len(docs) == 1
    assert docs[0].document_id == first.document_id


# ---------------------------------------------------------------------------
# 9.2 — happy path / atomicity / empty-document unit tests
# ---------------------------------------------------------------------------


def test_happy_path_ingests_and_stores(tmp_path) -> None:
    """A valid multi-page PDF is ingested with correct counts + provenance."""
    provider = _StubEmbeddingProvider(
        provider_name="sentence-transformers", model_name="all-MiniLM-L6-v2"
    )
    service, store = _make_service(tmp_path, embedding_provider=provider)

    data = _make_pdf(["alpha content", "beta content", "gamma content"])
    response = service.ingest(data, "report.pdf")

    assert response.status == "ingested"
    # document_id is a valid UUID string.
    import uuid as _uuid

    assert str(_uuid.UUID(response.document_id)) == response.document_id
    assert response.document_name == "report.pdf"
    assert response.page_count == 3
    assert response.chunk_count == 3  # fake chunker: one chunk per page

    # Store now contains the chunks.
    assert store.count() == 3
    docs = store.list_documents()
    assert len(docs) == 1
    record = docs[0]
    assert record.document_id == response.document_id
    assert record.embedding_provider == "sentence-transformers"
    assert record.embedding_model == "all-MiniLM-L6-v2"
    assert record.page_count == 3
    assert record.chunk_count == 3


def test_embedding_failure_stores_nothing_and_preserves_prior(tmp_path) -> None:
    """Decision C: an embedding failure writes nothing and preserves prior data."""
    store = _make_store(tmp_path)

    # First, ingest a good document with a working provider.
    good_service = Ingestion_Service(
        config=Config(),
        vector_store=store,
        embedding_provider=_StubEmbeddingProvider(),
        chunking_module=_FakeChunkingModule(),
    )
    first = good_service.ingest(_make_pdf(["good doc content"]), "good.pdf")
    assert first.status == "ingested"
    count_after_good = store.count()
    assert count_after_good >= 1

    # Now attempt an ingest whose embedding step fails.
    failing_service = Ingestion_Service(
        config=Config(),
        vector_store=store,
        embedding_provider=_FailingEmbeddingProvider(),
        chunking_module=_FakeChunkingModule(),
    )
    with pytest.raises(EmbeddingError) as exc_info:
        failing_service.ingest(_make_pdf(["would-be new doc"]), "new.pdf")
    # The wrapped message identifies the file and is value-free.
    assert "new.pdf" in str(exc_info.value)

    # Nothing new stored; the prior good document is unchanged.
    assert store.count() == count_after_good
    docs = store.list_documents()
    assert len(docs) == 1
    assert docs[0].document_id == first.document_id


def test_empty_document_rejected(tmp_path) -> None:
    """Decision D2: a valid PDF with no extractable text -> EmptyDocumentError."""
    service, store = _make_service(tmp_path)

    # All-blank pages -> fake chunker yields zero chunks.
    blank = _make_pdf([None, None])
    with pytest.raises(EmptyDocumentError) as exc_info:
        service.ingest(blank, "blank.pdf")

    assert exc_info.value.file == "blank.pdf"
    assert "blank.pdf" in str(exc_info.value)
    # Nothing stored.
    assert store.count() == 0
    assert store.list_documents() == []


# ---------------------------------------------------------------------------
# 9.3 — Property 6: embedding provenance recorded for every indexed unit
# Validates: Requirements 5.5
# ---------------------------------------------------------------------------

_page_text_strategy = st.text(
    alphabet=st.characters(min_codepoint=97, max_codepoint=122),
    min_size=1,
    max_size=20,
)


@settings(
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(
    pages=st.lists(_page_text_strategy, min_size=1, max_size=5),
    provider_name=st.sampled_from(["sentence-transformers", "openai", "prov-x"]),
    model_name=st.sampled_from(["all-MiniLM-L6-v2", "text-embedding-3-small"]),
)
def test_property_provenance_recorded(
    tmp_path_factory, pages, provider_name, model_name
) -> None:
    """Property 6: every stored chunk AND the record carry the active provenance.

    Validates: Requirements 5.5. Uses a fake chunker for speed and a fresh temp
    store per example. Reads chunk provenance via ``query(...)`` and the record
    via ``list_documents()``.
    """
    tmp_dir = tmp_path_factory.mktemp("provenance")
    provider = _StubEmbeddingProvider(
        provider_name=provider_name, model_name=model_name
    )
    service, store = _make_service(
        tmp_dir, embedding_provider=provider
    )

    data = _make_pdf(list(pages))
    response = service.ingest(data, "doc.pdf")
    assert response.status == "ingested"

    # Every stored chunk carries the active provider/model.
    retrieved = store.query(provider.embed_query("q"), top_k=response.chunk_count + 5)
    assert len(retrieved) == response.chunk_count
    for item in retrieved:
        assert item.embedding_provider == provider_name
        assert item.embedding_model == model_name

    # The document record carries the active provider/model.
    docs = store.list_documents()
    assert len(docs) == 1
    assert docs[0].embedding_provider == provider_name
    assert docs[0].embedding_model == model_name


# ---------------------------------------------------------------------------
# 9.4 — Property 7: ingestion is idempotent under identical content
# Validates: Requirements 11.6, 11.7, 11.8
# ---------------------------------------------------------------------------


@settings(
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(pages=st.lists(_page_text_strategy, min_size=1, max_size=4))
def test_property_duplicate_ingestion_is_idempotent(
    tmp_path_factory, pages
) -> None:
    """Property 7: ingesting identical bytes twice is idempotent.

    Validates: Requirements 11.6-8. First -> "ingested"; second -> "duplicate"
    referencing the same document_id; store state (count + id set) unchanged
    after the second. Fresh temp store per example.
    """
    tmp_dir = tmp_path_factory.mktemp("dedupe")
    service, store = _make_service(tmp_dir)

    data = _make_pdf(list(pages))

    first = service.ingest(data, "doc.pdf")
    assert first.status == "ingested"
    count_after_first = store.count()
    ids_after_first = {r.document_id for r in store.list_documents()}

    second = service.ingest(data, "doc.pdf")
    assert second.status == "duplicate"
    assert second.document_id == first.document_id
    assert second.page_count == first.page_count
    assert second.chunk_count == first.chunk_count

    # Store state is unchanged after the duplicate.
    assert store.count() == count_after_first
    assert {r.document_id for r in store.list_documents()} == ids_after_first


# ---------------------------------------------------------------------------
# 9.5 — Property 8: document listing reflects exactly the ingested set
# Validates: Requirements 12.1, 12.2
# ---------------------------------------------------------------------------


@settings(
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(bodies=st.lists(_page_text_strategy, min_size=0, max_size=5, unique=True))
def test_property_listing_reflects_ingested_set(
    tmp_path_factory, bodies
) -> None:
    """Property 8: list_documents returns exactly the ingested document_ids.

    Validates: Requirements 12.1-2. Distinct bodies => distinct bytes =>
    distinct hashes, so each is a genuinely new document yielding >= 1 chunk.
    An empty set leaves the store empty ([]). Fresh temp store per example.
    """
    tmp_dir = tmp_path_factory.mktemp("listing")
    service, store = _make_service(tmp_dir)

    ingested_ids: set[str] = set()
    for index, body in enumerate(bodies):
        # Each doc has one non-empty page => >= 1 chunk via the fake chunker.
        data = _make_pdf([body])
        response = service.ingest(data, f"doc-{index}.pdf")
        assert response.status == "ingested"
        ingested_ids.add(response.document_id)

    listed_ids = {record.document_id for record in store.list_documents()}
    assert listed_ids == ingested_ids
    if not bodies:
        assert store.list_documents() == []


# ---------------------------------------------------------------------------
# 9.6 — boundary unit tests
# ---------------------------------------------------------------------------


def test_boundary_just_over_size_limit_rejected(tmp_path) -> None:
    """A file exactly one byte over MAX_PDF_SIZE_MB is rejected; nothing stored."""
    config = dataclasses.replace(Config(), max_pdf_size_mb=1)
    service, store = _make_service(tmp_path, config=config)

    limit_bytes = 1 * 1024 * 1024
    just_over = b"x" * (limit_bytes + 1)
    with pytest.raises(FileTooLargeError):
        service.ingest(just_over, "over.pdf")
    assert store.count() == 0
    assert store.list_documents() == []


def test_boundary_at_size_limit_is_allowed_through_size_check(tmp_path) -> None:
    """A file exactly at the limit passes the size check (not rejected for size).

    A file exactly equal to the limit must NOT be rejected as too large. Since
    that many raw bytes are not a valid PDF, the parse step raises PDFParseError
    (proving the size gate let it through), and nothing is stored.
    """
    config = dataclasses.replace(Config(), max_pdf_size_mb=1)
    service, store = _make_service(tmp_path, config=config)

    at_limit = b"x" * (1 * 1024 * 1024)
    with pytest.raises(PDFParseError):
        service.ingest(at_limit, "at.pdf")
    # Not a FileTooLargeError; nothing stored either way.
    assert store.count() == 0


def test_boundary_just_over_page_limit_rejected(tmp_path) -> None:
    """A PDF one page over MAX_PDF_PAGES is rejected; nothing stored."""
    config = dataclasses.replace(Config(), max_pdf_pages=2)
    service, store = _make_service(tmp_path, config=config)

    three_page = _make_pdf(["p1", "p2", "p3"])
    with pytest.raises(TooManyPagesError) as exc_info:
        service.ingest(three_page, "pages.pdf")
    assert exc_info.value.page_count == 3
    assert store.count() == 0
    assert store.list_documents() == []


def test_boundary_at_page_limit_is_allowed(tmp_path) -> None:
    """A PDF exactly at MAX_PDF_PAGES is accepted (not rejected for pages)."""
    config = dataclasses.replace(Config(), max_pdf_pages=2)
    service, store = _make_service(tmp_path, config=config)

    two_page = _make_pdf(["p1 content", "p2 content"])
    response = service.ingest(two_page, "atpages.pdf")
    assert response.status == "ingested"
    assert response.page_count == 2
    assert store.count() == 2


# ---------------------------------------------------------------------------
# Integration-style test using the REAL Chunking_Module (wiring proof)
# ---------------------------------------------------------------------------


def test_integration_with_real_chunking_module(tmp_path) -> None:
    """Prove the wiring with the real tokenizer-backed Chunking_Module.

    Uses a tiny chunk size so a short page still yields chunk(s); the tokenizer
    is cached locally from Task 8. Stub embeddings keep it network-free.
    """
    store = _make_store(tmp_path)
    real_chunker = Chunking_Module(chunk_size_tokens=8, chunk_overlap_tokens=2)
    service = Ingestion_Service(
        config=Config(),
        vector_store=store,
        embedding_provider=_StubEmbeddingProvider(),
        chunking_module=real_chunker,
    )

    data = _make_pdf(["The quick brown fox jumps over the lazy dog repeatedly."])
    response = service.ingest(data, "real.pdf")

    assert response.status == "ingested"
    assert response.page_count == 1
    assert response.chunk_count >= 1
    assert store.count() == response.chunk_count
    docs = store.list_documents()
    assert len(docs) == 1
    assert docs[0].document_id == response.document_id

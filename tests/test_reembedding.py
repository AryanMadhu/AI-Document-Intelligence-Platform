"""Tests for document re-embedding (Task 13; R5.3, R15.6-7, R16.3).

All tests are deterministic and fully offline: embedding providers are stubbed
(a small fixed dimension, deterministic vectors) and every :class:`VectorStore`
is created under a pytest ``tmp_path`` temporary directory, so no real
``chroma_db/`` is ever touched and no network/model is loaded.

The store is populated directly via ``VectorStore.add_chunks(...)`` with chosen
provenance so each re-embedding scenario is fully controllable. The
``Ingestion_Service`` is built with an injected stub provider; ``reembed_document``
never parses or chunks, so no PDF parser or tokenizer is exercised.

Covers the approved Task 13 scenarios:
    1.  Successful re-embedding updates provenance.
    2.  Only the target document changes (others untouched).
    3.  Multi-chunk documents re-embed every chunk.
    4.  Already-compatible document re-embeds cleanly.
    5.  Embedding failure leaves prior state untouched (no deletion).
    6.  No partial state after failure (chunks identical).
    7.  Metadata preservation (hash/page_count/pages/sections/name/ingested_at).
    8.  Embedding identity updated (per-chunk + DocumentRecord).
    9.  Retrieval succeeds after previously raising ReembeddingRequiredError.
    10. Compatible-document retrieval stays functional across an unrelated reembed.
    11. Unknown document_id -> DocumentNotFoundError; store unchanged.
    12. Determinism (stub embeddings + tmp_path throughout).
    + A direct VectorStore unit test for the D7 length-mismatch ValueError.
"""

from __future__ import annotations

import dataclasses

import pytest

from src.config import Config
from src.embeddings.base import EmbeddingError, EmbeddingProvider
from src.ingestion import DocumentNotFoundError, Ingestion_Service
from src.models import Chunk
from src.retrieval import ReembeddingRequiredError, Retrieval_Service
from src.vectorstore import VectorStore

_DIM = 8

_OLD_PROVIDER = "old-prov"
_OLD_MODEL = "old-model"
_NEW_PROVIDER = "new-prov"
_NEW_MODEL = "new-model"


# ---------------------------------------------------------------------------
# Deterministic, dependency-free stubs
# ---------------------------------------------------------------------------


class _StubEmbeddingProvider(EmbeddingProvider):
    """Deterministic, offline embedding provider implementing the full ABC.

    Every text maps to a fixed ``_DIM``-length vector derived only from its
    content, so embeddings are reproducible across runs and processes. No torch,
    no network.
    """

    def __init__(
        self,
        provider_name: str = _NEW_PROVIDER,
        model_name: str = _NEW_MODEL,
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
        return [_stub_vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return _stub_vector(text)


class _FailingEmbeddingProvider(_StubEmbeddingProvider):
    """Stub whose ``embed_texts`` always raises :class:`EmbeddingError`.

    Used to prove re-embedding aborts BEFORE any store mutation when embedding
    generation fails.
    """

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        raise EmbeddingError("stub embedding failure")


def _stub_vector(text: str) -> list[float]:
    """Build a deterministic ``_DIM``-length vector from ``text``."""
    seed = float(len(text) % 7)
    return [float((seed + offset) % 7) for offset in range(_DIM)]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_store(tmp_dir) -> VectorStore:
    """Create an isolated VectorStore under ``tmp_dir`` (never real chroma_db/)."""
    return VectorStore(persist_path=str(tmp_dir), collection_name="documents")


def _config(**overrides) -> Config:
    """Build a default Config with overrides (no env/.env load)."""
    return dataclasses.replace(Config(), **overrides)


def _service(store: VectorStore, provider: EmbeddingProvider) -> Ingestion_Service:
    """Build an Ingestion_Service with the injected stub provider + store."""
    return Ingestion_Service(
        config=_config(),
        vector_store=store,
        embedding_provider=provider,
    )


def _add_document(
    store: VectorStore,
    document_id: str,
    document_name: str,
    chunks: list[Chunk],
    *,
    embedding_provider: str,
    embedding_model: str,
    page_count: int,
) -> None:
    """Populate ``store`` with ``chunks`` for one document, with chosen provenance."""
    embeddings = [_stub_vector(chunk.chunk_text) for chunk in chunks]
    store.add_chunks(
        document_id,
        chunks,
        embeddings,
        document_hash=f"hash-{document_id}",
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
        page_count=page_count,
    )


def _record_by_id(store: VectorStore, document_id: str):
    """Return the single DocumentRecord for ``document_id`` (or None)."""
    for record in store.list_documents():
        if record.document_id == document_id:
            return record
    return None


def _simple_chunks(document_name: str, n: int) -> list[Chunk]:
    """Build ``n`` deterministic chunks for ``document_name``."""
    return [
        Chunk(
            document_name=document_name,
            page_number=i + 1,
            section=None,
            chunk_text=f"text-{i}",
        )
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Scenario 1 — successful re-embedding updates provenance
# ---------------------------------------------------------------------------


def test_reembed_updates_provenance(tmp_path) -> None:
    """reembed_document returns a reembedded response and updates provenance."""
    store = _make_store(tmp_path)
    chunks = _simple_chunks("doc.pdf", 2)
    _add_document(
        store,
        "doc-1",
        "doc.pdf",
        chunks,
        embedding_provider=_OLD_PROVIDER,
        embedding_model=_OLD_MODEL,
        page_count=2,
    )

    service = _service(store, _StubEmbeddingProvider(_NEW_PROVIDER, _NEW_MODEL))
    response = service.reembed_document("doc-1")

    assert response.document_id == "doc-1"
    assert response.status == "reembedded"
    assert response.embedding_provider == _NEW_PROVIDER
    assert response.embedding_model == _NEW_MODEL
    assert response.chunk_count == 2

    record = _record_by_id(store, "doc-1")
    assert record is not None
    assert record.embedding_provider == _NEW_PROVIDER
    assert record.embedding_model == _NEW_MODEL


# ---------------------------------------------------------------------------
# Scenario 2 — multiple documents, only the target changes
# ---------------------------------------------------------------------------


def test_reembed_only_target_changes(tmp_path) -> None:
    """Re-embedding docA leaves docB's provenance and chunks untouched."""
    store = _make_store(tmp_path)
    chunks_a = _simple_chunks("a.pdf", 2)
    chunks_b = _simple_chunks("b.pdf", 3)
    _add_document(
        store, "doc-a", "a.pdf", chunks_a,
        embedding_provider=_OLD_PROVIDER, embedding_model=_OLD_MODEL, page_count=2,
    )
    _add_document(
        store, "doc-b", "b.pdf", chunks_b,
        embedding_provider=_OLD_PROVIDER, embedding_model=_OLD_MODEL, page_count=3,
    )

    service = _service(store, _StubEmbeddingProvider(_NEW_PROVIDER, _NEW_MODEL))
    service.reembed_document("doc-a")

    record_a = _record_by_id(store, "doc-a")
    record_b = _record_by_id(store, "doc-b")
    assert record_a.embedding_provider == _NEW_PROVIDER
    assert record_a.embedding_model == _NEW_MODEL

    # docB still on the old embedding space, same chunk count + texts.
    assert record_b.embedding_provider == _OLD_PROVIDER
    assert record_b.embedding_model == _OLD_MODEL
    assert record_b.chunk_count == 3
    texts_b = [c.chunk_text for c in store.get_document_chunks("doc-b")]
    assert texts_b == [c.chunk_text for c in chunks_b]


# ---------------------------------------------------------------------------
# Scenario 3 — multiple chunks all carry new provenance
# ---------------------------------------------------------------------------


def test_reembed_multiple_chunks(tmp_path) -> None:
    """A >=3-chunk document re-embeds every chunk; chunk_count unchanged."""
    store = _make_store(tmp_path)
    chunks = _simple_chunks("doc.pdf", 4)
    _add_document(
        store, "doc-1", "doc.pdf", chunks,
        embedding_provider=_OLD_PROVIDER, embedding_model=_OLD_MODEL, page_count=4,
    )

    active = _StubEmbeddingProvider(_NEW_PROVIDER, _NEW_MODEL)
    service = _service(store, active)
    service.reembed_document("doc-1")

    record = _record_by_id(store, "doc-1")
    assert record.chunk_count == 4

    # Every stored chunk vector now carries the new provenance (via query).
    results = store.query(active.embed_query("text-0"), top_k=4)
    assert len(results) == 4
    for item in results:
        assert item.embedding_provider == _NEW_PROVIDER
        assert item.embedding_model == _NEW_MODEL


# ---------------------------------------------------------------------------
# Scenario 4 — already-compatible document re-embeds cleanly
# ---------------------------------------------------------------------------


def test_reembed_already_compatible(tmp_path) -> None:
    """Re-embedding a doc already on the active provenance succeeds, no change."""
    store = _make_store(tmp_path)
    chunks = _simple_chunks("doc.pdf", 2)
    _add_document(
        store, "doc-1", "doc.pdf", chunks,
        embedding_provider=_NEW_PROVIDER, embedding_model=_NEW_MODEL, page_count=2,
    )

    service = _service(store, _StubEmbeddingProvider(_NEW_PROVIDER, _NEW_MODEL))
    response = service.reembed_document("doc-1")

    assert response.status == "reembedded"
    record = _record_by_id(store, "doc-1")
    assert record.embedding_provider == _NEW_PROVIDER
    assert record.embedding_model == _NEW_MODEL
    assert record.chunk_count == 2
    assert [c.chunk_text for c in store.get_document_chunks("doc-1")] == [
        c.chunk_text for c in chunks
    ]


# ---------------------------------------------------------------------------
# Scenario 5 — embedding failure leaves prior state untouched
# ---------------------------------------------------------------------------


def test_reembed_embedding_failure_leaves_state_untouched(tmp_path) -> None:
    """A failing active provider raises EmbeddingError and deletes nothing."""
    store = _make_store(tmp_path)
    chunks = _simple_chunks("doc.pdf", 3)
    _add_document(
        store, "doc-1", "doc.pdf", chunks,
        embedding_provider=_OLD_PROVIDER, embedding_model=_OLD_MODEL, page_count=3,
    )
    count_before = store.count()

    service = _service(store, _FailingEmbeddingProvider(_NEW_PROVIDER, _NEW_MODEL))
    with pytest.raises(EmbeddingError):
        service.reembed_document("doc-1")

    record = _record_by_id(store, "doc-1")
    assert record.embedding_provider == _OLD_PROVIDER
    assert record.embedding_model == _OLD_MODEL
    assert record.chunk_count == 3
    assert store.count() == count_before
    assert [c.chunk_text for c in store.get_document_chunks("doc-1")] == [
        c.chunk_text for c in chunks
    ]


# ---------------------------------------------------------------------------
# Scenario 6 — no partial state after failure (chunks identical)
# ---------------------------------------------------------------------------


def test_reembed_failure_no_partial_state(tmp_path) -> None:
    """After a failed re-embed, get_document_chunks returns the SAME chunks."""
    store = _make_store(tmp_path)
    chunks = [
        Chunk("doc.pdf", page_number=1, section="Intro", chunk_text="alpha"),
        Chunk("doc.pdf", page_number=2, section=None, chunk_text="beta"),
        Chunk("doc.pdf", page_number=3, section="End", chunk_text="gamma"),
    ]
    _add_document(
        store, "doc-1", "doc.pdf", chunks,
        embedding_provider=_OLD_PROVIDER, embedding_model=_OLD_MODEL, page_count=3,
    )
    before = store.get_document_chunks("doc-1")

    service = _service(store, _FailingEmbeddingProvider(_NEW_PROVIDER, _NEW_MODEL))
    with pytest.raises(EmbeddingError):
        service.reembed_document("doc-1")

    after = store.get_document_chunks("doc-1")
    assert len(after) == len(before) == 3
    assert [(c.chunk_text, c.page_number, c.section) for c in after] == [
        (c.chunk_text, c.page_number, c.section) for c in before
    ]


# ---------------------------------------------------------------------------
# Scenario 7 — metadata preservation (D3 + hash/pages/sections/name/id)
# ---------------------------------------------------------------------------


def test_reembed_preserves_metadata(tmp_path) -> None:
    """All document/chunk metadata is preserved except provider/model (D3)."""
    store = _make_store(tmp_path)
    chunks = [
        Chunk("report.pdf", page_number=1, section="A", chunk_text="one"),
        Chunk("report.pdf", page_number=2, section=None, chunk_text="two"),
        Chunk("report.pdf", page_number=5, section="B", chunk_text="three"),
    ]
    _add_document(
        store, "doc-1", "report.pdf", chunks,
        embedding_provider=_OLD_PROVIDER, embedding_model=_OLD_MODEL, page_count=7,
    )

    before_record = _record_by_id(store, "doc-1")
    before_chunks = store.get_document_chunks("doc-1")
    before_hash = before_record.document_hash
    before_page_count = before_record.page_count
    before_ingested_at = before_record.ingested_at
    before_pages = [c.page_number for c in before_chunks]
    before_sections = [c.section for c in before_chunks]
    before_texts = [c.chunk_text for c in before_chunks]

    service = _service(store, _StubEmbeddingProvider(_NEW_PROVIDER, _NEW_MODEL))
    service.reembed_document("doc-1")

    after_record = _record_by_id(store, "doc-1")
    after_chunks = store.get_document_chunks("doc-1")

    # Unchanged.
    assert after_record.document_id == "doc-1"
    assert after_record.document_name == "report.pdf"
    assert after_record.document_hash == before_hash
    assert after_record.page_count == before_page_count
    assert after_record.ingested_at == before_ingested_at  # D3: original kept
    assert [c.page_number for c in after_chunks] == before_pages
    assert [c.section for c in after_chunks] == before_sections
    assert [c.chunk_text for c in after_chunks] == before_texts
    assert None in before_sections  # exercised a None section

    # Changed.
    assert after_record.embedding_provider == _NEW_PROVIDER
    assert after_record.embedding_model == _NEW_MODEL


# ---------------------------------------------------------------------------
# Scenario 8 — embedding identity updated (per-chunk + DocumentRecord)
# ---------------------------------------------------------------------------


def test_reembed_embedding_identity_updated(tmp_path) -> None:
    """Both per-chunk provenance and the DocumentRecord equal the new active."""
    store = _make_store(tmp_path)
    chunks = _simple_chunks("doc.pdf", 2)
    _add_document(
        store, "doc-1", "doc.pdf", chunks,
        embedding_provider=_OLD_PROVIDER, embedding_model=_OLD_MODEL, page_count=2,
    )

    active = _StubEmbeddingProvider(_NEW_PROVIDER, _NEW_MODEL)
    service = _service(store, active)
    service.reembed_document("doc-1")

    # Per-chunk provenance via query.
    results = store.query(active.embed_query("text-0"), top_k=2)
    assert results
    for item in results:
        assert item.embedding_provider == _NEW_PROVIDER
        assert item.embedding_model == _NEW_MODEL

    # DocumentRecord provenance.
    record = _record_by_id(store, "doc-1")
    assert record.embedding_provider == _NEW_PROVIDER
    assert record.embedding_model == _NEW_MODEL


# ---------------------------------------------------------------------------
# Scenario 9 — retrieval works after previously raising ReembeddingRequiredError
# ---------------------------------------------------------------------------


def test_retrieval_recovers_after_reembed(tmp_path) -> None:
    """A doc that gated retrieval becomes retrievable once re-embedded."""
    store = _make_store(tmp_path)
    chunks = _simple_chunks("doc.pdf", 3)
    _add_document(
        store, "doc-1", "doc.pdf", chunks,
        embedding_provider=_OLD_PROVIDER, embedding_model=_OLD_MODEL, page_count=3,
    )

    active = _StubEmbeddingProvider(_NEW_PROVIDER, _NEW_MODEL)
    config = _config(top_k=5)
    retrieval = Retrieval_Service(config, store, active)

    # Before re-embedding: incompatible provenance -> gate raises.
    with pytest.raises(ReembeddingRequiredError):
        retrieval.retrieve("a question")

    # Re-embed with the SAME active provider/model.
    _service(store, active).reembed_document("doc-1")

    # Now retrieval succeeds, returning min(top_k, store_size).
    results = retrieval.retrieve("a question")
    assert len(results) == min(5, store.count())
    assert store.count() == 3


# ---------------------------------------------------------------------------
# Scenario 10 — compatible-document retrieval stays functional
# ---------------------------------------------------------------------------


def test_compatible_retrieval_unaffected_by_unrelated_reembed(tmp_path) -> None:
    """A compatible doc stays retrievable before and after an unrelated reembed."""
    store = _make_store(tmp_path)
    active = _StubEmbeddingProvider(_NEW_PROVIDER, _NEW_MODEL)

    # docA already on the active embedding space (compatible).
    _add_document(
        store, "doc-a", "a.pdf", _simple_chunks("a.pdf", 2),
        embedding_provider=_NEW_PROVIDER, embedding_model=_NEW_MODEL, page_count=2,
    )

    config = _config(top_k=5)
    retrieval = Retrieval_Service(config, store, active)

    # Works before.
    before = retrieval.retrieve("q")
    assert len(before) == min(5, store.count())

    # Re-embed docA (an operation on the compatible doc itself).
    _service(store, active).reembed_document("doc-a")

    # Still works after.
    after = retrieval.retrieve("q")
    assert len(after) == min(5, store.count())
    assert store.count() == 2


# ---------------------------------------------------------------------------
# Scenario 11 — unknown document_id -> DocumentNotFoundError, store unchanged
# ---------------------------------------------------------------------------


def test_reembed_unknown_document_raises(tmp_path) -> None:
    """An unknown document_id raises DocumentNotFoundError; store unchanged."""
    store = _make_store(tmp_path)
    _add_document(
        store, "doc-1", "doc.pdf", _simple_chunks("doc.pdf", 2),
        embedding_provider=_OLD_PROVIDER, embedding_model=_OLD_MODEL, page_count=2,
    )
    count_before = store.count()
    docs_before = store.list_documents()

    # VectorStore.get_document_chunks returns [] for an unknown id (D1).
    assert store.get_document_chunks("nope") == []

    service = _service(store, _StubEmbeddingProvider(_NEW_PROVIDER, _NEW_MODEL))
    with pytest.raises(DocumentNotFoundError) as exc_info:
        service.reembed_document("nope")
    assert exc_info.value.document_id == "nope"

    assert store.count() == count_before
    assert store.list_documents() == docs_before


# ---------------------------------------------------------------------------
# Scenario 12 — determinism
# ---------------------------------------------------------------------------


def test_reembed_is_deterministic(tmp_path) -> None:
    """Two independent stores produce identical re-embed outcomes."""
    results = []
    for name in ("run-a", "run-b"):
        store = _make_store(tmp_path / name)
        chunks = _simple_chunks("doc.pdf", 3)
        _add_document(
            store, "doc-1", "doc.pdf", chunks,
            embedding_provider=_OLD_PROVIDER, embedding_model=_OLD_MODEL,
            page_count=3,
        )
        service = _service(store, _StubEmbeddingProvider(_NEW_PROVIDER, _NEW_MODEL))
        response = service.reembed_document("doc-1")
        results.append(
            (
                response.chunk_count,
                response.embedding_provider,
                response.embedding_model,
                [c.chunk_text for c in store.get_document_chunks("doc-1")],
            )
        )
    assert results[0] == results[1]


# ---------------------------------------------------------------------------
# Direct VectorStore unit test — D7 length mismatch
# ---------------------------------------------------------------------------


def test_replace_document_embeddings_length_mismatch(tmp_path) -> None:
    """replace_document_embeddings raises ValueError on chunk/embedding mismatch (D7)."""
    store = _make_store(tmp_path)
    chunks = _simple_chunks("doc.pdf", 2)
    _add_document(
        store, "doc-1", "doc.pdf", chunks,
        embedding_provider=_OLD_PROVIDER, embedding_model=_OLD_MODEL, page_count=2,
    )

    # One fewer embedding than chunks.
    with pytest.raises(ValueError):
        store.replace_document_embeddings(
            "doc-1",
            chunks,
            [_stub_vector("text-0")],
            _NEW_PROVIDER,
            _NEW_MODEL,
        )

"""Tests for document deletion (Task 15A).

Covers both layers of the delete path:

* **VectorStore.delete_document** — scoped removal by ``document_id`` with a
  post-delete verification (raises on a residual/failed delete, never a silent
  partial delete).
* **Ingestion_Service.delete_document** — existence check first
  (``DocumentNotFoundError`` for an unknown id, store untouched), then delegate
  to the store; a storage failure propagates (never a false success).
* **Retrieval after deletion** — deleting a document actually removes its
  vectors from similarity search while the surviving document stays retrievable.

All tests are deterministic and fully offline: embeddings are stubbed (fixed
small dimension), and every :class:`VectorStore` is created under a pytest
``tmp_path`` temporary directory — never the real ``chroma_db/`` — mirroring the
existing vectorstore/reembedding test fixtures.
"""

from __future__ import annotations

import dataclasses

import pytest

from src.config import Config
from src.embeddings.base import EmbeddingProvider
from src.ingestion import DocumentNotFoundError, Ingestion_Service
from src.models import Chunk, DeleteResponse
from src.vectorstore import VectorStore

_DIM = 8


# ---------------------------------------------------------------------------
# Deterministic, dependency-free stubs / helpers (mirrors existing tests)
# ---------------------------------------------------------------------------


class _StubEmbeddingProvider(EmbeddingProvider):
    """Deterministic, offline embedding provider implementing the full ABC."""

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
        return [_stub_vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return _stub_vector(text)


def _stub_vector(text: str) -> list[float]:
    """Build a deterministic ``_DIM``-length vector from ``text``."""
    seed = float(len(text) % 7)
    return [float((seed + offset) % 7) for offset in range(_DIM)]


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


def _simple_chunks(document_name: str, n: int) -> list[Chunk]:
    """Build ``n`` deterministic chunks for ``document_name``."""
    return [
        Chunk(
            document_name=document_name,
            page_number=i + 1,
            section=None,
            chunk_text=f"{document_name}-text-{i}",
        )
        for i in range(n)
    ]


def _add_document(
    store: VectorStore,
    document_id: str,
    document_name: str,
    chunks: list[Chunk],
    *,
    embedding_provider: str = "stub-provider",
    embedding_model: str = "stub-model",
    page_count: int | None = None,
) -> None:
    """Populate ``store`` with ``chunks`` for one document."""
    embeddings = [_stub_vector(chunk.chunk_text) for chunk in chunks]
    store.add_chunks(
        document_id,
        chunks,
        embeddings,
        document_hash=f"hash-{document_id}",
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
        page_count=page_count if page_count is not None else len(chunks),
    )


# ===========================================================================
# VectorStore.delete_document
# ===========================================================================


def test_delete_removes_all_chunks_of_a_document(tmp_path) -> None:
    """Deleting a multi-chunk document removes every one of its chunks."""
    store = _make_store(tmp_path)
    _add_document(store, "doc-1", "doc.pdf", _simple_chunks("doc.pdf", 4))
    assert store.count() == 4

    store.delete_document("doc-1")

    assert store.count() == 0
    assert store.get_document_chunks("doc-1") == []


def test_delete_leaves_other_documents_untouched(tmp_path) -> None:
    """Deleting one document never touches the chunks of other documents."""
    store = _make_store(tmp_path)
    _add_document(store, "doc-a", "a.pdf", _simple_chunks("a.pdf", 3))
    _add_document(store, "doc-b", "b.pdf", _simple_chunks("b.pdf", 2))

    store.delete_document("doc-a")

    # docA gone; docB intact.
    assert store.get_document_chunks("doc-a") == []
    remaining = store.get_document_chunks("doc-b")
    assert [c.chunk_text for c in remaining] == [
        c.chunk_text for c in _simple_chunks("b.pdf", 2)
    ]


def test_delete_removes_document_from_list_documents(tmp_path) -> None:
    """A deleted document no longer appears in list_documents()."""
    store = _make_store(tmp_path)
    _add_document(store, "doc-a", "a.pdf", _simple_chunks("a.pdf", 2))
    _add_document(store, "doc-b", "b.pdf", _simple_chunks("b.pdf", 2))
    assert {r.document_id for r in store.list_documents()} == {"doc-a", "doc-b"}

    store.delete_document("doc-a")

    ids = {r.document_id for r in store.list_documents()}
    assert ids == {"doc-b"}


def test_delete_drops_count_by_exact_amount(tmp_path) -> None:
    """count() drops by exactly the number of chunks the deleted doc had."""
    store = _make_store(tmp_path)
    _add_document(store, "doc-a", "a.pdf", _simple_chunks("a.pdf", 3))
    _add_document(store, "doc-b", "b.pdf", _simple_chunks("b.pdf", 5))
    assert store.count() == 8

    store.delete_document("doc-a")

    assert store.count() == 5  # 8 - 3


def test_delete_unknown_id_is_a_store_level_noop(tmp_path) -> None:
    """At the STORE level, deleting an unknown id is a harmless no-op.

    The store does not decide "not found" — deleting a ``where`` match that no
    chunk satisfies removes nothing and the post-verification trivially passes.
    (The service layer is what raises DocumentNotFoundError for an unknown id.)
    """
    store = _make_store(tmp_path)
    _add_document(store, "doc-a", "a.pdf", _simple_chunks("a.pdf", 2))

    # No raise, and nothing else is disturbed.
    store.delete_document("does-not-exist")

    assert store.count() == 2
    assert {r.document_id for r in store.list_documents()} == {"doc-a"}


def test_delete_raises_when_verification_finds_residual_chunks(tmp_path) -> None:
    """A failed delete (residual chunks remain) raises, not a silent success.

    Simulated by monkeypatching the collection's ``delete`` to a no-op so the
    chunks survive; the post-delete verification must then detect the residual
    and raise RuntimeError rather than reporting success.
    """
    store = _make_store(tmp_path)
    _add_document(store, "doc-1", "doc.pdf", _simple_chunks("doc.pdf", 2))

    # Make the underlying delete do nothing -> chunks remain -> verify fails.
    store._collection.delete = lambda *args, **kwargs: None

    with pytest.raises(RuntimeError):
        store.delete_document("doc-1")

    # Nothing was removed (the simulated delete was a no-op).
    assert store.count() == 2


def test_delete_propagates_chroma_delete_exception(tmp_path) -> None:
    """An exception from the underlying delete is NOT swallowed."""
    store = _make_store(tmp_path)
    _add_document(store, "doc-1", "doc.pdf", _simple_chunks("doc.pdf", 1))

    def _boom(*args, **kwargs):
        raise RuntimeError("chroma delete exploded")

    store._collection.delete = _boom

    with pytest.raises(RuntimeError, match="chroma delete exploded"):
        store.delete_document("doc-1")


# ===========================================================================
# Retrieval after deletion (vectors actually removed from search)
# ===========================================================================


def test_retrieval_after_deletion_excludes_deleted_document(tmp_path) -> None:
    """After deleting one of two docs, search returns no deleted-doc chunk and
    still returns the surviving document's chunks."""
    store = _make_store(tmp_path)
    _add_document(store, "doc-a", "a.pdf", _simple_chunks("a.pdf", 3))
    _add_document(store, "doc-b", "b.pdf", _simple_chunks("b.pdf", 3))

    store.delete_document("doc-a")

    # Query with a stub vector; pull back more than remain to see everything.
    results = store.query(_stub_vector("a.pdf-text-0"), top_k=10)
    names = {r.chunk.document_name for r in results}

    # No chunk from the deleted document.
    assert "a.pdf" not in names
    # The surviving document's chunks are still retrievable.
    assert names == {"b.pdf"}
    assert len(results) == 3


# ===========================================================================
# Ingestion_Service.delete_document
# ===========================================================================


def test_service_delete_returns_delete_response(tmp_path) -> None:
    """A successful service delete returns DeleteResponse(status='deleted')."""
    store = _make_store(tmp_path)
    _add_document(store, "doc-1", "doc.pdf", _simple_chunks("doc.pdf", 2))

    service = _service(store, _StubEmbeddingProvider())
    response = service.delete_document("doc-1")

    assert isinstance(response, DeleteResponse)
    assert response.document_id == "doc-1"
    assert response.status == "deleted"
    assert isinstance(response.message, str) and response.message
    # Really gone from the store.
    assert store.get_document_chunks("doc-1") == []
    assert store.count() == 0


def test_service_delete_unknown_id_raises_and_leaves_store_unchanged(tmp_path) -> None:
    """An unknown document_id raises DocumentNotFoundError; store unchanged."""
    store = _make_store(tmp_path)
    _add_document(store, "doc-1", "doc.pdf", _simple_chunks("doc.pdf", 2))
    count_before = store.count()
    docs_before = store.list_documents()

    service = _service(store, _StubEmbeddingProvider())
    with pytest.raises(DocumentNotFoundError) as exc_info:
        service.delete_document("nope")
    assert exc_info.value.document_id == "nope"

    assert store.count() == count_before
    assert store.list_documents() == docs_before


def test_service_delete_propagates_store_failure_no_false_success(tmp_path) -> None:
    """A store delete failure propagates; the service never reports success."""
    store = _make_store(tmp_path)
    _add_document(store, "doc-1", "doc.pdf", _simple_chunks("doc.pdf", 2))

    # Underlying delete is a no-op -> store verification raises RuntimeError.
    store._collection.delete = lambda *args, **kwargs: None

    service = _service(store, _StubEmbeddingProvider())
    with pytest.raises(RuntimeError):
        service.delete_document("doc-1")

    # The chunks are still present (the simulated delete did nothing).
    assert store.count() == 2


def test_service_delete_only_target_removed(tmp_path) -> None:
    """Service delete of one document leaves the others fully intact."""
    store = _make_store(tmp_path)
    _add_document(store, "doc-a", "a.pdf", _simple_chunks("a.pdf", 2))
    _add_document(store, "doc-b", "b.pdf", _simple_chunks("b.pdf", 3))

    service = _service(store, _StubEmbeddingProvider())
    service.delete_document("doc-a")

    assert store.get_document_chunks("doc-a") == []
    record_ids = {r.document_id for r in store.list_documents()}
    assert record_ids == {"doc-b"}
    assert store.count() == 3

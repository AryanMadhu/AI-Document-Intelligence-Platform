"""Tests for the :class:`Retrieval_Service` (Task 10; R6, R15.6-7).

All tests are isolated and never touch the real ``chroma_db/``: unit tests build
a store under a pytest ``tmp_path`` temporary directory, and the Hypothesis
property tests share a single temp-dir-backed store that is emptied before every
example (see ``_reset_store``). Reusing one Chroma client across the many
property-test examples — instead of opening a brand-new persistent client per
example — is a deliberate accommodation for a native Chroma/Windows crash that
surfaces after many client lifecycles; emptying the store per example preserves
full logical isolation (each example starts from an empty store). The embedding
provider is stubbed (a small fixed dimension); no real model is loaded and no
network is used. The store is populated by calling ``VectorStore.add_chunks(...)``
directly with stub embeddings and chosen provenance so compatibility scenarios
are fully controllable.

Covers:
    * Property 5 (Top-k retrieval invariants; Validates Requirements 6.2, 6.3)
      via Hypothesis at >=100 iterations.
    * Property 11 (Embedding-space compatibility gating; Validates Requirements
      15.6, 15.7) via Hypothesis at >=100 iterations.
    * Edge-case unit tests: empty-store precedence (R6.5), Config default top_k
      (R6.3), Config override, and per-query override.
"""

from __future__ import annotations

import dataclasses

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from src.config import Config
from src.embeddings.base import EmbeddingProvider
from src.models import Chunk
from src.retrieval import ReembeddingRequiredError, Retrieval_Service
from src.vectorstore import VectorStore

_DIM = 8
_ACTIVE_PROVIDER = "stub-provider"
_ACTIVE_MODEL = "stub-model"


class _StubEmbeddingProvider(EmbeddingProvider):
    """Deterministic, dependency-free embedding provider for tests.

    ``embed_query`` returns a fixed vector so the store ranks against a stable
    reference. ``embed_texts`` derives a deterministic vector per text. No real
    model, no torch, no network.
    """

    def __init__(
        self,
        provider_name: str = _ACTIVE_PROVIDER,
        model_name: str = _ACTIVE_MODEL,
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
        return [_stub_vector(float(i)) for i in range(len(texts))]

    def embed_query(self, text: str) -> list[float]:
        return _stub_vector(0.0)


class _RaisingEmbeddingProvider(_StubEmbeddingProvider):
    """Stub whose ``embed_query`` raises, to prove empty-store short-circuits."""

    def embed_query(self, text: str) -> list[float]:  # pragma: no cover
        raise AssertionError("embed_query must not be called on an empty store")


def _stub_vector(seed: float) -> list[float]:
    """Build a deterministic ``_DIM``-length stub vector from ``seed``."""
    return [float((seed + offset) % 7) for offset in range(_DIM)]


def _make_store(tmp_dir) -> VectorStore:
    """Create an isolated VectorStore under ``tmp_dir``."""
    return VectorStore(persist_path=str(tmp_dir), collection_name="documents")


def _reset_store(store: VectorStore) -> None:
    """Empty ``store`` so a property-test example starts from an empty store.

    Deletes every stored chunk by id (a no-op when already empty), restoring
    ``count() == 0``. This gives each Hypothesis example the same clean slate a
    fresh store would, while reusing a single native Chroma client.
    """
    ids = store._collection.get().get("ids") or []
    if ids:
        store._collection.delete(ids=ids)


@pytest.fixture
def shared_store(tmp_path_factory) -> VectorStore:
    """A single VectorStore reused across one property test's examples.

    Backed by a per-test temp directory (never the real ``chroma_db/``). Each
    example calls :func:`_reset_store` first, so examples never cross-contaminate.
    One client is created per property test (not per example), which avoids a
    native Chroma/Windows crash seen when many clients are opened in a process.
    """
    tmp_dir = tmp_path_factory.mktemp("retrieval_shared")
    return _make_store(tmp_dir)


def _config(**overrides) -> Config:
    """Build a default Config with overrides (no env/.env load)."""
    return dataclasses.replace(Config(), **overrides)


def _add_document(
    store: VectorStore,
    document_id: str,
    document_name: str,
    n_chunks: int,
    *,
    embedding_provider: str,
    embedding_model: str,
) -> None:
    """Populate ``store`` with ``n_chunks`` chunks for one document."""
    chunks = [
        Chunk(document_name, page_number=i + 1, section=None, chunk_text=f"text-{i}")
        for i in range(n_chunks)
    ]
    embeddings = [_stub_vector(float(i)) for i in range(n_chunks)]
    store.add_chunks(
        document_id,
        chunks,
        embeddings,
        document_hash=f"hash-{document_id}",
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
        page_count=n_chunks,
    )


# ---------------------------------------------------------------------------
# Property 5 — Top-k retrieval invariants
# Validates: Requirements 6.2, 6.3
# ---------------------------------------------------------------------------


@settings(
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(
    n_chunks=st.integers(min_value=1, max_value=12),
    # None => rely on Config default; else a varied override (incl. k>N, k==N, k<N).
    top_k=st.one_of(st.none(), st.integers(min_value=1, max_value=20)),
)
def test_property_top_k_retrieval_invariants(shared_store, n_chunks, top_k) -> None:
    """Property 5: count == min(effective_top_k, store_size); scores non-increasing.

    Validates: Requirements 6.2, 6.3. A single compatible document (provenance
    == active stub provider/model, so gating passes) is populated with ``N``
    chunks. Querying with a varied ``top_k`` (including ``None`` to exercise the
    Config default) must return exactly ``min(effective_top_k, store_size)``
    chunks ranked in non-increasing score order. The shared store is emptied
    before each example so examples start from an empty store.
    """
    store = shared_store
    _reset_store(store)
    _add_document(
        store,
        "doc-1",
        "doc-1.pdf",
        n_chunks,
        embedding_provider=_ACTIVE_PROVIDER,
        embedding_model=_ACTIVE_MODEL,
    )

    config = _config(top_k=5)
    service = Retrieval_Service(config, store, _StubEmbeddingProvider())

    results = service.retrieve("a question", top_k=top_k)

    effective_top_k = top_k if top_k is not None else config.top_k
    store_size = store.count()
    assert store_size == n_chunks

    # Invariant: count == min(effective_top_k, store_size).
    assert len(results) == min(effective_top_k, store_size)

    # Invariant: scores are non-increasing (rely on store ranking; no re-sort).
    scores = [item.score for item in results]
    for earlier, later in zip(scores, scores[1:]):
        assert earlier >= later


# ---------------------------------------------------------------------------
# Property 11 — Embedding-space compatibility gating (strict, all-or-nothing)
# Validates: Requirements 15.6, 15.7
# ---------------------------------------------------------------------------

# Each document is (provider, model). "match" == active stub provider/model.
_provenance_strategy = st.tuples(
    st.sampled_from([_ACTIVE_PROVIDER, "other-provider"]),
    st.sampled_from([_ACTIVE_MODEL, "other-model"]),
)


@settings(
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(
    provenances=st.lists(_provenance_strategy, min_size=1, max_size=3),
    top_k=st.integers(min_value=1, max_value=10),
)
def test_property_compatibility_gating(shared_store, provenances, top_k) -> None:
    """Property 11: strict all-or-nothing embedding-space gating.

    Validates: Requirements 15.6, 15.7. Populate the store with documents whose
    provenance either matches the active stub provider/model or mismatches it.

    * If ALL documents match -> ``retrieve()`` returns results, with
      ``count == min(top_k, store_size)``.
    * If ANY document mismatches -> ``retrieve()`` raises
      ``ReembeddingRequiredError`` whose ``document_ids`` equal EXACTLY the set
      of mismatching document ids.

    The shared store is emptied before each example so examples start empty.
    """
    store = shared_store
    _reset_store(store)

    mismatching_ids: set[str] = set()
    total_chunks = 0
    for index, (provider, model) in enumerate(provenances):
        document_id = f"doc-{index}"
        n_chunks = (index % 3) + 1  # 1..3 chunks per document
        total_chunks += n_chunks
        _add_document(
            store,
            document_id,
            f"{document_id}.pdf",
            n_chunks,
            embedding_provider=provider,
            embedding_model=model,
        )
        if provider != _ACTIVE_PROVIDER or model != _ACTIVE_MODEL:
            mismatching_ids.add(document_id)

    config = _config(top_k=5)
    service = Retrieval_Service(config, store, _StubEmbeddingProvider())

    if mismatching_ids:
        with pytest.raises(ReembeddingRequiredError) as exc_info:
            service.retrieve("a question", top_k=top_k)
        assert set(exc_info.value.document_ids) == mismatching_ids
    else:
        results = service.retrieve("a question", top_k=top_k)
        assert len(results) == min(top_k, store.count())
        assert store.count() == total_chunks


# ---------------------------------------------------------------------------
# Edge-case unit tests
# ---------------------------------------------------------------------------


def test_empty_store_returns_empty_without_embedding(tmp_path) -> None:
    """Empty store -> retrieve returns [] and never embeds the query (R6.5).

    The provider's ``embed_query`` raises, proving the empty-store short-circuit
    happens BEFORE any embedding is attempted.
    """
    store = _make_store(tmp_path)
    assert store.count() == 0

    service = Retrieval_Service(_config(), store, _RaisingEmbeddingProvider())

    assert service.retrieve("q") == []


def test_default_top_k_from_config(tmp_path) -> None:
    """With no top_k arg and >5 compatible chunks, returns exactly config.top_k (R6.3)."""
    store = _make_store(tmp_path)
    _add_document(
        store,
        "doc-1",
        "doc-1.pdf",
        8,  # more than the default of 5
        embedding_provider=_ACTIVE_PROVIDER,
        embedding_model=_ACTIVE_MODEL,
    )

    config = _config(top_k=5)
    service = Retrieval_Service(config, store, _StubEmbeddingProvider())

    results = service.retrieve("q")
    assert len(results) == 5


def test_config_override_top_k(tmp_path) -> None:
    """A Config with top_k=3 makes retrieve return 3 results (R6.3)."""
    store = _make_store(tmp_path)
    _add_document(
        store,
        "doc-1",
        "doc-1.pdf",
        8,
        embedding_provider=_ACTIVE_PROVIDER,
        embedding_model=_ACTIVE_MODEL,
    )

    config = _config(top_k=3)
    service = Retrieval_Service(config, store, _StubEmbeddingProvider())

    results = service.retrieve("q")
    assert len(results) == 3


def test_per_query_override_top_k(tmp_path) -> None:
    """A per-query top_k overrides the Config default regardless of its value."""
    store = _make_store(tmp_path)
    _add_document(
        store,
        "doc-1",
        "doc-1.pdf",
        8,
        embedding_provider=_ACTIVE_PROVIDER,
        embedding_model=_ACTIVE_MODEL,
    )

    config = _config(top_k=5)
    service = Retrieval_Service(config, store, _StubEmbeddingProvider())

    results = service.retrieve("q", top_k=2)
    assert len(results) == 2


def test_reembedding_error_message_names_documents(tmp_path) -> None:
    """The error message names affected doc names and recorded-vs-active mismatch (D4)."""
    store = _make_store(tmp_path)
    _add_document(
        store,
        "doc-mismatch",
        "old-report.pdf",
        2,
        embedding_provider="other-provider",
        embedding_model="other-model",
    )

    service = Retrieval_Service(_config(), store, _StubEmbeddingProvider())

    with pytest.raises(ReembeddingRequiredError) as exc_info:
        service.retrieve("q")

    error = exc_info.value
    assert error.document_ids == ["doc-mismatch"]
    message = str(error)
    assert "old-report.pdf" in message
    assert "other-provider" in message
    assert "other-model" in message
    assert _ACTIVE_PROVIDER in message
    assert _ACTIVE_MODEL in message

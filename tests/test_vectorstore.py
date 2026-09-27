"""Tests for the persistent Chroma-backed :class:`VectorStore` wrapper.

All tests are isolated: every store is created under a pytest ``tmp_path`` /
``tmp_path_factory`` temporary directory, never the real ``chroma_db/``.
Embeddings are stubbed (fixed small dimension); no real embedding model is
loaded.

Covers:
    * Property 4 (Storage round-trip; Validates Requirements 5.2, 6.4) via
      Hypothesis at >=100 iterations.
    * Deterministic example / edge-case unit tests: persistence across restart,
      empty collection, ``document_exists`` (keyed on hash not id), idempotent
      replace (no stale chunks), full ``DocumentRecord`` reconstruction,
      ``RetrievedChunk`` shape / ``top_k`` bounds, ``section`` None mapping, and
      length-mismatch validation.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from src.models import Chunk, DocumentRecord, RetrievedChunk
from src.vectorstore import VectorStore

_DIM = 8


def _stub_embedding(seed: float) -> list[float]:
    """Build a deterministic ``_DIM``-length stub vector from ``seed``."""
    return [float((seed + offset) % 7) for offset in range(_DIM)]


def _make_store(tmp_path) -> VectorStore:
    """Create an isolated VectorStore under ``tmp_path``."""
    return VectorStore(persist_path=str(tmp_path), collection_name="documents")


# ---------------------------------------------------------------------------
# Property 4 — Storage and retrieval preserve chunk metadata (round trip)
# Validates: Requirements 5.2, 6.4
# ---------------------------------------------------------------------------

_chunk_strategy = st.builds(
    Chunk,
    document_name=st.text(min_size=1, max_size=30),
    page_number=st.integers(min_value=1, max_value=500),
    section=st.one_of(st.none(), st.text(min_size=1, max_size=15)),
    chunk_text=st.text(min_size=0, max_size=60),
)


@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(chunks=st.lists(_chunk_strategy, min_size=1, max_size=8))
def test_property_storage_round_trip(tmp_path_factory, chunks) -> None:
    """Property 4: stored chunk metadata is returned unchanged.

    Validates: Requirements 5.2, 6.4. A fresh, unique store is created per
    example so examples never cross-contaminate. After ``add_chunks``, querying
    the chunks back returns identical ``{document_name, page_number, section,
    chunk_text}`` (including ``section=None`` round-tripping to ``None``), and
    each chunk remains addressable by its owning ``document_id``.
    """
    tmp_dir = tmp_path_factory.mktemp("roundtrip")
    store = VectorStore(persist_path=str(tmp_dir), collection_name="documents")

    document_id = "doc-under-test"
    embeddings = [_stub_embedding(float(i)) for i in range(len(chunks))]

    store.add_chunks(
        document_id,
        chunks,
        embeddings,
        document_hash="hash-abc",
        embedding_provider="sentence-transformers",
        embedding_model="all-MiniLM-L6-v2",
        page_count=42,
    )

    # Retrieve every chunk back (top_k large enough to cover all).
    retrieved = store.query(_stub_embedding(0.0), top_k=len(chunks) + 5)
    assert len(retrieved) == len(chunks)

    # Sort key maps section None -> ("", "") so None and str stay comparable.
    def _key(document_name, page_number, section, chunk_text):
        return (
            document_name,
            page_number,
            "" if section is None else section,
            0 if section is None else 1,
            chunk_text,
        )

    expected = sorted(
        _key(c.document_name, c.page_number, c.section, c.chunk_text) for c in chunks
    )
    actual = sorted(
        _key(
            r.chunk.document_name,
            r.chunk.page_number,
            r.chunk.section,
            r.chunk.chunk_text,
        )
        for r in retrieved
    )
    assert actual == expected

    # Each chunk remains addressable by its owning document_id.
    documents = store.list_documents()
    assert [record.document_id for record in documents] == [document_id]
    assert documents[0].chunk_count == len(chunks)


# ---------------------------------------------------------------------------
# Example / edge-case unit tests
# ---------------------------------------------------------------------------


def test_persistence_across_restart(tmp_path) -> None:
    """Data written by one store is visible to a new store on the same path."""
    store1 = _make_store(tmp_path)
    chunks = [Chunk("doc.pdf", 1, "1.0", "hello"), Chunk("doc.pdf", 2, None, "world")]
    store1.add_chunks(
        "D1",
        chunks,
        [_stub_embedding(1.0), _stub_embedding(2.0)],
        document_hash="hash-1",
        embedding_provider="prov",
        embedding_model="model",
        page_count=2,
    )

    # New instance on the SAME path sees the data.
    store2 = _make_store(tmp_path)
    assert store2.count() == 2
    assert len(store2.list_documents()) == 1
    assert len(store2.query(_stub_embedding(1.0), top_k=5)) == 2


def test_empty_collection_is_deterministic(tmp_path) -> None:
    """A fresh store reports empty results deterministically."""
    store = _make_store(tmp_path)
    assert store.count() == 0
    assert store.list_documents() == []
    assert store.query(_stub_embedding(0.0), top_k=5) == []


def test_document_exists_is_keyed_on_hash_not_id(tmp_path) -> None:
    """document_exists matches on document_hash, never document_id."""
    store = _make_store(tmp_path)
    document_id = "id-123"
    document_hash = "hash-xyz"
    store.add_chunks(
        document_id,
        [Chunk("doc.pdf", 1, None, "text")],
        [_stub_embedding(1.0)],
        document_hash=document_hash,
        embedding_provider="prov",
        embedding_model="model",
        page_count=1,
    )

    assert document_id != document_hash
    assert store.document_exists(document_hash) is True
    assert store.document_exists("unknown-hash") is False
    # Passing the document_id as if it were a hash must NOT match.
    assert store.document_exists(document_id) is False


def test_idempotent_replace_leaves_no_stale_chunks(tmp_path) -> None:
    """Re-adding a document_id with fewer chunks removes the old ones."""
    store = _make_store(tmp_path)
    three = [
        Chunk("doc.pdf", 1, None, "a"),
        Chunk("doc.pdf", 2, None, "b"),
        Chunk("doc.pdf", 3, None, "c"),
    ]
    store.add_chunks(
        "D",
        three,
        [_stub_embedding(float(i)) for i in range(3)],
        document_hash="hash-1",
        embedding_provider="prov",
        embedding_model="model",
        page_count=3,
    )
    assert store.count() == 3

    # Re-add same document_id with a single chunk.
    store.add_chunks(
        "D",
        [Chunk("doc.pdf", 1, None, "only")],
        [_stub_embedding(0.0)],
        document_hash="hash-1",
        embedding_provider="prov",
        embedding_model="model",
        page_count=1,
    )
    assert store.count() == 1
    documents = store.list_documents()
    assert len(documents) == 1
    assert documents[0].chunk_count == 1


def test_list_documents_reconstructs_complete_record(tmp_path) -> None:
    """list_documents rebuilds a fully-populated DocumentRecord."""
    store = _make_store(tmp_path)
    ingested_at = datetime(2024, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    store.add_chunks(
        "doc-id-1",
        [Chunk("report.pdf", 1, "1.1", "alpha"), Chunk("report.pdf", 2, None, "beta")],
        [_stub_embedding(1.0), _stub_embedding(2.0)],
        document_hash="sha-1",
        embedding_provider="sentence-transformers",
        embedding_model="all-MiniLM-L6-v2",
        page_count=7,
        ingested_at=ingested_at,
    )

    documents = store.list_documents()
    assert len(documents) == 1
    record = documents[0]
    assert isinstance(record, DocumentRecord)
    assert record.document_id == "doc-id-1"
    assert record.document_name == "report.pdf"
    assert record.document_hash == "sha-1"
    assert record.page_count == 7
    assert record.chunk_count == 2
    assert record.embedding_provider == "sentence-transformers"
    assert record.embedding_model == "all-MiniLM-L6-v2"
    assert isinstance(record.ingested_at, datetime)
    assert record.ingested_at == ingested_at


def test_query_returns_retrieved_chunks_with_provenance_and_bounds(tmp_path) -> None:
    """query returns RetrievedChunk objects, honors top_k, and guards k<=0."""
    store = _make_store(tmp_path)
    chunks = [Chunk("doc.pdf", i + 1, None, f"text-{i}") for i in range(4)]
    store.add_chunks(
        "D",
        chunks,
        [_stub_embedding(float(i)) for i in range(4)],
        document_hash="hash-1",
        embedding_provider="prov-x",
        embedding_model="model-y",
        page_count=4,
    )

    results = store.query(_stub_embedding(0.0), top_k=2)
    assert len(results) == 2
    for item in results:
        assert isinstance(item, RetrievedChunk)
        assert isinstance(item.chunk, Chunk)
        assert item.embedding_provider == "prov-x"
        assert item.embedding_model == "model-y"
        assert isinstance(item.score, float)

    # top_k cannot exceed stored count.
    assert len(store.query(_stub_embedding(0.0), top_k=100)) == 4
    # Guard: non-positive top_k returns [].
    assert store.query(_stub_embedding(0.0), top_k=0) == []
    assert store.query(_stub_embedding(0.0), top_k=-3) == []


def test_section_none_maps_round_trip(tmp_path) -> None:
    """section=None round-trips to None; a real section is preserved."""
    store = _make_store(tmp_path)
    store.add_chunks(
        "D",
        [Chunk("doc.pdf", 1, None, "no-section"), Chunk("doc.pdf", 2, "1.2", "with-section")],
        [_stub_embedding(1.0), _stub_embedding(2.0)],
        document_hash="hash-1",
        embedding_provider="prov",
        embedding_model="model",
        page_count=2,
    )

    results = store.query(_stub_embedding(1.0), top_k=5)
    by_text = {r.chunk.chunk_text: r.chunk.section for r in results}
    assert by_text["no-section"] is None
    assert by_text["with-section"] == "1.2"


def test_add_chunks_length_mismatch_raises_value_error(tmp_path) -> None:
    """A chunks/embeddings length mismatch raises ValueError."""
    store = _make_store(tmp_path)
    with pytest.raises(ValueError):
        store.add_chunks(
            "D",
            [Chunk("doc.pdf", 1, None, "a"), Chunk("doc.pdf", 2, None, "b")],
            [_stub_embedding(1.0)],  # only one embedding for two chunks
            document_hash="hash-1",
            embedding_provider="prov",
            embedding_model="model",
            page_count=2,
        )

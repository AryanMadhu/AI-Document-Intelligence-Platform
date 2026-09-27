"""Persistent Chroma vector store wrapper.

This module wraps a **persistent** Chroma client behind a thin, backend-neutral
``VectorStore`` API so the store can later be swapped for pgvector (or any other
backend) without leaking Chroma types out of this module (R6.2, design:
Vector_Store wrapper). Chroma is imported at module level here — that is
intentional and confined to this module — but no ``chromadb`` type ever appears
in a public method signature or return value. Callers only ever see the domain
models from :mod:`src.models` (:class:`Chunk`, :class:`DocumentRecord`,
:class:`RetrievedChunk`).

Keying convention (design: addressing by ``document_id``, dedupe by
``document_hash``):

* Chroma record id — a **composite** ``f"{document_id}:{i}"`` string, kept
  deliberately distinct from both ``document_id`` and ``document_hash`` so the
  internal id never collides with either domain identifier.
* ``document_id`` — the stable UUID a document is *addressed* by; stored as a
  metadata field on every chunk (never used as the Chroma id).
* ``document_hash`` — the SHA-256 of the file bytes, used **only** for
  duplicate detection via :meth:`document_exists` (never used to address a
  document, never used as the Chroma id).

Embedding ownership: this store never computes vectors. The collection is
created with the embedding function explicitly disabled (``None``) so Chroma
performs no server-side embedding; callers always supply explicit ``embeddings``
produced by the active embedding provider (Task 4). The store only persists and
returns the vectors it is handed.

``section`` sentinel mapping: Chroma metadata values must be scalar and
non-``None``. ``Chunk.section`` may be ``None``, so on write a ``None`` section
is stored as the empty string ``""``; on read the empty string is mapped back to
``None`` so the round-trip preserves ``section is None``.
"""

from __future__ import annotations

from datetime import datetime, timezone

import chromadb

from src.models import Chunk, DocumentRecord, RetrievedChunk

# Sentinel stored in Chroma metadata for a ``None`` section (Chroma disallows
# ``None`` metadata values). Mapped back to ``None`` on read.
_SECTION_NONE_SENTINEL = ""


class VectorStore:
    """Thin, backend-neutral wrapper over a persistent Chroma collection.

    The wrapper hides all Chroma-specific types: every public method accepts
    and returns only plain values and :mod:`src.models` domain objects, so the
    backend can be replaced without touching callers.

    Persistence: backed by ``chromadb.PersistentClient(path=persist_path)``, so
    opening a new :class:`VectorStore` on the same ``persist_path`` sees data
    written by an earlier instance (survives restarts).
    """

    def __init__(
        self,
        persist_path: str = "chroma_db",
        collection_name: str = "documents",
    ) -> None:
        """Open (or create) a persistent Chroma collection.

        Args:
            persist_path: Filesystem directory Chroma persists to. Reusing the
                same path in a new instance reopens the existing data.
            collection_name: Name of the collection holding all chunks.

        Notes:
            The collection is created with ``embedding_function=None`` so Chroma
            never computes vectors; the caller always supplies explicit
            ``embeddings`` (see :meth:`add_chunks`).
        """
        self._persist_path = persist_path
        self._collection_name = collection_name
        self._client = chromadb.PersistentClient(path=persist_path)
        # embedding_function=None => we always pass explicit embeddings; Chroma
        # performs no server-side embedding of its own.
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            embedding_function=None,
        )

    def add_chunks(
        self,
        document_id: str,
        chunks: list[Chunk],
        embeddings: list[list[float]],
        *,
        document_hash: str,
        embedding_provider: str,
        embedding_model: str,
        page_count: int,
        ingested_at: datetime | None = None,
    ) -> None:
        """Persist a document's chunks and their vectors (idempotent replace).

        The positional signature ``(document_id, chunks, embeddings)`` matches
        the design exactly. The keyword-only arguments carry the document-level
        metadata needed to later reconstruct a complete
        :class:`~src.models.DocumentRecord` in :meth:`list_documents`.

        Idempotent, document-scoped replace: any records already stored under
        ``document_id`` are deleted *before* the new chunks are added. Re-adding
        the same ``document_id`` with fewer chunks therefore leaves no orphaned
        (stale) chunks behind.

        Chroma record ids are the composite ``f"{document_id}:{i}"`` — distinct
        from both ``document_id`` and ``document_hash``. ``document_id`` and
        ``document_hash`` live only in metadata.

        Args:
            document_id: Stable UUID the document is addressed by.
            chunks: The chunks to store (may be empty — see Notes).
            embeddings: One vector per chunk, in the same order as ``chunks``.
            document_hash: SHA-256 of the file bytes (dedupe key).
            embedding_provider: Provenance provider for these embeddings.
            embedding_model: Provenance model for these embeddings.
            page_count: Number of pages in the source document.
            ingested_at: Ingestion timestamp; defaults to
                ``datetime.now(timezone.utc)`` when ``None``. Stored as an
                ISO-8601 string in metadata.

        Raises:
            ValueError: If ``len(chunks) != len(embeddings)``. The message is
                value-free (carries no chunk text or vector data).

        Notes:
            Empty input is handled gracefully: the document-scoped delete still
            runs (clearing any prior chunks), but Chroma's ``add`` is skipped so
            it is never called with empty lists.
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                "chunks and embeddings must have the same length"
            )

        resolved_ingested_at = ingested_at or datetime.now(timezone.utc)
        ingested_at_iso = resolved_ingested_at.isoformat()

        # Document-scoped, idempotent replace: clear any prior records for this
        # document_id so a re-add with fewer chunks leaves no orphans.
        self._collection.delete(where={"document_id": document_id})

        # Guard against empty input: Chroma rejects add() with empty lists.
        if not chunks:
            return

        ids: list[str] = [f"{document_id}:{i}" for i in range(len(chunks))]
        metadatas: list[dict[str, object]] = []
        for chunk in chunks:
            # section None -> "" sentinel (Chroma metadata cannot be None);
            # mapped back to None on read.
            section_value = (
                chunk.section
                if chunk.section is not None
                else _SECTION_NONE_SENTINEL
            )
            metadatas.append(
                {
                    "document_name": chunk.document_name,
                    "page_number": chunk.page_number,
                    "section": section_value,
                    "chunk_text": chunk.chunk_text,
                    "embedding_provider": embedding_provider,
                    "embedding_model": embedding_model,
                    "document_id": document_id,
                    "document_hash": document_hash,
                    # Document-level fields needed to rebuild DocumentRecord.
                    "page_count": page_count,
                    "ingested_at": ingested_at_iso,
                }
            )

        self._collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def query(
        self,
        query_embedding: list[float],
        top_k: int,
    ) -> list[RetrievedChunk]:
        """Return the ``top_k`` most similar chunks to ``query_embedding``.

        Score convention: Chroma returns a *distance* per hit. This wrapper
        converts distance to a similarity score deterministically as
        ``score = 1.0 - distance`` (smaller distance => larger score). Callers
        and tests should rely only on the *ordering* of scores (non-increasing),
        not on the exact metric value.

        Args:
            query_embedding: The query vector to search with.
            top_k: Maximum number of results to return.

        Returns:
            A list of :class:`~src.models.RetrievedChunk`, ordered as Chroma
            ranks them (most similar first). Returns ``[]`` when the collection
            is empty, when there are no results, or when ``top_k <= 0``. Never
            returns Chroma dicts/objects.
        """
        if top_k <= 0:
            return []
        if self._collection.count() == 0:
            return []

        result = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
        )

        # Chroma nests results one level per query embedding; we sent one.
        metadatas_batches = result.get("metadatas") or []
        distances_batches = result.get("distances") or []
        if not metadatas_batches:
            return []

        metadatas = metadatas_batches[0] or []
        distances = distances_batches[0] if distances_batches else []

        retrieved: list[RetrievedChunk] = []
        for index, metadata in enumerate(metadatas):
            distance = distances[index] if index < len(distances) else 0.0
            score = 1.0 - float(distance)
            retrieved.append(
                RetrievedChunk(
                    chunk=self._chunk_from_metadata(metadata),
                    score=score,
                    embedding_provider=str(metadata["embedding_provider"]),
                    embedding_model=str(metadata["embedding_model"]),
                )
            )
        return retrieved

    def list_documents(self) -> list[DocumentRecord]:
        """Reconstruct one :class:`DocumentRecord` per stored document.

        All chunk metadata is read and grouped by ``document_id``. For each
        group a complete record is rebuilt: ``chunk_count`` is the number of
        chunks in the group; the remaining fields are taken from the group's
        metadata (``ingested_at`` is parsed from its ISO-8601 string back to a
        :class:`datetime`).

        Returns:
            The reconstructed records, sorted deterministically by
            ``(document_name, document_id)`` so results are stable (Chroma does
            not guarantee ordering). Returns ``[]`` when the store is empty.
        """
        stored = self._collection.get(include=["metadatas"])
        metadatas = stored.get("metadatas") or []

        grouped: dict[str, dict[str, object]] = {}
        counts: dict[str, int] = {}
        for metadata in metadatas:
            document_id = str(metadata["document_id"])
            counts[document_id] = counts.get(document_id, 0) + 1
            if document_id not in grouped:
                grouped[document_id] = metadata

        records: list[DocumentRecord] = []
        for document_id, metadata in grouped.items():
            records.append(
                DocumentRecord(
                    document_id=document_id,
                    document_name=str(metadata["document_name"]),
                    document_hash=str(metadata["document_hash"]),
                    page_count=int(metadata["page_count"]),
                    chunk_count=counts[document_id],
                    embedding_provider=str(metadata["embedding_provider"]),
                    embedding_model=str(metadata["embedding_model"]),
                    ingested_at=datetime.fromisoformat(
                        str(metadata["ingested_at"])
                    ),
                )
            )

        records.sort(key=lambda record: (record.document_name, record.document_id))
        return records

    def document_exists(self, document_hash: str) -> bool:
        """Report whether any stored chunk carries ``document_hash``.

        This is the duplicate-detection check and is keyed on the SHA-256
        ``document_hash`` — **never** on ``document_id``. It answers "have I
        already ingested these exact bytes?".

        Args:
            document_hash: The SHA-256 hash of the candidate file's bytes.

        Returns:
            ``True`` if at least one stored chunk has this ``document_hash``,
            otherwise ``False``.
        """
        found = self._collection.get(
            where={"document_hash": document_hash},
            limit=1,
        )
        return bool(found.get("ids"))

    def count(self) -> int:
        """Return the number of stored chunks.

        Returns:
            The chunk count across all documents (``0`` for an empty store).
        """
        return int(self._collection.count())

    @staticmethod
    def _chunk_from_metadata(metadata: dict[str, object]) -> Chunk:
        """Rebuild a :class:`Chunk` from stored metadata.

        Applies the ``section`` sentinel mapping in reverse: the empty-string
        sentinel is mapped back to ``None`` so an originally-``None`` section
        round-trips to ``None``.

        Args:
            metadata: A single chunk's stored metadata dict.

        Returns:
            The reconstructed :class:`Chunk`.
        """
        raw_section = metadata.get("section")
        section = (
            None
            if raw_section is None or raw_section == _SECTION_NONE_SENTINEL
            else str(raw_section)
        )
        return Chunk(
            document_name=str(metadata["document_name"]),
            page_number=int(metadata["page_number"]),
            section=section,
            chunk_text=str(metadata["chunk_text"]),
        )

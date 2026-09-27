"""Retrieval service: embed a question and return the top-k similar chunks.

The :class:`Retrieval_Service` is the read side of the query pipeline. It embeds
a user question with the *active* :class:`~src.embeddings.base.EmbeddingProvider`
and asks the :class:`~src.vectorstore.VectorStore` for the ``top_k`` most similar
chunks (R6.1, R6.2, R6.3, R6.4).

Before it ever queries, the service enforces the **embedding-space
compatibility check** (R15.6, R15.7). Vectors produced by different
providers/models live in different vector spaces, so mixing them yields
meaningless similarity scores. To keep results trustworthy the check is
**strict and all-or-nothing**: if *any* indexed document's recorded
``(embedding_provider, embedding_model)`` differs from the active
provider/model, the service raises :class:`ReembeddingRequiredError` naming
*every* affected document and returns no results at all — never a partial or
filtered result set.

``retrieve()`` step order (see the method docstring for detail):
    1. Resolve the effective ``top_k`` (per-query override, else ``Config.top_k``).
    2. Empty-store precedence: an empty store returns ``[]`` immediately — no
       embedding is computed and no compatibility error is raised (R6.5).
    3. Compatibility gating over *all* documents (strict, all-or-nothing).
    4. Embed the question with the active provider (R6.1).
    5. Query the vector store for the top-k ranked chunks (R6.2) and return them
       unchanged; the store already returns them in non-increasing score order.

This component holds no banking- or domain-specific logic (R15.5); it only wires
the embedding provider to the vector store.
"""

from __future__ import annotations

from src.config import Config
from src.embeddings.base import EmbeddingProvider, build_embedding_provider
from src.models import DocumentRecord, RetrievedChunk
from src.vectorstore import VectorStore


class ReembeddingRequiredError(Exception):
    """Raised when indexed documents are incompatible with the active embedding space.

    Per the strict, all-or-nothing compatibility policy (R15.6, R15.7), this is
    raised whenever *any* indexed document's recorded
    ``(embedding_provider, embedding_model)`` differs from the active
    provider/model. When raised, retrieval returns no results at all.

    The primary carried payload is :attr:`document_ids` — the ids of every
    affected (incompatible) document, so callers (e.g. the re-embed flow) know
    exactly which documents to re-embed. The human-readable message names the
    affected document *names* and the recorded-vs-active provider/model mismatch
    so a human can understand what happened; it never contains any secret.

    Attributes:
        document_ids: The ``document_id`` of every incompatible document.
    """

    def __init__(
        self,
        document_ids: list[str],
        details: str | None = None,
    ) -> None:
        """Build the error from the affected document ids and a message.

        Args:
            document_ids: The ids of every incompatible document (the payload).
            details: Optional pre-built, human-readable, secret-free message. A
                default message naming the affected ids is used when omitted.
        """
        self.document_ids: list[str] = list(document_ids)
        message = details or (
            "Re-embedding required for documents: "
            f"{', '.join(self.document_ids)}"
        )
        super().__init__(message)


class Retrieval_Service:
    """Embed a question and return the top-k most similar chunks (R6, R15.6-7).

    The service embeds the incoming question with the active
    :class:`~src.embeddings.base.EmbeddingProvider` and returns the ``top_k``
    ranked chunks from the :class:`~src.vectorstore.VectorStore`. It first
    enforces the strict, all-or-nothing embedding-space compatibility check so
    similarity scores are only ever computed within a single, consistent vector
    space.
    """

    def __init__(
        self,
        config: Config,
        vector_store: VectorStore,
        embedding_provider: EmbeddingProvider | None = None,
    ) -> None:
        """Wire the service to its configuration, store, and embedding provider.

        Args:
            config: Active configuration. ``config.top_k`` supplies the default
                number of chunks to return (R6.3).
            vector_store: The store queried for similar chunks and inspected for
                embedding-space compatibility.
            embedding_provider: The active provider used to embed questions.
                Defaults to :func:`build_embedding_provider` applied to
                ``config`` (which lazily loads the heavy backend only when it is
                actually used).
        """
        self._config = config
        self._vector_store = vector_store
        self._embedding_provider = embedding_provider or build_embedding_provider(
            config
        )

    def retrieve(
        self,
        question: str,
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        """Return the top-k chunks most similar to ``question``.

        Steps, in order:
            1. **Resolve top_k** — use ``top_k`` when provided, else the
               configured ``Config.top_k`` default (R6.3).
            2. **Empty-store precedence** — if the store holds no chunks, return
               ``[]`` immediately. No embedding is computed and no compatibility
               error is raised, because there is nothing to be incompatible with
               (R6.5).
            3. **Compatibility gating** — inspect *every* document from
               ``vector_store.list_documents()``. Any document whose recorded
               ``embedding_provider``/``embedding_model`` differs from the active
               provider/model makes the whole store incompatible: raise
               :class:`ReembeddingRequiredError` listing all affected documents
               and return nothing (strict, all-or-nothing; R15.6, R15.7).
            4. **Embed the question** with the active provider (R6.1).
            5. **Query** the store for the top-k ranked chunks (R6.2) and return
               them unchanged — the store already ranks them in non-increasing
               score order (R6.4).

        Args:
            question: The natural-language question to search with.
            top_k: Optional per-query override for the number of chunks to
                return; ``None`` falls back to ``Config.top_k``.

        Returns:
            The ranked :class:`~src.models.RetrievedChunk` list (at most
            ``min(effective_top_k, store_size)`` items), each carrying its chunk
            metadata ``{document_name, page_number, section, chunk_text}`` and
            embedding provenance. Empty when the store is empty (R6.5).

        Raises:
            ReembeddingRequiredError: If any indexed document is incompatible
                with the active embedding space (R15.6, R15.7).
        """
        # 1. Resolve the effective top_k (per-query override, else Config).
        effective_top_k = top_k if top_k is not None else self._config.top_k

        # 2. Empty-store precedence: nothing to embed against, nothing to check.
        if self._vector_store.count() == 0:
            return []

        # 3. Strict, all-or-nothing embedding-space compatibility gating.
        active_provider = self._embedding_provider.provider_name
        active_model = self._embedding_provider.model_name

        documents = self._vector_store.list_documents()
        if not documents:  # Defensive: treated the same as an empty store.
            return []

        affected: list[DocumentRecord] = [
            record
            for record in documents
            if record.embedding_provider != active_provider
            or record.embedding_model != active_model
        ]
        if affected:
            raise ReembeddingRequiredError(
                [record.document_id for record in affected],
                self._build_incompatibility_message(
                    affected, active_provider, active_model
                ),
            )

        # 4. Embed the question with the active provider.
        query_vector = self._embedding_provider.embed_query(question)

        # 5. Query the store; it returns min(top_k, store_size) chunks already
        #    ranked in non-increasing score order. Return unchanged (no re-sort).
        return self._vector_store.query(query_vector, effective_top_k)

    @staticmethod
    def _build_incompatibility_message(
        affected: list[DocumentRecord],
        active_provider: str,
        active_model: str,
    ) -> str:
        """Build a human-readable, secret-free incompatibility message.

        The message names each affected document and shows its recorded
        provider/model versus the active provider/model, so a human can see why
        re-embedding is required. It contains no credential values.

        Args:
            affected: The incompatible document records.
            active_provider: The active embedding provider name.
            active_model: The active embedding model name.

        Returns:
            A single-line, human-readable summary of the mismatch.
        """
        parts = [
            f"'{record.document_name}' (id={record.document_id}) recorded "
            f"{record.embedding_provider}/{record.embedding_model}"
            for record in affected
        ]
        return (
            "Re-embedding required: the following document(s) were indexed with "
            "an embedding provider/model that differs from the active "
            f"{active_provider}/{active_model}: "
            + "; ".join(parts)
            + ". Re-embed them with the active provider/model before querying."
        )

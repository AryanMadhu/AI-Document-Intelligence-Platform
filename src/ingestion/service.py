"""Ingestion orchestration: limits, hashing, dedupe, provenance (R11, R5, R16).

This module implements the design's ``Ingestion_Service`` — the component that
turns raw PDF bytes into stored, embedded, provenance-tagged chunks. It wires
together the already-built pieces (:class:`~src.ingestion.pdf_parser.PDF_Parser`,
:class:`~src.ingestion.chunking.Chunking_Module`, the active
:class:`~src.embeddings.base.EmbeddingProvider`, and the
:class:`~src.vectorstore.VectorStore`) behind a single :meth:`Ingestion_Service.ingest`
entry point, and enforces the request-flow described in design's
"Ingestion (POST /documents/upload)" steps 1-8.

Orchestration order (design steps 1-8; enforced exactly by :meth:`ingest`):

1. **Size check (before parse).** Bytes larger than ``MAX_PDF_SIZE_MB`` are
   rejected with :class:`FileTooLargeError`; nothing is stored (R11.4-5).
2. **Hash + dedupe.** The SHA-256 ``document_hash`` of the bytes is computed
   (R11.6). If the store already contains it, the existing document's
   ``document_id`` is looked up and a ``status="duplicate"`` response is
   returned; no new id is generated and nothing is written (R11.6-8).
3. **Parse.** :class:`PDFParseError` from the parser propagates unchanged for
   invalid/corrupt PDFs (R3.3, R16.2).
4. **Page check (after parse).** ``page_count = len(pages)``; more than
   ``MAX_PDF_PAGES`` raises :class:`TooManyPagesError`; nothing stored (R11.5).
5. **Chunk.** The pages are split into page-scoped chunks.
6. **Empty check.** A valid PDF that yields zero chunks is rejected with
   :class:`EmptyDocumentError`; a zero-chunk document is never stored.
7. **Embed all, then store.** Every chunk's text is embedded in one batch
   *before* anything is written. If embedding fails, nothing is stored and
   previously-ingested documents are untouched (embed-then-store atomicity).
8. **Store.** A fresh UUID ``document_id`` is generated and all chunks + vectors
   are persisted in a single :meth:`VectorStore.add_chunks` call, tagged with
   provenance read from the *active* provider (never hard-coded, R5.5).

Security: every exception message is value-free (identifies the file and the
relevant limit/reason, never any secret or credential).
"""

from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timezone

from src.config import Config
from src.embeddings.base import (
    EmbeddingError,
    EmbeddingProvider,
    build_embedding_provider,
)
from src.ingestion.chunking import Chunking_Module
from src.ingestion.pdf_parser import PDF_Parser, PDFParseError  # noqa: F401 (re-export via package)
from src.models import UploadResponse
from src.vectorstore import VectorStore


# ---------------------------------------------------------------------------
# Typed service-level exceptions
# ---------------------------------------------------------------------------


class IngestionError(Exception):
    """Base class for all ingestion service-level errors.

    All subclasses carry value-free messages that identify the offending file
    and the relevant limit/reason, and never contain any secret (R16).
    """


class FileTooLargeError(IngestionError):
    """Raised when uploaded bytes exceed ``MAX_PDF_SIZE_MB`` (R11.4-5).

    Raised *before* parsing so an oversized file is never opened, and nothing
    is stored.

    Attributes:
        file: Filename of the rejected upload.
        limit_mb: The configured size limit in megabytes.
    """

    def __init__(self, file: str, limit_mb: int) -> None:
        """Initialize with the offending file and the size limit.

        Args:
            file: Filename of the rejected upload.
            limit_mb: The configured ``MAX_PDF_SIZE_MB`` limit.
        """
        self.file = file
        self.limit_mb = limit_mb
        super().__init__(
            f"File '{file}' exceeds the maximum PDF size of {limit_mb} MB."
        )


class TooManyPagesError(IngestionError):
    """Raised when a parsed PDF exceeds ``MAX_PDF_PAGES`` (R11.4-5).

    Raised *after* parsing (page count is known then) but *before* any storage,
    so nothing is stored.

    Attributes:
        file: Filename of the rejected upload.
        page_count: The document's actual page count.
        limit_pages: The configured page limit.
    """

    def __init__(self, file: str, page_count: int, limit_pages: int) -> None:
        """Initialize with the offending file, its page count, and the limit.

        Args:
            file: Filename of the rejected upload.
            page_count: The document's actual page count.
            limit_pages: The configured ``MAX_PDF_PAGES`` limit.
        """
        self.file = file
        self.page_count = page_count
        self.limit_pages = limit_pages
        super().__init__(
            f"File '{file}' has {page_count} pages, exceeding the maximum of "
            f"{limit_pages} pages."
        )


class EmptyDocumentError(IngestionError):
    """Raised when a valid PDF yields zero chunks (decision D2).

    A PDF that parses successfully but produces no extractable text/chunks (for
    example, all pages are blank) is rejected here rather than stored as a
    zero-chunk document.

    Attributes:
        file: Filename of the rejected upload.
    """

    def __init__(self, file: str) -> None:
        """Initialize with the offending file.

        Args:
            file: Filename of the rejected upload.
        """
        self.file = file
        super().__init__(
            f"File '{file}' contains no extractable text; no chunks were found."
        )


# ---------------------------------------------------------------------------
# Module-level helper
# ---------------------------------------------------------------------------


def compute_document_hash(file_bytes: bytes) -> str:
    """Return the SHA-256 hex digest of ``file_bytes`` (the Document_Hash).

    This is the ``document_hash`` used *only* for duplicate detection (R11.6);
    it is never used to address a document.

    Args:
        file_bytes: The raw file bytes to hash.

    Returns:
        The lowercase hexadecimal SHA-256 digest of ``file_bytes``.
    """
    return hashlib.sha256(file_bytes).hexdigest()


# ---------------------------------------------------------------------------
# Ingestion_Service
# ---------------------------------------------------------------------------


class Ingestion_Service:
    """Orchestrate PDF ingestion end to end (design: Ingestion request flow).

    The service enforces size/page limits, computes the SHA-256
    ``document_hash`` for dedupe, parses, chunks, embeds via the active
    :class:`~src.embeddings.base.EmbeddingProvider`, and stores chunks + vectors
    in the :class:`~src.vectorstore.VectorStore`, tagging every stored unit with
    provenance (provider/model) read from the active provider.

    Collaborators are injectable for testing; sensible defaults are built when
    not supplied. Size/page limits are read from :class:`~src.config.Config`
    (never hard-coded in the orchestration logic).
    """

    def __init__(
        self,
        config: Config,
        vector_store: VectorStore,
        embedding_provider: EmbeddingProvider | None = None,
        pdf_parser: PDF_Parser | None = None,
        chunking_module: Chunking_Module | None = None,
    ) -> None:
        """Store collaborators and read the configured limits.

        Args:
            config: Active configuration. Supplies ``max_pdf_size_mb`` and
                ``max_pdf_pages`` (and is passed to a default
                :class:`Chunking_Module`).
            vector_store: The store chunks and vectors are persisted to /
                deduped against.
            embedding_provider: The active embedding provider. When ``None`` it
                is built via :func:`build_embedding_provider`.
            pdf_parser: PDF parser; defaults to a fresh :class:`PDF_Parser`.
            chunking_module: Chunker; defaults to a fresh
                :class:`Chunking_Module` bound to ``config``.
        """
        self._config = config
        self._vector_store = vector_store
        self._embedding_provider = (
            embedding_provider
            if embedding_provider is not None
            else build_embedding_provider(config)
        )
        self._pdf_parser = pdf_parser if pdf_parser is not None else PDF_Parser()
        self._chunking_module = (
            chunking_module
            if chunking_module is not None
            else Chunking_Module(config=config)
        )

        # Read limits from config — never hard-code 5/50 in the logic.
        self._max_pdf_size_mb = config.max_pdf_size_mb
        self._max_pdf_pages = config.max_pdf_pages

    def ingest(self, file_bytes: bytes, filename: str) -> UploadResponse:
        """Ingest one PDF, or report it as a duplicate (design steps 1-8).

        The steps run in exactly the order below; each rejection happens before
        any write, and embedding happens before any store, so a failed ingest
        never leaves a partial write and never disturbs previously-ingested
        documents.

        Args:
            file_bytes: The raw PDF bytes to ingest.
            filename: The source filename (used for identity and error
                messages).

        Returns:
            An :class:`~src.models.UploadResponse`. ``status="ingested"`` for a
            newly-stored document (carrying its fresh ``document_id``), or
            ``status="duplicate"`` referencing the existing document's
            ``document_id`` when the exact bytes were already ingested.

        Raises:
            FileTooLargeError: If the bytes exceed ``MAX_PDF_SIZE_MB``.
            PDFParseError: If the bytes are not a valid/parseable PDF (R3.3);
                propagated unchanged from the parser.
            TooManyPagesError: If the parsed page count exceeds
                ``MAX_PDF_PAGES``.
            EmptyDocumentError: If a valid PDF yields zero chunks (decision D2).
            EmbeddingError: If embedding fails; nothing is stored (propagated
                with the filename identified via chaining).
        """
        # 1. SIZE CHECK (before parse). Nothing stored on rejection.
        max_bytes = self._max_pdf_size_mb * 1024 * 1024
        if len(file_bytes) > max_bytes:
            raise FileTooLargeError(filename, self._max_pdf_size_mb)

        # 2. HASH + DEDUPE. On a hash match, return the existing document's id;
        # write nothing and generate no new id (R11.6-8).
        document_hash = compute_document_hash(file_bytes)
        if self._vector_store.document_exists(document_hash):
            existing = self._find_existing_record(document_hash)
            if existing is not None:
                return UploadResponse(
                    document_id=existing.document_id,
                    document_name=filename,
                    status="duplicate",
                    page_count=existing.page_count,
                    chunk_count=existing.chunk_count,
                    message="Document already ingested.",
                )
            # Defensive: document_exists said True but no matching record was
            # found. Fall through and ingest as a new document rather than fail.

        # 3. PARSE. Invalid PDF -> PDFParseError propagates unchanged (R3.3).
        pages = self._pdf_parser.parse(file_bytes, filename)

        # 4. PAGE CHECK (after parse). page_count = len(pages) (decision B).
        page_count = len(pages)
        if page_count > self._max_pdf_pages:
            raise TooManyPagesError(filename, page_count, self._max_pdf_pages)

        # 5. CHUNK.
        chunks = self._chunking_module.chunk_pages(pages, document_name=filename)

        # 6. EMPTY CHECK (decision D2). Never store a zero-chunk document.
        if not chunks:
            raise EmptyDocumentError(filename)

        # 7. EMBED ALL (before any store). On failure, nothing has been written
        # yet, so previously-ingested documents remain untouched (atomicity).
        try:
            embeddings = self._embedding_provider.embed_texts(
                [chunk.chunk_text for chunk in chunks]
            )
        except EmbeddingError as exc:
            # Re-raise naming the file for identification (R5.3/R16.3); chained
            # and value-free. Nothing was stored.
            raise EmbeddingError(
                f"Embedding failed for document '{filename}'; nothing was stored."
            ) from exc

        # 8. STORE. Fresh document_id; provenance from the ACTIVE provider.
        document_id = str(uuid.uuid4())
        self._vector_store.add_chunks(
            document_id,
            chunks,
            embeddings,
            document_hash=document_hash,
            embedding_provider=self._embedding_provider.provider_name,
            embedding_model=self._embedding_provider.model_name,
            page_count=page_count,
            ingested_at=datetime.now(timezone.utc),
        )

        # 9. RETURN.
        return UploadResponse(
            document_id=document_id,
            document_name=filename,
            status="ingested",
            page_count=page_count,
            chunk_count=len(chunks),
            message="Ingested successfully.",
        )

    def _find_existing_record(self, document_hash: str):
        """Find the stored document whose ``document_hash`` matches (decision A1).

        Rather than adding a new :class:`VectorStore` method, this lists all
        documents and matches on ``document_hash`` to recover the existing
        ``document_id`` (Task 6 stays unchanged).

        Args:
            document_hash: The SHA-256 hash to match on.

        Returns:
            The matching :class:`~src.models.DocumentRecord`, or ``None`` if no
            stored document carries that hash.
        """
        for record in self._vector_store.list_documents():
            if record.document_hash == document_hash:
                return record
        return None

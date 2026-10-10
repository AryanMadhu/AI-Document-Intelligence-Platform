"""FastAPI backend and error-to-HTTP mapping (Task 14; R10, R11, R12, R15, R16).

This module wires the already-built services (Task 9 ingestion, Task 10
retrieval, Task 12 query) behind a thin HTTP surface. It is deliberately
*import-safe* (decision D7): importing ``src.api.app`` — and even calling
:func:`create_app` — constructs **nothing heavy**. No ``VectorStore`` is opened,
no embedding/LLM provider is built, no model is downloaded, no network call is
made, and no Chroma directory is created at import or app-construction time.

How import-safety is guaranteed:
    * ``create_app`` only *registers* routes and exception handlers; it never
      calls any provider factory or service constructor.
    * Every service/provider is produced by a module-level, ``functools.lru_cache``
      wrapped builder (``_get_*``). Those builders run **on first request**, when
      FastAPI resolves the route's ``Depends(...)`` provider, and the result is
      cached (a process-wide singleton) for every subsequent request.
    * The thin ``*_dep`` callables are what the routes depend on and what tests
      override via ``app.dependency_overrides`` to inject fakes/stubs.

Design decisions implemented here:
    * D1  — a successful upload returns HTTP 200.
    * D2  — ``TOO_MANY_PAGES`` -> 413, ``EMPTY_DOCUMENT`` -> 422.
    * D3  — ``QueryRequest.top_k`` is accepted but intentionally **not** wired
      into ``Query_Service`` in this task; the ``/query`` route calls
      ``service.answer(question)`` with the question only (see the route).
    * D4  — a custom ``RequestValidationError`` handler returns the project's
      ``ErrorResponse`` envelope (422, ``VALIDATION_ERROR``).
    * D5  — a *missing* ``question`` field is a validation error (422); an
      *empty/whitespace* question value is owned by ``Query_Service`` which
      raises ``EmptyQuestionError`` -> 400 ``EMPTY_QUESTION``. The route never
      re-checks whitespace.
    * D7  — lazy, cached DI (above).
    * D8  — a single ``src/api/app.py`` module (no separate routers/errors).
    * D10 — no ``/retrieve`` HTTP endpoint.

Error handling (Task 14.2): there is **no per-route try/except**. Every domain,
framework, and unexpected error is translated to an ``ErrorResponse`` body by an
app-level exception handler. The three categories are kept separate:
    * Category A (domain) — one handler per typed service exception, each with
      its own mapped status + stable error code.
    * Category B (framework) — ``RequestValidationError`` -> a generic 422.
    * Category C (unexpected) — a catch-all ``Exception`` handler -> a generic
      500 that never leaks a traceback, path, secret, or ``repr``.
"""

from __future__ import annotations

from functools import lru_cache

from fastapi import Depends, FastAPI, File, Request, UploadFile, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from src.config import Config, MissingCredentialError
from src.embeddings.base import EmbeddingError, build_embedding_provider
from src.ingestion import (
    DocumentNotFoundError,
    EmptyDocumentError,
    FileTooLargeError,
    Ingestion_Service,
    PDFParseError,
    TooManyPagesError,
)
from src.llm.base import LLMError, build_llm_provider
from src.models import (
    DocumentListResponse,
    ErrorResponse,
    QueryRequest,
    QueryResponse,
    ReembedResponse,
    UploadResponse,
)
from src.query import EmptyQuestionError, Query_Service
from src.retrieval import ReembeddingRequiredError, Retrieval_Service
from src.vectorstore import VectorStore


# Status-code constants. Prefer the current (non-deprecated) names and fall back
# to the historical ones so the module works across Starlette versions without
# emitting deprecation warnings on newer ones. The numeric values are identical
# (413 and 422); only the constant names changed.
_HTTP_413_CONTENT_TOO_LARGE = getattr(
    status, "HTTP_413_CONTENT_TOO_LARGE", 413
)
_HTTP_422_UNPROCESSABLE = getattr(
    status, "HTTP_422_UNPROCESSABLE_CONTENT", 422
)


# ---------------------------------------------------------------------------
# Lazy, cached builders (decision D7).
#
# None of these run at import time. Each is wrapped in ``functools.lru_cache``
# so the first request through the corresponding ``Depends`` provider builds the
# object exactly once and every later request reuses the cached singleton. This
# is what keeps ``import src.api.app`` and ``create_app()`` free of any model
# download / network / Chroma directory creation.
# ---------------------------------------------------------------------------


@lru_cache
def _get_config() -> Config:
    """Build (once) and cache the active configuration from the environment."""
    return Config.from_env()


@lru_cache
def _get_vector_store() -> VectorStore:
    """Open (once) and cache the persistent vector store (default chroma_db path)."""
    return VectorStore()


@lru_cache
def _get_embedding_provider():
    """Build (once) and cache the active embedding provider from config."""
    return build_embedding_provider(_get_config())


@lru_cache
def _get_ingestion_service() -> Ingestion_Service:
    """Build (once) and cache the ingestion service and its collaborators."""
    return Ingestion_Service(
        config=_get_config(),
        vector_store=_get_vector_store(),
        embedding_provider=_get_embedding_provider(),
    )


@lru_cache
def _get_retrieval_service() -> Retrieval_Service:
    """Build (once) and cache the retrieval service and its collaborators."""
    return Retrieval_Service(
        _get_config(),
        _get_vector_store(),
        _get_embedding_provider(),
    )


@lru_cache
def _get_query_service() -> Query_Service:
    """Build (once) and cache the query service and its collaborators."""
    return Query_Service(
        _get_config(),
        retrieval_service=_get_retrieval_service(),
        llm_provider=build_llm_provider(_get_config()),
    )


# ---------------------------------------------------------------------------
# Thin dependency callables.
#
# The routes depend on THESE functions (not the ``_get_*`` builders directly),
# so tests can swap in fakes via ``app.dependency_overrides[<dep>] = ...``
# without touching the cached singletons.
# ---------------------------------------------------------------------------


def config_dep() -> Config:
    """Dependency: return the cached active configuration."""
    return _get_config()


def vector_store_dep() -> VectorStore:
    """Dependency: return the cached vector store."""
    return _get_vector_store()


def ingestion_service_dep() -> Ingestion_Service:
    """Dependency: return the cached ingestion service."""
    return _get_ingestion_service()


def query_service_dep() -> Query_Service:
    """Dependency: return the cached query service."""
    return _get_query_service()


# ---------------------------------------------------------------------------
# Error-to-HTTP mapping helper (Task 14.2).
# ---------------------------------------------------------------------------


def _error_response(error_code: str, detail: str, status_code: int) -> JSONResponse:
    """Build a ``JSONResponse`` carrying the ``ErrorResponse`` envelope.

    Every error path in the app funnels through this single helper so all error
    bodies share the exact ``{"error": ..., "detail": ...}`` shape. ``detail``
    is always a pre-vetted, credential-safe message chosen by the caller — raw
    exception text is never passed through unless it is already known to be
    value-free.

    Args:
        error_code: Stable machine-readable code, e.g. ``"EMPTY_QUESTION"``.
        detail: Human-readable, credential-safe message.
        status_code: The HTTP status to return.

    Returns:
        A ``JSONResponse`` with the serialized ``ErrorResponse`` body.
    """
    body = ErrorResponse(error=error_code, detail=detail).model_dump()
    return JSONResponse(status_code=status_code, content=body)


def _register_exception_handlers(app: FastAPI) -> None:
    """Register all app-level exception handlers (the three categories).

    No route contains a try/except; translating an exception to an HTTP
    response is done exclusively here. The handlers are grouped into:

    * **Category A — domain exceptions** (one handler each). Each maps a typed
      service exception to its design-mandated status + stable error code, with
      a safe ``detail`` built from value-free attributes (file names, limits,
      document ids) — never from ``repr`` or raw provider payloads.
    * **Category B — framework validation**: ``RequestValidationError`` -> a
      generic 422 ``VALIDATION_ERROR`` (the raw pydantic error tree, which can
      contain internal paths, is never dumped into ``detail``).
    * **Category C — the unexpected catch-all**: any other ``Exception`` -> a
      generic 500 ``INTERNAL_ERROR`` that leaks nothing.
    """

    # ----- Category A: domain exceptions --------------------------------

    @app.exception_handler(EmptyQuestionError)
    async def _handle_empty_question(request: Request, exc: EmptyQuestionError):
        # D5: an empty/whitespace question VALUE is owned by Query_Service.
        return _error_response(
            "EMPTY_QUESTION",
            "A question is required.",
            status.HTTP_400_BAD_REQUEST,
        )

    @app.exception_handler(PDFParseError)
    async def _handle_pdf_parse(request: Request, exc: PDFParseError):
        # exc.file is a filename (safe addressing handle), not a secret.
        file_name = getattr(exc, "file", None)
        detail = "The file could not be parsed as a valid PDF."
        if file_name:
            detail += f" ({file_name})"
        return _error_response(
            "INVALID_PDF", detail, status.HTTP_400_BAD_REQUEST
        )

    @app.exception_handler(FileTooLargeError)
    async def _handle_file_too_large(request: Request, exc: FileTooLargeError):
        return _error_response(
            "FILE_TOO_LARGE",
            f"File exceeds the maximum size of {exc.limit_mb} MB.",
            _HTTP_413_CONTENT_TOO_LARGE,
        )

    @app.exception_handler(TooManyPagesError)
    async def _handle_too_many_pages(request: Request, exc: TooManyPagesError):
        # D2: TOO_MANY_PAGES -> 413.
        return _error_response(
            "TOO_MANY_PAGES",
            f"File exceeds the maximum of {exc.limit_pages} pages.",
            _HTTP_413_CONTENT_TOO_LARGE,
        )

    @app.exception_handler(EmptyDocumentError)
    async def _handle_empty_document(request: Request, exc: EmptyDocumentError):
        # D2: EMPTY_DOCUMENT -> 422.
        return _error_response(
            "EMPTY_DOCUMENT",
            "The document contains no extractable text.",
            _HTTP_422_UNPROCESSABLE,
        )

    @app.exception_handler(ReembeddingRequiredError)
    async def _handle_reembedding_required(
        request: Request, exc: ReembeddingRequiredError
    ):
        # ErrorResponse has only {error, detail}; the affected document ids are
        # safe addressing handles, so they go into the detail string (no new
        # Task-3 model fields are added).
        ids = ", ".join(exc.document_ids)
        return _error_response(
            "REEMBEDDING_REQUIRED",
            "Re-embedding is required before querying the following "
            f"document(s): {ids}",
            status.HTTP_409_CONFLICT,
        )

    @app.exception_handler(LLMError)
    async def _handle_llm_error(request: Request, exc: LLMError):
        # Never echo exc text — provider payloads may be sensitive.
        return _error_response(
            "LLM_FAILED",
            "Answer generation failed.",
            status.HTTP_502_BAD_GATEWAY,
        )

    @app.exception_handler(EmbeddingError)
    async def _handle_embedding_error(request: Request, exc: EmbeddingError):
        # Never echo exc text.
        return _error_response(
            "EMBEDDING_FAILED",
            "Embedding generation failed.",
            status.HTTP_502_BAD_GATEWAY,
        )

    @app.exception_handler(MissingCredentialError)
    async def _handle_missing_credential(
        request: Request, exc: MissingCredentialError
    ):
        # NOTE (reachability): this handler is currently UNREACHABLE in the MVP.
        # No request path lets a MissingCredentialError propagate to the API:
        #   * The Gemini provider (src/llm/gemini_provider.py `_get_client`)
        #     catches a missing GEMINI_API_KEY from `config.require_credential`
        #     and RE-RAISES it as LLMError(str(exc)) — so a missing Gemini key
        #     surfaces as LLMError -> 502 LLM_FAILED, not here.
        #   * The MVP embedding provider (sentence-transformers) needs no
        #     credential and never calls `require_credential`.
        # The handler is kept registered for a FUTURE provider that lets a
        # MissingCredentialError propagate unchanged. str(exc) is value-free /
        # name-only per Task 2 (Config's message names the key, never its value)
        # — safe to include so the response identifies the missing key by name.
        return _error_response(
            "MISSING_CREDENTIAL",
            str(exc),
            status.HTTP_500_INTERNAL_SERVER_ERROR,
        )

    @app.exception_handler(DocumentNotFoundError)
    async def _handle_document_not_found(
        request: Request, exc: DocumentNotFoundError
    ):
        return _error_response(
            "DOCUMENT_NOT_FOUND",
            "No document found with the given id.",
            status.HTTP_404_NOT_FOUND,
        )

    # ----- Category B: framework validation (D4) ------------------------

    @app.exception_handler(RequestValidationError)
    async def _handle_validation_error(
        request: Request, exc: RequestValidationError
    ):
        # Keep the detail generic/safe: do NOT dump the raw pydantic error tree
        # (it can contain internal field paths). A missing ``question`` field
        # (D5) lands here as a 422.
        return _error_response(
            "VALIDATION_ERROR",
            "The request was invalid.",
            _HTTP_422_UNPROCESSABLE,
        )

    # ----- Category C: unexpected catch-all -----------------------------

    @app.exception_handler(Exception)
    async def _handle_unexpected(request: Request, exc: Exception):
        # NEVER include a traceback, file path, env var, secret, provider/chroma
        # internal, or exc repr. A single opaque, safe message.
        return _error_response(
            "INTERNAL_ERROR",
            "An unexpected error occurred.",
            status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


# ---------------------------------------------------------------------------
# Upload handler (extracted so the size guard — including the None-size
# bounded-read fallback — is unit-testable without a TestClient).
# ---------------------------------------------------------------------------


async def _handle_upload(
    file: UploadFile,
    service: Ingestion_Service,
    config: Config,
) -> UploadResponse:
    """Run the upload size guard, then delegate to the ingestion service.

    A size guard runs **before** the body is handed to the service:

    * **Known size** — when the client reports ``file.size`` (TestClient and
      most real clients send a ``Content-Length``), reject immediately with a
      :class:`FileTooLargeError` (mapped to 413) if it exceeds the limit,
      *without* reading the whole upload; otherwise read the full body.
    * **Unknown size (fallback)** — when ``file.size`` is ``None`` (a missing or
      unreliable ``Content-Length``), the body is **never** pulled into memory
      unbounded: at most ``max_bytes + 1`` bytes are read and the request is
      rejected if that bounded read exceeds the limit. The ``+ 1`` lets a body
      exactly at the limit through while still detecting the first over-limit
      byte.

    Task 9's ``len(file_bytes)`` check remains the authoritative backstop. This
    helper holds no other business logic; all errors bubble to the app-level
    handlers.

    Args:
        file: The uploaded file.
        service: The ingestion service to delegate to once the guard passes.
        config: The active configuration (supplies ``max_pdf_size_mb``).

    Returns:
        The :class:`UploadResponse` produced by the ingestion service.

    Raises:
        FileTooLargeError: If the upload exceeds the configured size limit (via
            either the known-size pre-check or the unknown-size bounded read).
    """
    max_bytes = config.max_pdf_size_mb * 1024 * 1024
    if file.size is not None:
        if file.size > max_bytes:
            raise FileTooLargeError(file.filename or "upload", config.max_pdf_size_mb)
        data = await file.read()
    else:
        # size unknown (missing/unreliable Content-Length): never pull an unbounded
        # body into memory — read at most max_bytes + 1 bytes and reject if over.
        data = await file.read(max_bytes + 1)
        if len(data) > max_bytes:
            raise FileTooLargeError(file.filename or "upload", config.max_pdf_size_mb)
    filename = file.filename or "upload"
    return service.ingest(data, filename)


# ---------------------------------------------------------------------------
# App factory (decision D8).
# ---------------------------------------------------------------------------


def create_app() -> FastAPI:
    """Build and return the FastAPI app with routes + exception handlers.

    ``create_app`` performs **wiring only** (decision D7): it registers the
    routes and the exception handlers and returns the app. It constructs no
    service, provider, or vector store — those are built lazily on first request
    through the cached ``Depends`` providers. This keeps both importing the
    module and calling this factory free of any model/network/Chroma side
    effect, which the import-safety test asserts.

    Returns:
        A fully-wired :class:`fastapi.FastAPI` application.
    """
    app = FastAPI(title="AI Document Intelligence Platform")

    _register_exception_handlers(app)

    @app.post(
        "/documents/upload",
        response_model=UploadResponse,
        status_code=status.HTTP_200_OK,
    )
    async def upload_document(
        file: UploadFile = File(...),
        service: Ingestion_Service = Depends(ingestion_service_dep),
        config: Config = Depends(config_dep),
    ) -> UploadResponse:
        """Ingest an uploaded PDF (D1: success is 200; duplicate is 200).

        A size guard runs **before** the body is handed to the service (see
        :func:`_handle_upload`): when the client reports ``file.size`` and it
        exceeds the limit, a :class:`FileTooLargeError` is raised (mapped to
        413) *without* reading the whole upload; when ``file.size`` is ``None``
        (missing/unreliable ``Content-Length``) a **bounded** read of at most
        ``max_bytes + 1`` bytes is used so an unbounded body is never pulled
        into memory. Task 9's ``len(file_bytes)`` check remains the
        defense-in-depth backstop. The route holds no business logic beyond
        delegating to the handler; all errors bubble to the app-level handlers.
        """
        return await _handle_upload(file, service, config)

    @app.get("/documents", response_model=DocumentListResponse)
    async def list_documents(
        store: VectorStore = Depends(vector_store_dep),
    ) -> DocumentListResponse:
        """List all ingested documents (empty list when none)."""
        return DocumentListResponse(documents=store.list_documents())

    @app.post("/query", response_model=QueryResponse)
    async def query(
        request: QueryRequest,
        service: Query_Service = Depends(query_service_dep),
    ) -> QueryResponse:
        """Answer a grounded question (D5: empty-question handling is the service's).

        D3 limitation: ``request.top_k`` is accepted on the request model but is
        intentionally **not** wired into ``Query_Service`` in this task. The
        service is called with the question ONLY; any ``top_k`` sent by the
        client is accepted-but-unused and does not change behavior. (Wiring a
        per-query ``top_k`` into the query pipeline is deferred.)
        """
        # NOTE (D3): deliberately NOT passing request.top_k — accepted-but-unused.
        return service.answer(request.question)

    @app.post(
        "/documents/{document_id}/reembed",
        response_model=ReembedResponse,
        status_code=status.HTTP_200_OK,
    )
    async def reembed_document(
        document_id: str,
        service: Ingestion_Service = Depends(ingestion_service_dep),
    ) -> ReembedResponse:
        """Re-embed a stored document addressed by its stable ``document_id``."""
        return service.reembed_document(document_id)

    return app


# Module-level, uvicorn-compatible app. Building it constructs nothing heavy
# (create_app only wires routes/handlers), so ``uvicorn src.api.app:app`` and
# ``import src.api.app`` are both import-safe (D7).
app = create_app()

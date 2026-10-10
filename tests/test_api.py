"""API tests for the FastAPI backend (Task 14; R10, R11, R12, R15, R16).

All tests are deterministic and fully offline. The HTTP endpoints are exercised
with :class:`fastapi.testclient.TestClient`, and every service/provider is
replaced via ``app.dependency_overrides`` so no real embedding model, LLM,
network, or persistent Chroma store is ever touched.

Two layers of coverage:

* **Unit-level (fakes)** — the bulk of the suite overrides the thin dependency
  callables (``ingestion_service_dep``, ``query_service_dep``,
  ``vector_store_dep``, ``config_dep``) with hand-built fakes that return (or
  raise) exactly the Task-3 models / typed service exceptions needed to drive
  each endpoint and each error-to-HTTP mapping. This isolates the API wiring
  and the exception handlers from the heavy services.
* **Import-safety** — asserts that importing ``src.api.app`` and calling
  ``create_app()`` constructs no provider / store (decision D7), by monkeypatching
  the provider/store builders to blow up if called.
* **Integration (14.3)** — builds the app with REAL ``Ingestion_Service`` +
  ``Retrieval_Service`` + ``Query_Service`` over a REAL temp ``VectorStore``
  (``tmp_path``), but with STUB embedding + STUB LLM providers, ingests a small
  real in-memory ``fitz`` PDF through ``POST /documents/upload``, then
  ``POST /query`` and asserts every citation traces back to an ingested chunk.
"""

from __future__ import annotations

import asyncio
import dataclasses
import os
from datetime import datetime, timezone

import fitz  # PyMuPDF — used only to build a tiny real PDF for the integration test
import pytest
from fastapi.testclient import TestClient

# Workspace root (this file lives in tests/); used to run the import-safety
# subprocess from a directory where ``src`` is importable.
_WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import src.api.app as app_module
from src.api.app import (
    _handle_upload,
    config_dep,
    create_app,
    ingestion_service_dep,
    query_service_dep,
    vector_store_dep,
)
from src.config import Config, MissingCredentialError
from src.embeddings.base import EmbeddingError, EmbeddingProvider
from src.ingestion import (
    EmptyDocumentError,
    FileTooLargeError,
    Ingestion_Service,
    PDFParseError,
    TooManyPagesError,
)
from src.ingestion.service import DocumentNotFoundError
from src.llm.base import LLMError, LLMProvider
from src.llm.gemini_provider import GeminiLLMProvider
from src.models import (
    Chunk,
    Citation,
    DocumentRecord,
    QueryResponse,
    ReembedResponse,
    RetrievedChunk,
    UploadResponse,
)
from src.query import NO_ANSWER_RESPONSE, EmptyQuestionError, Query_Service
from src.retrieval import ReembeddingRequiredError, Retrieval_Service
from src.vectorstore import VectorStore


# ===========================================================================
# Autouse guard: no REAL _get_* builder may run during a test (D7 safety net)
# ===========================================================================


# The six real, lru_cache-wrapped builders on the app module. If any of them
# ran during a test, something bypassed the dependency overrides and a REAL
# provider / store / service (or a real credential read) was constructed — which
# could download a model, hit the network, or create a persistent chroma_db/.
_REAL_BUILDER_NAMES = (
    "_get_config",
    "_get_vector_store",
    "_get_embedding_provider",
    "_get_ingestion_service",
    "_get_retrieval_service",
    "_get_query_service",
)


@pytest.fixture(autouse=True)
def _guard_no_real_builder_runs(monkeypatch):
    """Fail any test in which a REAL ``_get_*`` builder is invoked.

    Every test in this module injects fakes/stubs via
    ``app.dependency_overrides`` (which bypass the ``_get_*`` builders entirely)
    or monkeypatches the builders itself (the import-safety tests). So in a
    correct test NO real builder should ever run.

    Why *record-and-delegate* rather than *raise*: a raise inside request
    handling would be swallowed by the catch-all ``Exception`` handler and
    turned into an HTTP 500, so the request might still "succeed" while a real
    builder secretly ran. Recording the invocation and asserting at teardown
    detects a stray real-builder call regardless of how the error would have
    been handled.

    Each builder's ``lru_cache`` is cleared at setup AND teardown so a cached
    singleton from a prior test can neither mask a call here nor leak into the
    next test. The caches are cleared on the ORIGINAL ``lru_cache`` objects
    (captured before wrapping), because ``monkeypatch.setattr`` replaces the
    attribute with a plain wrapper that has no ``cache_clear``.

    Tests that intentionally replace the builders (the in-process import-safety
    test) override this wrapper with their own; their replacement wins and the
    recorded-invocations list stays empty, so no false teardown failure occurs.
    """
    invoked: list[str] = []
    originals = {}

    for name in _REAL_BUILDER_NAMES:
        original = getattr(app_module, name)
        originals[name] = original
        # Clear the real cache BEFORE wrapping (wrapper has no cache_clear).
        original.cache_clear()

        def _make_wrapper(builder_name: str, builder):
            def _wrapper(*args, **kwargs):
                invoked.append(builder_name)
                return builder(*args, **kwargs)

            return _wrapper

        monkeypatch.setattr(app_module, name, _make_wrapper(name, original))

    try:
        yield
    finally:
        # monkeypatch restores the originals automatically on teardown; clear
        # the real caches again so nothing leaks into the next test.
        for original in originals.values():
            original.cache_clear()
        assert invoked == [], (
            "a real _get_* builder ran during the test "
            f"(invoked: {invoked}); dependency overrides should bypass them"
        )


# ===========================================================================
# Fakes (no real model / gemini / network / persistent chroma)
# ===========================================================================


class _FakeIngestionService:
    """Fake ``Ingestion_Service`` whose ``ingest``/``reembed_document`` are scripted.

    ``ingest`` and ``reembed_document`` either return a preset response or raise
    a preset exception. ``ingest_called`` records whether ``ingest`` ran — the
    oversized-upload test uses it to prove the route's size guard rejected the
    request *before* delegating to the service.
    """

    def __init__(
        self,
        *,
        ingest_result: UploadResponse | None = None,
        ingest_error: Exception | None = None,
        reembed_result: ReembedResponse | None = None,
        reembed_error: Exception | None = None,
    ) -> None:
        self._ingest_result = ingest_result
        self._ingest_error = ingest_error
        self._reembed_result = reembed_result
        self._reembed_error = reembed_error
        self.ingest_called = False
        self.last_ingest_args: tuple[bytes, str] | None = None

    def ingest(self, data: bytes, filename: str) -> UploadResponse:
        self.ingest_called = True
        self.last_ingest_args = (data, filename)
        if self._ingest_error is not None:
            raise self._ingest_error
        assert self._ingest_result is not None
        return self._ingest_result

    def reembed_document(self, document_id: str) -> ReembedResponse:
        if self._reembed_error is not None:
            raise self._reembed_error
        assert self._reembed_result is not None
        return self._reembed_result


class _FakeQueryService:
    """Fake ``Query_Service`` whose ``answer`` is scripted and records its args.

    ``questions`` records every question passed to ``answer`` so a test can prove
    the ``/query`` route called the service with the question ONLY (D3 — no
    ``top_k`` is forwarded).
    """

    def __init__(
        self,
        *,
        result: QueryResponse | None = None,
        error: Exception | None = None,
    ) -> None:
        self._result = result
        self._error = error
        self.questions: list[str] = []

    def answer(self, question: str) -> QueryResponse:
        self.questions.append(question)
        if self._error is not None:
            raise self._error
        assert self._result is not None
        return self._result


class _FakeVectorStore:
    """Fake vector store exposing just ``list_documents`` for GET /documents."""

    def __init__(self, documents: list[DocumentRecord]) -> None:
        self._documents = documents

    def list_documents(self) -> list[DocumentRecord]:
        return list(self._documents)


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------


def _upload_response(status_value: str = "ingested") -> UploadResponse:
    return UploadResponse(
        document_id="doc-123",
        document_name="sample.pdf",
        status=status_value,
        page_count=3,
        chunk_count=5,
        message="ok",
    )


def _reembed_response() -> ReembedResponse:
    return ReembedResponse(
        document_id="doc-123",
        status="reembedded",
        embedding_provider="sentence-transformers",
        embedding_model="all-MiniLM-L6-v2",
        chunk_count=5,
    )


def _document_record(document_id: str, name: str) -> DocumentRecord:
    return DocumentRecord(
        document_id=document_id,
        document_name=name,
        document_hash=f"hash-{document_id}",
        page_count=2,
        chunk_count=4,
        embedding_provider="sentence-transformers",
        embedding_model="all-MiniLM-L6-v2",
        ingested_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
    )


# ---------------------------------------------------------------------------
# App / client builders
# ---------------------------------------------------------------------------


def _client_with_overrides(**overrides) -> TestClient:
    """Build a fresh app and apply the given ``dependency_overrides``.

    Args:
        **overrides: Mapping of dependency callable name -> replacement factory,
            using the keyword keys ``ingestion``, ``query``, ``store``,
            ``config``.

    Returns:
        A ``TestClient`` wrapping the overridden app.
    """
    app = create_app()
    if "ingestion" in overrides:
        app.dependency_overrides[ingestion_service_dep] = overrides["ingestion"]
    if "query" in overrides:
        app.dependency_overrides[query_service_dep] = overrides["query"]
    if "store" in overrides:
        app.dependency_overrides[vector_store_dep] = overrides["store"]
    if "config" in overrides:
        app.dependency_overrides[config_dep] = overrides["config"]
    return TestClient(app)


def _tiny_pdf_bytes() -> bytes:
    """Return a tiny, grounded, in-memory PDF (two pages with known text)."""
    doc = fitz.open()
    try:
        page1 = doc.new_page()
        page1.insert_text(
            (72, 72),
            "The capital of France is Paris. Paris sits on the Seine river.",
        )
        page2 = doc.new_page()
        page2.insert_text(
            (72, 72),
            "The Eiffel Tower is located in Paris and was completed in 1889.",
        )
        return doc.tobytes()
    finally:
        doc.close()


# ===========================================================================
# Success paths
# ===========================================================================


def test_upload_success_returns_200_with_document_id() -> None:
    """A successful ingest returns 200 (D1) with the document_id in the body."""
    fake = _FakeIngestionService(ingest_result=_upload_response("ingested"))
    client = _client_with_overrides(ingestion=lambda: fake, config=lambda: Config())

    resp = client.post(
        "/documents/upload",
        files={"file": ("sample.pdf", b"%PDF-fake-bytes", "application/pdf")},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["document_id"] == "doc-123"
    assert body["status"] == "ingested"
    assert fake.ingest_called is True


def test_upload_duplicate_returns_200() -> None:
    """A duplicate upload returns 200 with status='duplicate' (no special path)."""
    fake = _FakeIngestionService(ingest_result=_upload_response("duplicate"))
    client = _client_with_overrides(ingestion=lambda: fake, config=lambda: Config())

    resp = client.post(
        "/documents/upload",
        files={"file": ("sample.pdf", b"%PDF-dup", "application/pdf")},
    )

    assert resp.status_code == 200
    assert resp.json()["status"] == "duplicate"


def test_query_success_returns_citations() -> None:
    """A grounded answer returns 200 with the citations present."""
    response = QueryResponse(
        answer="Paris is the capital of France.",
        citations=[Citation(document="sample.pdf", page=1, excerpt="capital of France is Paris")],
    )
    fake = _FakeQueryService(result=response)
    client = _client_with_overrides(query=lambda: fake)

    resp = client.post("/query", json={"question": "What is the capital of France?"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "Paris is the capital of France."
    assert len(body["citations"]) == 1
    assert body["citations"][0]["document"] == "sample.pdf"


def test_query_no_answer_returns_empty_citations() -> None:
    """A no-answer response returns 200 with an empty citations list."""
    response = QueryResponse(answer=NO_ANSWER_RESPONSE, citations=[])
    fake = _FakeQueryService(result=response)
    client = _client_with_overrides(query=lambda: fake)

    resp = client.post("/query", json={"question": "Totally unrelated question?"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == NO_ANSWER_RESPONSE
    assert body["citations"] == []


def test_reembed_success_returns_200() -> None:
    """A successful re-embed returns 200 echoing the document_id."""
    fake = _FakeIngestionService(reembed_result=_reembed_response())
    client = _client_with_overrides(ingestion=lambda: fake)

    resp = client.post("/documents/doc-123/reembed")

    assert resp.status_code == 200
    body = resp.json()
    assert body["document_id"] == "doc-123"
    assert body["status"] == "reembedded"


def test_list_documents_populated() -> None:
    """GET /documents returns the ingested set when documents exist."""
    docs = [_document_record("doc-a", "a.pdf"), _document_record("doc-b", "b.pdf")]
    store = _FakeVectorStore(docs)
    client = _client_with_overrides(store=lambda: store)

    resp = client.get("/documents")

    assert resp.status_code == 200
    body = resp.json()
    assert len(body["documents"]) == 2
    ids = {d["document_id"] for d in body["documents"]}
    assert ids == {"doc-a", "doc-b"}


def test_list_documents_empty() -> None:
    """GET /documents returns an empty list when none are ingested."""
    store = _FakeVectorStore([])
    client = _client_with_overrides(store=lambda: store)

    resp = client.get("/documents")

    assert resp.status_code == 200
    assert resp.json()["documents"] == []


# ===========================================================================
# Validation (framework, Category B) — D5
# ===========================================================================


def test_query_missing_question_field_returns_422_validation_error() -> None:
    """A missing ``question`` field is a validation error -> 422 (D5)."""
    fake = _FakeQueryService(result=QueryResponse(answer="x", citations=[]))
    client = _client_with_overrides(query=lambda: fake)

    resp = client.post("/query", json={})

    assert resp.status_code == 422
    body = resp.json()
    assert body["error"] == "VALIDATION_ERROR"
    assert "detail" in body
    # The fake must NOT have been called — validation precedes the dependency.
    assert fake.questions == []


def test_query_wrong_type_question_returns_422_validation_error() -> None:
    """A wrong-typed ``question`` value is a validation error -> 422 (D5)."""
    fake = _FakeQueryService(result=QueryResponse(answer="x", citations=[]))
    client = _client_with_overrides(query=lambda: fake)

    resp = client.post("/query", json={"question": 123, "nested": {"a": 1}})

    # 123 coerces? pydantic v2 is strict for int->str in this context: it is a
    # validation error. Either way the contract is a 422 VALIDATION_ERROR body.
    assert resp.status_code == 422
    assert resp.json()["error"] == "VALIDATION_ERROR"


def test_query_empty_question_value_returns_400_empty_question() -> None:
    """An empty/whitespace question VALUE -> service raises -> 400 EMPTY_QUESTION (D5)."""
    fake = _FakeQueryService(error=EmptyQuestionError("a question is required"))
    client = _client_with_overrides(query=lambda: fake)

    resp = client.post("/query", json={"question": "   "})

    assert resp.status_code == 400
    body = resp.json()
    assert body["error"] == "EMPTY_QUESTION"
    assert body["detail"] == "A question is required."
    # Proof the service (not the route) owned the whitespace check.
    assert fake.questions == ["   "]


# ===========================================================================
# Size guard (runs BEFORE read / delegation)
# ===========================================================================


def test_oversized_upload_rejected_before_read_413() -> None:
    """An oversized upload is rejected 413 by the size guard BEFORE ingest runs."""
    fake = _FakeIngestionService(ingest_result=_upload_response())
    small_limit_config = dataclasses.replace(Config(), max_pdf_size_mb=1)

    client = _client_with_overrides(
        ingestion=lambda: fake,
        config=lambda: small_limit_config,
    )

    # >1MB payload; TestClient sets content-length so UploadFile.size is set.
    oversized = b"a" * (2 * 1024 * 1024)
    resp = client.post(
        "/documents/upload",
        files={"file": ("big.pdf", oversized, "application/pdf")},
    )

    assert resp.status_code == 413
    body = resp.json()
    assert body["error"] == "FILE_TOO_LARGE"
    assert "1 MB" in body["detail"]
    # The guard rejected BEFORE reading/delegating: ingest was never called.
    assert fake.ingest_called is False


# ===========================================================================
# Domain error -> HTTP mapping (Category A)
# ===========================================================================


def test_invalid_pdf_returns_400_invalid_pdf() -> None:
    """A PDFParseError maps to 400 INVALID_PDF."""
    fake = _FakeIngestionService(ingest_error=PDFParseError("f.pdf", "bad"))
    client = _client_with_overrides(ingestion=lambda: fake, config=lambda: Config())

    resp = client.post(
        "/documents/upload",
        files={"file": ("f.pdf", b"not-a-pdf", "application/pdf")},
    )

    assert resp.status_code == 400
    body = resp.json()
    assert body["error"] == "INVALID_PDF"
    assert "f.pdf" in body["detail"]


def test_too_many_pages_returns_413() -> None:
    """A TooManyPagesError maps to 413 TOO_MANY_PAGES (D2)."""
    fake = _FakeIngestionService(ingest_error=TooManyPagesError("f.pdf", 60, 50))
    client = _client_with_overrides(ingestion=lambda: fake, config=lambda: Config())

    resp = client.post(
        "/documents/upload",
        files={"file": ("f.pdf", b"%PDF", "application/pdf")},
    )

    assert resp.status_code == 413
    body = resp.json()
    assert body["error"] == "TOO_MANY_PAGES"
    assert "50 pages" in body["detail"]


def test_empty_document_returns_422() -> None:
    """An EmptyDocumentError maps to 422 EMPTY_DOCUMENT (D2)."""
    fake = _FakeIngestionService(ingest_error=EmptyDocumentError("f.pdf"))
    client = _client_with_overrides(ingestion=lambda: fake, config=lambda: Config())

    resp = client.post(
        "/documents/upload",
        files={"file": ("f.pdf", b"%PDF", "application/pdf")},
    )

    assert resp.status_code == 422
    body = resp.json()
    assert body["error"] == "EMPTY_DOCUMENT"


def test_reembedding_required_returns_409_with_ids() -> None:
    """A ReembeddingRequiredError maps to 409 and names the affected ids."""
    fake = _FakeQueryService(error=ReembeddingRequiredError(["doc-1", "doc-2"]))
    client = _client_with_overrides(query=lambda: fake)

    resp = client.post("/query", json={"question": "anything"})

    assert resp.status_code == 409
    body = resp.json()
    assert body["error"] == "REEMBEDDING_REQUIRED"
    assert "doc-1" in body["detail"]
    assert "doc-2" in body["detail"]


def test_document_not_found_returns_404() -> None:
    """A DocumentNotFoundError on re-embed maps to 404 DOCUMENT_NOT_FOUND."""
    fake = _FakeIngestionService(reembed_error=DocumentNotFoundError("nope"))
    client = _client_with_overrides(ingestion=lambda: fake)

    resp = client.post("/documents/nope/reembed")

    assert resp.status_code == 404
    body = resp.json()
    assert body["error"] == "DOCUMENT_NOT_FOUND"


def test_llm_failure_returns_502_without_echoing_exception() -> None:
    """An LLMError maps to 502 LLM_FAILED and does NOT echo the exception text."""
    secret_text = "gemini-internal-secret-trace-xyz"
    fake = _FakeQueryService(error=LLMError(secret_text))
    client = _client_with_overrides(query=lambda: fake)

    resp = client.post("/query", json={"question": "anything"})

    assert resp.status_code == 502
    body = resp.json()
    assert body["error"] == "LLM_FAILED"
    assert body["detail"] == "Answer generation failed."
    assert secret_text not in body["detail"]


def test_embedding_failure_returns_502_without_echoing_exception() -> None:
    """An EmbeddingError maps to 502 EMBEDDING_FAILED and does NOT echo exc text."""
    secret_text = "secret-ish-embedding-detail"
    fake = _FakeIngestionService(ingest_error=EmbeddingError(secret_text))
    client = _client_with_overrides(ingestion=lambda: fake, config=lambda: Config())

    resp = client.post(
        "/documents/upload",
        files={"file": ("f.pdf", b"%PDF", "application/pdf")},
    )

    assert resp.status_code == 502
    body = resp.json()
    assert body["error"] == "EMBEDDING_FAILED"
    assert body["detail"] == "Embedding generation failed."
    assert secret_text not in body["detail"]


# ===========================================================================
# Unexpected error (Category C)
# ===========================================================================


def test_unexpected_error_returns_500_and_leaks_nothing() -> None:
    """A plain Exception maps to 500 INTERNAL_ERROR; no internals leak."""
    fake = _FakeQueryService(error=Exception("boom with /path and secret"))
    app = create_app()
    app.dependency_overrides[query_service_dep] = lambda: fake
    # The catch-all Exception handler produces the 500 response at runtime; the
    # TestClient must NOT re-raise the server exception, so we can assert the
    # client-facing body a real HTTP client would receive.
    client = TestClient(app, raise_server_exceptions=False)

    resp = client.post("/query", json={"question": "anything"})

    assert resp.status_code == 500
    body = resp.json()
    assert body["error"] == "INTERNAL_ERROR"
    assert body["detail"] == "An unexpected error occurred."
    serialized = resp.text
    for leaked in ("boom", "/path", "secret", "Traceback"):
        assert leaked not in serialized


# ===========================================================================
# D3 — top_k accepted-but-unused
# ===========================================================================


def test_query_top_k_in_body_is_accepted_but_unused() -> None:
    """A request top_k still returns 200 and is NOT forwarded to the service (D3)."""
    response = QueryResponse(
        answer="Grounded answer.",
        citations=[Citation(document="sample.pdf", page=1, excerpt="excerpt")],
    )
    fake = _FakeQueryService(result=response)
    client = _client_with_overrides(query=lambda: fake)

    resp = client.post("/query", json={"question": "What?", "top_k": 99})

    assert resp.status_code == 200
    # The route called answer(question) with the question ONLY — no top_k.
    assert fake.questions == ["What?"]


# ===========================================================================
# Import-safety (D7)
# ===========================================================================


def test_fresh_import_builds_nothing_heavy() -> None:
    """A FRESH import of src.api.app in a child process constructs no provider/store.

    This proves decision D7 at the strongest level: in a brand-new interpreter,
    merely ``import src.api.app`` and calling ``create_app()`` must not build a
    VectorStore / embedding / LLM provider, download a model, hit the network,
    or create a ``chroma_db/`` directory. A subprocess is used (rather than
    ``importlib.reload`` in-process) so it cannot disturb the shared module
    identity other tests rely on. The provider/store builders are replaced with
    bombs before the import runs; if the import or ``create_app()`` triggered any
    of them, the child would exit non-zero.
    """
    import subprocess
    import sys
    import textwrap

    program = textwrap.dedent(
        """
        import sys

        # Pre-register the module so the first real import of src.api.app binds
        # these bombs; if construction happens at import/create time they fire.
        def _boom(*a, **k):
            raise AssertionError("heavy construction happened at import/create time")

        import src.api.app as a
        a.build_embedding_provider = _boom
        a.build_llm_provider = _boom
        a.VectorStore = _boom
        app = a.create_app()
        assert type(app).__name__ == "FastAPI"
        print("OK")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", program],
        capture_output=True,
        text=True,
        cwd=_WORKSPACE_ROOT,
    )
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout


def test_create_app_builds_nothing_heavy(monkeypatch) -> None:
    """In-process: create_app() alone constructs no provider/store (D7).

    Replaces the provider/store builders on the already-imported module with
    bombs, then calls ``create_app()``. The factory must only wire routes and
    handlers — construction happens on first request through ``Depends``, never
    here — so no bomb should fire.
    """

    def _boom(*args, **kwargs):  # pragma: no cover - must never be called here
        raise AssertionError("heavy construction happened at create_app time")

    monkeypatch.setattr(app_module, "build_embedding_provider", _boom)
    monkeypatch.setattr(app_module, "build_llm_provider", _boom)
    monkeypatch.setattr(app_module, "VectorStore", _boom)

    app = app_module.create_app()
    assert app.__class__.__name__ == "FastAPI"


# ===========================================================================
# Integration test (14.3) — real services, stub providers, temp store
# ===========================================================================


class _StubEmbeddingProvider(EmbeddingProvider):
    """Deterministic offline embedding provider (fixed small dimension).

    Vectors depend only on text content, so ingestion and query embed into the
    same space and similarity ranking is reproducible. No torch, no network.
    """

    _DIM = 16

    def __init__(
        self,
        provider_name: str = "stub-embed",
        model_name: str = "stub-embed-v1",
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
        return self._DIM

    def _vector(self, text: str) -> list[float]:
        # Simple bag-of-chars hashing into a fixed-dim vector; deterministic.
        vec = [0.0] * self._DIM
        for ch in text.lower():
            vec[ord(ch) % self._DIM] += 1.0
        return vec

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


class _StubLLMProvider(LLMProvider):
    """Stub LLM that returns a grounded answer (never the no-answer sentinel).

    Returning a non-sentinel answer makes the query service produce citations,
    so the integration test can assert citation provenance.
    """

    @property
    def provider_name(self) -> str:
        return "stub-llm"

    @property
    def model_name(self) -> str:
        return "stub-llm-v1"

    def generate(self, prompt: str) -> str:
        return "Paris is the capital of France, grounded in the provided context."


def test_integration_upload_then_query_citations_trace_to_chunks(tmp_path) -> None:
    """Real ingest + query over a temp store with stub providers; citations valid.

    Builds the app with REAL ``Ingestion_Service`` / ``Retrieval_Service`` /
    ``Query_Service`` over a REAL ``VectorStore`` under ``tmp_path`` (never a
    persistent ``chroma_db/``), but with stub embedding + stub LLM providers.
    Uploads a tiny real in-memory PDF, queries it, and asserts every citation's
    (document, page, excerpt) traces back to an ingested chunk.
    """
    config = Config()  # defaults; no env/.env dependence for the stubbed path
    store = VectorStore(persist_path=str(tmp_path / "chroma"), collection_name="documents")
    embedder = _StubEmbeddingProvider()

    ingestion = Ingestion_Service(
        config=config,
        vector_store=store,
        embedding_provider=embedder,
    )
    retrieval = Retrieval_Service(config, store, embedder)
    query = Query_Service(
        config,
        retrieval_service=retrieval,
        llm_provider=_StubLLMProvider(),
    )

    app = create_app()
    app.dependency_overrides[ingestion_service_dep] = lambda: ingestion
    app.dependency_overrides[query_service_dep] = lambda: query
    app.dependency_overrides[vector_store_dep] = lambda: store
    # The upload route also depends on config_dep; override it with the same
    # test Config so the real _get_config builder is never reached (the autouse
    # guard asserts no real _get_* builder runs during any test).
    app.dependency_overrides[config_dep] = lambda: config
    client = TestClient(app)

    # 1. Ingest a tiny real PDF through the HTTP endpoint.
    upload = client.post(
        "/documents/upload",
        files={"file": ("france.pdf", _tiny_pdf_bytes(), "application/pdf")},
    )
    assert upload.status_code == 200
    assert upload.json()["status"] == "ingested"
    document_id = upload.json()["document_id"]

    # Collect the ingested chunks (document, page, text) directly from the store.
    stored_chunks = store.get_document_chunks(document_id)
    assert stored_chunks, "expected the PDF to produce at least one chunk"
    stored_lookup = {
        (c.document_name, c.page_number, c.chunk_text) for c in stored_chunks
    }

    # 2. Query through the HTTP endpoint.
    resp = client.post("/query", json={"question": "What is the capital of France?"})
    assert resp.status_code == 200
    body = resp.json()
    assert "answer" in body and "citations" in body
    assert body["citations"], "a grounded answer should carry citations"

    # 3. Every citation traces to an ingested chunk (document, page, excerpt).
    for citation in body["citations"]:
        key = (citation["document"], citation["page"], citation["excerpt"])
        assert key in stored_lookup, f"citation {key} does not trace to an ingested chunk"


# ===========================================================================
# Size guard — None-size bounded-read fallback (direct _handle_upload)
#
# TestClient always sets a Content-Length, so UploadFile.size is never None
# through HTTP. To exercise the unknown-size fallback we call the extracted
# module-level handler ``_handle_upload`` directly with a fake UploadFile whose
# ``size`` is None and whose ``read(n)`` records the n it was called with.
# ===========================================================================


class _FakeUploadFile:
    """Minimal UploadFile-like double with ``size=None`` and a recording ``read``.

    ``read(n)`` records every ``n`` it is called with in ``read_calls`` and
    returns the preset ``content``. The ``size is None`` branch of the handler
    is what drives the bounded read of ``max_bytes + 1``.
    """

    def __init__(self, content: bytes, filename: str = "unknown.pdf") -> None:
        self.size = None
        self.filename = filename
        self._content = content
        self.read_calls: list[int] = []

    async def read(self, n: int = -1) -> bytes:
        self.read_calls.append(n)
        return self._content


def test_handle_upload_none_size_oversized_rejected_before_ingest() -> None:
    """Unknown-size (None) upload over the limit: bounded read, reject, no ingest.

    The handler must read at most ``max_bytes + 1`` bytes (never an unbounded
    body), detect that the bounded read exceeded the limit, raise
    ``FileTooLargeError``, and never call the ingestion service.
    """
    config = dataclasses.replace(Config(), max_pdf_size_mb=1)
    max_bytes = config.max_pdf_size_mb * 1024 * 1024  # 1 MiB
    # read() returns MORE than the limit (max_bytes + 1 bytes).
    fake_file = _FakeUploadFile(content=b"a" * (max_bytes + 1))
    fake_service = _FakeIngestionService(ingest_result=_upload_response())

    with pytest.raises(FileTooLargeError):
        asyncio.run(_handle_upload(fake_file, fake_service, config))

    # The read was BOUNDED to exactly max_bytes + 1 (never unbounded read(-1)).
    assert fake_file.read_calls == [max_bytes + 1]
    # The guard rejected BEFORE delegating: ingest was never called.
    assert fake_service.ingest_called is False


def test_handle_upload_none_size_acceptable_delegates_to_ingest() -> None:
    """Unknown-size (None) upload within the limit: bounded read, then ingest.

    A body at or under the limit passes the bounded read and is forwarded to the
    ingestion service with the exact bytes that were read; the handler returns
    the service's ``UploadResponse``.
    """
    config = dataclasses.replace(Config(), max_pdf_size_mb=1)
    max_bytes = config.max_pdf_size_mb * 1024 * 1024
    data = b"%PDF small"  # well under the limit
    fake_file = _FakeUploadFile(content=data)
    expected = _upload_response("ingested")
    fake_service = _FakeIngestionService(ingest_result=expected)

    result = asyncio.run(_handle_upload(fake_file, fake_service, config))

    # Still a BOUNDED read of exactly max_bytes + 1 (not read(-1)).
    assert fake_file.read_calls == [max_bytes + 1]
    # ingest WAS called with the bytes that were read + the filename.
    assert fake_service.ingest_called is True
    assert fake_service.last_ingest_args == (data, "unknown.pdf")
    assert result is expected


# ===========================================================================
# MISSING_CREDENTIAL — reachable path (502) + handler-unit (name-only, no leak)
#
# PROVEN FACT (investigation): in the MVP no request path raises
# MissingCredentialError to the API. The Gemini provider catches a missing
# GEMINI_API_KEY from config.require_credential and RE-RAISES it as
# LLMError(str(exc)) (name-only), so the reachable missing-credential path is
# missing GEMINI_API_KEY -> LLMError -> 502 LLM_FAILED. The embedding factory
# (sentence-transformers, MVP) needs no credential. The MISSING_CREDENTIAL
# handler is therefore currently UNREACHABLE but kept for a future provider.
# ===========================================================================


class _StubRetrievalReturnsChunk:
    """Retrieval stub returning exactly one chunk so generate() is reached.

    Its ``retrieve`` signature matches ``Retrieval_Service.retrieve`` closely
    enough for ``Query_Service`` (which calls ``retrieve(question)``).
    """

    def retrieve(self, question: str, top_k: int | None = None):
        chunk = Chunk(
            document_name="doc.pdf",
            page_number=1,
            section=None,
            chunk_text="some grounded context text",
        )
        return [
            RetrievedChunk(
                chunk=chunk,
                score=0.99,
                embedding_provider="sentence-transformers",
                embedding_model="all-MiniLM-L6-v2",
            )
        ]


def test_missing_gemini_key_reachable_path_returns_502_and_leaks_no_secret(
    monkeypatch,
) -> None:
    """Missing GEMINI_API_KEY surfaces as LLMError -> 502 LLM_FAILED; no leak.

    This drives the REAL reachable path: a REAL ``Query_Service`` whose
    ``llm_provider`` is a REAL ``GeminiLLMProvider`` built on a ``Config`` with
    ``gemini_api_key=None`` and a retrieval STUB returning one chunk (so
    ``generate()`` is reached). ``GeminiLLMProvider._get_client`` calls
    ``config.require_credential("GEMINI_API_KEY")`` which raises, is re-raised as
    ``LLMError``, mapped by the handler to 502 ``LLM_FAILED``. No google SDK is
    imported (``_get_client`` raises before constructing ``genai.Client``), so
    the path is deterministic and offline.

    A recognizable DECOY secret is placed in the environment to prove no env
    secret ever leaks into the error body.
    """
    # Decoy env secret that must NEVER appear in any response body.
    monkeypatch.setenv("OPENAI_API_KEY", "SECRET-DECOY-SHOULD-NOT-APPEAR-123")
    # Ensure the Gemini key is genuinely absent for this path.
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    # Config with gemini_api_key=None (do not read .env — construct explicitly).
    no_key_config = dataclasses.replace(Config(), gemini_api_key=None)
    real_gemini = GeminiLLMProvider(config=no_key_config)
    real_query_service = Query_Service(
        no_key_config,
        retrieval_service=_StubRetrievalReturnsChunk(),
        llm_provider=real_gemini,
    )

    client = _client_with_overrides(query=lambda: real_query_service)

    resp = client.post("/query", json={"question": "What is the capital of France?"})

    assert resp.status_code == 502
    body = resp.json()
    assert body["error"] == "LLM_FAILED"
    # Fixed, value-free detail — contains neither the key name nor key material.
    assert body["detail"] == "Answer generation failed."
    assert "GEMINI_API_KEY" not in resp.text
    assert "SECRET-DECOY-SHOULD-NOT-APPEAR-123" not in resp.text


def test_missing_credential_handler_detail_is_name_only_no_value() -> None:
    """Handler-unit: MISSING_CREDENTIAL names the key only, leaks no value (500).

    Although the handler is unreachable in the MVP (see the reachable-path test
    above), this proves its body is name-only: a fake service raises a
    ``MissingCredentialError`` (whose message, per Config, names the key only),
    and the response detail must contain the key NAME but no secret VALUE.
    """
    key_name = "GEMINI_API_KEY"
    secret_value = "SUPERSECRET-VALUE-MUST-NOT-APPEAR"
    # Config's real message is name-only; simulate it faithfully.
    fake = _FakeQueryService(error=MissingCredentialError(f"{key_name} is not set"))
    client = _client_with_overrides(query=lambda: fake)

    resp = client.post("/query", json={"question": "anything"})

    assert resp.status_code == 500
    body = resp.json()
    assert body["error"] == "MISSING_CREDENTIAL"
    # Names the key (name-only requirement) ...
    assert key_name in body["detail"]
    # ... but never a secret value.
    assert secret_value not in resp.text


# ===========================================================================
# RequestValidationError does not echo client-sent input (secret-safety)
# ===========================================================================


def test_validation_error_body_does_not_echo_client_input() -> None:
    """A 422 VALIDATION_ERROR body is fixed and never serializes client input.

    The custom ``RequestValidationError`` handler must return the fixed envelope
    and must NOT dump ``exc.errors()`` / ``exc.body`` / the raw request input. A
    distinctive token sent in the (invalid) request must not appear anywhere in
    the response.
    """
    fake = _FakeQueryService(result=QueryResponse(answer="x", citations=[]))
    client = _client_with_overrides(query=lambda: fake)

    resp = client.post(
        "/query",
        json={"question": 123, "leak_me": "VALIDATION-LEAK-TOKEN"},
    )

    assert resp.status_code == 422
    body = resp.json()
    # Exactly the fixed VALIDATION_ERROR envelope.
    assert body == {"error": "VALIDATION_ERROR", "detail": "The request was invalid."}
    # The distinctive client-sent token never surfaces.
    assert "VALIDATION-LEAK-TOKEN" not in resp.text
    # The service was never reached — validation precedes the dependency.
    assert fake.questions == []

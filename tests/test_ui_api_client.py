"""Unit tests for ui.api_client — the requests-based backend client.

Fully offline: a fake ``requests.Session`` is injected into the client so no
real network call ever happens. Covers success + error translation for all
three endpoints, transport-error handling (ConnectionError / Timeout), and
malformed bodies, plus assertions on the exact request shape (``file`` form
field, ``{"question": ...}`` JSON with no top_k, and the (5, 60) timeout).
"""

from __future__ import annotations

import requests

from ui import api_client as AC


class FakeResponse:
    """Minimal stand-in for requests.Response."""

    def __init__(self, status_code: int, json_body=None, raise_value_error=False):
        self.status_code = status_code
        self._json_body = json_body
        self._raise_value_error = raise_value_error

    def json(self):
        if self._raise_value_error:
            raise ValueError("no json")
        return self._json_body


class FakeSession:
    """Records the last request and returns a programmed response (or raises)."""

    def __init__(self, response=None, exc=None):
        self._response = response
        self._exc = exc
        self.last_call = None

    def request(self, method, url, timeout=None, **kwargs):
        self.last_call = {
            "method": method,
            "url": url,
            "timeout": timeout,
            "kwargs": kwargs,
        }
        if self._exc is not None:
            raise self._exc
        return self._response


def _client_with_session(session) -> AC.ApiClient:
    client = AC.ApiClient(base_url="http://test-backend:1234")
    client.session = session
    return client


# ---------------------------------------------------------------------------
# upload_document
# ---------------------------------------------------------------------------
class TestUpload:
    def test_upload_ingested(self):
        body = {
            "document_id": "id",
            "document_name": "a.pdf",
            "status": "ingested",
            "page_count": 2,
            "chunk_count": 5,
            "message": "ok",
        }
        session = FakeSession(FakeResponse(200, body))
        client = _client_with_session(session)
        result = client.upload_document("a.pdf", b"%PDF-1.4")
        assert isinstance(result, AC.ApiOk)
        assert result.payload["status"] == "ingested"

    def test_upload_duplicate(self):
        body = {
            "document_id": "id",
            "document_name": "a.pdf",
            "status": "duplicate",
            "page_count": 2,
            "chunk_count": 5,
            "message": "dup",
        }
        session = FakeSession(FakeResponse(200, body))
        result = _client_with_session(session).upload_document("a.pdf", b"x")
        assert isinstance(result, AC.ApiOk)
        assert result.payload["status"] == "duplicate"

    def test_upload_uses_file_form_field_and_timeout(self):
        session = FakeSession(FakeResponse(200, {"document_id": "i"}))
        client = _client_with_session(session)
        client.upload_document("doc.pdf", b"bytes")
        call = session.last_call
        assert call["method"] == "POST"
        assert call["url"].endswith("/documents/upload")
        assert call["timeout"] == (5, 60)
        # The multipart form field MUST be named exactly "file".
        files = call["kwargs"]["files"]
        assert "file" in files
        assert files["file"][0] == "doc.pdf"

    def test_upload_413_file_too_large(self):
        body = {"error": "FILE_TOO_LARGE", "detail": "File exceeds the maximum size of 5 MB."}
        session = FakeSession(FakeResponse(413, body))
        result = _client_with_session(session).upload_document("a.pdf", b"x")
        assert isinstance(result, AC.ApiError)
        assert result.code == "FILE_TOO_LARGE"
        assert result.status_code == 413
        assert result.detail == "File exceeds the maximum size of 5 MB."

    def test_upload_400_invalid_pdf(self):
        body = {"error": "INVALID_PDF", "detail": "bad"}
        session = FakeSession(FakeResponse(400, body))
        result = _client_with_session(session).upload_document("a.pdf", b"x")
        assert isinstance(result, AC.ApiError)
        assert result.code == "INVALID_PDF"
        assert result.status_code == 400


# ---------------------------------------------------------------------------
# list_documents
# ---------------------------------------------------------------------------
class TestList:
    def test_list_populated(self):
        body = {"documents": [{"document_name": "a.pdf"}]}
        session = FakeSession(FakeResponse(200, body))
        result = _client_with_session(session).list_documents()
        assert isinstance(result, AC.ApiOk)
        assert len(result.payload["documents"]) == 1

    def test_list_empty(self):
        session = FakeSession(FakeResponse(200, {"documents": []}))
        result = _client_with_session(session).list_documents()
        assert isinstance(result, AC.ApiOk)
        assert result.payload["documents"] == []

    def test_list_uses_get(self):
        session = FakeSession(FakeResponse(200, {"documents": []}))
        _client_with_session(session).list_documents()
        assert session.last_call["method"] == "GET"
        assert session.last_call["url"].endswith("/documents")


# ---------------------------------------------------------------------------
# query
# ---------------------------------------------------------------------------
class TestQuery:
    def test_query_with_citations(self):
        body = {
            "answer": "yes",
            "citations": [{"document": "a.pdf", "page": 1, "excerpt": "x"}],
        }
        session = FakeSession(FakeResponse(200, body))
        result = _client_with_session(session).query("why?")
        assert isinstance(result, AC.ApiOk)
        assert result.payload["citations"]

    def test_query_empty_citations(self):
        body = {"answer": "idk", "citations": []}
        session = FakeSession(FakeResponse(200, body))
        result = _client_with_session(session).query("why?")
        assert isinstance(result, AC.ApiOk)
        assert result.payload["citations"] == []

    def test_query_sends_question_only_no_top_k(self):
        session = FakeSession(FakeResponse(200, {"answer": "a", "citations": []}))
        client = _client_with_session(session)
        client.query("what is x?")
        call = session.last_call
        assert call["method"] == "POST"
        assert call["url"].endswith("/query")
        assert call["timeout"] == (5, 60)
        json_body = call["kwargs"]["json"]
        assert json_body == {"question": "what is x?"}
        assert "top_k" not in json_body

    def test_query_empty_question_400(self):
        body = {"error": "EMPTY_QUESTION", "detail": "A question is required."}
        session = FakeSession(FakeResponse(400, body))
        result = _client_with_session(session).query("")
        assert isinstance(result, AC.ApiError)
        assert result.code == "EMPTY_QUESTION"


# ---------------------------------------------------------------------------
# delete_document
# ---------------------------------------------------------------------------
class TestDelete:
    def test_delete_success(self):
        body = {
            "document_id": "doc-1",
            "status": "deleted",
            "message": "Document deleted.",
        }
        session = FakeSession(FakeResponse(200, body))
        result = _client_with_session(session).delete_document("doc-1")
        assert isinstance(result, AC.ApiOk)
        assert result.payload["status"] == "deleted"
        assert result.payload["document_id"] == "doc-1"

    def test_delete_uses_delete_method_and_path(self):
        session = FakeSession(
            FakeResponse(200, {"document_id": "abc", "status": "deleted", "message": "ok"})
        )
        client = _client_with_session(session)
        client.delete_document("abc")
        call = session.last_call
        assert call["method"] == "DELETE"
        assert call["url"].endswith("/documents/abc")
        assert call["timeout"] == (5, 60)

    def test_delete_404_document_not_found(self):
        body = {"error": "DOCUMENT_NOT_FOUND", "detail": "No document found with the given id."}
        session = FakeSession(FakeResponse(404, body))
        result = _client_with_session(session).delete_document("missing")
        assert isinstance(result, AC.ApiError)
        assert result.code == "DOCUMENT_NOT_FOUND"
        assert result.status_code == 404

    def test_delete_500_server_error(self):
        body = {"error": "INTERNAL_ERROR", "detail": "An unexpected error occurred."}
        session = FakeSession(FakeResponse(500, body))
        result = _client_with_session(session).delete_document("doc-1")
        assert isinstance(result, AC.ApiError)
        assert result.code == "INTERNAL_ERROR"
        assert result.status_code == 500


# ---------------------------------------------------------------------------
# transport errors + malformed bodies
# ---------------------------------------------------------------------------
class TestTransport:
    def test_connection_error(self):
        session = FakeSession(exc=requests.ConnectionError("boom"))
        result = _client_with_session(session).list_documents()
        assert isinstance(result, AC.ApiError)
        assert result.code == "CONNECTION_ERROR"
        assert result.status_code is None

    def test_timeout(self):
        session = FakeSession(exc=requests.Timeout("slow"))
        result = _client_with_session(session).query("q")
        assert isinstance(result, AC.ApiError)
        assert result.code == "TIMEOUT"

    def test_other_request_exception(self):
        session = FakeSession(exc=requests.RequestException("weird"))
        result = _client_with_session(session).list_documents()
        assert isinstance(result, AC.ApiError)
        assert result.code == "UNEXPECTED_RESPONSE"

    def test_non_json_success_body(self):
        session = FakeSession(FakeResponse(200, raise_value_error=True))
        result = _client_with_session(session).list_documents()
        assert isinstance(result, AC.ApiError)
        assert result.code == "UNEXPECTED_RESPONSE"

    def test_non_dict_success_body(self):
        session = FakeSession(FakeResponse(200, ["not", "a", "dict"]))
        result = _client_with_session(session).list_documents()
        assert isinstance(result, AC.ApiError)
        assert result.code == "UNEXPECTED_RESPONSE"

    def test_non_json_error_body(self):
        session = FakeSession(FakeResponse(500, raise_value_error=True))
        result = _client_with_session(session).list_documents()
        assert isinstance(result, AC.ApiError)
        assert result.code == "UNEXPECTED_RESPONSE"
        assert result.status_code == 500

    def test_error_body_without_code(self):
        session = FakeSession(FakeResponse(500, {"nope": "x"}))
        result = _client_with_session(session).list_documents()
        assert isinstance(result, AC.ApiError)
        assert result.code == "UNEXPECTED_RESPONSE"


# ---------------------------------------------------------------------------
# base url + module client
# ---------------------------------------------------------------------------
class TestConfig:
    def test_default_base_url(self, monkeypatch):
        monkeypatch.delenv("API_BASE_URL", raising=False)
        assert AC.resolve_base_url() == "http://127.0.0.1:8000"

    def test_env_base_url_trims_slash(self, monkeypatch):
        monkeypatch.setenv("API_BASE_URL", "http://example:9000/")
        assert AC.resolve_base_url() == "http://example:9000"

    def test_set_and_get_client(self):
        sentinel = object()
        AC.set_client(sentinel)
        assert AC.get_client() is sentinel
        # reset so other tests build a real (but un-called) client lazily
        AC.set_client(None)

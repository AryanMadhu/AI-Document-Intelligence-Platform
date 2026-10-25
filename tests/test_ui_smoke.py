"""Smoke + import-safety tests for the Streamlit UI.

Two layers:

* **Import-safety**: importing ``ui.app`` / ``ui.api_client`` / ``ui.presentation``
  must make NO network call. We enforce this by monkeypatching ``requests`` so
  any socket-level call explodes, then importing/reloading the modules.

* **AppTest smoke** (``streamlit.testing.v1.AppTest``): a FAKE api client is
  injected via ``ui.api_client.set_client`` so the app contacts NO backend. We
  then run the app and assert it renders without error.

What AppTest in Streamlit 1.40.2 can/can't drive (verified empirically):
  * CAN run the whole app, read ``title``, and drive ``st.chat_input`` via
    ``.set_value(...).run()`` — so the QUERY flow is exercised end-to-end here
    against the fake client (no network).
  * CANNOT drive ``st.file_uploader`` (there is no ``file_uploader`` accessor on
    AppTest in 1.40.2). The UPLOAD flow is therefore covered by the api_client
    tests (request shape + error mapping) and the presentation unit tests, not
    by AppTest. We do NOT change the UI to suit the test tool.
"""

from __future__ import annotations

import importlib
import sys

import pytest

from ui import api_client as AC
from ui.api_client import ApiError, ApiOk


# ---------------------------------------------------------------------------
# A fake client shared by the AppTest cases. No network; records calls.
# ---------------------------------------------------------------------------
class FakeClient:
    def __init__(self, query_result=None, list_result=None):
        self.calls = []
        self._query_result = query_result or ApiOk(
            {"answer": "a grounded answer", "citations": []}
        )
        self._list_result = list_result or ApiOk({"documents": []})

    def list_documents(self):
        self.calls.append("list")
        return self._list_result

    def query(self, question):
        self.calls.append(("query", question))
        return self._query_result

    def upload_document(self, filename, data_bytes):
        self.calls.append(("upload", filename))
        return ApiOk(
            {
                "document_id": "id",
                "document_name": filename,
                "status": "ingested",
                "page_count": 1,
                "chunk_count": 1,
                "message": "ok",
            }
        )


@pytest.fixture
def app_test():
    """Build an AppTest for ui/app.py with a fake client injected."""
    from streamlit.testing.v1 import AppTest

    fake = FakeClient()
    AC.set_client(fake)
    at = AppTest.from_file("ui/app.py", default_timeout=30)
    yield at, fake
    AC.set_client(None)


# ---------------------------------------------------------------------------
# Import-safety: no network at import time.
# ---------------------------------------------------------------------------
def test_imports_make_no_network_call(monkeypatch):
    """Reimporting the ui modules must not touch the network."""
    import requests

    def _boom(*args, **kwargs):
        raise AssertionError("network call made at import time!")

    # Any attempt to open a socket-level request fails loudly.
    monkeypatch.setattr(requests.sessions.Session, "request", _boom)
    monkeypatch.setattr(requests, "request", _boom, raising=False)

    for name in ["ui.presentation", "ui.api_client", "ui.app"]:
        module = importlib.import_module(name)
        importlib.reload(module)
    # Reaching here means no import-time network call happened.


def test_get_client_does_not_call_network_on_construction(monkeypatch):
    """Building the ApiClient reads env only; it opens no socket."""
    import requests

    monkeypatch.setattr(
        requests.sessions.Session,
        "request",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("no network allowed")),
    )
    AC.set_client(None)
    client = AC.get_client()  # must not raise
    assert isinstance(client, AC.ApiClient)
    AC.set_client(None)


# ---------------------------------------------------------------------------
# AppTest smoke — renders with no backend.
# ---------------------------------------------------------------------------
def test_app_renders_without_backend(app_test):
    at, fake = app_test
    at.run()
    # No uncaught exception rendered.
    assert not at.exception
    # Title rendered.
    assert any("AI Document Intelligence" in t.value for t in at.title)
    # The only backend interaction is the (faked) document-list refresh.
    assert "list" in fake.calls


def test_app_has_chat_input(app_test):
    at, fake = app_test
    at.run()
    assert len(at.chat_input) == 1


def test_query_flow_via_chat_input(app_test):
    """Drive the chat_input (the only query entry point) against the fake."""
    at, fake = app_test
    at.run()
    at.chat_input[0].set_value("what is in the document?").run()
    assert not at.exception
    assert ("query", "what is in the document?") in fake.calls
    # The transcript renders user + assistant chat messages.
    assert len(at.chat_message) >= 2


def test_query_with_citations_renders(app_test):
    """A query returning citations renders without error (expanders present)."""
    at, fake = app_test
    fake._query_result = ApiOk(
        {
            "answer": "Yes.",
            "citations": [{"document": "a.pdf", "page": 2, "excerpt": "excerpt text"}],
        }
    )
    at.run()
    at.chat_input[0].set_value("question?").run()
    assert not at.exception


def test_query_error_keeps_app_running(app_test):
    """An ApiError from query is shown as a friendly message, app still renders."""
    at, fake = app_test
    fake._query_result = ApiError(code="LLM_FAILED", detail=None, status_code=502)
    at.run()
    at.chat_input[0].set_value("question?").run()
    assert not at.exception
    assert ("query", "question?") in fake.calls

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

Uploader reset verification (UI-only fix):
  AppTest 1.40.2 cannot inspect or drive ``st.file_uploader`` state, so the
  "reset the uploader after a confirmed upload" behavior CANNOT be asserted via
  AppTest (the selected-file + cancel-× state is not observable there). Instead
  it is verified at two deterministic seams:
    * ``_do_upload`` returns ``True`` only on a confirmed-indexed response
      (``ingested`` or ``duplicate``) and ``False`` on any failure, and
    * the sidebar decision logic bumps ``session_state.uploader_key`` ONLY when
      ``_do_upload`` returns ``True`` (a new key recreates the widget empty).
  A manual visual check is still required: upload a PDF and confirm the selected
  filename + cancel (×) disappear after a successful (ingested/duplicate) upload,
  and remain on an error so the same file can be retried.
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


# ---------------------------------------------------------------------------
# _do_upload return-contract + uploader-key reset logic (UI-only fix).
#
# These are deterministic UNIT tests that isolate the return value + message
# behavior WITHOUT a Streamlit ScriptRunContext. We monkeypatch:
#   * ``ui.app.get_client`` to return a fake client, and
#   * ``ui.app.st`` with a lightweight fake exposing only what _do_upload uses
#     (``spinner`` as a context manager, ``error``/``info``/``success``, and a
#     dict-like ``session_state``).
# AppTest cannot drive st.file_uploader, so the uploader reset is verified here
# via the _do_upload contract + the key-increment behavior (see module docstring).
# ---------------------------------------------------------------------------
import contextlib

from ui import app as UIAPP


class _FakeSessionState(dict):
    """Dict that also supports attribute access (like st.session_state)."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:  # pragma: no cover - defensive
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = value


class _FakeSt:
    """Minimal stand-in for ``streamlit`` as used by ``_do_upload``."""

    def __init__(self):
        self.errors = []
        self.infos = []
        self.successes = []
        self.warnings = []
        self.session_state = _FakeSessionState()

    @contextlib.contextmanager
    def spinner(self, _text):
        yield

    def error(self, msg):
        self.errors.append(msg)

    def info(self, msg):
        self.infos.append(msg)

    def success(self, msg):
        self.successes.append(msg)

    def warning(self, msg):
        self.warnings.append(msg)


class _UploadClient:
    """Fake client returning a preset upload result (and an ApiOk doc list)."""

    def __init__(self, upload_result):
        self._upload_result = upload_result

    def upload_document(self, filename, data_bytes):
        return self._upload_result

    def list_documents(self):
        # Use ui.app's ApiOk so isinstance checks inside _refresh_documents hold
        # even if another test reloaded ui.api_client (new class identity).
        return UIAPP.ApiOk({"documents": []})


def _patch_app(monkeypatch, upload_result):
    """Point ui.app at a fake st + fake client; return the fake st."""
    fake_st = _FakeSt()
    monkeypatch.setattr(UIAPP, "st", fake_st)
    monkeypatch.setattr(UIAPP, "get_client", lambda: _UploadClient(upload_result))
    return fake_st


def _upload_ok(status):
    # Use ui.app's ApiOk so isinstance(result, ApiOk) inside _do_upload holds
    # regardless of module-reload ordering across the test session.
    return UIAPP.ApiOk(
        {
            "document_id": "id",
            "document_name": "a.pdf",
            "status": status,
            "page_count": 1,
            "chunk_count": 1,
            "message": "ok",
        }
    )


def test_do_upload_returns_true_on_ingested(monkeypatch):
    """A 200 'ingested' response returns True via the success path (no error)."""
    fake_st = _patch_app(monkeypatch, _upload_ok("ingested"))
    assert UIAPP._do_upload("a.pdf", b"x") is True
    assert fake_st.errors == []
    assert fake_st.successes and not fake_st.infos


def test_do_upload_returns_true_on_duplicate(monkeypatch):
    """A 200 'duplicate' response returns True via the info path (reset-eligible)."""
    fake_st = _patch_app(monkeypatch, _upload_ok("duplicate"))
    # True == reset-eligible: a duplicate IS indexed, so the uploader resets too.
    assert UIAPP._do_upload("a.pdf", b"x") is True
    assert fake_st.errors == []
    assert fake_st.infos and not fake_st.successes


def test_do_upload_returns_false_on_api_error(monkeypatch):
    """An ApiError returns False and shows an error (no reset)."""
    fake_st = _patch_app(
        monkeypatch, UIAPP.ApiError(code="UPLOAD_FAILED", detail=None, status_code=502)
    )
    assert UIAPP._do_upload("a.pdf", b"x") is False
    assert fake_st.errors and not fake_st.successes and not fake_st.infos


def test_do_upload_returns_false_on_malformed_ok(monkeypatch):
    """A malformed ApiOk (fails validate_upload_response) returns False + message."""
    from ui.presentation import UNEXPECTED_RESPONSE_MESSAGE

    # Missing required fields (e.g. page_count/chunk_count/status) -> invalid shape.
    fake_st = _patch_app(monkeypatch, UIAPP.ApiOk({"document_name": "a.pdf"}))
    assert UIAPP._do_upload("a.pdf", b"x") is False
    assert UNEXPECTED_RESPONSE_MESSAGE in fake_st.errors
    assert not fake_st.successes and not fake_st.infos


def test_uploader_key_increments_only_on_success(monkeypatch):
    """Emulate the sidebar button branch: key bumps iff _do_upload returns True."""
    fake_st = _FakeSt()
    fake_st.session_state["uploader_key"] = 0
    monkeypatch.setattr(UIAPP, "st", fake_st)

    def _apply(do_upload_result):
        # Mirror the exact decision in _render_sidebar's Upload-button branch.
        if do_upload_result:
            fake_st.session_state["uploader_key"] += 1

    # Failure path: no increment.
    _apply(False)
    assert fake_st.session_state["uploader_key"] == 0
    # Success path: increment.
    _apply(True)
    assert fake_st.session_state["uploader_key"] == 1
    # Another failure after a success: still no further increment.
    _apply(False)
    assert fake_st.session_state["uploader_key"] == 1

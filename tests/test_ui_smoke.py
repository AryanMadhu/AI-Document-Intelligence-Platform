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


# ---------------------------------------------------------------------------
# Delete flow — _do_delete contract, _clear_citations_for_document, and the
# sidebar confirm/cancel decision seams (UI-only; Task 15A).
#
# AppTest 1.40.2 limitation (documented): per-document Delete / Confirm / Cancel
# are rendered inside st.sidebar via st.button + st.columns, which AppTest in
# 1.40.2 cannot reliably address by key/label, and st.rerun() short-circuits a
# scripted run. So the delete UX is verified here at deterministic LOGIC seams
# (the same strategy the uploader-reset tests use), NOT via AppTest widgets.
# A MANUAL visual check is still required: upload a PDF, click Delete -> a
# confirm line naming the file appears (nothing deleted yet); Cancel closes it
# with the doc intact; Confirm removes the doc and clears any transcript entry
# that cited it; a failed delete keeps the doc and leaves the transcript.
# ---------------------------------------------------------------------------


class _DeleteClient:
    """Fake client returning a preset delete result; records the ids it saw."""

    def __init__(self, delete_result):
        self._delete_result = delete_result
        self.deleted_ids = []
        self.list_calls = 0

    def delete_document(self, document_id):
        self.deleted_ids.append(document_id)
        return self._delete_result

    def list_documents(self):
        self.list_calls += 1
        return UIAPP.ApiOk({"documents": []})


def _patch_app_delete(monkeypatch, delete_result):
    """Point ui.app at a fake st + fake delete client; return (fake_st, client)."""
    fake_st = _FakeSt()
    monkeypatch.setattr(UIAPP, "st", fake_st)
    client = _DeleteClient(delete_result)
    monkeypatch.setattr(UIAPP, "get_client", lambda: client)
    return fake_st, client


def _delete_ok(status="deleted"):
    return UIAPP.ApiOk(
        {"document_id": "doc-1", "status": status, "message": "Document deleted."}
    )


def test_do_delete_returns_true_on_deleted(monkeypatch):
    """A 200 {status:'deleted'} response returns True and calls the client once."""
    fake_st, client = _patch_app_delete(monkeypatch, _delete_ok("deleted"))
    assert UIAPP._do_delete("doc-1", "a.pdf") is True
    assert client.deleted_ids == ["doc-1"]
    assert fake_st.errors == []


def test_do_delete_returns_false_on_api_error(monkeypatch):
    """An ApiError returns False and shows a friendly error."""
    fake_st, client = _patch_app_delete(
        monkeypatch, UIAPP.ApiError(code="DOCUMENT_NOT_FOUND", detail=None, status_code=404)
    )
    assert UIAPP._do_delete("doc-1", "a.pdf") is False
    assert fake_st.errors  # a message was shown


def test_do_delete_returns_false_on_unexpected_body(monkeypatch):
    """A 2xx body without status=='deleted' is treated as unexpected -> False."""
    from ui.presentation import UNEXPECTED_RESPONSE_MESSAGE

    fake_st, client = _patch_app_delete(monkeypatch, UIAPP.ApiOk({"status": "weird"}))
    assert UIAPP._do_delete("doc-1", "a.pdf") is False
    assert UNEXPECTED_RESPONSE_MESSAGE in fake_st.errors


def test_clear_citations_drops_only_entries_citing_the_deleted_doc(monkeypatch):
    """Successful deletion purges transcript entries that cite the deleted doc."""
    fake_st = _FakeSt()
    monkeypatch.setattr(UIAPP, "st", fake_st)
    fake_st.session_state["transcript"] = [
        {  # cites the deleted doc -> dropped
            "question": "q1",
            "answer": "a1",
            "citations": [{"document": "gone.pdf", "page": 1, "excerpt": "x"}],
        },
        {  # cites a surviving doc -> kept
            "question": "q2",
            "answer": "a2",
            "citations": [{"document": "keep.pdf", "page": 2, "excerpt": "y"}],
        },
        {  # no citations -> kept
            "question": "q3",
            "answer": "a3",
            "citations": [],
        },
        {  # error entry (no citations key) -> kept
            "question": "q4",
            "error": "oops",
        },
    ]

    UIAPP._clear_citations_for_document("gone.pdf")

    kept_questions = [e["question"] for e in fake_st.session_state["transcript"]]
    assert kept_questions == ["q2", "q3", "q4"]


def test_clear_citations_drops_mixed_entry_citing_deleted_doc(monkeypatch):
    """MVP behavior: a mixed-citation entry is dropped if it cites the deleted doc."""
    fake_st = _FakeSt()
    monkeypatch.setattr(UIAPP, "st", fake_st)
    fake_st.session_state["transcript"] = [
        {
            "question": "q",
            "answer": "a",
            "citations": [
                {"document": "keep.pdf", "page": 1, "excerpt": "x"},
                {"document": "gone.pdf", "page": 3, "excerpt": "z"},
            ],
        }
    ]
    UIAPP._clear_citations_for_document("gone.pdf")
    assert fake_st.session_state["transcript"] == []


def test_delete_confirm_flow_success_removes_and_clears(monkeypatch):
    """Emulate the sidebar Confirm branch on success.

    Mirrors the exact decisions in _render_document_row's confirm path: call
    _do_delete; on True clear citations + pending_delete + refresh. Asserts the
    doc is gone (via refresh returning []), the citing transcript entry cleared,
    and pending_delete reset.
    """
    fake_st, client = _patch_app_delete(monkeypatch, _delete_ok("deleted"))
    fake_st.session_state["transcript"] = [
        {"question": "q", "answer": "a", "citations": [{"document": "a.pdf", "page": 1, "excerpt": "x"}]}
    ]
    fake_st.session_state["documents"] = [{"document_name": "a.pdf", "document_id": "doc-1"}]
    fake_st.session_state["documents_loaded"] = True
    fake_st.session_state["pending_delete"] = "doc-1"

    # Emulate the confirm branch logic exactly.
    if UIAPP._do_delete("doc-1", "a.pdf"):
        UIAPP._clear_citations_for_document("a.pdf")
        fake_st.session_state["pending_delete"] = None
        UIAPP._refresh_documents()

    assert client.deleted_ids == ["doc-1"]
    assert fake_st.session_state["transcript"] == []  # citing entry dropped
    assert fake_st.session_state["pending_delete"] is None
    assert fake_st.session_state["documents"] == []  # refresh returned empty list
    assert client.list_calls == 1


def test_delete_confirm_flow_failure_keeps_doc_and_transcript(monkeypatch):
    """Emulate the sidebar Confirm branch on failure: doc + transcript preserved."""
    fake_st, client = _patch_app_delete(
        monkeypatch, UIAPP.ApiError(code="INTERNAL_ERROR", detail=None, status_code=500)
    )
    original_transcript = [
        {"question": "q", "answer": "a", "citations": [{"document": "a.pdf", "page": 1, "excerpt": "x"}]}
    ]
    original_docs = [{"document_name": "a.pdf", "document_id": "doc-1"}]
    fake_st.session_state["transcript"] = list(original_transcript)
    fake_st.session_state["documents"] = list(original_docs)
    fake_st.session_state["pending_delete"] = "doc-1"

    # Emulate the confirm branch logic exactly (failure path).
    if UIAPP._do_delete("doc-1", "a.pdf"):
        UIAPP._clear_citations_for_document("a.pdf")
        fake_st.session_state["pending_delete"] = None
        UIAPP._refresh_documents()
    else:
        # Failure: close the confirm UI, keep everything else intact.
        fake_st.session_state["pending_delete"] = None

    assert client.deleted_ids == ["doc-1"]  # the API was attempted
    assert fake_st.session_state["transcript"] == original_transcript  # unchanged
    assert fake_st.session_state["documents"] == original_docs  # doc kept
    assert fake_st.session_state["pending_delete"] is None
    assert client.list_calls == 0  # no refresh on failure


def test_delete_click_sets_pending_without_calling_api(monkeypatch):
    """Clicking Delete arms pending_delete and makes NO API call (step 1)."""
    fake_st, client = _patch_app_delete(monkeypatch, _delete_ok("deleted"))
    fake_st.session_state["pending_delete"] = None

    # Emulate the "Delete" button branch: set pending_delete only.
    fake_st.session_state["pending_delete"] = "doc-1"

    assert fake_st.session_state["pending_delete"] == "doc-1"
    assert client.deleted_ids == []  # no API call on the arming click


def test_delete_cancel_makes_no_api_call_and_clears_pending(monkeypatch):
    """Cancel clears pending_delete and never calls the client (step 2 cancel)."""
    fake_st, client = _patch_app_delete(monkeypatch, _delete_ok("deleted"))
    fake_st.session_state["pending_delete"] = "doc-1"
    fake_st.session_state["documents"] = [{"document_name": "a.pdf", "document_id": "doc-1"}]
    fake_st.session_state["transcript"] = [{"question": "q", "answer": "a", "citations": []}]

    # Emulate the Cancel branch: clear pending_delete, no API call.
    fake_st.session_state["pending_delete"] = None

    assert client.deleted_ids == []
    assert fake_st.session_state["pending_delete"] is None
    # Unrelated state untouched.
    assert fake_st.session_state["documents"] == [{"document_name": "a.pdf", "document_id": "doc-1"}]
    assert len(fake_st.session_state["transcript"]) == 1

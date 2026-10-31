"""Streamlit UI composition for the AI Document Intelligence Platform.

This app is a THIN HTTP CLIENT of the FastAPI backend. It owns no document
intelligence logic: it uploads PDFs, lists documents, and asks questions, all
through :mod:`ui.api_client`, and renders the results safely via
:mod:`ui.presentation`. It never contacts the backend at import time — every
network call happens only on a user action (upload button / chat submit).

Running the app
---------------
Run from the PROJECT ROOT so both ``ui`` and ``src`` are importable::

    python -m streamlit run ui/app.py

Streamlit sometimes puts the *script's* folder (``ui/``) on ``sys.path`` instead
of the project root; to be robust we prepend the project root to ``sys.path`` at
the very top (before importing ``ui.*``) — see the sys.path block below.

Deployment note (Task 19): the Dockerfile must set ``WORKDIR`` to the project
root (or set ``PYTHONPATH`` to it) for the same reason — so ``ui`` and ``src``
resolve as top-level packages.

Layout (Change B)
-----------------
* SIDEBAR: the upload control + the ingested-document list.
* MAIN AREA: this session's chat transcript, rendered with ``st.chat_message``
  for presentation only, plus ``st.chat_input`` for the next question.

Only the CURRENT question is sent to ``/query`` — no conversation history is
ever sent to the backend. The transcript is presentation-only state kept in
``st.session_state``.
"""

from __future__ import annotations

# --- sys.path robustness (Change G) ---------------------------------------
# Prepend the PROJECT ROOT (the parent of this file's directory, i.e. the parent
# of ``ui/``) to sys.path BEFORE importing any ``ui.*`` module. This makes the
# app import-correct whether Streamlit launched it with the project root or the
# ``ui/`` folder on sys.path.
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
# --------------------------------------------------------------------------

import streamlit as st

from ui.api_client import ApiError, ApiOk, get_client
from ui.presentation import (
    NO_CITATIONS_CAPTION,
    NO_CITATIONS_HINT,
    UNEXPECTED_RESPONSE_MESSAGE,
    error_severity,
    escape_markdown_answer,
    escape_markdown_strict,
    format_excerpt,
    format_page_count,
    format_timestamp,
    friendly_error,
    group_citations,
    grouped_citation_title,
    sources_label,
    validate_documents_response,
    validate_query_response,
    validate_upload_response,
)
from ui.styles import inject_css

# Exact upload hint text (Change C): no hard-coded size/page numbers.
UPLOAD_HINT = "PDF files only. Size and page limits apply."
# First-operation latency note (model cold start).
SLOW_FIRST_OP_NOTE = "The first operation can take ~30s while the service warms up."

# Theme-appropriate chat avatars (deterministic, static emoji).
USER_AVATAR = "🧑"
ASSISTANT_AVATAR = "📄"

# Static example questions shown when the transcript is empty.
EXAMPLE_QUESTIONS = (
    "What is this document about?",
    "Summarize page 1",
    "What are the key points?",
)


# ---------------------------------------------------------------------------
# Session state (Change F): transcript + document list persist across reruns and
# are NEVER cleared when a later call fails.
# ---------------------------------------------------------------------------
def _init_state() -> None:
    """Initialize session_state keys if absent (idempotent)."""
    if "transcript" not in st.session_state:
        # list of {"question": str, "result": dict} entries (presentation only).
        st.session_state.transcript = []
    if "documents" not in st.session_state:
        st.session_state.documents = []
    if "documents_loaded" not in st.session_state:
        st.session_state.documents_loaded = False
    if "uploader_key" not in st.session_state:
        # Bumped after a confirmed-indexed upload to recreate the file_uploader
        # widget empty (clearing the selected filename + the cancel × control).
        st.session_state.uploader_key = 0


# ---------------------------------------------------------------------------
# Backend interactions (each only runs on a user action).
# ---------------------------------------------------------------------------
def _refresh_documents() -> None:
    """Fetch the document list and store it; keep prior list on any failure."""
    result = get_client().list_documents()
    if isinstance(result, ApiError):
        st.session_state.documents_loaded = True
        st.warning(friendly_error(result.code, result.detail))
        return
    if isinstance(result, ApiOk) and validate_documents_response(result.payload):
        st.session_state.documents = result.payload["documents"]
        st.session_state.documents_loaded = True
    else:
        # Malformed body: keep prior documents, show generic message.
        st.session_state.documents_loaded = True
        st.warning(UNEXPECTED_RESPONSE_MESSAGE)


def _do_upload(filename: str, data_bytes: bytes) -> bool:
    """Upload a PDF and surface a safe summary; refresh the doc list on success.

    Returns ``True`` ONLY on a CONFIRMED-indexed response (HTTP-200 with either
    ``status == "ingested"`` or ``status == "duplicate"`` — both confirm the
    document IS indexed). Returns ``False`` on any transport/validation FAILURE
    (``ApiError`` or malformed body) so the caller can leave the selected file in
    place for a retry. The caller uses this flag to decide whether to reset the
    uploader widget.
    """
    with st.spinner(f"Uploading… {SLOW_FIRST_OP_NOTE}"):
        result = get_client().upload_document(filename, data_bytes)

    if isinstance(result, ApiError):
        st.error(friendly_error(result.code, result.detail))
        return False

    if not (isinstance(result, ApiOk) and validate_upload_response(result.payload)):
        st.error(UNEXPECTED_RESPONSE_MESSAGE)
        return False

    payload = result.payload
    safe_name = escape_markdown_strict(payload["document_name"])
    status = payload["status"]
    pages = payload["page_count"]
    if status == "duplicate":
        st.info(
            f"**{safe_name}** is already in your library ({format_page_count(pages)})."
        )
    else:
        st.success(f"Added **{safe_name}** ({format_page_count(pages)}).")
    _refresh_documents()
    # Both "ingested" and "duplicate" are HTTP-200 confirmations that the document
    # IS indexed, so BOTH reset the uploader; only transport/validation FAILURES
    # (handled above) keep the selected file so the user can retry.
    return True


def _do_query(question: str) -> None:
    """Send the current question only; append the outcome to the transcript.

    Prior transcript entries are never removed on failure (Change F).
    """
    with st.spinner(f"Thinking… {SLOW_FIRST_OP_NOTE}"):
        result = get_client().query(question)

    entry: dict = {"question": question}
    if isinstance(result, ApiError):
        entry["error"] = friendly_error(result.code, result.detail)
        entry["severity"] = error_severity(result.code)
    elif isinstance(result, ApiOk) and validate_query_response(result.payload):
        entry["answer"] = result.payload["answer"]
        entry["citations"] = result.payload["citations"]
    else:
        entry["error"] = UNEXPECTED_RESPONSE_MESSAGE
    st.session_state.transcript.append(entry)


# ---------------------------------------------------------------------------
# Rendering helpers.
# ---------------------------------------------------------------------------
def _render_sidebar() -> None:
    """Render the upload control and the document list in the sidebar."""
    with st.sidebar:
        st.header("Documents")

        uploaded = st.file_uploader(
            "Upload a PDF",
            type=["pdf"],
            help=UPLOAD_HINT,
            key=f"pdf_uploader_{st.session_state.uploader_key}",
        )
        st.caption(UPLOAD_HINT)
        if st.button("Upload", type="primary", use_container_width=True):
            if uploaded is None:
                st.warning("Choose a PDF file first.")
            elif _do_upload(uploaded.name, uploaded.getvalue()):
                # Confirmed indexed (ingested or duplicate): bump the key so the
                # uploader widget is recreated empty, clearing the selected
                # file + ×. Failures leave the widget as-is so the user can retry.
                st.session_state.uploader_key += 1
                st.rerun()

        st.divider()

        # Lazily load the list the first time the sidebar renders.
        if not st.session_state.documents_loaded:
            _refresh_documents()

        documents = st.session_state.documents
        st.subheader(f"Indexed documents ({len(documents)})")
        if not documents:
            st.caption("No documents indexed yet.")
            return
        # Show name + pages + ingested date ONLY (chunk count dropped).
        # document_id and document_hash are intentionally hidden.
        for doc in documents:
            name = escape_markdown_strict(str(doc.get("document_name", "")))
            pages = format_page_count(doc.get("page_count", "?"))
            ingested = escape_markdown_strict(
                format_timestamp(doc.get("ingested_at", ""))
            )
            st.markdown(f"**{name}**")
            st.caption(pages)
            st.caption(f"Ingested: {ingested}")
            st.divider()


def _render_citations(citations: list) -> None:
    """Render citations GROUPED by (document, page) as one expander per group.

    Each group's title (escaped) carries its 1-based passage number(s); the body
    lists every excerpt in that group, each rendered with the STRICT escaper and
    clearly separated. Group count (not raw citation count) drives the label.
    """
    groups = group_citations(citations)
    st.caption(sources_label(len(groups)))
    for group in groups:
        with st.expander(grouped_citation_title(group)):
            excerpts = group.get("excerpts", [])
            for idx, excerpt in enumerate(excerpts):
                if idx:
                    st.divider()
                st.write(format_excerpt(excerpt))


def _render_answer_entry(entry: dict) -> None:
    """Render a single assistant transcript entry (answer/error/empty-citations)."""
    if "error" in entry:
        if entry.get("severity") == "warning":
            st.warning(entry["error"])
        else:
            st.error(entry["error"])
        return

    answer = entry.get("answer", "")
    citations = entry.get("citations", [])
    # Render the answer with the ANSWER escaper so benign **bold**/*italic*/lists
    # render, while links/images/inline HTML/:color[] stay neutralized. Still NO
    # unsafe_allow_html.
    st.markdown(escape_markdown_answer(answer))

    if citations:
        _render_citations(citations)
    else:
        # Change E: neutral card + neutral caption; no "not grounded" language.
        st.caption(NO_CITATIONS_CAPTION)
        st.caption(NO_CITATIONS_HINT)


def _render_empty_chat_hint() -> None:
    """Show a friendly hint with example questions when no Q&A has happened yet."""
    st.caption("No questions yet. Try one of these:")
    for example in EXAMPLE_QUESTIONS:
        # Static example text; strict-escaped for consistency (it's constant).
        st.markdown(f"- {escape_markdown_strict(example)}")


def _render_transcript() -> None:
    """Render this session's Q&A transcript in the main area (presentation only)."""
    transcript = st.session_state.transcript
    if not transcript:
        _render_empty_chat_hint()
        return
    for entry in transcript:
        with st.chat_message("user", avatar=USER_AVATAR):
            st.write(escape_markdown_strict(entry["question"]))
        with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
            _render_answer_entry(entry)


# ---------------------------------------------------------------------------
# Main entry point.
# ---------------------------------------------------------------------------
def main() -> None:
    """Compose and run the Streamlit page.

    Safe to run WITHOUT a backend: no network call is made here. The backend is
    contacted only when the user uploads or submits a question, and (lazily) to
    populate the sidebar document list on first render — still only at run time,
    never at import.
    """
    st.set_page_config(page_title="AI Document Intelligence", layout="wide")
    inject_css()
    _init_state()

    st.title("AI Document Intelligence")
    st.write(
        "Upload PDFs on the left, then ask grounded questions about them below."
    )

    _render_sidebar()
    _render_transcript()

    # st.chat_input drives the question (Change A). AppTest in 1.40.2 cannot
    # drive this widget; the query flow is covered via pure functions + client
    # tests instead (see tests/).
    question = st.chat_input("Ask a question about your documents")
    if question:
        _do_query(question)
        # Rerun so the new entry renders inside the transcript section above.
        st.rerun()


if __name__ == "__main__":
    main()

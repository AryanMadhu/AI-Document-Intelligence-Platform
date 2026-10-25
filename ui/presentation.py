"""Pure presentation + safety logic for the UI (NO Streamlit, NO network).

Everything here is a pure function of its inputs. This module is deliberately
free of any Streamlit import and never makes a network call, which makes it
fully unit-testable offline and keeps the security-critical logic isolated.

Responsibilities:

* **Markdown/HTML injection defense** — two escapers:
  :func:`escape_markdown_strict` and :func:`escape_markdown_answer`.
  Backend answers and excerpts are untrusted text that we render in a
  Markdown-capable surface. A naive render of a payload like
  ``![x](http://evil/leak)`` or ``[click](http://evil)`` would make the browser
  fetch an attacker URL (an ``<img>``/``<a>``), which can leak that the user
  viewed the content and can be used for tracking/exfiltration.

  - :func:`escape_markdown_strict` backslash-escapes every character that can
    *start* Markdown link/image syntax or inline HTML
    (``[ ] ( ) ! ` * _ < > # ~ |``), so the payload renders as literal text
    instead of an active element. This is used for VERBATIM text: citation
    excerpts, document names, filenames, and the user's own question echo.
    ``escape_markdown`` is kept as an alias of this strict escaper for backward
    compatibility.
  - :func:`escape_markdown_answer` escapes ONLY ``\\ [ ] <``. This lets benign
    formatting (``**bold**``, ``*italic*``, ``_em_``, ``#`` headings, ``|``
    tables, lists, ``*`` bullet items) render in the LLM answer, while still
    neutralizing links ``[a](url)``, images ``![a](url)``, reference links
    (``[a][r]`` + ``[r]: url``), inline HTML (``<img ...>``), and Streamlit
    color/directive syntax (``:red[x]``) — all of which require ``[``, ``]`` or
    ``<`` that this escaper neutralizes.
* **Error-code -> friendly message mapping** (:func:`friendly_error`). Maps the
  backend's stable error codes (and client-side transport failures) to safe,
  generic, user-facing messages. For ``FILE_TOO_LARGE`` / ``TOO_MANY_PAGES`` the
  backend's ``detail`` is appended ONLY when it matches an exact, known-safe
  regex; anything else is dropped. No raw JSON body, document id, traceback,
  path, or secret is ever surfaced.
* **Response-shape validation** (:func:`validate_query_response`,
  :func:`validate_documents_response`, :func:`validate_upload_response`). Guards
  rendering so a malformed body never reaches the UI; callers fall back to a
  generic "unexpected response" message and keep prior state.
* **Display formatting** (:func:`page_badge_text`, :func:`format_citation_title`,
  :func:`format_excerpt`, :func:`format_timestamp`, :func:`group_citations`,
  :func:`grouped_citation_title`, :func:`sources_label`). All color decisions
  live in :mod:`ui.styles`; this module emits no color/CSS.
"""

from __future__ import annotations

import re
from datetime import timezone
from datetime import datetime
from typing import Any

# ---------------------------------------------------------------------------
# Generic, user-facing message for anything we won't/can't surface safely.
# ---------------------------------------------------------------------------
UNEXPECTED_RESPONSE_MESSAGE = "Unexpected response from the server."

# Neutral caption shown when an answer comes back with no citations (Change E).
NO_CITATIONS_CAPTION = "No supporting citations were returned for this response"
NO_CITATIONS_HINT = (
    "Try rephrasing your question, or upload a document relevant to it."
)


# ---------------------------------------------------------------------------
# Markdown / HTML injection defense (two escapers).
# ---------------------------------------------------------------------------

# STRICT: characters that can START Markdown link/image syntax or open inline
# HTML. Backslash-escaping each one makes Markdown treat it as a literal
# character, so link/image/HTML constructs are neutralized (rendered as text,
# not active elements). ``$`` is escaped too so a dollar amount like ``$5``
# renders as ``\$5`` and cannot trigger Streamlit's ``$...$`` LaTeX/math
# rendering. Order matters: the backslash itself is escaped FIRST so we don't
# double-escape the escapes we add afterwards.
_MD_ESCAPE_CHARS = "[]()!`*_<>#~|$"

# ANSWER: the MINIMAL set needed to make any active element impossible to form
# while leaving benign emphasis/heading/list/table syntax intact. Every
# Markdown link/image/reference-link, every inline HTML tag, and Streamlit's
# ``:color[...]`` directive REQUIRES at least one of ``[``, ``]`` or ``<`` —
# escaping those three (plus the backslash itself, first) is sufficient to
# neutralize them all, while ``*``/``_``/``#``/``|`` stay live so ``**bold**``,
# ``*italic*``, ``_em_``, ``#`` headings, ``|`` tables and ``*`` bullets render.
# ``$`` is also escaped so a dollar amount like ``$5`` renders as ``\$5`` and
# cannot trigger Streamlit's ``$...$`` LaTeX/math rendering.
_MD_ANSWER_ESCAPE_CHARS = "[]<$"


def escape_markdown_strict(text: Any) -> str:
    """Neutralize Markdown/HTML so VERBATIM untrusted text renders literally.

    Approach: backslash-escape the backslash first (so later escapes aren't
    themselves re-escaped), then backslash-escape every character that can begin
    a Markdown link/image or an inline HTML tag:
    ``[ ] ( ) ! ` * _ < > # ~ | $``. ``$`` is escaped so a dollar amount like
    ``$5`` renders as ``\\$5`` and cannot trigger Streamlit's ``$...$``
    LaTeX/math rendering. The result, rendered through a Markdown surface, shows
    the original characters verbatim and does NOT produce an ``<a>``, ``<img>``,
    or any HTML element — so a payload such as
    ``![a](http://evil/x.png)`` or ``<img src=x onerror=...>`` cannot cause the
    browser to fetch an attacker URL or execute anything.

    Use this for text that must appear EXACTLY as given: citation excerpts,
    document names, filenames, and the user's own question echo.

    Non-string inputs are coerced to ``str`` defensively so a malformed backend
    value can never raise here.

    Args:
        text: Untrusted VERBATIM text (excerpt, document name, filename, echo).

    Returns:
        The escaped string, safe to hand to ``st.markdown`` WITHOUT
        ``unsafe_allow_html`` (or to ``st.write``/``st.text``).
    """
    if not isinstance(text, str):
        text = str(text)
    # Escape the backslash first.
    text = text.replace("\\", "\\\\")
    for ch in _MD_ESCAPE_CHARS:
        text = text.replace(ch, "\\" + ch)
    return text


def escape_markdown_answer(text: Any) -> str:
    """Neutralize ACTIVE elements in the LLM answer while keeping formatting.

    Escapes ONLY ``\\``, ``[``, ``]``, ``<`` and ``$`` (backslash first). This is
    the minimal set that makes it impossible to form any active element:

    * links ``[a](url)`` and reference links ``[a][r]`` / ``[r]: url`` — need
      ``[``/``]`` (escaped, so they render as literal brackets);
    * images ``![a](url)`` — need ``[``/``]`` (escaped);
    * inline HTML such as ``<img ...>`` — needs ``<`` (escaped);
    * Streamlit color/directive syntax ``:red[x]`` — needs ``[``/``]`` (escaped);
    * Streamlit ``$...$`` LaTeX/math — needs ``$`` (escaped), so a dollar amount
      like ``$5`` renders as ``\\$5`` instead of being treated as math.

    Meanwhile ``*``, ``_``, ``#``, ``|`` and list markers are left untouched, so
    ``**bold**``, ``*italic*``, ``_em_``, ``#`` headings, ``|`` tables and
    ``*`` bullet items all render normally. Use this ONLY for the trusted-shape
    LLM answer body — never for verbatim excerpts/names/filenames (use
    :func:`escape_markdown_strict` for those).

    Non-string inputs are coerced to ``str`` defensively.

    Args:
        text: The LLM answer text.

    Returns:
        The escaped string, safe to hand to ``st.markdown`` WITHOUT
        ``unsafe_allow_html``, with benign formatting preserved.
    """
    if not isinstance(text, str):
        text = str(text)
    # Escape the backslash first.
    text = text.replace("\\", "\\\\")
    for ch in _MD_ANSWER_ESCAPE_CHARS:
        text = text.replace(ch, "\\" + ch)
    return text


# Backward-compatible alias: existing callers/tests referencing
# ``escape_markdown`` get the STRICT behavior unchanged.
escape_markdown = escape_markdown_strict


# ---------------------------------------------------------------------------
# Change C — regex-gated detail for size/page limits.
#
# The ONLY backend ``detail`` strings we are willing to surface verbatim are the
# two limit messages, and ONLY when they match these exact patterns (anchored,
# no trailing junk). Anything else -> show the generic message alone. This stops
# an attacker-tampered or unexpected detail (e.g. one carrying an HTML payload,
# a path, or a secret) from ever reaching the screen.
# ---------------------------------------------------------------------------
_FILE_TOO_LARGE_DETAIL_RE = re.compile(r"^File exceeds the maximum size of \d+ MB\.$")
_TOO_MANY_PAGES_DETAIL_RE = re.compile(r"^File exceeds the maximum of \d+ pages\.$")

# Generic, safe, user-facing messages per error code. NEVER include ids,
# tracebacks, paths, secrets, or raw backend bodies.
_GENERIC_MESSAGES: dict[str, str] = {
    "EMPTY_QUESTION": "Please enter a question.",
    "INVALID_PDF": "That file could not be read as a valid PDF. Please upload a valid PDF.",
    "FILE_TOO_LARGE": "That file is too large to upload.",
    "TOO_MANY_PAGES": "That document has too many pages to upload.",
    "EMPTY_DOCUMENT": "That document contains no extractable text, so it can't be indexed.",
    "REEMBEDDING_REQUIRED": "Some documents need re-indexing; please contact the administrator.",
    "LLM_FAILED": "The assistant couldn't answer right now. Please try again.",
    "QUOTA_EXCEEDED": (
        "The usage limit for the assistant has been reached. Please check your "
        "quota and try again after it resets. Daily limits reset once per day; "
        "per-minute limits reset within a minute."
    ),
    "LLM_AUTH_FAILED": "The assistant isn't configured correctly. Please contact the administrator.",
    "SERVICE_BUSY": "The assistant is busy right now. Please wait a moment and try again.",
    "EMBEDDING_FAILED": "The document couldn't be processed right now. Please try again.",
    "DOCUMENT_NOT_FOUND": "That document could not be found.",
    "VALIDATION_ERROR": "The request was invalid. Please check your input and try again.",
    "INTERNAL_ERROR": "Something went wrong on the server. Please try again.",
    "MISSING_CREDENTIAL": "The server is not configured correctly. Please contact the administrator.",
    # Client-side transport failures (produced by ui.api_client, not the backend).
    "CONNECTION_ERROR": (
        "Can't reach the backend service. Make sure it's running and try again."
    ),
    "TIMEOUT": "The assistant is taking too long to respond. Please try again.",
    "UNEXPECTED_RESPONSE": UNEXPECTED_RESPONSE_MESSAGE,
}

# Error codes that describe a RECOVERABLE / transient condition — the UI renders
# these in a gentler "warning" (amber) style rather than a hard "error" (red)
# one. Everything NOT listed here is treated as a hard failure ("error").
_RECOVERABLE_CODES = frozenset(
    {
        "SERVICE_BUSY",
        "TIMEOUT",
        "CONNECTION_ERROR",
        "REEMBEDDING_REQUIRED",
        # Quota is recoverable once it resets (amber/warning), unlike the hard
        # LLM_AUTH_FAILED config error which stays red/error.
        "QUOTA_EXCEEDED",
    }
)


def error_severity(error_code: Any) -> str:
    """Return the display severity for an error code: ``"warning"`` or ``"error"``.

    Recoverable / transient conditions ({``SERVICE_BUSY``, ``TIMEOUT``,
    ``CONNECTION_ERROR``, ``REEMBEDDING_REQUIRED``}) return ``"warning"`` so the
    UI can show them in a gentler amber style; everything else (including
    unknown / non-string codes) returns ``"error"`` for the hard-failure red
    style. Pure function — no color/CSS lives here.

    Args:
        error_code: The backend/client error code (e.g. ``"SERVICE_BUSY"``).

    Returns:
        ``"warning"`` for recoverable codes, otherwise ``"error"``.
    """
    if isinstance(error_code, str) and error_code in _RECOVERABLE_CODES:
        return "warning"
    return "error"

# Fallback for any code we don't recognize -> generic, never leaks the code's
# payload or any detail.
_UNKNOWN_CODE_MESSAGE = "Something went wrong. Please try again."


def friendly_error(error_code: Any, detail: Any = None) -> str:
    """Map an error code (+ optional backend detail) to a safe user message.

    Only ``FILE_TOO_LARGE`` and ``TOO_MANY_PAGES`` may have the backend
    ``detail`` appended, and ONLY when ``detail`` matches the exact,
    whitelisted regex for that code (Change C). For every other code — and for a
    non-matching detail — the generic message is returned ALONE. Unknown codes
    return a generic fallback. No raw detail, JSON, id, path, or secret is ever
    included for a non-whitelisted detail.

    Args:
        error_code: The backend/client error code (e.g. ``"FILE_TOO_LARGE"``).
        detail: The backend ``detail`` string, if any. Ignored unless the code
            is one of the two whitelisted limit codes AND it matches the regex.

    Returns:
        A safe, user-facing message string.
    """
    if not isinstance(error_code, str):
        return _UNKNOWN_CODE_MESSAGE

    generic = _GENERIC_MESSAGES.get(error_code)
    if generic is None:
        return _UNKNOWN_CODE_MESSAGE

    # Regex-gated detail append for the two limit codes only.
    if isinstance(detail, str):
        if error_code == "FILE_TOO_LARGE" and _FILE_TOO_LARGE_DETAIL_RE.match(detail):
            return f"{generic} {detail}"
        if error_code == "TOO_MANY_PAGES" and _TOO_MANY_PAGES_DETAIL_RE.match(detail):
            return f"{generic} {detail}"

    return generic


# ---------------------------------------------------------------------------
# Change F — response-shape validation helpers.
#
# Each validator returns True only when the body is a dict with the expected
# fields present and correctly typed. Callers that get False must show the
# generic "unexpected response" message and keep prior state.
# ---------------------------------------------------------------------------


def _is_citation(value: Any) -> bool:
    """Return True if ``value`` is a well-formed citation object."""
    if not isinstance(value, dict):
        return False
    if not isinstance(value.get("document"), str):
        return False
    # page may be int or str (Citation.page is int | str).
    page = value.get("page")
    if not isinstance(page, (int, str)) or isinstance(page, bool):
        return False
    if not isinstance(value.get("excerpt"), str):
        return False
    return True


def validate_query_response(body: Any) -> bool:
    """Validate a ``POST /query`` success body.

    Requires a dict with a string ``answer`` and a list ``citations`` where each
    citation has string ``document``, int-or-str ``page``, and string
    ``excerpt``.
    """
    if not isinstance(body, dict):
        return False
    if not isinstance(body.get("answer"), str):
        return False
    citations = body.get("citations")
    if not isinstance(citations, list):
        return False
    return all(_is_citation(c) for c in citations)


def validate_upload_response(body: Any) -> bool:
    """Validate a ``POST /documents/upload`` success body (UploadResponse shape)."""
    if not isinstance(body, dict):
        return False
    required_str = ("document_id", "document_name", "status", "message")
    for key in required_str:
        if not isinstance(body.get(key), str):
            return False
    for key in ("page_count", "chunk_count"):
        val = body.get(key)
        if not isinstance(val, int) or isinstance(val, bool):
            return False
    return True


def _is_document_record(value: Any) -> bool:
    """Return True if ``value`` is a well-formed DocumentRecord object."""
    if not isinstance(value, dict):
        return False
    # Only the fields the UI actually shows are strictly required; but we
    # validate the core identity/count fields to catch a malformed body.
    if not isinstance(value.get("document_name"), str):
        return False
    for key in ("page_count", "chunk_count"):
        val = value.get(key)
        if not isinstance(val, int) or isinstance(val, bool):
            return False
    if not isinstance(value.get("ingested_at"), str):
        return False
    return True


def validate_documents_response(body: Any) -> bool:
    """Validate a ``GET /documents`` success body.

    Requires a dict with a list ``documents`` (possibly empty) whose entries are
    well-formed document records.
    """
    if not isinstance(body, dict):
        return False
    documents = body.get("documents")
    if not isinstance(documents, list):
        return False
    return all(_is_document_record(d) for d in documents)


# ---------------------------------------------------------------------------
# Display formatting.
# ---------------------------------------------------------------------------


def page_badge_text(page: Any) -> str:
    """Return the short badge label for a citation page (handles int | str).

    * An int page ``7`` -> ``"p. 7"``.
    * A string page (e.g. a section identifier like ``"Intro"``) -> that string
      as-is (trimmed). Blank/None -> a neutral placeholder.
    """
    if isinstance(page, bool):
        # bool is an int subclass; treat defensively as unknown.
        return "p. ?"
    if isinstance(page, int):
        return f"p. {page}"
    if isinstance(page, str):
        stripped = page.strip()
        return stripped if stripped else "p. ?"
    return "p. ?"


def format_citation_title(document: Any, page: Any) -> str:
    """Build the (ESCAPED) expander title for a citation: name + page badge.

    Both the document name and the page label are escaped so neither can inject
    Markdown/HTML. Rendered as normal Markdown text (no ``unsafe_allow_html``).
    """
    safe_doc = escape_markdown(document if isinstance(document, str) else str(document))
    safe_badge = escape_markdown(page_badge_text(page))
    return f"{safe_doc} — {safe_badge}"


def format_excerpt(excerpt: Any) -> str:
    """Return the ESCAPED (strict) excerpt text, safe for Markdown rendering."""
    return escape_markdown_strict(
        excerpt if isinstance(excerpt, str) else str(excerpt)
    )


def format_timestamp(value: Any) -> str:
    """Format an ISO-8601 timestamp as ``"18 Oct 2026 (UTC)"`` for display.

    Parses ``value`` with :func:`datetime.datetime.fromisoformat` (handling a
    trailing ``Z`` by swapping it for ``+00:00``), converts to UTC, and renders
    ``"%-d"`` is avoided for cross-platform safety — the day is emitted without a
    leading zero via ``int()``. Examples that parse: ``2026-10-18T13:15:28Z``,
    ``2026-10-18T13:15:28.589634Z``, ``2026-10-18T13:15:28`` (no Z / no micros).

    Fallback (never raises): on ANY malformed / empty / non-string input, return
    the original string if it is a NON-EMPTY string, otherwise return ``""``.

    Args:
        value: An ISO-8601 timestamp string (or anything, defensively).

    Returns:
        ``"D Mon YYYY (UTC)"`` on success, else the safe fallback above.
    """
    if not isinstance(value, str) or not value.strip():
        # Non-string or empty -> empty string (the only safe fallback).
        return value if isinstance(value, str) else ""

    raw = value.strip()
    iso = raw[:-1] + "+00:00" if raw.endswith("Z") else raw
    try:
        parsed = datetime.fromisoformat(iso)
    except (ValueError, TypeError):
        # Malformed but non-empty string -> return it unchanged.
        return raw

    # Treat naive timestamps as UTC; convert aware ones to UTC for display.
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    parsed = parsed.astimezone(timezone.utc)
    return f"{parsed.day} {parsed.strftime('%b %Y')} (UTC)"


def group_citations(citations: Any) -> list[dict]:
    """Group a response-ordered citation list by ``(document, page)``.

    Citations arrive in RESPONSE ORDER, which (confirmed) equals prompt passage
    order — so the Nth citation (1-based) is prompt passage N. We group adjacent
    or repeated citations that share the same ``(document, page)`` key while
    PRESERVING first-appearance order of the groups, and for each group collect:

    * ``document`` / ``page`` — the shared key values (page kept as-is, so a
      string page groups by that string);
    * ``excerpts`` — every excerpt in that group, in original order;
    * ``passages`` — the 1-based index (``index + 1``) of each citation in the
      ORIGINAL input list, in original order.

    Deterministic. An empty / non-list input yields ``[]``.

    Args:
        citations: The ``citations`` list from a ``/query`` response.

    Returns:
        A list of ``{"document", "page", "excerpts": [...], "passages": [...]}``
        dicts in first-appearance order.
    """
    if not isinstance(citations, list):
        return []

    groups: list[dict] = []
    index: dict[tuple, dict] = {}
    for i, cite in enumerate(citations):
        if not isinstance(cite, dict):
            continue
        document = cite.get("document")
        page = cite.get("page")
        excerpt = cite.get("excerpt")
        key = (document, page)
        passage_number = i + 1  # 1-based passage number == response position.
        if key not in index:
            group = {
                "document": document,
                "page": page,
                "excerpts": [],
                "passages": [],
            }
            index[key] = group
            groups.append(group)
        index[key]["excerpts"].append(excerpt)
        index[key]["passages"].append(passage_number)
    return groups


def _passage_suffix(passages: Any) -> str:
    """Build the passage suffix: ``" · passage 2"`` or ``" · passages 2, 3"``.

    Uses sorted, de-duplicated passage numbers. Numeric, so inherently safe (no
    escaping needed). Returns ``""`` if there are no passages.
    """
    if not isinstance(passages, (list, tuple)) or not passages:
        return ""
    unique = sorted({p for p in passages if isinstance(p, int)})
    if not unique:
        return ""
    if len(unique) == 1:
        return f" · passage {unique[0]}"
    return " · passages " + ", ".join(str(p) for p in unique)


def grouped_citation_title(group: Any) -> str:
    """Build the ESCAPED (strict) expander title for a grouped citation.

    Shape: ``"{safe_doc} — {safe_badge}{passage_suffix}"`` where the document
    name and page badge are escaped with the STRICT escaper (verbatim, inert)
    and the passage suffix is numeric text (``" · passage 2"`` /
    ``" · passages 2, 3"``) which is safe by construction.

    Args:
        group: A dict as produced by :func:`group_citations`.

    Returns:
        The escaped title string (safe for plain Markdown, no HTML).
    """
    if not isinstance(group, dict):
        group = {}
    document = group.get("document")
    page = group.get("page")
    safe_doc = escape_markdown_strict(
        document if isinstance(document, str) else str(document)
    )
    safe_badge = escape_markdown_strict(page_badge_text(page))
    return f"{safe_doc} — {safe_badge}{_passage_suffix(group.get('passages'))}"


def sources_label(n: Any) -> str:
    """Return the sources header label, e.g. ``"Sources (3)"``."""
    return f"Sources ({n})"


def format_page_count(page_count: Any) -> str:
    """Return a grammatically correct page-count label.

    * An int equal to ``1`` -> ``"1 page"`` (singular).
    * Any other int -> ``"{n} pages"`` (e.g. ``0`` -> ``"0 pages"``,
      ``2`` -> ``"2 pages"``).
    * A missing / non-int value (e.g. the ``"?"`` fallback the sidebar uses) ->
      ``"{value} pages"`` defensively, so this never raises.

    Args:
        page_count: The document's page count (int, or a fallback like ``"?"``).

    Returns:
        A display string such as ``"1 page"`` or ``"3 pages"``.
    """
    if isinstance(page_count, int) and not isinstance(page_count, bool):
        if page_count == 1:
            return "1 page"
        return f"{page_count} pages"
    return f"{page_count} pages"

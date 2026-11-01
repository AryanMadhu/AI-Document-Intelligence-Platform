"""HTTP client for the FastAPI backend (``requests``-based, Streamlit-free).

This module wraps the three backend endpoints the UI uses. It is deliberately
side-effect-free at import time: constructing :class:`ApiClient` reads the
``API_BASE_URL`` environment variable but makes NO network call, and importing
the module alone does nothing. All network calls happen only when a method is
invoked (i.e. on a user action).

Design points (per approved decisions):

* **Transport**: ``requests`` with a fixed timeout tuple ``(5, 60)`` —
  connect=5s, read=60s. There is NO automatic upload retry.
* **Base URL**: from the ``API_BASE_URL`` env var, defaulting to
  ``http://127.0.0.1:8000``. The URL is never printed or logged.
* **Contracts**:
    - upload: ``POST /documents/upload`` multipart with the form field named
      exactly ``file``.
    - list:   ``GET /documents``.
    - query:  ``POST /query`` with JSON body ``{"question": <str>}`` only — no
      ``top_k``, no conversation history.
* **Typed results, never raw raises**: every method returns either
  :class:`ApiOk` (with the parsed JSON payload) or :class:`ApiError` (with a
  stable code + safe message). ``requests.ConnectionError`` ->
  ``CONNECTION_ERROR``; ``requests.Timeout`` -> ``TIMEOUT``; a non-JSON body or
  any other transport problem -> ``UNEXPECTED_RESPONSE``. The backend's error
  envelope ``{"error": CODE, "detail": ...}`` is parsed and surfaced as an
  ``ApiError`` carrying that code and the raw detail (the UI layer decides what,
  if any, of the detail is safe to show — see ui.presentation.friendly_error).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import requests

# Fixed timeout tuple: (connect=5s, read=60s). The first operation after a cold
# start can be slow (model load), hence the generous read timeout.
_TIMEOUT: tuple[int, int] = (5, 60)

_DEFAULT_BASE_URL = "http://127.0.0.1:8000"


@dataclass(frozen=True)
class ApiOk:
    """A successful API result carrying the parsed JSON payload."""

    payload: dict[str, Any]


@dataclass(frozen=True)
class ApiError:
    """A failed API result with a stable code and a safe-to-map message.

    Attributes:
        code: Stable error code — either a backend code (e.g. ``FILE_TOO_LARGE``)
            or a client-side transport code (``CONNECTION_ERROR``, ``TIMEOUT``,
            ``UNEXPECTED_RESPONSE``).
        detail: The backend ``detail`` string when present, else ``None``. The
            UI layer (ui.presentation) decides whether any of it is safe to
            surface; this client does not render anything.
        status_code: The HTTP status if an HTTP response was received, else
            ``None`` (transport failures).
    """

    code: str
    detail: str | None = None
    status_code: int | None = None


def resolve_base_url() -> str:
    """Return the backend base URL from ``API_BASE_URL`` (default localhost:8000).

    The value is read from the environment and the trailing slash is trimmed so
    paths join cleanly. The URL is never logged or printed.
    """
    return os.getenv("API_BASE_URL", _DEFAULT_BASE_URL).rstrip("/")


def _parse_error(response: requests.Response) -> ApiError:
    """Translate a non-2xx response into a typed :class:`ApiError`.

    Attempts to read the backend error envelope ``{"error", "detail"}``. If the
    body isn't JSON or doesn't carry an ``error`` code, falls back to a generic
    ``UNEXPECTED_RESPONSE`` so nothing unvetted is surfaced.
    """
    try:
        body = response.json()
    except ValueError:
        return ApiError(code="UNEXPECTED_RESPONSE", status_code=response.status_code)
    if isinstance(body, dict) and isinstance(body.get("error"), str):
        detail = body.get("detail")
        return ApiError(
            code=body["error"],
            detail=detail if isinstance(detail, str) else None,
            status_code=response.status_code,
        )
    return ApiError(code="UNEXPECTED_RESPONSE", status_code=response.status_code)


def _parse_success(response: requests.Response) -> ApiOk | ApiError:
    """Parse a 2xx response body as JSON into :class:`ApiOk`.

    A 2xx with a non-JSON or non-dict body is treated as
    ``UNEXPECTED_RESPONSE`` (schema is validated further by the UI layer).
    """
    try:
        body = response.json()
    except ValueError:
        return ApiError(code="UNEXPECTED_RESPONSE", status_code=response.status_code)
    if not isinstance(body, dict):
        return ApiError(code="UNEXPECTED_RESPONSE", status_code=response.status_code)
    return ApiOk(payload=body)


class ApiClient:
    """Thin ``requests`` wrapper over the backend HTTP contracts.

    Construction reads the base URL from the environment but performs NO network
    I/O. A :class:`requests.Session` is used for connection reuse; it is created
    eagerly but opens no socket until a request is actually made. Tests inject a
    fake by replacing the module-level client (see :func:`get_client`) or by
    monkeypatching ``self.session``.
    """

    def __init__(self, base_url: str | None = None) -> None:
        self.base_url = (base_url or resolve_base_url()).rstrip("/")
        self.session = requests.Session()

    # -- internal request helpers -------------------------------------------

    def _post(self, path: str, **kwargs: Any) -> ApiOk | ApiError:
        """POST and translate the outcome into a typed result."""
        return self._request("POST", path, **kwargs)

    def _get(self, path: str, **kwargs: Any) -> ApiOk | ApiError:
        """GET and translate the outcome into a typed result."""
        return self._request("GET", path, **kwargs)

    def _delete(self, path: str, **kwargs: Any) -> ApiOk | ApiError:
        """DELETE and translate the outcome into a typed result."""
        return self._request("DELETE", path, **kwargs)

    def _request(self, method: str, path: str, **kwargs: Any) -> ApiOk | ApiError:
        """Run an HTTP request, converting transport errors to typed results.

        Never raises on a transport failure: ``ConnectionError`` ->
        ``CONNECTION_ERROR``, ``Timeout`` -> ``TIMEOUT``, any other
        ``RequestException`` -> ``UNEXPECTED_RESPONSE``.
        """
        url = f"{self.base_url}{path}"
        try:
            response = self.session.request(
                method, url, timeout=_TIMEOUT, **kwargs
            )
        except requests.Timeout:
            return ApiError(code="TIMEOUT")
        except requests.ConnectionError:
            return ApiError(code="CONNECTION_ERROR")
        except requests.RequestException:
            return ApiError(code="UNEXPECTED_RESPONSE")

        if response.status_code >= 400:
            return _parse_error(response)
        return _parse_success(response)

    # -- public contract methods --------------------------------------------

    def upload_document(self, filename: str, data_bytes: bytes) -> ApiOk | ApiError:
        """Upload a PDF via ``POST /documents/upload``.

        Sends multipart/form-data with the form field named EXACTLY ``file``
        (matching the backend contract). No retry on failure.
        """
        files = {"file": (filename, data_bytes, "application/pdf")}
        return self._post("/documents/upload", files=files)

    def list_documents(self) -> ApiOk | ApiError:
        """List ingested documents via ``GET /documents``."""
        return self._get("/documents")

    def query(self, question: str) -> ApiOk | ApiError:
        """Ask a question via ``POST /query``.

        Sends JSON ``{"question": question}`` ONLY — never ``top_k`` and never
        any conversation history.
        """
        return self._post("/query", json={"question": question})

    def delete_document(self, document_id: str) -> ApiOk | ApiError:
        """Delete a document via ``DELETE /documents/{document_id}``.

        The backend returns a JSON object body (HTTP 200) on success, so
        ``ApiOk.payload`` holds ``{document_id, status, message}``. Errors reuse
        the shared handling: a backend ``{error, detail}`` envelope -> an
        ``ApiError`` with that code (e.g. ``DOCUMENT_NOT_FOUND`` for a 404), and
        transport failures -> ``CONNECTION_ERROR`` / ``TIMEOUT`` /
        ``UNEXPECTED_RESPONSE``.
        """
        return self._delete(f"/documents/{document_id}")


# ---------------------------------------------------------------------------
# Replaceable module-level client (so the app and tests share one instance and
# tests can inject a fake without any network). Lazily constructed on first use;
# NOT built at import time.
# ---------------------------------------------------------------------------
_client: ApiClient | None = None


def get_client() -> ApiClient:
    """Return the process-wide :class:`ApiClient`, building it on first use.

    Lazy construction keeps import side-effect-free. Tests can override the
    client by calling :func:`set_client` with a fake, or by monkeypatching this
    function — no network is required.
    """
    global _client
    if _client is None:
        _client = ApiClient()
    return _client


def set_client(client: Any) -> None:
    """Install a client instance (used by tests to inject a fake)."""
    global _client
    _client = client

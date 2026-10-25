"""UI package: a thin Streamlit HTTP client for the FastAPI backend.

This package is intentionally lightweight. Importing it has NO side effects:
no network calls, no backend contact, no heavy construction. The three leaf
modules split responsibilities cleanly:

* :mod:`ui.presentation` — pure, Streamlit-free, network-free functions
  (security-critical Markdown/HTML escaping, error-code-to-message mapping,
  response-shape validation, display formatting).
* :mod:`ui.api_client` — a ``requests``-based client wrapping the backend HTTP
  contracts; returns typed results/errors and never raises raw transport
  exceptions. No Streamlit import.
* :mod:`ui.styles` — the single static CSS block (the only place
  ``unsafe_allow_html=True`` is used).
* :mod:`ui.app` — the Streamlit composition layer that wires the above together.
"""

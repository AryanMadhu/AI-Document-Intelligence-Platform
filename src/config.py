"""Typed application configuration for the AI Document Intelligence Platform.

This module centralizes all runtime-tunable settings and per-provider
credentials. Settings are loaded from environment variables (populated from a
local ``.env`` file in development via ``python-dotenv``, or from
provider-injected environment variables in deployment) and fall back to the
design's finalized defaults when unset.

Security notes (R16.4, R17.3, R17.4):
    * Credentials are read from the environment ONLY and are never hard-coded.
    * A missing required credential is reported BY NAME ONLY; its value is
      never logged, echoed, or embedded in any error, string, or ``repr``.
    * ``Config.__repr__`` deliberately omits credential values so secrets can
      never leak into logs.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from typing import Optional

from dotenv import load_dotenv


class MissingCredentialError(Exception):
    """Raised when a required provider credential is absent or empty.

    The error message names the missing credential key only and never contains
    the credential's value (R16.4, R17.4).
    """


def _get_int(env_name: str, default: int) -> int:
    """Return an integer setting from the environment, or ``default``.

    The raw environment value is a string; it is coerced to ``int``. An empty
    or unset variable falls back to ``default``.

    Args:
        env_name: The environment variable name to read.
        default: The value to use when the variable is unset or empty.

    Returns:
        The parsed integer value.

    Raises:
        ValueError: If the environment value is present but not a valid int.
    """
    raw = os.environ.get(env_name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw.strip())


def _get_str(env_name: str, default: str) -> str:
    """Return a string setting from the environment, or ``default``.

    An unset or empty (whitespace-only) variable falls back to ``default``.
    """
    raw = os.environ.get(env_name)
    if raw is None or raw.strip() == "":
        return default
    return raw


def _get_optional_str(env_name: str) -> Optional[str]:
    """Return a credential string from the environment, or ``None`` if unset/empty."""
    raw = os.environ.get(env_name)
    if raw is None or raw.strip() == "":
        return None
    return raw


@dataclass
class Config:
    """Typed, immutable-ish snapshot of platform configuration.

    Non-secret settings appear in ``repr``; credential fields are marked
    ``repr=False`` so their values can never leak into logs.

    Attributes:
        embedding_provider: Key selecting the embedding provider implementation.
        embedding_model: Model identifier for the embedding provider.
        llm_provider: Key selecting the LLM provider implementation.
        llm_model: Model identifier for the LLM provider.
        top_k: Number of chunks to retrieve per query.
        chunk_size_tokens: Target chunk size in tokens.
        chunk_overlap_tokens: Overlap between consecutive chunks in tokens.
        max_pdf_size_mb: Maximum accepted PDF size in megabytes.
        max_pdf_pages: Maximum accepted PDF page count.
        gemini_api_key: Google Gemini credential (env-only, may be ``None``).
        openai_api_key: OpenAI credential (env-only, may be ``None``).
        anthropic_api_key: Anthropic credential (env-only, may be ``None``).
    """

    embedding_provider: str = "sentence-transformers"
    embedding_model: str = "all-MiniLM-L6-v2"
    llm_provider: str = "gemini"
    llm_model: str = "gemini-flash-latest"
    top_k: int = 5
    chunk_size_tokens: int = 700
    chunk_overlap_tokens: int = 90
    max_pdf_size_mb: int = 5
    max_pdf_pages: int = 50

    # Credentials: env-only, never rendered in repr to prevent secret leakage.
    gemini_api_key: Optional[str] = field(default=None, repr=False)
    openai_api_key: Optional[str] = field(default=None, repr=False)
    anthropic_api_key: Optional[str] = field(default=None, repr=False)

    # Names of fields that hold secrets; used by credential lookups.
    _CREDENTIAL_FIELDS = ("gemini_api_key", "openai_api_key", "anthropic_api_key")

    @classmethod
    def from_env(cls, load_env_file: bool = True) -> "Config":
        """Build a ``Config`` from environment variables (and optionally ``.env``).

        ``load_dotenv()`` is called so a developer's local ``.env`` is picked
        up, but variables already set in the process take precedence (this is
        ``python-dotenv``'s default ``override=False`` behavior). Integer
        settings are coerced from their string representations.

        Args:
            load_env_file: When ``True`` (default), load a local ``.env`` file.

        Returns:
            A fully-populated ``Config`` instance.
        """
        if load_env_file:
            load_dotenv()

        return cls(
            embedding_provider=_get_str("EMBEDDING_PROVIDER", "sentence-transformers"),
            embedding_model=_get_str("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
            llm_provider=_get_str("LLM_PROVIDER", "gemini"),
            llm_model=_get_str("LLM_MODEL", "gemini-flash-latest"),
            top_k=_get_int("TOP_K", 5),
            chunk_size_tokens=_get_int("CHUNK_SIZE_TOKENS", 700),
            chunk_overlap_tokens=_get_int("CHUNK_OVERLAP_TOKENS", 90),
            max_pdf_size_mb=_get_int("MAX_PDF_SIZE_MB", 5),
            max_pdf_pages=_get_int("MAX_PDF_PAGES", 50),
            gemini_api_key=_get_optional_str("GEMINI_API_KEY"),
            openai_api_key=_get_optional_str("OPENAI_API_KEY"),
            anthropic_api_key=_get_optional_str("ANTHROPIC_API_KEY"),
        )

    def require_credential(self, key_name: str) -> str:
        """Return a required credential value, or raise a value-free error.

        Args:
            key_name: The credential's environment variable name (e.g.
                ``"GEMINI_API_KEY"``) or the equivalent attribute name (e.g.
                ``"gemini_api_key"``).

        Returns:
            The non-empty credential value.

        Raises:
            MissingCredentialError: If the credential is unset or empty. The
                message names the key only and never contains its value
                (R16.4, R17.4).
        """
        attr_name = key_name.lower()
        if attr_name not in self._CREDENTIAL_FIELDS:
            raise MissingCredentialError(f"{key_name} is not a known credential")

        value = getattr(self, attr_name)
        if value is None or value == "":
            # Report by NAME ONLY — never include the (absent) value.
            raise MissingCredentialError(f"{key_name.upper()} is not set")
        return value

    def __repr__(self) -> str:  # pragma: no cover - trivial formatting
        """Return a repr containing only non-secret settings.

        Credential fields are intentionally excluded so that no secret value
        can leak into logs (R16.4, R17.4).
        """
        parts = [
            f"{f.name}={getattr(self, f.name)!r}"
            for f in fields(self)
            if f.repr and f.name not in self._CREDENTIAL_FIELDS
        ]
        return f"Config({', '.join(parts)})"

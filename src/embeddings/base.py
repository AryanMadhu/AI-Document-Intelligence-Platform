"""Embedding provider interface and Config-driven factory.

This module defines the ``EmbeddingProvider`` abstraction — the swappable seam
that Requirement 15 mandates for embedding models — plus the
``build_embedding_provider`` factory that selects a concrete implementation
from a :class:`~src.config.Config` value (never from hard-coded application
logic).

Security notes (R16.4):
    * The factory reports an unknown provider or a missing credential BY NAME
      / by the offending selector string only. No credential value is ever
      included in an ``EmbeddingError`` message.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.config import Config


class EmbeddingError(Exception):
    """Raised when embedding fails or the factory is misconfigured.

    Messages are always value-free: they may name a provider selector string
    or a required credential key, but they never contain a credential's value
    (R16.4).
    """


class EmbeddingProvider(ABC):
    """Converts text into vector embeddings. Swappable via ``Config``.

    Concrete implementations expose a stable ``provider_name`` and
    ``model_name`` (recorded as embedding provenance, R5.5, and used by the
    embedding-space compatibility check, R15.6) plus the embedding
    ``dimension``. They must implement both batch embedding (for ingestion)
    and single-query embedding (for retrieval).
    """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Stable provider identifier.

        Returns:
            A stable string id, e.g. ``'sentence-transformers'`` or
            ``'openai'``.
        """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Stable embedding model identifier.

        Returns:
            The model id, e.g. ``'all-MiniLM-L6-v2'``.
        """

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Embedding vector dimensionality.

        Returns:
            The number of components in each embedding vector.
        """

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of chunk texts (ingestion path).

        Args:
            texts: The chunk texts to embed. An empty list yields ``[]``.

        Returns:
            One embedding vector (a ``list[float]``) per input text, in order.

        Raises:
            EmbeddingError: If the underlying model fails to embed the batch.
        """

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Embed a single question (retrieval path).

        Args:
            text: The query string to embed.

        Returns:
            A single embedding vector as a flat ``list[float]``.

        Raises:
            EmbeddingError: If the underlying model fails to embed the query.
        """


def build_embedding_provider(config: Config) -> EmbeddingProvider:
    """Build the active :class:`EmbeddingProvider` from configuration.

    Provider selection is driven entirely by ``config.embedding_provider`` —
    it is never hard-coded in application logic. Switching providers is a
    configuration (``.env``) change, not a source change (R15.3).

    The concrete implementation module is imported *locally*, inside the
    branch that selects it, so that merely building a differently-configured
    provider (or importing this module) does not drag in heavy dependencies
    such as ``torch`` (R15 seam; keeps the factory and tests fast).

    Args:
        config: The active configuration. ``config.embedding_provider``
            selects the implementation and ``config.embedding_model`` supplies
            the model identifier.

    Returns:
        A concrete ``EmbeddingProvider`` instance.

    Raises:
        EmbeddingError: If ``config.embedding_provider`` names no known
            provider. The message includes the offending provider string but
            never any credential value (R16.4).
    """
    provider_key = config.embedding_provider

    if provider_key == "sentence-transformers":
        # Local import so torch/sentence-transformers load only when selected.
        from src.embeddings.sentence_transformers_provider import (
            SentenceTransformersEmbeddingProvider,
        )

        # SentenceTransformers runs locally and needs no credential.
        return SentenceTransformersEmbeddingProvider(model_name=config.embedding_model)

    # Unknown/unsupported provider: name the offending selector, no secret.
    raise EmbeddingError(f"Unsupported embedding provider: {provider_key!r}")


def _require_credential(config: Config, key_name: str) -> str:
    """Return a required credential, or raise a named, value-free ``EmbeddingError``.

    Helper reserved for future credential-bearing embedding providers (e.g. an
    OpenAI provider). It delegates to :meth:`Config.require_credential` and
    re-raises any missing-credential failure as an ``EmbeddingError`` that
    names the credential key ONLY — never its value (R16.4). No credential-
    bearing provider ships in the MVP, so this is unused today but keeps the
    factory ready to fail safely when one is added.

    Args:
        config: The active configuration holding credentials.
        key_name: The credential env var / attribute name, e.g.
            ``"OPENAI_API_KEY"``.

    Returns:
        The non-empty credential value.

    Raises:
        EmbeddingError: If the credential is unset or empty. The message names
            the key only.
    """
    try:
        return config.require_credential(key_name)
    except Exception as exc:  # MissingCredentialError (value-free message)
        raise EmbeddingError(str(exc)) from exc

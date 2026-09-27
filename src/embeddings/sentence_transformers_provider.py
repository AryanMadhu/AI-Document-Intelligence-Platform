"""Local embedding provider backed by ``sentence-transformers`` (MVP default).

``SentenceTransformersEmbeddingProvider`` embeds text with a local
``sentence-transformers`` model (default ``all-MiniLM-L6-v2``, 384-dim). It is
the MVP default embedding provider: it is free, needs no credential, and runs
in-process (R5.1, R15.1).

Lazy loading: the heavy ``SentenceTransformer`` model (and, transitively,
``torch``) is NOT constructed in ``__init__``. It is loaded on first use via
:meth:`_get_model`, which caches the instance. Importing this module or merely
constructing the provider therefore stays cheap and imports no torch — the
~90MB model download happens only when embedding actually runs.
"""

from __future__ import annotations

from typing import Any, Optional

from src.embeddings.base import EmbeddingError, EmbeddingProvider

# Dimensionality of the default model (all-MiniLM-L6-v2). Used as the fallback
# when the model has not been loaded yet so that ``dimension`` need not force a
# (potentially expensive) model load.
_DEFAULT_MODEL_DIMENSION = 384


class SentenceTransformersEmbeddingProvider(EmbeddingProvider):
    """Embeds text using a local ``sentence-transformers`` model.

    The underlying model is lazy-loaded and cached on first embed call. No
    domain-specific logic lives here (R15.5) — it is a thin adapter over the
    library.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        """Initialize the provider without loading the model.

        Args:
            model_name: The ``sentence-transformers`` model identifier to load
                on first use. Defaults to ``'all-MiniLM-L6-v2'`` (384-dim).
        """
        self._model_name = model_name
        # Cached model instance; ``None`` until the first embed call.
        self._model: Optional[Any] = None

    def _get_model(self) -> Any:
        """Return the cached model, loading it on first use.

        ``from sentence_transformers import SentenceTransformer`` is imported
        HERE (not at module top) so importing this module or constructing the
        provider does not import ``torch``.

        Returns:
            The loaded ``SentenceTransformer`` instance (cached).

        Raises:
            EmbeddingError: If the model cannot be imported or loaded.
        """
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self._model_name)
            except Exception as exc:
                raise EmbeddingError(
                    "Failed to load the sentence-transformers embedding model"
                ) from exc
        return self._model

    @property
    def provider_name(self) -> str:
        """Return the stable provider id ``'sentence-transformers'``."""
        return "sentence-transformers"

    @property
    def model_name(self) -> str:
        """Return the configured model identifier."""
        return self._model_name

    @property
    def dimension(self) -> int:
        """Return the embedding vector dimensionality.

        When the model is already loaded, the dimension is derived from it via
        ``get_sentence_embedding_dimension()``. When the model has not been
        loaded, the known default-model dimension (384 for
        ``all-MiniLM-L6-v2``) is returned without forcing a load, so callers
        that only need the dimension stay fast.

        Returns:
            The embedding dimensionality (384 for the default model).
        """
        if self._model is not None:
            derived = self._model.get_sentence_embedding_dimension()
            if derived is not None:
                return int(derived)
        return _DEFAULT_MODEL_DIMENSION

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts in a single ``model.encode`` call.

        Args:
            texts: The chunk texts to embed. An empty list returns ``[]``
                without loading the model.

        Returns:
            One embedding vector per input text, as ``list[list[float]]`` of
            plain Python floats (converted from numpy via ``.tolist()``).

        Raises:
            EmbeddingError: If the underlying model fails to embed the batch.
        """
        if not texts:
            return []
        try:
            model = self._get_model()
            embeddings = model.encode(texts, convert_to_numpy=True)
            return [list(map(float, row)) for row in embeddings.tolist()]
        except EmbeddingError:
            raise
        except Exception as exc:
            raise EmbeddingError("Failed to embed the batch of texts") from exc

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string.

        Args:
            text: The query string to embed.

        Returns:
            A flat ``list[float]`` of plain Python floats.

        Raises:
            EmbeddingError: If the underlying model fails to embed the query.
        """
        try:
            model = self._get_model()
            embedding = model.encode(text, convert_to_numpy=True)
            return [float(value) for value in embedding.tolist()]
        except EmbeddingError:
            raise
        except Exception as exc:
            raise EmbeddingError("Failed to embed the query") from exc

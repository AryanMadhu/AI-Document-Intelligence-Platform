"""Fast, deterministic unit tests for the embedding factory and provider.

These tests never download or load the real ``sentence-transformers`` model.
The heavy model is replaced with a fake via ``_get_model`` monkeypatching, so
the suite runs with no network access and without importing torch.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from src.config import Config
from src.embeddings.base import (
    EmbeddingError,
    EmbeddingProvider,
    build_embedding_provider,
)
from src.embeddings.sentence_transformers_provider import (
    SentenceTransformersEmbeddingProvider,
)


class _FakeModel:
    """Stand-in for ``SentenceTransformer`` used in tests (no torch, no I/O)."""

    def __init__(self, dimension: int = 384) -> None:
        self._dimension = dimension

    def get_sentence_embedding_dimension(self) -> int:
        return self._dimension

    def encode(self, data, convert_to_numpy: bool = True):
        if isinstance(data, str):
            # Single query -> flat vector.
            return np.array([0.1, 0.2, 0.3], dtype=np.float32)
        # Batch -> one row per text.
        return np.array([[0.1, 0.2, 0.3] for _ in data], dtype=np.float32)


class _RaisingModel:
    """Fake model whose ``encode`` always raises, to exercise error handling."""

    def get_sentence_embedding_dimension(self) -> int:
        return 384

    def encode(self, data, convert_to_numpy: bool = True):
        raise RuntimeError("boom")


def _config(**overrides) -> Config:
    """Build a default Config, applying any field overrides (no env/.env load)."""
    return dataclasses.replace(Config(), **overrides)


def test_factory_default_returns_sentence_transformers_provider() -> None:
    """Default config maps to a SentenceTransformersEmbeddingProvider (R15.3)."""
    config = _config(
        embedding_provider="sentence-transformers",
        embedding_model="all-MiniLM-L6-v2",
    )

    provider = build_embedding_provider(config)

    assert isinstance(provider, SentenceTransformersEmbeddingProvider)
    assert provider.provider_name == "sentence-transformers"
    assert provider.model_name == "all-MiniLM-L6-v2"
    # No model load was triggered by construction.
    assert provider._model is None


def test_dimension_is_384_without_loading_real_model(monkeypatch) -> None:
    """dimension returns 384 for the default model without a real load."""
    provider = SentenceTransformersEmbeddingProvider()

    # Unloaded: returns the known default-model dimension.
    assert provider.dimension == 384

    # Loaded (stubbed): derives 384 from the model.
    monkeypatch.setattr(provider, "_get_model", lambda: _FakeModel(384))
    provider._model = _FakeModel(384)
    assert provider.dimension == 384


def test_factory_unknown_provider_raises_named_error() -> None:
    """An unknown provider string raises EmbeddingError naming the string (R16.4)."""
    config = _config(embedding_provider="does-not-exist")

    with pytest.raises(EmbeddingError) as exc_info:
        build_embedding_provider(config)

    message = str(exc_info.value)
    assert "does-not-exist" in message
    # No secret leaked (there is none, but assert the message stays selector-only).
    assert "API_KEY" not in message.upper()


def test_embed_texts_delegates_and_returns_python_floats(monkeypatch) -> None:
    """embed_texts returns list[list[float]] of Python floats with expected shape."""
    provider = SentenceTransformersEmbeddingProvider()
    monkeypatch.setattr(provider, "_get_model", lambda: _FakeModel())

    result = provider.embed_texts(["a", "b"])

    assert isinstance(result, list)
    assert len(result) == 2
    for row in result:
        assert isinstance(row, list)
        assert len(row) == 3
        assert all(isinstance(value, float) for value in row)


def test_embed_texts_empty_returns_empty_without_loading(monkeypatch) -> None:
    """embed_texts([]) returns [] and never loads the model."""

    def _fail():  # pragma: no cover - must not be called
        raise AssertionError("_get_model should not be called for empty input")

    provider = SentenceTransformersEmbeddingProvider()
    monkeypatch.setattr(provider, "_get_model", _fail)

    assert provider.embed_texts([]) == []


def test_embed_query_returns_flat_list_of_floats(monkeypatch) -> None:
    """embed_query returns a flat list[float] of Python floats."""
    provider = SentenceTransformersEmbeddingProvider()
    monkeypatch.setattr(provider, "_get_model", lambda: _FakeModel())

    result = provider.embed_query("q")

    assert isinstance(result, list)
    assert len(result) == 3
    assert all(isinstance(value, float) for value in result)


def test_embed_texts_wraps_underlying_failure(monkeypatch) -> None:
    """A failing encode surfaces as EmbeddingError for batch embedding."""
    provider = SentenceTransformersEmbeddingProvider()
    monkeypatch.setattr(provider, "_get_model", lambda: _RaisingModel())

    with pytest.raises(EmbeddingError):
        provider.embed_texts(["a", "b"])


def test_embed_query_wraps_underlying_failure(monkeypatch) -> None:
    """A failing encode surfaces as EmbeddingError for query embedding."""
    provider = SentenceTransformersEmbeddingProvider()
    monkeypatch.setattr(provider, "_get_model", lambda: _RaisingModel())

    with pytest.raises(EmbeddingError):
        provider.embed_query("q")


def test_embedding_provider_is_abstract() -> None:
    """EmbeddingProvider cannot be instantiated directly."""
    with pytest.raises(TypeError):
        EmbeddingProvider()  # type: ignore[abstract]

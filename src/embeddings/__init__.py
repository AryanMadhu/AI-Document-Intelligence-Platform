"""Embedding providers and factory.

Public names are re-exported for convenience. Only the lightweight interface
and factory module (``base``) is imported at package-import time; the concrete
``sentence-transformers`` implementation (and torch) is imported lazily inside
the factory, so importing this package does NOT import torch.
"""

from src.embeddings.base import (
    EmbeddingError,
    EmbeddingProvider,
    build_embedding_provider,
)

__all__ = [
    "EmbeddingError",
    "EmbeddingProvider",
    "build_embedding_provider",
]

"""Retrieval service package.

Re-exports the retrieval read-side API: :class:`Retrieval_Service` and the
:class:`ReembeddingRequiredError` it raises when indexed documents are
incompatible with the active embedding space (R15.6-7).

Importing this package is lightweight: ``service.py`` only imports
:func:`~src.embeddings.base.build_embedding_provider`, which lazily loads the
heavy embedding backend (torch/sentence-transformers) only when a provider is
actually built — so merely importing the class does not drag in those deps.
"""

from src.retrieval.service import ReembeddingRequiredError, Retrieval_Service

__all__ = ["Retrieval_Service", "ReembeddingRequiredError"]

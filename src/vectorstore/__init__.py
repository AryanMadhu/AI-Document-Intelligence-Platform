"""Vector store wrapper (Chroma).

Re-exports :class:`~src.vectorstore.chroma_store.VectorStore`. The import is
kept minimal; the concrete module imports ``chromadb`` at its own module level.
"""

from src.vectorstore.chroma_store import VectorStore

__all__ = ["VectorStore"]

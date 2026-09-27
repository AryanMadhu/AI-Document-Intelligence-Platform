"""PDF ingestion: validation, parsing, chunking, hashing.

Importing this package stays lightweight: ``Chunking_Module`` lazily imports
``transformers`` inside a method, so importing the class here does not pull in
``transformers``/``torch`` at package import time.
"""

from src.ingestion.chunking import Chunking_Module, ChunkingError
from src.ingestion.pdf_parser import PageText, PDF_Parser, PDFParseError

__all__ = [
    "PDF_Parser",
    "PageText",
    "PDFParseError",
    "Chunking_Module",
    "ChunkingError",
]

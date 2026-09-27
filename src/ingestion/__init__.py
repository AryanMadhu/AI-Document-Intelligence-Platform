"""PDF ingestion: validation, parsing, chunking, hashing, orchestration.

Importing this package stays lightweight: ``Chunking_Module`` lazily imports
``transformers`` inside a method, and the ``Ingestion_Service`` factory only
pulls the heavy embedding provider in when built, so importing the classes here
does not pull in ``transformers``/``torch`` at package import time.
"""

from src.ingestion.chunking import Chunking_Module, ChunkingError
from src.ingestion.pdf_parser import PageText, PDF_Parser, PDFParseError
from src.ingestion.service import (
    EmptyDocumentError,
    FileTooLargeError,
    Ingestion_Service,
    IngestionError,
    TooManyPagesError,
    compute_document_hash,
)

__all__ = [
    "PDF_Parser",
    "PageText",
    "PDFParseError",
    "Chunking_Module",
    "ChunkingError",
    "Ingestion_Service",
    "compute_document_hash",
    "IngestionError",
    "FileTooLargeError",
    "TooManyPagesError",
    "EmptyDocumentError",
]

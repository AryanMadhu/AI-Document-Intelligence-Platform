"""PDF parsing with page preservation (R3).

This module implements the design's ``PDF_Parser`` component. It uses
**PyMuPDF** (imported as ``fitz``) to extract per-page text from a PDF while
preserving the originating page number for every unit of text, so downstream
components can cite the exact page a fact came from (R3.1, R3.2).

Design notes:

* **Signature decision — bytes + filename.** ``PDF_Parser.parse`` takes the
  raw PDF *bytes* plus a ``filename`` used only for error identification. The
  ``Ingestion_Service`` (a later task) already holds the file bytes — it
  computes the SHA-256 ``document_hash`` and enforces the size limit *before*
  parsing — so passing bytes avoids a second read of the file and keeps the
  parser pure and easy to test. PyMuPDF opens an in-memory byte stream via
  ``fitz.open(stream=file_bytes, filetype="pdf")``.
* **Empty pages are recorded, never skipped (R3.4).** A page whose extraction
  yields ``None`` or whitespace-only text is recorded as an *empty* page
  (``text=""``) and processing continues with the remaining pages. An
  individual page that fails to extract is likewise treated as empty rather
  than aborting the whole document.
* **Invalid input raises ``PDFParseError`` (R3.3).** Non-PDF or corrupt bytes
  that cannot be opened raise :class:`PDFParseError`, which carries both the
  offending filename and a concise, traceback-free reason.

``fitz`` is imported only in this module; no other module depends on PyMuPDF.
"""

from __future__ import annotations

from dataclasses import dataclass

import fitz  # PyMuPDF; imported only here


@dataclass
class PageText:
    """The extracted text of a single PDF page (R3.2).

    This is an internal model (a plain ``dataclass``, consistent with the
    system-internal models in :mod:`src.models`); it never crosses an HTTP
    boundary.

    Attributes:
        page_number: 1-based page number the text originated from.
        text: The page's extracted text. An empty string ``""`` denotes a page
            with no extractable text (R3.4).
    """

    page_number: int
    text: str


class PDFParseError(Exception):
    """Raised when input cannot be opened/parsed as a valid PDF (R3.3).

    The error carries both the offending file identifier and the reason so the
    ``Ingestion_Service`` can reject the file and surface a message that
    identifies the file and the cause. Both are accessible as attributes and
    appear in ``str(exception)``.

    Attributes:
        file: Identifier (filename) of the file that failed to parse.
        reason: Concise, human-readable reason (no full traceback).
    """

    def __init__(self, file: str, reason: str) -> None:
        """Initialize the error with the offending file and a concise reason.

        Args:
            file: Identifier (filename) of the file that failed to parse.
            reason: Concise, human-readable reason for the failure.
        """
        self.file = file
        self.reason = reason
        super().__init__(f"Failed to parse PDF '{file}': {reason}")


class PDF_Parser:
    """Extract per-page text from a PDF using PyMuPDF (R3).

    The parser is stateless and pure: it takes raw PDF bytes and returns a list
    of :class:`PageText`, one per page, in page order. It encodes no
    domain-specific logic.
    """

    def parse(self, file_bytes: bytes, filename: str) -> list[PageText]:
        """Extract text from every page of a PDF, preserving page numbers.

        Args:
            file_bytes: The raw bytes of the PDF file to parse.
            filename: The source filename, used only to identify the file in a
                :class:`PDFParseError`. Not read from disk.

        Returns:
            A list of :class:`PageText`, one entry per page in page order, with
            1-based ``page_number`` values (R3.1, R3.2). Pages with no
            extractable text are included with ``text=""`` (R3.4).

        Raises:
            PDFParseError: If ``file_bytes`` cannot be opened or parsed as a
                valid PDF (R3.3). The error carries ``filename`` and a concise
                reason; the underlying exception is chained via ``from``.
        """
        try:
            document = fitz.open(stream=file_bytes, filetype="pdf")
        except Exception as exc:  # noqa: BLE001 - normalize any open failure
            raise PDFParseError(filename, str(exc) or exc.__class__.__name__) from exc

        try:
            pages: list[PageText] = []
            for index in range(document.page_count):
                page_number = index + 1
                try:
                    extracted = document[index].get_text()
                except Exception:  # noqa: BLE001 - a bad page must not abort the doc
                    # An individual page that fails to extract is recorded as
                    # empty so processing continues (R3.4).
                    extracted = None

                text = self._normalize(extracted)
                pages.append(PageText(page_number=page_number, text=text))
            return pages
        finally:
            # Release the underlying file handle / native resources.
            document.close()

    @staticmethod
    def _normalize(extracted: str | None) -> str:
        """Normalize extracted page text to the empty-page convention.

        Treats ``None`` or whitespace-only extraction as an empty page,
        returning ``""`` (R3.4). Otherwise returns the extracted text
        unchanged.

        Args:
            extracted: Raw text returned by ``page.get_text()``, possibly
                ``None``.

        Returns:
            The extracted text, or ``""`` when there is no extractable text.
        """
        if extracted is None or not extracted.strip():
            return ""
        return extracted

"""Unit tests for :class:`src.ingestion.PDF_Parser` (R3).

PyMuPDF is available, so these tests build tiny *real* PDFs in memory with
``fitz`` and round-trip them through the parser — no fixture files on disk and
no network access. The root ``conftest.py`` makes ``src`` importable.

Covers:
    * Normal multi-page PDF: 1-based page numbers in order, inserted words
      present (R3.1, R3.2).
    * Empty-text page recorded and processing continues (R3.4).
    * Corrupt / non-PDF input raises ``PDFParseError`` with file + reason
      (R3.3).
    * Edge: a single-page PDF behaves deterministically.
"""

from __future__ import annotations

import fitz  # PyMuPDF — used only to construct real PDFs for the tests
import pytest

from src.ingestion import PageText, PDF_Parser, PDFParseError


def _make_pdf(pages: list[str | None]) -> bytes:
    """Build valid PDF bytes from a list of page contents.

    Args:
        pages: One entry per page. A string is inserted as page text; ``None``
            produces a blank page with no inserted text.

    Returns:
        The serialized PDF as bytes (round-trips through ``PDF_Parser.parse``).
    """
    doc = fitz.open()
    try:
        for content in pages:
            page = doc.new_page()
            if content is not None:
                page.insert_text((72, 72), content)
        return doc.tobytes()
    finally:
        doc.close()


def _normalize_ws(text: str) -> str:
    """Collapse all whitespace runs to single spaces and strip the ends."""
    return " ".join(text.split())


def test_normal_multipage_pdf_preserves_order_and_page_numbers() -> None:
    """A 3-page PDF yields 3 PageText in order with 1-based numbers (R3.1/R3.2)."""
    pages = ["Hello page one", "Second page text", "Third"]
    data = _make_pdf(pages)

    result = PDF_Parser().parse(data, "sample.pdf")

    assert [p.page_number for p in result] == [1, 2, 3]
    assert all(isinstance(p, PageText) for p in result)

    for page_text, original in zip(result, pages):
        normalized = _normalize_ws(page_text.text)
        for word in original.split():
            assert word in normalized


def test_empty_text_page_recorded_and_processing_continues() -> None:
    """A blank page is recorded as empty text and other pages still parse (R3.4)."""
    # Page 1 has text, page 2 is blank, page 3 has text.
    data = _make_pdf(["First page content", None, "Last page content"])

    result = PDF_Parser().parse(data, "with_blank.pdf")

    # Every page is present with correct 1-based numbering; none skipped.
    assert [p.page_number for p in result] == [1, 2, 3]

    # The blank page normalizes to "".
    assert result[1].text == ""

    # Surrounding pages retain their text (processing did not stop at the blank).
    assert "First" in _normalize_ws(result[0].text)
    assert "Last" in _normalize_ws(result[2].text)


def test_whitespace_only_page_normalized_to_empty() -> None:
    """A page whose only text is whitespace normalizes to '' (R3.4)."""
    data = _make_pdf(["   \n\t  ", "Real content"])

    result = PDF_Parser().parse(data, "whitespace.pdf")

    assert result[0].text == ""
    assert "Real" in _normalize_ws(result[1].text)


def test_corrupt_input_raises_pdf_parse_error_with_file_and_reason() -> None:
    """Non-PDF bytes raise PDFParseError carrying file + reason (R3.3)."""
    with pytest.raises(PDFParseError) as exc_info:
        PDF_Parser().parse(b"this is not a pdf", "bad.pdf")

    err = exc_info.value
    assert err.file == "bad.pdf"
    assert err.reason  # non-empty reason populated
    assert "bad.pdf" in str(err)


def test_truncated_pdf_bytes_raise_pdf_parse_error() -> None:
    """Truncated PDF bytes also raise PDFParseError (R3.3)."""
    valid = _make_pdf(["some content"])
    truncated = valid[: len(valid) // 3]

    with pytest.raises(PDFParseError) as exc_info:
        PDF_Parser().parse(truncated, "truncated.pdf")

    assert exc_info.value.file == "truncated.pdf"
    assert "truncated.pdf" in str(exc_info.value)


def test_single_page_pdf_returns_one_page_text() -> None:
    """A single-page PDF returns exactly one PageText (edge case)."""
    data = _make_pdf(["Only page"])

    result = PDF_Parser().parse(data, "single.pdf")

    assert len(result) == 1
    assert result[0].page_number == 1
    assert "Only" in _normalize_ws(result[0].text)


def test_pdf_parse_error_shape() -> None:
    """PDFParseError exposes file/reason attributes and a readable message."""
    err = PDFParseError("f.pdf", "some reason")

    assert err.file == "f.pdf"
    assert err.reason == "some reason"
    assert "f.pdf" in str(err)
    assert "some reason" in str(err)

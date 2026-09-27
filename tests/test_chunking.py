"""Tests for :class:`src.ingestion.Chunking_Module` (R4).

Covers the design's Properties 1-3 (Hypothesis, >=100 iterations each) plus
deterministic edge-case unit tests (R4.4).

Tokenizer / performance
-----------------------
Per the user's approved decision, the tests load the *real* tokenizer bundled
with ``transformers`` (``AutoTokenizer.from_pretrained("all-MiniLM-L6-v2")``) via
``Chunking_Module``. To avoid reloading it per Hypothesis example, a single
module-scoped ``Chunking_Module`` is constructed once (``CHUNKER``) and reused
across every example and test.

Assertion method — window construction, not re-encoded text
-----------------------------------------------------------
``chunk_text`` is a *decode* of a window of token ids, so *re-encoding* it can
drift from the original window length by a tiny amount (decode/re-encode
round-trip). The size/overlap properties (P1, P2) therefore assert on the
**deterministic construction windows** exposed by
``Chunking_Module._window_token_spans`` (the exact spans used to build the
chunks), which is more robust than fuzzy substring matching on decoded text.
P3 (metadata) asserts on the produced :class:`~src.models.Chunk` objects.

Page-boundary preservation and the "final chunk per page" exemption
-------------------------------------------------------------------
Chunks never cross a page break, so the last chunk of each page is that page's
natural short/"final" chunk. P1 asserts that every chunk *except the page's
final one* is within [500, 800] tokens (default size 700), and P2 asserts that
consecutive same-page windows overlap by exactly ``chunk_overlap_tokens``.
"""

from __future__ import annotations

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from src.ingestion.chunking import Chunking_Module
from src.ingestion.pdf_parser import PageText
from src.models import Chunk

# One shared chunker for the whole module: loads the real tokenizer ONCE and
# reuses it across every Hypothesis example and unit test (default size 700 /
# overlap 90 resolved from Config).
CHUNKER = Chunking_Module()

# The configured defaults, read from the shared chunker (never hard-coded here
# beyond referencing the resolved values).
CHUNK_SIZE = CHUNKER.chunk_size_tokens
CHUNK_OVERLAP = CHUNKER.chunk_overlap_tokens

# Bounded word-based text generator. Joining N random ASCII "words" keeps token
# counts in a reasonable range (a few thousand tokens max), so examples run
# fast while still producing multi-chunk pages.
_WORDS = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyz", min_size=1, max_size=8
)


def _text_of_words(min_words: int, max_words: int) -> st.SearchStrategy[str]:
    """Strategy producing space-joined random words within a word-count range."""
    return st.lists(_WORDS, min_size=min_words, max_size=max_words).map(" ".join)


def _token_count(chunker: Chunking_Module, text: str) -> int:
    """Return the number of tokens in ``text`` using the chunker's tokenizer."""
    return len(chunker._encode(text))


# ---------------------------------------------------------------------------
# Property 1 — chunk size stays within configured bounds (Validates R4.1)
# ---------------------------------------------------------------------------
@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(text=_text_of_words(0, 900))
def test_property1_non_final_chunk_size_within_bounds(text: str) -> None:
    """Every non-final chunk of a single page is within [500, 800] tokens.

    Asserted on the deterministic construction windows (not re-encoded text).
    The page's final chunk is exempt (it holds the remainder).
    """
    ids = CHUNKER._encode(text)
    spans = CHUNKER._window_token_spans(len(ids))

    if len(spans) <= 1:
        # Single (final) chunk, or empty page: nothing to assert for P1.
        return

    for start, end in spans[:-1]:
        window_len = end - start
        # With default size 700 non-final windows are exactly 700; assert the
        # required 500-800 band regardless.
        assert window_len == CHUNK_SIZE
        assert 500 <= window_len <= 800


# ---------------------------------------------------------------------------
# Property 2 — consecutive chunks overlap within the configured range (R4.2)
# ---------------------------------------------------------------------------
@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(text=_text_of_words(200, 900))
def test_property2_consecutive_same_page_overlap_within_range(text: str) -> None:
    """Consecutive same-page chunks overlap within 10-15% of chunk size.

    Overlap is measured from the deterministic construction windows: for
    consecutive spans ``(s0, e0)`` and ``(s1, e1)`` the shared token count is
    ``e0 - s1``. With default size 700 / overlap 90 this is 90 => 90/700 ≈
    12.86%, which lies in [0.10, 0.15].
    """
    ids = CHUNKER._encode(text)
    spans = CHUNKER._window_token_spans(len(ids))

    if len(spans) < 2:
        # Need >=2 chunks for a meaningful overlap assertion; skip short cases.
        return

    for (s0, e0), (s1, e1) in zip(spans, spans[1:]):
        overlap = e0 - s1
        assert overlap == CHUNK_OVERLAP
        fraction = overlap / CHUNK_SIZE
        assert 0.10 <= fraction <= 0.15


# ---------------------------------------------------------------------------
# Property 3 — every chunk carries complete metadata (R4.3, R4.4)
# ---------------------------------------------------------------------------
@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    text=_text_of_words(0, 900),
    page_number=st.integers(min_value=1, max_value=50),
    document_name=st.text(
        alphabet="abcdefghijklmnopqrstuvwxyz0123456789_", min_size=1, max_size=20
    ).map(lambda s: f"{s}.pdf"),
    section=st.one_of(st.none(), st.text(min_size=1, max_size=12)),
)
def test_property3_every_chunk_carries_complete_metadata(
    text: str, page_number: int, document_name: str, section: str | None
) -> None:
    """Each produced chunk carries complete, valid metadata (with/without section).

    ``page_number`` here doubles as a one-page "document range" ([1, page_number]
    trivially contains page_number). Covers both a provided section string and
    ``section is None``.
    """
    chunks = CHUNKER.chunk_page(
        page_text=text,
        page_number=page_number,
        document_name=document_name,
        section=section,
    )

    for chunk in chunks:
        assert isinstance(chunk, Chunk)
        # non-empty document_name
        assert chunk.document_name == document_name
        assert chunk.document_name != ""
        # page_number within the document's page range (single page here)
        assert chunk.page_number == page_number
        assert 1 <= chunk.page_number
        # chunk_text is the decoded window: a non-empty str for non-empty pages
        assert isinstance(chunk.chunk_text, str)
        assert chunk.chunk_text != ""
        # section is the provided identifier or None
        assert chunk.section == section


# ---------------------------------------------------------------------------
# Property 3 companion — page range across a multi-page document (R4.3)
# ---------------------------------------------------------------------------
@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    pages=st.lists(_text_of_words(0, 200), min_size=1, max_size=4),
    document_name=st.text(
        alphabet="abcdefghijklmnopqrstuvwxyz0123456789_", min_size=1, max_size=20
    ).map(lambda s: f"{s}.pdf"),
)
def test_property3_page_number_within_document_range(
    pages: list[str], document_name: str
) -> None:
    """Every chunk's page_number lies within [1, number_of_pages] (R4.3)."""
    page_texts = [
        PageText(page_number=i + 1, text=t) for i, t in enumerate(pages)
    ]
    n_pages = len(page_texts)

    chunks = CHUNKER.chunk_pages(page_texts, document_name)

    for chunk in chunks:
        assert 1 <= chunk.page_number <= n_pages
        assert chunk.section is None  # MVP: chunk_pages sets section None
        assert chunk.document_name == document_name


# ---------------------------------------------------------------------------
# 8.5 — Edge-case unit tests (deterministic) (Validates R4.4)
# ---------------------------------------------------------------------------
def test_empty_text_returns_no_chunks() -> None:
    """Empty text yields no chunks."""
    assert CHUNKER.chunk_page("", page_number=1, document_name="d.pdf") == []


@pytest.mark.parametrize("blank", ["", "   ", "\n\t  \r\n", "     \t"])
def test_whitespace_only_text_returns_no_chunks(blank: str) -> None:
    """Whitespace-only text yields no chunks."""
    assert CHUNKER.chunk_page(blank, page_number=3, document_name="d.pdf") == []


def test_single_short_page_yields_one_chunk_with_all_tokens() -> None:
    """A page with fewer tokens than chunk_size yields exactly one chunk."""
    text = "the quick brown fox jumps over the lazy dog"
    assert _token_count(CHUNKER, text) < CHUNK_SIZE

    chunks = CHUNKER.chunk_page(text, page_number=7, document_name="short.pdf")

    assert len(chunks) == 1
    assert chunks[0].page_number == 7
    assert chunks[0].document_name == "short.pdf"
    assert chunks[0].section is None
    assert chunks[0].chunk_text.strip() != ""


def test_section_less_input_sets_section_none() -> None:
    """Omitting section leaves section as None (R4.4)."""
    chunks = CHUNKER.chunk_page(
        "some real content here", page_number=1, document_name="d.pdf"
    )
    assert chunks
    assert all(c.section is None for c in chunks)


def test_section_string_is_preserved() -> None:
    """A provided section identifier is preserved on every chunk."""
    chunks = CHUNKER.chunk_page(
        "some real content here",
        page_number=1,
        document_name="d.pdf",
        section="Introduction",
    )
    assert chunks
    assert all(c.section == "Introduction" for c in chunks)


def test_exact_boundary_token_count_yields_one_chunk() -> None:
    """Text whose token count == chunk_size yields exactly one chunk.

    The window-span logic is the deterministic source of truth: a token_count
    equal to CHUNK_SIZE produces a single full window.
    """
    assert CHUNKER._window_token_spans(CHUNK_SIZE) == [(0, CHUNK_SIZE)]


def test_one_over_boundary_yields_two_chunks_with_correct_overlap() -> None:
    """token_count == chunk_size + 1 yields two chunks overlapping by overlap.

    The second (final) window holds the remainder; the shared span between the
    two windows equals CHUNK_OVERLAP.
    """
    spans = CHUNKER._window_token_spans(CHUNK_SIZE + 1)

    assert len(spans) == 2
    (s0, e0), (s1, e1) = spans
    assert (s0, e0) == (0, CHUNK_SIZE)
    step = CHUNK_SIZE - CHUNK_OVERLAP
    assert s1 == step
    assert e1 == CHUNK_SIZE + 1
    # overlap between the two windows equals the configured overlap
    assert e0 - s1 == CHUNK_OVERLAP


def test_chunk_pages_preserves_page_number_and_order() -> None:
    """chunk_pages preserves per-chunk page_number and page order.

    Each chunk maps to exactly one page and pages appear in ascending order.
    """
    # Build pages that each produce >=2 chunks so ordering is exercised.
    long_a = " ".join(["alpha"] * 1200)
    long_b = " ".join(["beta"] * 1200)
    pages = [
        PageText(page_number=1, text=long_a),
        PageText(page_number=2, text=""),  # empty page -> no chunks
        PageText(page_number=3, text=long_b),
    ]

    chunks = CHUNKER.chunk_pages(pages, "multi.pdf")

    # Empty page 2 contributes nothing.
    observed_pages = [c.page_number for c in chunks]
    assert 2 not in observed_pages
    assert set(observed_pages) == {1, 3}

    # Page numbers are non-decreasing (order preserved, no interleaving).
    assert observed_pages == sorted(observed_pages)

    # Each chunk maps to exactly one page and carries the document name.
    for c in chunks:
        assert c.document_name == "multi.pdf"
        assert c.page_number in (1, 3)
        assert c.section is None


def test_chunk_pages_never_merges_across_page_break() -> None:
    """A chunk is always attributable to a single page (no cross-page merge)."""
    pages = [
        PageText(page_number=1, text="first page only text here"),
        PageText(page_number=2, text="second page only text here"),
    ]
    chunks = CHUNKER.chunk_pages(pages, "two.pdf")
    # Each page is short -> exactly one chunk per non-empty page.
    assert [c.page_number for c in chunks] == [1, 2]

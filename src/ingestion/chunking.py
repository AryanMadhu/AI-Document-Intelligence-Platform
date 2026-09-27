"""Token-based, page-boundary-preserving chunking (R4).

This module implements the design's ``Chunking_Module`` component. It splits a
document's extracted text into overlapping, token-sized :class:`~src.models.Chunk`
objects, attaching the metadata ``{document_name, page_number, section,
chunk_text}`` to each one (R4.3, R4.4).

Design tradeoff — page-boundary-preserving chunking (R4, R9.2)
--------------------------------------------------------------
Chunking deliberately **never merges text across a page break**: each chunk
maps to exactly one page. This favours *citation accuracy* — a chunk never
straddles two pages, so a citation can name the exact source page with no
ambiguity. The cost is a small reduction in semantic continuity when a passage
spans a page break; ``chunk_overlap_tokens`` mitigates this *within* a page but
never across a page boundary.

Concretely, :meth:`Chunking_Module.chunk_page` slides a fixed-size window over a
single page's token ids:

* window size = ``chunk_size_tokens`` (default 700, from :class:`~src.config.Config`),
* step = ``chunk_size_tokens - chunk_overlap_tokens`` (default 610),

so consecutive chunks *of the same page* overlap by exactly
``chunk_overlap_tokens`` (default 90 ≈ 12.9% of 700, within R4.2's 10-15%). The
**last chunk of each page** holds the page's remainder and may therefore be
shorter than ``chunk_size_tokens`` — it is the natural "final" chunk exempted by
Property 1 (R4.1). Overlap applies only *between consecutive chunks of the same
page*; there is no overlap across a page boundary.

Tokenizer decision and the 512 model-max note
----------------------------------------------
Token counting and splitting use the tokenizer bundled with ``transformers`` —
specifically ``AutoTokenizer.from_pretrained("all-MiniLM-L6-v2")`` (the
configured embedding model's tokenizer). No third-party splitter library is
introduced.

The tokenizer is loaded **lazily** (on first use, then cached) and
``transformers`` is imported *inside* the loader, so importing this module does
not pull in ``transformers``/``torch``.

Important: ``all-MiniLM-L6-v2`` reports ``model_max_length == 512``. That is the
*model input cap*, **not** our chunk target. Our chunk target is 500-800 tokens
(default 700) and may exceed 512. We use the tokenizer purely as a
counter/splitter here — we are **not** feeding text into the model — so we must
avoid truncation. We therefore call ``encode(text, add_special_tokens=False)``
**without** ``truncation=True`` (encode does not truncate by default). When a
sequence is longer than 512, ``transformers`` may emit a benign
"Token indices sequence length is longer than ... model_max_length" warning;
because we only count/split, that warning is harmless and is suppressed for our
use (see :meth:`_encode`). We do **not** shrink the chunk size to 512.

Token counting / splitting mechanics
-------------------------------------
* Count / split input: ``ids = tokenizer.encode(text, add_special_tokens=False)``;
  ``count = len(ids)``.
* Chunk text: ``tokenizer.decode(window_ids)`` for each window of ids.

Because the produced ``chunk_text`` is a *decode* of the window ids, a
subsequent *re-encode* of that text can differ from the original window length
by a tiny amount (decode/re-encode round-trip drift). The property tests
therefore assert on the **original window sizes used to build the chunks**
(deterministic), not on re-encoded ``chunk_text`` — see :mod:`tests.test_chunking`.
The internal helper :meth:`_window_token_spans` exposes those deterministic
window spans for tests.

No domain-specific logic lives here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from src.config import Config
from src.ingestion.pdf_parser import PageText
from src.models import Chunk

if TYPE_CHECKING:  # pragma: no cover - typing only, no runtime import
    from transformers import PreTrainedTokenizerBase


class ChunkingError(Exception):
    """Raised when the chunking tokenizer cannot be constructed.

    The message names the tokenizer/model but never contains any secret; the
    underlying exception is chained via ``from`` for debugging.
    """


class Chunking_Module:
    """Split page text into overlapping, token-sized, page-scoped chunks (R4).

    The chunk size and overlap come from :class:`~src.config.Config` (or from
    explicit constructor arguments), never from literals baked into the
    splitting logic. The tokenizer is loaded lazily on first use.

    Attributes:
        chunk_size_tokens: Target window size in tokens (default 700 via
            ``Config.chunk_size_tokens``).
        chunk_overlap_tokens: Token overlap between consecutive same-page
            chunks (default 90 via ``Config.chunk_overlap_tokens``).
    """

    #: The tokenizer model used purely for counting/splitting (the configured
    #: embedding model's tokenizer). See the module docstring for the 512 note.
    _TOKENIZER_MODEL = "all-MiniLM-L6-v2"

    #: Fully-qualified fallback repo id for the same tokenizer. The bare model
    #: name above resolves to a canonical Hub repo that requires a network hit;
    #: the ``sentence-transformers/`` org repo is what ``sentence-transformers``
    #: itself downloads and caches locally, so preferring it makes the tokenizer
    #: load offline (from cache) and identical in vocabulary/behaviour.
    _TOKENIZER_MODEL_FALLBACK = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(
        self,
        config: Config | None = None,
        chunk_size_tokens: int | None = None,
        chunk_overlap_tokens: int | None = None,
    ) -> None:
        """Resolve chunk size / overlap and defer tokenizer loading.

        Resolution order for each setting: the explicit argument if provided,
        otherwise the corresponding :class:`~src.config.Config` value. When
        ``config`` is ``None`` a default ``Config.from_env()`` is used.

        Args:
            config: Configuration source; ``Config.from_env()`` when ``None``.
            chunk_size_tokens: Explicit override for the window size in tokens.
            chunk_overlap_tokens: Explicit override for the overlap in tokens.

        Raises:
            ValueError: If the resolved sizes are invalid (non-positive size,
                negative overlap, or overlap >= size which would prevent the
                window from advancing).
        """
        if chunk_size_tokens is None or chunk_overlap_tokens is None:
            resolved_config = config if config is not None else Config.from_env()
            if chunk_size_tokens is None:
                chunk_size_tokens = resolved_config.chunk_size_tokens
            if chunk_overlap_tokens is None:
                chunk_overlap_tokens = resolved_config.chunk_overlap_tokens

        if chunk_size_tokens <= 0:
            raise ValueError("chunk_size_tokens must be positive")
        if chunk_overlap_tokens < 0:
            raise ValueError("chunk_overlap_tokens must be non-negative")
        if chunk_overlap_tokens >= chunk_size_tokens:
            raise ValueError(
                "chunk_overlap_tokens must be smaller than chunk_size_tokens "
                "so the window advances"
            )

        self.chunk_size_tokens: int = chunk_size_tokens
        self.chunk_overlap_tokens: int = chunk_overlap_tokens

        # Lazily initialised; see _get_tokenizer.
        self._tokenizer: Optional["PreTrainedTokenizerBase"] = None

    # ------------------------------------------------------------------
    # Tokenizer (lazy)
    # ------------------------------------------------------------------
    def _get_tokenizer(self) -> "PreTrainedTokenizerBase":
        """Lazily construct and cache the tokenizer.

        ``transformers`` is imported here (not at module top) so importing this
        module does not import ``transformers``/``torch``.

        Returns:
            The cached ``AutoTokenizer`` for ``all-MiniLM-L6-v2``.

        Raises:
            ChunkingError: If the tokenizer cannot be loaded.
        """
        if self._tokenizer is None:
            from transformers import AutoTokenizer  # local import (lazy)

            # Prefer the configured model name; if it cannot be resolved, fall
            # back to the fully-qualified ``sentence-transformers/`` repo, which
            # is the same tokenizer and is the one cached locally by
            # sentence-transformers.
            #
            # First pass: cache-only (local_files_only=True) so a fully cached
            # tokenizer loads instantly with no network hit — this avoids Hub
            # rate-limiting/latency entirely when the files are already present.
            # Second pass (if nothing was cached): allow the normal download.
            candidates = (self._TOKENIZER_MODEL, self._TOKENIZER_MODEL_FALLBACK)
            first_error: Exception | None = None
            for local_only in (True, False):
                for model_id in candidates:
                    try:
                        self._tokenizer = AutoTokenizer.from_pretrained(
                            model_id, local_files_only=local_only
                        )
                        break
                    except Exception as exc:  # noqa: BLE001 - try next candidate
                        if first_error is None:
                            first_error = exc
                if self._tokenizer is not None:
                    break
            if self._tokenizer is None:
                raise ChunkingError(
                    f"Failed to load tokenizer "
                    f"'{self._TOKENIZER_MODEL}' (and fallback "
                    f"'{self._TOKENIZER_MODEL_FALLBACK}'): {first_error}"
                ) from first_error
        return self._tokenizer

    def _encode(self, text: str) -> list[int]:
        """Encode text to token ids for counting/splitting (no truncation).

        Uses ``add_special_tokens=False`` and does NOT pass ``truncation=True``,
        so sequences longer than the model's 512 cap are counted in full (see
        the module docstring). The benign "Token indices sequence length is
        longer than ... model_max_length" warning is suppressed here because we
        only count/split and never feed the model.

        Args:
            text: The text to tokenize.

        Returns:
            The list of token ids (without special tokens).
        """
        tokenizer = self._get_tokenizer()
        import logging
        import warnings

        # The "Token indices sequence length is longer than the specified
        # maximum sequence length (N > 512)" notice is emitted via the
        # transformers *logger* (not the warnings module) and is benign for our
        # counting/splitting use (we never feed the model). Silence both the
        # logger and any warnings for the duration of this call only.
        tokenization_logger = logging.getLogger(
            "transformers.tokenization_utils_base"
        )
        previous_level = tokenization_logger.level
        tokenization_logger.setLevel(logging.ERROR)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return tokenizer.encode(text, add_special_tokens=False)
        finally:
            tokenization_logger.setLevel(previous_level)

    def _decode(self, ids: list[int]) -> str:
        """Decode a slice of token ids back into text for ``chunk_text``."""
        tokenizer = self._get_tokenizer()
        return tokenizer.decode(ids)

    def _window_token_spans(self, token_count: int) -> list[tuple[int, int]]:
        """Return the deterministic ``(start, end)`` id spans for one page.

        This is the single source of truth for how a page's token ids are
        windowed, and is used both by :meth:`chunk_page` and by the property
        tests (so tests assert on the *construction* windows rather than on
        re-encoded ``chunk_text``, avoiding decode/re-encode drift).

        Semantics (page-boundary preserving):
            * ``window size = chunk_size_tokens``
            * ``step = chunk_size_tokens - chunk_overlap_tokens``
            * every span except the last is exactly ``chunk_size_tokens`` long
            * consecutive spans overlap by exactly ``chunk_overlap_tokens``
            * the last span holds the remainder (may be shorter)

        Args:
            token_count: Number of tokens on the page.

        Returns:
            A list of ``(start, end)`` half-open index spans into the page's
            token id list. Empty when ``token_count == 0``.
        """
        if token_count <= 0:
            return []
        if token_count <= self.chunk_size_tokens:
            return [(0, token_count)]

        step = self.chunk_size_tokens - self.chunk_overlap_tokens
        spans: list[tuple[int, int]] = []
        start = 0
        while start < token_count:
            end = min(start + self.chunk_size_tokens, token_count)
            spans.append((start, end))
            if end == token_count:
                break
            start += step
        return spans

    # ------------------------------------------------------------------
    # Public chunking API
    # ------------------------------------------------------------------
    def chunk_page(
        self,
        page_text: str,
        page_number: int,
        document_name: str,
        section: str | None = None,
    ) -> list[Chunk]:
        """Split ONE page's text into token-windowed chunks (R4.1-R4.4).

        All chunks produced here carry the *same* ``page_number`` — chunks never
        cross a page break (page-boundary preservation). Within the page,
        consecutive chunks overlap by ``chunk_overlap_tokens`` and every chunk
        except the last is exactly ``chunk_size_tokens`` tokens; the last chunk
        holds the remainder.

        Args:
            page_text: The page's extracted text.
            page_number: 1-based page number attached to every produced chunk.
            document_name: Source PDF filename attached to every chunk.
            section: Section identifier, or ``None`` when unavailable (R4.4).

        Returns:
            A list of :class:`~src.models.Chunk` in order. An empty or
            whitespace-only page yields ``[]`` (no chunks). A page with fewer
            tokens than ``chunk_size_tokens`` yields exactly one chunk holding
            all its tokens.
        """
        if page_text is None or not page_text.strip():
            return []

        token_ids = self._encode(page_text)
        if not token_ids:
            return []

        chunks: list[Chunk] = []
        for start, end in self._window_token_spans(len(token_ids)):
            window_ids = token_ids[start:end]
            chunk_text = self._decode(window_ids)
            chunks.append(
                Chunk(
                    document_name=document_name,
                    page_number=page_number,
                    section=section,
                    chunk_text=chunk_text,
                )
            )
        return chunks

    def chunk_pages(
        self, pages: list[PageText], document_name: str
    ) -> list[Chunk]:
        """Chunk every page of a document in order, preserving page boundaries.

        Iterates the pages in order and calls :meth:`chunk_page` for each,
        concatenating the results. ``section`` is ``None`` for the MVP (PDF
        section detection is out of scope). Each produced chunk maps to exactly
        one page; text is never merged across a page break.

        Args:
            pages: Per-page text in page order (from :class:`PDF_Parser`).
            document_name: Source PDF filename attached to every chunk.

        Returns:
            All chunks for the document, in page order.
        """
        all_chunks: list[Chunk] = []
        for page in pages:
            all_chunks.extend(
                self.chunk_page(
                    page_text=page.text,
                    page_number=page.page_number,
                    document_name=document_name,
                    section=None,
                )
            )
        return all_chunks

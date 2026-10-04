"""Response formatting: build ``{answer, citations[]}`` from in-context chunks (R9).

:class:`Response_Formatter` turns an LLM answer plus the retrieved chunks that
were placed into the prompt context into a :class:`~src.models.QueryResponse`.
Every citation it emits is derived ONLY from an in-context retrieved chunk — the
formatter never invents a citation outside the retrieved set (R9.1, R9.2, R9.4).

Grounding honesty: Citation validity is structurally guaranteed, but citation
relevance and LLM factuality are not independently verified in the MVP.
"""

from __future__ import annotations

from src.models import Citation, QueryResponse, RetrievedChunk


class Response_Formatter:
    """Assemble a cited :class:`~src.models.QueryResponse` from in-context chunks.

    Citations are built one-per-in-context-chunk (D2), each carrying the chunk's
    full ``chunk_text`` as its excerpt (D3, no truncation). When the answer is
    the no-answer sentinel, the citation list is empty (R9.3).
    """

    def format(
        self,
        answer: str,
        chunks: list[RetrievedChunk],
    ) -> QueryResponse:
        """Build the final response from an answer and its in-context chunks.

        If ``answer`` (after ``.strip()``) exactly equals the no-answer sentinel
        :data:`~src.query.service.NO_ANSWER_RESPONSE`, the result carries the
        sentinel text and an empty citation list (R9.3). Otherwise, one
        :class:`~src.models.Citation` is built per retrieved chunk:
        ``Citation(document=chunk_document_name, page=page_number,
        excerpt=chunk_text)`` — the excerpt is the chunk's full text with no
        truncation (D2, D3). ``Citation.page`` accepts ``int | str``; the chunk's
        integer ``page_number`` is passed through unchanged.

        Args:
            answer: The LLM answer text (or the no-answer sentinel).
            chunks: The retrieved chunks that were placed in the LLM context.

        Returns:
            A :class:`~src.models.QueryResponse` whose citations each trace to an
            in-context chunk (empty for the no-answer sentinel).
        """
        # Imported here (not at module top) to keep a single source of truth for
        # the sentinel in service.py while avoiding a circular import.
        from src.query.service import NO_ANSWER_RESPONSE

        stripped = answer.strip()
        if stripped == NO_ANSWER_RESPONSE:
            return QueryResponse(answer=NO_ANSWER_RESPONSE, citations=[])

        citations = [
            Citation(
                document=retrieved.chunk.document_name,
                page=retrieved.chunk.page_number,
                excerpt=retrieved.chunk.chunk_text,
            )
            for retrieved in chunks
        ]
        return QueryResponse(answer=answer, citations=citations)

"""Tests for the Query_Service, grounded prompt builder, and formatter (Task 12).

All tests are fully deterministic and offline: there is NO real Gemini, no
network, no embedding model, and no vector store. Retrieval is a duck-typed stub
exposing ``retrieve(question, top_k=None)`` (it does not subclass anything), and
the LLM is a stub implementing the :class:`~src.llm.base.LLMProvider` ABC. Both
stubs track whether they were called so the "LLM/retrieval not invoked"
guarantees can be asserted directly.

Covers:
    * Property 9 (grounded prompt content; Validates Requirements 8.1, 8.2) via
      Hypothesis at >=100 examples.
    * Property 10 (citation validity; Validates Requirements 9.1, 9.2, 9.3, 9.4,
      10.2) via Hypothesis at >=100 examples, covering both the grounded and the
      no-answer-sentinel branches.
    * Empty/whitespace question -> EmptyQuestionError, LLM + retrieval NOT called.
    * Empty retrieval -> No_Answer_Response, empty citations, LLM NOT called.
    * Exact no-answer sentinel from the LLM -> empty citations.
    * LLMError propagates unchanged.
    * ReembeddingRequiredError propagates unchanged, LLM NOT called.
"""

from __future__ import annotations

import dataclasses

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.config import Config
from src.llm.base import LLMError, LLMProvider
from src.models import Chunk, QueryResponse, RetrievedChunk
from src.query import (
    NO_ANSWER_RESPONSE,
    EmptyQuestionError,
    Query_Service,
    Response_Formatter,
    build_grounded_prompt,
)
from src.retrieval import ReembeddingRequiredError

# A stable fragment of the answer-only grounding instruction. Asserting this
# substring proves the "answer using only the context" instruction is present
# without being brittle about exact line wrapping.
_ANSWER_ONLY_FRAGMENT = "using ONLY the numbered context passages"


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _StubLLMProvider(LLMProvider):
    """Deterministic LLM stub. Returns a canned answer and records invocation."""

    def __init__(self, answer: str) -> None:
        self._answer = answer
        self.called = False
        self.last_prompt: str | None = None

    @property
    def provider_name(self) -> str:
        return "stub-llm"

    @property
    def model_name(self) -> str:
        return "stub-model"

    def generate(self, prompt: str) -> str:
        self.called = True
        self.last_prompt = prompt
        return self._answer


class _RaisingLLMProvider(_StubLLMProvider):
    """LLM stub whose ``generate`` raises :class:`LLMError` (to prove D5)."""

    def __init__(self) -> None:
        super().__init__(answer="unused")

    def generate(self, prompt: str) -> str:
        self.called = True
        self.last_prompt = prompt
        raise LLMError("stub provider failure")


class _NeverCalledLLMProvider(_StubLLMProvider):
    """LLM stub whose ``generate`` must never run; fails loudly if it does."""

    def __init__(self) -> None:
        super().__init__(answer="must-not-be-used")

    def generate(self, prompt: str) -> str:  # pragma: no cover - guard
        self.called = True
        raise AssertionError("LLM.generate must not be called in this scenario")


class _StubRetrievalService:
    """Duck-typed retrieval stub exposing ``retrieve(question, top_k=None)``.

    Returns a preset chunk list, or raises a preset exception. Tracks whether
    ``retrieve`` was called so "retrieval not invoked" can be asserted.
    """

    def __init__(
        self,
        chunks: list[RetrievedChunk] | None = None,
        error: Exception | None = None,
    ) -> None:
        self._chunks = chunks or []
        self._error = error
        self.called = False
        self.last_question: str | None = None

    def retrieve(
        self,
        question: str,
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        self.called = True
        self.last_question = question
        if self._error is not None:
            raise self._error
        return list(self._chunks)


def _config() -> Config:
    """A default Config (no env/.env load)."""
    return dataclasses.replace(Config())


def _make_chunk(
    document_name: str,
    page_number: int,
    chunk_text: str,
    section: str | None = None,
) -> RetrievedChunk:
    """Build a RetrievedChunk via src.models with stub provenance."""
    return RetrievedChunk(
        chunk=Chunk(
            document_name=document_name,
            page_number=page_number,
            section=section,
            chunk_text=chunk_text,
        ),
        score=1.0,
        embedding_provider="stub-provider",
        embedding_model="stub-model",
    )


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

# Chunk text that is non-empty and substring-checkable. Allow a broad range of
# printable characters but strip control chars that could collide with the
# prompt template structure; min_size=1 keeps every chunk_text present/checkable.
_chunk_text_strategy = st.text(
    alphabet=st.characters(
        min_codepoint=33,
        max_codepoint=126,
    ),
    min_size=1,
    max_size=40,
)

_document_name_strategy = st.text(
    alphabet=st.characters(min_codepoint=97, max_codepoint=122),
    min_size=1,
    max_size=12,
).map(lambda s: f"{s}.pdf")


@st.composite
def _retrieved_chunks(draw, min_size: int = 1, max_size: int = 6):
    """Draw a non-empty, bounded list of varied RetrievedChunks."""
    n = draw(st.integers(min_value=min_size, max_value=max_size))
    chunks: list[RetrievedChunk] = []
    for _ in range(n):
        chunks.append(
            _make_chunk(
                document_name=draw(_document_name_strategy),
                page_number=draw(st.integers(min_value=1, max_value=500)),
                chunk_text=draw(_chunk_text_strategy),
            )
        )
    return chunks


# ---------------------------------------------------------------------------
# Property 9 — grounded prompt content
# Validates: Requirements 8.1, 8.2
# ---------------------------------------------------------------------------


@settings(max_examples=100, deadline=None)
@given(
    chunks=_retrieved_chunks(),
    question=st.text(
        alphabet=st.characters(min_codepoint=33, max_codepoint=126),
        min_size=1,
        max_size=40,
    ),
)
def test_property_grounded_prompt_contains_context_and_instructions(
    chunks, question
) -> None:
    """Property 9: prompt contains every chunk's text, the question, and both
    grounding instructions.

    Validates: Requirements 8.1, 8.2. For any non-empty set of retrieved chunks,
    the grounded prompt must contain (a) every chunk's ``chunk_text``, (b) the
    question, (c) the answer-only-from-context instruction, and (d) the exact
    no-answer sentinel (the "return No_Answer_Response otherwise" instruction).
    """
    prompt = build_grounded_prompt(question, chunks)

    # (a) every chunk's text appears.
    for retrieved in chunks:
        assert retrieved.chunk.chunk_text in prompt

    # (b) the question appears.
    assert question in prompt

    # (c) the answer-only grounding instruction appears (stable fragment).
    assert _ANSWER_ONLY_FRAGMENT in prompt

    # (d) the exact no-answer sentinel (return-no-answer-otherwise) appears.
    assert NO_ANSWER_RESPONSE in prompt


# ---------------------------------------------------------------------------
# Property 10 — citation validity
# Validates: Requirements 9.1, 9.2, 9.3, 9.4, 10.2
# ---------------------------------------------------------------------------


@settings(max_examples=100, deadline=None)
@given(chunks=_retrieved_chunks())
def test_property_citations_derive_from_in_context_chunks_grounded(chunks) -> None:
    """Property 10 (grounded branch): every citation maps to an in-context chunk.

    Validates: Requirements 9.1, 9.2, 9.4, 10.2. With a grounded (non-sentinel)
    answer routed through Query_Service.answer, citations must be non-empty and
    each ``(document, page, excerpt)`` must equal some in-context chunk's
    ``(document_name, page_number, chunk_text)``.
    """
    grounded_answer = "A grounded answer drawn from the context."
    llm = _StubLLMProvider(grounded_answer)
    retrieval = _StubRetrievalService(chunks=chunks)
    service = Query_Service(_config(), retrieval_service=retrieval, llm_provider=llm)

    response = service.answer("what is this about?")

    assert isinstance(response, QueryResponse)
    assert response.answer == grounded_answer
    assert len(response.citations) == len(chunks)

    in_context = {
        (c.chunk.document_name, c.chunk.page_number, c.chunk.chunk_text)
        for c in chunks
    }
    for citation in response.citations:
        assert (citation.document, citation.page, citation.excerpt) in in_context


@settings(max_examples=100, deadline=None)
@given(chunks=_retrieved_chunks())
def test_property_citations_empty_on_no_answer_sentinel(chunks) -> None:
    """Property 10 (sentinel branch): the no-answer sentinel yields no citations.

    Validates: Requirements 9.3. When the formatter receives the exact no-answer
    sentinel as the answer, the citation list SHALL be empty regardless of how
    many chunks were in context.
    """
    formatter = Response_Formatter()
    response = formatter.format(NO_ANSWER_RESPONSE, chunks)

    assert response.answer == NO_ANSWER_RESPONSE
    assert response.citations == []


# ---------------------------------------------------------------------------
# Edge-case unit tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_question", [None, "", "   ", "\t\n  "])
def test_empty_question_raises_without_invoking_llm_or_retrieval(bad_question) -> None:
    """Empty/whitespace question -> EmptyQuestionError; LLM + retrieval NOT called.

    Validates: R16.1 (D4). The guard runs before any retrieval or generation, so
    both stubs must record ``called is False``.
    """
    llm = _NeverCalledLLMProvider()
    retrieval = _StubRetrievalService(chunks=[_make_chunk("d.pdf", 1, "text")])
    service = Query_Service(_config(), retrieval_service=retrieval, llm_provider=llm)

    with pytest.raises(EmptyQuestionError):
        service.answer(bad_question)  # type: ignore[arg-type]

    assert llm.called is False
    assert retrieval.called is False


def test_empty_retrieval_returns_no_answer_without_llm() -> None:
    """Empty retrieval -> No_Answer_Response, empty citations, LLM NOT called.

    Validates: R8.4, R9.3. The stub retrieval returns ``[]`` so answer() returns
    the sentinel with no citations and never calls the LLM.
    """
    llm = _NeverCalledLLMProvider()
    retrieval = _StubRetrievalService(chunks=[])
    service = Query_Service(_config(), retrieval_service=retrieval, llm_provider=llm)

    response = service.answer("anything?")

    assert response.answer == NO_ANSWER_RESPONSE
    assert response.citations == []
    assert retrieval.called is True
    assert llm.called is False


def test_llm_returns_exact_sentinel_yields_empty_citations() -> None:
    """LLM returns EXACTLY the sentinel -> empty citations, answer == sentinel.

    Validates: R8.4, R9.3. At least one chunk is retrieved, but the model's exact
    no-answer sentinel suppresses all citations.
    """
    llm = _StubLLMProvider(NO_ANSWER_RESPONSE)
    retrieval = _StubRetrievalService(
        chunks=[_make_chunk("d.pdf", 2, "some context text")]
    )
    service = Query_Service(_config(), retrieval_service=retrieval, llm_provider=llm)

    response = service.answer("a real question?")

    assert llm.called is True
    assert response.answer == NO_ANSWER_RESPONSE
    assert response.citations == []


def test_llm_error_propagates_unchanged() -> None:
    """LLMError from generate propagates unchanged out of answer() (D5)."""
    llm = _RaisingLLMProvider()
    retrieval = _StubRetrievalService(
        chunks=[_make_chunk("d.pdf", 1, "context text")]
    )
    service = Query_Service(_config(), retrieval_service=retrieval, llm_provider=llm)

    with pytest.raises(LLMError):
        service.answer("a question?")

    assert llm.called is True


def test_reembedding_required_error_propagates_without_llm() -> None:
    """ReembeddingRequiredError from retrieval propagates unchanged (D6); LLM NOT called."""
    llm = _NeverCalledLLMProvider()
    retrieval = _StubRetrievalService(
        error=ReembeddingRequiredError(["doc-1"])
    )
    service = Query_Service(_config(), retrieval_service=retrieval, llm_provider=llm)

    with pytest.raises(ReembeddingRequiredError) as exc_info:
        service.answer("a question?")

    assert exc_info.value.document_ids == ["doc-1"]
    assert retrieval.called is True
    assert llm.called is False


def test_missing_retrieval_service_raises_value_error() -> None:
    """D7: a None retrieval_service raises a clear ValueError (never builds a store)."""
    with pytest.raises(ValueError, match="Retrieval_Service is required"):
        Query_Service(_config(), retrieval_service=None, llm_provider=_StubLLMProvider("x"))

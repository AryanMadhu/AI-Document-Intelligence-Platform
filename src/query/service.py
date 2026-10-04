"""Query orchestration and the grounded prompt builder (Task 12; R8, R9, R16.1).

:class:`Query_Service` is the write-nothing read side of the answer pipeline. It
orchestrates: validate question -> retrieve top-k chunks -> build a grounded
prompt -> call the LLM -> assemble ``{answer, citations}``. It holds no domain
logic and never touches the vector store or an embedding provider directly: the
only abstractions it uses are the injected :class:`~src.retrieval.Retrieval_Service`
(the sole gateway to retrieval, which itself owns the store) and the
:class:`~src.llm.base.LLMProvider` (the sole gateway to generation). The caller
(e.g. the API layer) is responsible for constructing and injecting a configured
``Retrieval_Service``; ``Query_Service`` will not build a store on its own.

Grounding honesty (important — do not overstate what the MVP guarantees):
    The grounded prompt is an *instruction* to the LLM to answer only from the
    provided context and to return the no-answer sentinel otherwise. It is NOT a
    hard technical constraint and does not, by itself, make it impossible for
    the model to hallucinate. What the MVP guarantees *structurally* is citation
    validity: citations are generated ONLY from the retrieved chunks that were
    placed into the LLM context (see :mod:`src.query.formatter`), never invented
    outside that set.

    Citation validity is structurally guaranteed, but citation relevance and
    LLM factuality are not independently verified in the MVP.
"""

from __future__ import annotations

from src.config import Config
from src.llm.base import LLMProvider, build_llm_provider
from src.models import QueryResponse, RetrievedChunk
from src.query.formatter import Response_Formatter
from src.retrieval import Retrieval_Service

# D1 — the single no-answer sentinel. This EXACT string is used both in the
# prompt (the instruction tells the model to reply with exactly this text) and
# for detection. Detection is an EXACT match after ``.strip()`` — not
# case-insensitive and not a substring test. There is one source of truth for
# this constant; the formatter imports it from here.
NO_ANSWER_RESPONSE = "I don't know based on the provided documents"


class EmptyQuestionError(Exception):
    """Raised when the question is ``None``, empty, or whitespace-only (D4, R16.1).

    This is raised by :meth:`Query_Service.answer` *before* any retrieval or LLM
    call, so neither the retrieval service nor the LLM is ever invoked for an
    empty question. The message is value-free (carries no user content).
    """


def build_grounded_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    """Build the design's grounded prompt from a question and retrieved chunks.

    This is a module-level, side-effect-free pure function so it can be tested
    directly (Property 9). The returned prompt embeds every retrieved chunk's
    text (each tagged with its source document and page) and both grounding
    instructions: answer using ONLY the context, and reply with EXACTLY the
    :data:`NO_ANSWER_RESPONSE` sentinel when the context does not contain the
    answer (R8.1, R8.2).

    The prompt is an instruction, not a hard constraint — see the module
    docstring on grounding honesty.

    Args:
        question: The user's natural-language question.
        chunks: The retrieved chunks to ground the answer in; each is rendered
            as one numbered context line (1-based).

    Returns:
        The fully-assembled grounded prompt string.
    """
    context_lines = [
        f"[{index}] (document: {retrieved.chunk.document_name}, "
        f"page: {retrieved.chunk.page_number}) {retrieved.chunk.chunk_text}"
        for index, retrieved in enumerate(chunks, start=1)
    ]
    context_block = "\n".join(context_lines)

    return (
        "You are a document question-answering assistant. Answer the user's "
        "question\n"
        "using ONLY the numbered context passages below. Do not use any outside "
        "knowledge.\n"
        "If the answer is not contained in the context, reply with EXACTLY:\n"
        f'"{NO_ANSWER_RESPONSE}"\n'
        "\n"
        "Context passages:\n"
        f"{context_block}\n"
        "\n"
        f"Question: {question}\n"
        "\n"
        "Answer using only the context above. Do not fabricate citations or facts."
    )


class Query_Service:
    """Orchestrate retrieve -> grounded prompt -> LLM -> citation assembly (R8, R9).

    ``Query_Service`` never constructs or touches a :class:`VectorStore` or an
    embedding provider directly. Retrieval is reached ONLY through the injected
    :class:`~src.retrieval.Retrieval_Service`, and generation ONLY through the
    injected :class:`~src.llm.base.LLMProvider`. This keeps the service decoupled
    from storage concerns; the caller/API wires in a configured retrieval
    service.
    """

    def __init__(
        self,
        config: Config,
        retrieval_service: Retrieval_Service | None = None,
        llm_provider: LLMProvider | None = None,
    ) -> None:
        """Wire the service to its configuration, retrieval service, and LLM.

        Args:
            config: The active configuration (read-only here).
            retrieval_service: The sole gateway to retrieval. **Required**: if
                ``None`` a :class:`ValueError` is raised, because building one
                would require constructing a vector store and this service
                deliberately never touches the store directly.
            llm_provider: The sole gateway to generation. Defaults to
                :func:`build_llm_provider` applied to ``config`` (which does not
                touch any store and lazily loads its SDK only on first use).

        Raises:
            ValueError: If ``retrieval_service`` is ``None``.
        """
        if retrieval_service is None:
            raise ValueError("a Retrieval_Service is required")

        self._config = config
        self._retrieval_service = retrieval_service
        self._llm_provider = llm_provider or build_llm_provider(config)
        self._formatter = Response_Formatter()

    def answer(self, question: str) -> QueryResponse:
        """Produce a grounded, cited answer for ``question`` (R8, R9, R16.1).

        Exact step order:
            1. **Validate** — if ``question`` is ``None``, empty, or
               whitespace-only, raise :class:`EmptyQuestionError`. Neither the
               retrieval service nor the LLM is called (D4, R16.1).
            2. **Retrieve** — ask the injected ``Retrieval_Service`` for the
               top-k chunks. A :class:`~src.retrieval.ReembeddingRequiredError`
               raised here propagates unchanged (D6).
            3. **Empty retrieval** — if no chunks come back, return
               :data:`NO_ANSWER_RESPONSE` with empty citations. The LLM is NOT
               called (R8.4, R9.3).
            4. **Build the grounded prompt** from the question and chunks.
            5. **Generate** — call ``LLMProvider.generate``. An
               :class:`~src.llm.base.LLMError` raised here propagates unchanged
               (D5).
            6. **No-answer sentinel** — if the generated text exactly equals
               :data:`NO_ANSWER_RESPONSE` after ``.strip()``, return it with
               empty citations.
            7. **Assemble citations** — otherwise delegate to
               :class:`Response_Formatter`, building one citation per in-context
               chunk (D2) whose excerpt is the chunk's full ``chunk_text`` (D3).

        Args:
            question: The user's natural-language question.

        Returns:
            A :class:`~src.models.QueryResponse` carrying the answer and its
            citations (empty when the context does not support an answer).

        Raises:
            EmptyQuestionError: If ``question`` is ``None``/empty/whitespace.
            ReembeddingRequiredError: Propagated unchanged from retrieval (D6).
            LLMError: Propagated unchanged from the LLM (D5).
        """
        # 1. Validate FIRST — before any retrieval or LLM call (D4, R16.1).
        if question is None or question.strip() == "":
            raise EmptyQuestionError("a question is required")

        # 2. Retrieve top-k. ReembeddingRequiredError propagates unchanged (D6).
        chunks = self._retrieval_service.retrieve(question)

        # 3. Empty retrieval -> no-answer, empty citations, LLM NOT called.
        if not chunks:
            return QueryResponse(answer=NO_ANSWER_RESPONSE, citations=[])

        # 4. Build the grounded prompt.
        prompt = build_grounded_prompt(question, chunks)

        # 5. Generate. LLMError propagates unchanged (D5).
        answer_text = self._llm_provider.generate(prompt)

        # 6. Exact no-answer sentinel detection (strip, exact match) -> empty
        #    citations.
        if answer_text.strip() == NO_ANSWER_RESPONSE:
            return QueryResponse(answer=NO_ANSWER_RESPONSE, citations=[])

        # 7. Assemble the final response; citations derive ONLY from in-context
        #    chunks (D2/D3).
        return self._formatter.format(answer_text, chunks)

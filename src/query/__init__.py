"""Query orchestration, grounded prompting, citations (Task 12).

Re-exports the query write-side API: :class:`Query_Service` (orchestration),
:class:`Response_Formatter` (citation assembly), :class:`EmptyQuestionError`
(empty-question guard), the pure :func:`build_grounded_prompt` builder, and the
single :data:`NO_ANSWER_RESPONSE` sentinel.

Importing this package is lightweight: ``service.py`` imports
:func:`~src.llm.base.build_llm_provider`, which lazily loads the heavy LLM SDK
only when a provider is actually built — so importing these classes does not
drag in torch/transformers or the Google SDK.
"""

from src.query.formatter import Response_Formatter
from src.query.service import (
    NO_ANSWER_RESPONSE,
    EmptyQuestionError,
    Query_Service,
    build_grounded_prompt,
)

__all__ = [
    "Query_Service",
    "Response_Formatter",
    "EmptyQuestionError",
    "build_grounded_prompt",
    "NO_ANSWER_RESPONSE",
]

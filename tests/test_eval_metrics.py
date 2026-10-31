"""Offline, deterministic unit tests for the evaluation metrics (Task 17.2).

These tests import ONLY the pure helper functions from ``eval/run_eval.py`` and
exercise them against hand-computed values. They make NO network/model/LLM
calls: the retrieval/query "services" used in the ``evaluate_question`` tests are
tiny local fakes, and all chunk/citation objects are the real ``src.models``
dataclasses/pydantic models constructed directly.

Why ``importlib.util`` instead of ``from eval.run_eval import ...``:
    The project's ``eval/`` directory is a plain data/script folder with NO
    ``__init__.py`` and its name shadows Python's builtin ``eval``. Importing it
    as a package from the test suite is awkward and fragile. Loading the module
    from its file path with ``importlib.util`` is the clean, dependency-free way
    to reach the pure functions WITHOUT creating any new production file
    (e.g. an ``eval/__init__.py``). The module's own top-of-file sys.path shim
    makes its ``src`` imports resolve regardless.

**Validates: Requirements 14.2, 14.3, 14.4**
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys

import pytest

from src.models import Chunk, Citation, RetrievedChunk

# --- Load eval/run_eval.py by file path (see module docstring) -------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS_DIR)
_RUN_EVAL_PATH = os.path.join(_ROOT, "eval", "run_eval.py")

_MODULE_NAME = "eval_run_eval_under_test"
_spec = importlib.util.spec_from_file_location(_MODULE_NAME, _RUN_EVAL_PATH)
assert _spec is not None and _spec.loader is not None
run_eval = importlib.util.module_from_spec(_spec)
# Register in sys.modules BEFORE exec so that `@dataclass` introspection (which
# looks the defining module up in sys.modules by __module__) resolves cleanly.
sys.modules[_MODULE_NAME] = run_eval
_spec.loader.exec_module(run_eval)
# --------------------------------------------------------------------------

NO_ANSWER = "I don't know based on the provided documents"


# ---------------------------------------------------------------------------
# Small builders for the real src.models objects (no network).
# ---------------------------------------------------------------------------


def make_chunk(document_name: str, page_number: int, chunk_text: str) -> RetrievedChunk:
    """Build a RetrievedChunk wrapping a real Chunk (no services involved)."""
    return RetrievedChunk(
        chunk=Chunk(
            document_name=document_name,
            page_number=page_number,
            section=None,
            chunk_text=chunk_text,
        ),
        score=0.9,
        embedding_provider="sentence-transformers",
        embedding_model="all-MiniLM-L6-v2",
    )


class FakeResponse:
    """A stand-in for QueryResponse with just the fields the harness reads."""

    def __init__(self, answer: str, citations: list) -> None:
        self.answer = answer
        self.citations = citations


class FakeRetrieval:
    """Fake retrieval service: returns canned chunks or raises."""

    def __init__(self, result=None, exc: Exception | None = None) -> None:
        self._result = result or []
        self._exc = exc

    def retrieve(self, question: str):
        if self._exc is not None:
            raise self._exc
        return self._result


class FakeQuery:
    """Fake query service: returns a canned response or raises."""

    def __init__(self, response: FakeResponse | None = None, exc: Exception | None = None) -> None:
        self._response = response
        self._exc = exc

    def answer(self, question: str):
        if self._exc is not None:
            raise self._exc
        return self._response


# ===========================================================================
# normalize_text
# ===========================================================================


def test_normalize_text_collapses_whitespace_strips_lowercases():
    assert run_eval.normalize_text("  Hello\n\tWORLD   foo ") == "hello world foo"
    assert run_eval.normalize_text("Multiple\n\n\nlines\there") == "multiple lines here"


def test_normalize_text_coerces_non_str():
    assert run_eval.normalize_text(123) == "123"


# ===========================================================================
# load_eval_set
# ===========================================================================


def _write_json(tmp_path, obj) -> str:
    path = os.path.join(str(tmp_path), "eval_set.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(obj, handle)
    return path


def _valid_entry(**overrides):
    entry = {
        "id": "ML-01",
        "question": "What is training?",
        "essential_facts": ["fact one", "fact two"],
        "source_document": "doc.pdf",
        "source_page": 1,
    }
    entry.update(overrides)
    return entry


def test_load_eval_set_valid(tmp_path):
    path = _write_json(tmp_path, [_valid_entry(), _valid_entry(id="ML-02", source_page=2)])
    questions = run_eval.load_eval_set(path)
    assert len(questions) == 2
    assert questions[0].id == "ML-01"
    assert questions[0].essential_facts == ["fact one", "fact two"]
    assert questions[1].source_page == 2


def test_load_eval_set_rejects_non_list(tmp_path):
    path = _write_json(tmp_path, {"id": "ML-01"})
    with pytest.raises(ValueError, match="top-level JSON array"):
        run_eval.load_eval_set(path)


def test_load_eval_set_missing_field_names_id(tmp_path):
    entry = _valid_entry()
    del entry["source_document"]
    path = _write_json(tmp_path, [entry])
    with pytest.raises(ValueError, match="id='ML-01'"):
        run_eval.load_eval_set(path)


def test_load_eval_set_missing_id_field_names_index(tmp_path):
    entry = _valid_entry()
    del entry["id"]
    path = _write_json(tmp_path, [entry])
    with pytest.raises(ValueError, match="index 0"):
        run_eval.load_eval_set(path)


def test_load_eval_set_rejects_zero_page(tmp_path):
    path = _write_json(tmp_path, [_valid_entry(source_page=0)])
    with pytest.raises(ValueError, match="source_page"):
        run_eval.load_eval_set(path)


def test_load_eval_set_rejects_string_page(tmp_path):
    path = _write_json(tmp_path, [_valid_entry(source_page="1")])
    with pytest.raises(ValueError, match="source_page"):
        run_eval.load_eval_set(path)


def test_load_eval_set_rejects_non_list_facts(tmp_path):
    path = _write_json(tmp_path, [_valid_entry(essential_facts="not a list")])
    with pytest.raises(ValueError, match="essential_facts"):
        run_eval.load_eval_set(path)


def test_load_eval_set_rejects_empty_facts(tmp_path):
    path = _write_json(tmp_path, [_valid_entry(essential_facts=[])])
    with pytest.raises(ValueError, match="essential_facts"):
        run_eval.load_eval_set(path)


# ===========================================================================
# retrieval_hit
# ===========================================================================


def test_retrieval_hit_true_when_doc_and_page_present():
    retrieved = [make_chunk("a.pdf", 1, "x"), make_chunk("doc.pdf", 2, "y")]
    assert run_eval.retrieval_hit("doc.pdf", 2, retrieved) is True


def test_retrieval_hit_miss_wrong_page():
    retrieved = [make_chunk("doc.pdf", 1, "x")]
    assert run_eval.retrieval_hit("doc.pdf", 2, retrieved) is False


def test_retrieval_hit_miss_wrong_doc():
    retrieved = [make_chunk("other.pdf", 2, "x")]
    assert run_eval.retrieval_hit("doc.pdf", 2, retrieved) is False


def test_retrieval_hit_miss_empty():
    assert run_eval.retrieval_hit("doc.pdf", 1, []) is False


def test_retrieval_hit_is_case_sensitive():
    retrieved = [make_chunk("Doc.PDF", 1, "x")]
    assert run_eval.retrieval_hit("doc.pdf", 1, retrieved) is False


# ===========================================================================
# fact_coverage
# ===========================================================================


def test_fact_coverage_all_present():
    answer = "Training adjusts the model. Inference uses the trained model."
    facts = ["training adjusts the model", "inference uses the trained model"]
    assert run_eval.fact_coverage(answer, facts) == (2, 2)


def test_fact_coverage_none_present():
    assert run_eval.fact_coverage("totally unrelated", ["alpha", "beta", "gamma"]) == (0, 3)


def test_fact_coverage_partial():
    answer = "Classification predicts a category."
    facts = ["classification predicts a category", "regression predicts a number"]
    assert run_eval.fact_coverage(answer, facts) == (1, 2)


def test_fact_coverage_newline_in_fact_matches_single_line_answer():
    # A fact that contains an internal newline still matches an answer that has
    # the same words on one line (both normalize to single spaces).
    answer = "The input layer receives feature values here."
    facts = ["the input layer\nreceives feature values"]
    assert run_eval.fact_coverage(answer, facts) == (1, 1)


def test_fact_coverage_is_case_insensitive():
    assert run_eval.fact_coverage("HELLO WORLD", ["hello world"]) == (1, 1)


# ===========================================================================
# citation_is_supported
# ===========================================================================


def test_citation_is_supported_match():
    retrieved = [make_chunk("doc.pdf", 2, "A full sentence that supports the claim.")]
    cite = Citation(document="doc.pdf", page=2, excerpt="supports the claim")
    assert run_eval.citation_is_supported(cite, retrieved) is True


def test_citation_is_supported_doc_differs():
    retrieved = [make_chunk("doc.pdf", 2, "supports the claim")]
    cite = Citation(document="other.pdf", page=2, excerpt="supports the claim")
    assert run_eval.citation_is_supported(cite, retrieved) is False


def test_citation_is_supported_page_differs():
    retrieved = [make_chunk("doc.pdf", 2, "supports the claim")]
    cite = Citation(document="doc.pdf", page=3, excerpt="supports the claim")
    assert run_eval.citation_is_supported(cite, retrieved) is False


def test_citation_is_supported_excerpt_not_in_chunk():
    retrieved = [make_chunk("doc.pdf", 2, "a totally different sentence")]
    cite = Citation(document="doc.pdf", page=2, excerpt="supports the claim")
    assert run_eval.citation_is_supported(cite, retrieved) is False


def test_citation_is_supported_robust_int_str_page_compare():
    # Chunk page is int 2; Citation page comes through as str "2" -> still matches.
    retrieved = [make_chunk("doc.pdf", 2, "supports the claim")]
    cite = Citation(document="doc.pdf", page="2", excerpt="supports the claim")
    assert run_eval.citation_is_supported(cite, retrieved) is True


# ===========================================================================
# citation_validity (the refined rules)
# ===========================================================================


def test_citation_validity_no_answer_zero_citations_is_valid():
    assert run_eval.citation_validity(NO_ANSWER, [], [], NO_ANSWER) is True


def test_citation_validity_no_answer_with_citations_is_invalid():
    bogus = Citation(document="doc.pdf", page=1, excerpt="x")
    assert run_eval.citation_validity(NO_ANSWER, [bogus], [], NO_ANSWER) is False


def test_citation_validity_ordinary_answer_zero_citations_is_invalid():
    retrieved = [make_chunk("doc.pdf", 1, "some supporting text")]
    assert run_eval.citation_validity("A real answer.", [], retrieved, NO_ANSWER) is False


def test_citation_validity_ordinary_answer_all_supported_is_valid():
    retrieved = [make_chunk("doc.pdf", 1, "the earth orbits the sun")]
    cite = Citation(document="doc.pdf", page=1, excerpt="orbits the sun")
    assert run_eval.citation_validity("A real answer.", [cite], retrieved, NO_ANSWER) is True


def test_citation_validity_ordinary_answer_unrelated_doc_is_invalid():
    retrieved = [make_chunk("doc.pdf", 1, "the earth orbits the sun")]
    cite = Citation(document="unrelated.pdf", page=9, excerpt="orbits the sun")
    assert run_eval.citation_validity("A real answer.", [cite], retrieved, NO_ANSWER) is False


def test_citation_validity_ordinary_answer_unsupported_excerpt_is_invalid():
    retrieved = [make_chunk("doc.pdf", 1, "the earth orbits the sun")]
    cite = Citation(document="doc.pdf", page=1, excerpt="made up excerpt not present")
    assert run_eval.citation_validity("A real answer.", [cite], retrieved, NO_ANSWER) is False


# ===========================================================================
# Aggregation transparency
# ===========================================================================


def test_aggregate_retrieval_counts_and_score():
    rows = [
        {"retrieval_status": "evaluated", "hit": True},
        {"retrieval_status": "evaluated", "hit": False},
        {"retrieval_status": "evaluated", "hit": True},
        {"retrieval_status": "error", "hit": None},
    ]
    result = run_eval.aggregate_retrieval(rows)
    assert result.numerator == 2
    assert result.denominator == 3  # error excluded; miss stays in denominator
    assert result.score == pytest.approx(2 / 3)
    assert result.evaluated == 3
    assert result.errors == 1
    assert 0.0 <= result.score <= 1.0


def test_aggregate_retrieval_zero_denominator_is_none_not_divide_by_zero():
    rows = [{"retrieval_status": "error", "hit": None}]
    result = run_eval.aggregate_retrieval(rows)
    assert result.denominator == 0
    assert result.score is None


def test_aggregate_fact_coverage_mean_and_micro():
    rows = [
        {"answer_status": "evaluated", "fact_present": 2, "fact_total": 2},  # 1.0
        {"answer_status": "evaluated", "fact_present": 1, "fact_total": 2},  # 0.5
        {"answer_status": "error", "fact_present": None, "fact_total": 3},
    ]
    result, extra = run_eval.aggregate_fact_coverage(rows)
    # Mean of fractions = (1.0 + 0.5) / 2 = 0.75
    assert result.score == pytest.approx(0.75)
    assert result.evaluated == 2
    assert result.errors == 1
    assert 0.0 <= result.score <= 1.0
    # Micro-average raw sums = 3 present / 4 total = 0.75 here too, reported separately.
    assert extra["micro_average_present_sum"] == 3
    assert extra["micro_average_total_sum"] == 4
    assert extra["micro_average_score"] == pytest.approx(0.75)


def test_aggregate_citation_validity_counts():
    rows = [
        {"citation_valid": True},
        {"citation_valid": False},
        {"citation_valid": None},  # not computable -> excluded
    ]
    result = run_eval.aggregate_citation_validity(rows)
    assert result.numerator == 1
    assert result.denominator == 2
    assert result.score == pytest.approx(0.5)
    assert result.evaluated == 2
    assert result.errors == 1


def test_aggregate_citation_validity_zero_denominator_is_none():
    rows = [{"citation_valid": None}]
    result = run_eval.aggregate_citation_validity(rows)
    assert result.denominator == 0
    assert result.score is None


# ===========================================================================
# evaluate_question with FAKE services (no network / no real services)
# ===========================================================================


def _eval_question():
    from src.models import EvalQuestion

    return EvalQuestion(
        id="ML-01",
        question="q?",
        essential_facts=["orbits the sun"],
        source_document="doc.pdf",
        source_page=1,
    )


def test_evaluate_question_retrieval_error_is_not_fabricated_hit_or_miss():
    q = _eval_question()
    retrieval = FakeRetrieval(exc=RuntimeError("boom"))
    query = FakeQuery(response=FakeResponse("the earth orbits the sun", []))
    row = run_eval.evaluate_question(q, retrieval, query, NO_ANSWER)
    assert row["retrieval_status"] == "error"
    assert row["hit"] is None  # NOT True, NOT False
    assert row["retrieved_pairs"] == []
    assert row["retrieval_error_reason"].startswith("RuntimeError")
    # Citation validity needs retrieval -> not computable when retrieval errored.
    assert row["citation_valid"] is None


def test_evaluate_question_llm_error_coverage_and_citation_not_evaluated_not_zero():
    q = _eval_question()
    retrieved = [make_chunk("doc.pdf", 1, "the earth orbits the sun")]
    retrieval = FakeRetrieval(result=retrieved)
    query = FakeQuery(exc=RuntimeError("llm down"))
    row = run_eval.evaluate_question(q, retrieval, query, NO_ANSWER)
    # Retrieval still evaluated independently.
    assert row["retrieval_status"] == "evaluated"
    assert row["hit"] is True
    # Answer failed -> coverage NOT 0, citation NOT fabricated.
    assert row["answer_status"] == "error"
    assert row["fact_present"] is None
    assert row["citation_valid"] is None
    assert row["answer_error_reason"].startswith("RuntimeError")


def test_evaluate_question_happy_path():
    q = _eval_question()
    retrieved = [make_chunk("doc.pdf", 1, "the earth orbits the sun")]
    retrieval = FakeRetrieval(result=retrieved)
    cite = Citation(document="doc.pdf", page=1, excerpt="orbits the sun")
    query = FakeQuery(response=FakeResponse("the earth orbits the sun", [cite]))
    row = run_eval.evaluate_question(q, retrieval, query, NO_ANSWER)
    assert row["retrieval_status"] == "evaluated"
    assert row["hit"] is True
    assert row["answer_status"] == "evaluated"
    assert row["fact_present"] == 1
    assert row["fact_total"] == 1
    assert row["citation_valid"] is True

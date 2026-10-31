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


# ===========================================================================
# Token helpers for the OPT-IN overlap matcher (_tokenize / _singularize /
# _content_tokens). Deterministic, stdlib-only, no network.
# ===========================================================================


def test_tokenize_splits_on_non_alphanumeric_and_lowercases():
    assert run_eval._tokenize("Input layer: Receives feature-values!") == [
        "input",
        "layer",
        "receives",
        "feature",
        "values",
    ]
    # Digits are kept; punctuation/whitespace are separators.
    assert run_eval._tokenize("Top-K = 5, k2") == ["top", "k", "5", "k2"]


def test_tokenize_coerces_non_str():
    assert run_eval._tokenize(123) == ["123"]


def test_singularize_regular_plurals():
    # ies -> y
    assert run_eval._singularize("categories") == "category"
    # plain trailing s -> stripped (words not ending in es/ss)
    assert run_eval._singularize("models") == "model"
    assert run_eval._singularize("numbers") == "number"
    # es -> stripped (len > 3); "classes" -> "class"
    assert run_eval._singularize("classes") == "class"


def test_singularize_es_rule_can_over_stem_documented_heuristic():
    # DOCUMENTED QUIRK: the ordered rules check "es" before plain "s", so an
    # "-es" word whose singular is just "-e" (e.g. "value") over-stems to
    # "valu". This is acceptable because overlap matching compares the SAME
    # stemmer on both sides: the answer's "values" also stems to "valu", so the
    # fact/answer tokens still agree. It is NOT a real lemmatizer.
    assert run_eval._singularize("values") == "valu"


def test_singularize_leaves_ss_and_short_words_alone():
    # "class" ends in "ss" -> unchanged (not stripped to "clas").
    assert run_eval._singularize("class") == "class"
    assert run_eval._singularize("process") == "process"
    # Short words are untouched.
    assert run_eval._singularize("is") == "is"
    assert run_eval._singularize("as") == "as"
    # len == 3 boundary: "yes" should be left as-is (rule requires len > 3).
    assert run_eval._singularize("yes") == "yes"


def test_content_tokens_drops_stopwords_and_singularizes():
    # "a" and "the" are stopwords; "categories" singularized to "category".
    assert run_eval._content_tokens("Classification predicts a category") == [
        "classification",
        "predict",
        "category",
    ]
    # "that is" filler removed (both stopwords), plural normalized.
    assert run_eval._content_tokens("information that is unavailable") == [
        "information",
        "unavailable",
    ]


# ===========================================================================
# fact_covered_by_overlap (the opt-in matcher)
# ===========================================================================


def test_overlap_matches_plural_singular_divergence():
    fact = "Classification predicts a category"
    answer = "Classification predicts categories into discrete labels."
    assert run_eval.fact_covered_by_overlap(fact, answer, 0.85) is True


def test_overlap_matches_reformatting():
    fact = "The input layer receives feature values"
    answer = "Input layer: Receives feature values from the dataset."
    assert run_eval.fact_covered_by_overlap(fact, answer, 0.85) is True


def test_overlap_matches_inserted_filler_words():
    fact = (
        "Data leakage occurs when information unavailable at prediction time "
        "influences training or evaluation"
    )
    answer = (
        "Data leakage occurs when information that is unavailable at prediction "
        "time influences training or evaluation of the model."
    )
    assert run_eval.fact_covered_by_overlap(fact, answer, 0.85) is True


def test_overlap_absent_fact_is_false():
    fact = "Gradient descent minimizes the loss function"
    answer = "Bananas are a good source of potassium and fiber."
    assert run_eval.fact_covered_by_overlap(fact, answer, 0.85) is False


def test_overlap_threshold_half_tokens_is_false_at_085():
    # Fact has 4 distinct content tokens; answer contains only 2 of them
    # -> ratio 0.5 < 0.85 -> not covered.
    fact = "alpha beta gamma delta"
    answer = "alpha beta omega sigma"
    assert run_eval.fact_covered_by_overlap(fact, answer, 0.85) is False
    # Same fact clears a lower threshold.
    assert run_eval.fact_covered_by_overlap(fact, answer, 0.5) is True


def test_overlap_all_stopword_fact_falls_back_to_substring():
    # A fact made entirely of stopwords has no content tokens -> fall back to
    # the strict normalized-substring behavior.
    fact = "it is the"  # all stopwords under _STOPWORDS
    assert run_eval._content_tokens(fact) == []
    # Substring present (case/space-insensitive) -> covered.
    assert run_eval.fact_covered_by_overlap(fact, "Yes, IT IS THE answer here.", 0.85) is True
    # Substring absent -> not covered.
    assert run_eval.fact_covered_by_overlap(fact, "completely different text", 0.85) is False


# ===========================================================================
# fact_coverage mode switching (regression: default stays substring)
# ===========================================================================


def test_fact_coverage_default_is_substring_regression():
    # Paraphrase that is NOT a substring: strict default scores it missing,
    # exactly as the pre-refinement behavior did.
    answer = "Classification predicts categories."
    facts = ["classification predicts a category"]
    assert run_eval.fact_coverage(answer, facts) == (0, 1)
    assert run_eval.fact_coverage(answer, facts, mode="substring") == (0, 1)


def test_fact_coverage_overlap_credits_paraphrase_higher_than_substring():
    answer = "Classification predicts categories into discrete labels."
    facts = ["classification predicts a category", "regression predicts a number"]
    substring = run_eval.fact_coverage(answer, facts, mode="substring")
    overlap = run_eval.fact_coverage(answer, facts, mode="overlap")
    assert substring == (0, 2)
    assert overlap == (1, 2)
    assert overlap[0] > substring[0]


def test_fact_coverage_rejects_unknown_mode():
    with pytest.raises(ValueError, match="substring"):
        run_eval.fact_coverage("x", ["y"], mode="semantic")


# ===========================================================================
# evaluate_question with coverage_mode="overlap" (FAKE services, no network)
# ===========================================================================


def test_evaluate_question_overlap_credits_paraphrase_substring_scores_zero():
    from src.models import EvalQuestion

    q = EvalQuestion(
        id="ML-02",
        question="What does classification predict?",
        essential_facts=["classification predicts a category"],
        source_document="doc.pdf",
        source_page=1,
    )
    retrieved = [make_chunk("doc.pdf", 1, "classification predicts categories")]
    retrieval = FakeRetrieval(result=retrieved)
    cite = Citation(document="doc.pdf", page=1, excerpt="classification predicts categories")
    query = FakeQuery(
        response=FakeResponse("Classification predicts categories into labels.", [cite])
    )

    # Default substring mode scores the paraphrase as 0 present.
    sub_row = run_eval.evaluate_question(q, retrieval, query, NO_ANSWER)
    assert sub_row["fact_present"] == 0

    # Overlap mode credits the same paraphrased correct answer.
    ov_row = run_eval.evaluate_question(
        q, retrieval, query, NO_ANSWER, coverage_mode="overlap"
    )
    assert ov_row["fact_present"] == 1
    assert ov_row["fact_total"] == 1
    assert ov_row["answer_status"] == "evaluated"

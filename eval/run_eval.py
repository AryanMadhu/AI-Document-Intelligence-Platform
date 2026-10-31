"""Evaluation harness for the RAG pipeline (Task 17; R14).

This script runs every question in the version-controlled Eval_Set
(``eval/eval_set.json``) through the production query pipeline and computes
three independent, deterministic metrics:

* **Retrieval_Hit_Rate** (R14.2) — does the recorded correct source
  (``source_document`` + ``source_page``) appear among the top-k retrieved
  chunks?
* **Answer_Key_Fact_Coverage** (R14.3) — deterministic normalized substring
  matching of each question's ``essential_facts`` against the produced answer.
* **Citation_Validity** (R14.4) — does every citation trace to an in-context
  retrieved chunk (correct document + page) whose text supports the excerpt?

Design honesty / transparency (REFINEMENT 4): every metric is reported with its
numerator, denominator, score, and the count of questions that were *evaluated*
versus *errored / not_evaluated*. A retrieval or LLM service EXCEPTION never
fabricates a hit/miss or a 0 coverage — those questions are excluded from the
metric's denominator and surfaced explicitly. A plain retrieval MISS (the
expected document/page simply isn't in the top-k, or the document isn't indexed
at all) is a real miss counted in the denominator — only a retrieval *service
exception* is an "error".

Import safety: importing this module runs NOTHING heavy. All metric logic lives
in pure, side-effect-free functions (importable by the offline test suite); the
live orchestration (building the production stack, making Gemini/embedding
calls, writing result files) is confined to :func:`main` and guarded by the
``if __name__ == "__main__"`` block.

Run it from the PROJECT ROOT so ``src`` imports resolve::

    .venv\\Scripts\\python.exe eval\\run_eval.py

Results are written (git-ignored) to ``eval/results/`` in two timestamped
formats: a human-readable ``.md`` and a machine-readable ``.json`` for cross-run
comparison. Secrets (API keys, credential values) are NEVER printed or written.
"""

from __future__ import annotations

# --- sys.path robustness --------------------------------------------------
# Prepend the PROJECT ROOT (the parent of this file's directory, i.e. the parent
# of ``eval/``) to sys.path BEFORE importing any ``src.*`` module, mirroring the
# block in ui/app.py. This makes the script import-correct when launched as
# ``python eval/run_eval.py`` from the project root (where Python otherwise puts
# the script's own folder, ``eval/``, on sys.path instead of the root).
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
# --------------------------------------------------------------------------

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from src.models import EvalQuestion, RetrievedChunk

# Default location of the version-controlled Eval_Set (R2.3).
_DEFAULT_EVAL_SET = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_set.json")
# Git-ignored directory for timestamped result files (both .md and .json).
_RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

# The exact keys an Eval_Set entry must carry (and only these).
_REQUIRED_KEYS = frozenset(
    {"id", "question", "essential_facts", "source_document", "source_page"}
)


# ===========================================================================
# PURE METRIC HELPERS (imported by the offline test suite — NO side effects,
# NO network, NO model/LLM construction).
# ===========================================================================


def normalize_text(s: Any) -> str:
    """Normalize text for deterministic matching.

    Collapses every run of whitespace (spaces, tabs, newlines) to a single
    space, strips leading/trailing whitespace, and lowercases. Non-string
    inputs are coerced via ``str()`` so the function never raises on odd input.

    Args:
        s: The value to normalize (coerced to ``str`` if not already).

    Returns:
        The normalized string.
    """
    if not isinstance(s, str):
        s = str(s)
    return re.sub(r"\s+", " ", s).strip().lower()


def load_eval_set(path: str) -> list[EvalQuestion]:
    """Load and validate the Eval_Set JSON file into ``EvalQuestion`` objects.

    The file must be a TOP-LEVEL JSON array. Each entry must carry EXACTLY the
    five required keys with correct types:

    * ``id`` — non-empty ``str``
    * ``question`` — non-empty ``str``
    * ``essential_facts`` — a non-empty ``list`` of non-empty ``str``
    * ``source_document`` — non-empty ``str``
    * ``source_page`` — ``int`` >= 1 (``bool`` rejected; ``"1"`` rejected)

    Args:
        path: Filesystem path to the Eval_Set JSON file.

    Returns:
        The parsed ``list[EvalQuestion]`` preserving file order.

    Raises:
        ValueError: If the top level is not a list, or any entry is malformed.
            The message names the offending entry by its ``id`` when available,
            otherwise by its 0-based index.
    """
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, list):
        raise ValueError(
            f"Eval_Set at {path!r} must be a top-level JSON array, "
            f"got {type(data).__name__}."
        )

    questions: list[EvalQuestion] = []
    for index, raw in enumerate(data):
        # A label used only for error messages (id if present/string, else index).
        label = (
            f"id={raw['id']!r}"
            if isinstance(raw, dict) and isinstance(raw.get("id"), str)
            else f"index {index}"
        )

        if not isinstance(raw, dict):
            raise ValueError(
                f"Eval_Set entry at {label} must be a JSON object, "
                f"got {type(raw).__name__}."
            )

        keys = set(raw.keys())
        if keys != _REQUIRED_KEYS:
            missing = _REQUIRED_KEYS - keys
            extra = keys - _REQUIRED_KEYS
            problems = []
            if missing:
                problems.append(f"missing {sorted(missing)}")
            if extra:
                problems.append(f"unexpected {sorted(extra)}")
            raise ValueError(
                f"Eval_Set entry at {label} has the wrong keys: "
                + "; ".join(problems)
                + "."
            )

        q_id = raw["id"]
        question = raw["question"]
        essential_facts = raw["essential_facts"]
        source_document = raw["source_document"]
        source_page = raw["source_page"]

        if not isinstance(q_id, str) or q_id.strip() == "":
            raise ValueError(f"Eval_Set entry at {label}: 'id' must be a non-empty string.")
        if not isinstance(question, str) or question.strip() == "":
            raise ValueError(
                f"Eval_Set entry at {label}: 'question' must be a non-empty string."
            )
        if not isinstance(source_document, str) or source_document.strip() == "":
            raise ValueError(
                f"Eval_Set entry at {label}: 'source_document' must be a non-empty string."
            )
        # bool is a subclass of int — reject it explicitly so True/False aren't pages.
        if not isinstance(source_page, int) or isinstance(source_page, bool):
            raise ValueError(
                f"Eval_Set entry at {label}: 'source_page' must be an int (got "
                f"{type(source_page).__name__})."
            )
        if source_page < 1:
            raise ValueError(
                f"Eval_Set entry at {label}: 'source_page' must be >= 1 (got {source_page})."
            )
        if not isinstance(essential_facts, list) or len(essential_facts) == 0:
            raise ValueError(
                f"Eval_Set entry at {label}: 'essential_facts' must be a non-empty list."
            )
        for fact_index, fact in enumerate(essential_facts):
            if not isinstance(fact, str) or fact.strip() == "":
                raise ValueError(
                    f"Eval_Set entry at {label}: essential_facts[{fact_index}] "
                    "must be a non-empty string."
                )

        questions.append(
            EvalQuestion(
                id=q_id,
                question=question,
                essential_facts=list(essential_facts),
                source_document=source_document,
                source_page=source_page,
            )
        )

    return questions


def retrieval_hit(
    expected_document: str,
    expected_page: int,
    retrieved: list[RetrievedChunk],
) -> bool:
    """Return whether the expected source appears among the retrieved chunks.

    A hit requires an EXACT (case-sensitive) match of ``document_name`` AND an
    exact match of ``page_number``.

    Args:
        expected_document: The recorded correct source document name.
        expected_page: The recorded correct source page (1-based int).
        retrieved: The top-k retrieved chunks to search.

    Returns:
        ``True`` iff some chunk matches both the document and the page.
    """
    for item in retrieved:
        chunk = item.chunk
        if chunk.document_name == expected_document and chunk.page_number == expected_page:
            return True
    return False


def fact_coverage(answer: str, essential_facts: list[str]) -> tuple[int, int]:
    """Return ``(num_present, total)`` essential facts covered by the answer.

    A fact is present iff its normalized form is a substring of the normalized
    answer (deterministic, case-insensitive, whitespace-insensitive — see
    :func:`normalize_text`). This is a conservative lower bound on correctness:
    a paraphrase that doesn't appear as a substring scores as missing (a
    documented MVP limitation; an LLM-as-judge is Stretch).

    Args:
        answer: The produced answer text.
        essential_facts: The predefined key facts a correct answer must cover.

    Returns:
        ``(num_present, total)`` where ``total == len(essential_facts)``.
    """
    normalized_answer = normalize_text(answer)
    present = 0
    for fact in essential_facts:
        if normalize_text(fact) in normalized_answer:
            present += 1
    return present, len(essential_facts)


def citation_is_supported(citation: Any, retrieved: list[RetrievedChunk]) -> bool:
    """Return whether a citation traces to a supporting in-context chunk.

    A citation is supported iff SOME retrieved chunk matches ALL of:

    * ``chunk.document_name == citation.document`` (exact, case-sensitive),
    * ``str(chunk.page_number) == str(citation.page)`` (robust int/str compare,
      since ``Citation.page`` may be an ``int`` or a section-id ``str``), and
    * the normalized citation excerpt is a substring of the normalized chunk
      text (the chunk's text supports the excerpt).

    Args:
        citation: A ``Citation``-like object with ``.document``, ``.page``,
            ``.excerpt``.
        retrieved: The in-context retrieved chunks the citation must trace to.

    Returns:
        ``True`` iff at least one retrieved chunk supports the citation.
    """
    normalized_excerpt = normalize_text(citation.excerpt)
    for item in retrieved:
        chunk = item.chunk
        if (
            chunk.document_name == citation.document
            and str(chunk.page_number) == str(citation.page)
            and normalized_excerpt in normalize_text(chunk.chunk_text)
        ):
            return True
    return False


def citation_validity(
    answer: str,
    citations: list,
    retrieved: list[RetrievedChunk],
    no_answer_sentinel: str,
) -> bool:
    """Return whether an answer's citations are valid (R14.4).

    Rules:

    * **Confirmed no-answer** — if ``answer.strip() == no_answer_sentinel`` the
      result is valid iff there are ZERO citations (a no-answer response carries
      no citations by construction).
    * **Ordinary answer** — an ordinary answer with ZERO citations is INVALID
      (no free pass). Otherwise it is valid iff EVERY citation is supported by
      an in-context chunk (see :func:`citation_is_supported`); any citation to
      an unrelated document, an unrelated page, or with an unsupported excerpt
      makes the whole answer's citations invalid.

    Args:
        answer: The produced answer text.
        citations: The citations attached to the answer.
        retrieved: The in-context retrieved chunks citations must trace to.
        no_answer_sentinel: The exact no-answer sentinel string.

    Returns:
        ``True`` iff the citation set is valid under the rules above.
    """
    if answer.strip() == no_answer_sentinel:
        return len(citations) == 0

    if len(citations) == 0:
        return False

    return all(citation_is_supported(c, retrieved) for c in citations)


# ---------------------------------------------------------------------------
# Pure aggregation helpers (explicit numerator/denominator/score/counts).
# ---------------------------------------------------------------------------


@dataclass
class MetricResult:
    """A transparent metric summary: numerator/denominator/score + counts.

    Attributes:
        numerator: The metric numerator (hits / facts present / valid cites).
        denominator: The metric denominator (evaluated units).
        score: ``numerator / denominator``, or ``None`` when the denominator is
            0 (never divide by zero — report NA instead).
        evaluated: Number of questions that contributed to the metric.
        errors: Number of questions excluded (errored / not_evaluated).
    """

    numerator: float
    denominator: float
    score: float | None
    evaluated: int
    errors: int

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dict of this metric."""
        return {
            "numerator": self.numerator,
            "denominator": self.denominator,
            "score": self.score,
            "evaluated": self.evaluated,
            "errors": self.errors,
        }


def _safe_score(numerator: float, denominator: float) -> float | None:
    """Return ``numerator / denominator`` or ``None`` when ``denominator`` is 0."""
    if denominator == 0:
        return None
    return numerator / denominator


def aggregate_retrieval(rows: list[dict[str, Any]]) -> MetricResult:
    """Aggregate Retrieval_Hit_Rate over the per-question rows.

    Only questions whose retrieval was EVALUATED count. A plain miss
    (``hit is False``) stays in the denominator; only retrieval *service
    exceptions* (``retrieval_status == 'error'``) are excluded as errors.

    Args:
        rows: The per-question result dicts.

    Returns:
        A :class:`MetricResult` with hits/evaluated and the error count.
    """
    evaluated = [r for r in rows if r.get("retrieval_status") == "evaluated"]
    errors = sum(1 for r in rows if r.get("retrieval_status") == "error")
    hits = sum(1 for r in evaluated if r.get("hit") is True)
    denominator = len(evaluated)
    return MetricResult(
        numerator=float(hits),
        denominator=float(denominator),
        score=_safe_score(hits, denominator),
        evaluated=len(evaluated),
        errors=errors,
    )


def aggregate_fact_coverage(rows: list[dict[str, Any]]) -> tuple[MetricResult, dict[str, Any]]:
    """Aggregate Answer_Key_Fact_Coverage over the per-question rows.

    The headline score is the design's **mean across questions**: the average of
    each evaluated question's ``(fact_present / fact_total)`` fraction. The
    returned ``extra`` dict ALSO carries the raw micro-average sums
    (``present_sum`` / ``total_sum``) for transparency, clearly labeled so a
    reader can tell the two apart.

    Args:
        rows: The per-question result dicts.

    Returns:
        ``(MetricResult, extra)`` where ``MetricResult`` holds the mean-of-
        fractions headline and ``extra`` holds the micro-average sums.
    """
    evaluated = [r for r in rows if r.get("answer_status") == "evaluated"]
    errors = sum(1 for r in rows if r.get("answer_status") != "evaluated")

    per_question_fractions: list[float] = []
    present_sum = 0
    total_sum = 0
    for r in evaluated:
        present = int(r.get("fact_present") or 0)
        total = int(r.get("fact_total") or 0)
        present_sum += present
        total_sum += total
        if total > 0:
            per_question_fractions.append(present / total)

    denominator = len(per_question_fractions)
    numerator = sum(per_question_fractions)
    result = MetricResult(
        numerator=numerator,
        denominator=float(denominator),
        score=_safe_score(numerator, denominator),
        evaluated=len(evaluated),
        errors=errors,
    )
    extra = {
        "aggregation": "mean_of_per_question_fractions",
        "micro_average_present_sum": present_sum,
        "micro_average_total_sum": total_sum,
        "micro_average_score": _safe_score(present_sum, total_sum),
    }
    return result, extra


def aggregate_citation_validity(rows: list[dict[str, Any]]) -> MetricResult:
    """Aggregate Citation_Validity over the per-question rows.

    Only questions where citation validity could actually be computed count —
    i.e. both the answer AND the retrieval were evaluated, so
    ``citation_valid`` is a real bool (not ``None``). Questions where it could
    not be computed are the errors.

    Args:
        rows: The per-question result dicts.

    Returns:
        A :class:`MetricResult` with valid/evaluated and the not_evaluated count.
    """
    computable = [r for r in rows if isinstance(r.get("citation_valid"), bool)]
    errors = len(rows) - len(computable)
    valid = sum(1 for r in computable if r.get("citation_valid") is True)
    denominator = len(computable)
    return MetricResult(
        numerator=float(valid),
        denominator=float(denominator),
        score=_safe_score(valid, denominator),
        evaluated=len(computable),
        errors=errors,
    )


# ---------------------------------------------------------------------------
# Safe error-reason formatting (NEVER leak secrets / payloads).
# ---------------------------------------------------------------------------


def _safe_reason(exc: Exception) -> str:
    """Return a short, credential-safe reason string for an exception.

    Includes the exception's type name and a truncated message. This project's
    domain exceptions are built with value-free messages (R16.4); the message is
    still truncated defensively so an unexpected third-party exception cannot
    dump a large or sensitive payload into the results.

    Args:
        exc: The caught exception.

    Returns:
        ``"<TypeName>: <truncated message>"`` (or just the type name).
    """
    message = str(exc).strip()
    if len(message) > 200:
        message = message[:200] + "…"
    return f"{type(exc).__name__}: {message}" if message else type(exc).__name__


# ===========================================================================
# PER-QUESTION ORCHESTRATION (live; called only from main()).
# ===========================================================================


def evaluate_question(
    q: EvalQuestion,
    retrieval_service: Any,
    query_service: Any,
    no_answer_sentinel: str,
    source_present: bool | None = None,
) -> dict[str, Any]:
    """Evaluate one Eval_Set question end-to-end and return a result row.

    Retrieval and answer steps are each wrapped in try/except so a service
    exception never fabricates a hit/miss or a 0 coverage. A retrieval *service
    exception* sets ``retrieval_status='error'`` and ``hit=None``; a plain miss
    (expected doc/page not in the top-k, or the document isn't indexed) is a
    real ``hit=False`` with ``retrieval_status='evaluated'``. When retrieval
    errored, citation validity is also not computable (``citation_valid=None``).

    Args:
        q: The question to evaluate.
        retrieval_service: Object with ``.retrieve(question) -> list[RetrievedChunk]``.
        query_service: Object with ``.answer(question) -> QueryResponse``.
        no_answer_sentinel: The exact no-answer sentinel string.
        source_present: Whether ``q.source_document`` is indexed in the store
            (transparency flag; ``None`` when unknown). Does NOT change a miss
            into an error.

    Returns:
        A per-question result dict (JSON-serializable).
    """
    row: dict[str, Any] = {
        "id": q.id,
        "question": q.question,
        "source_document": q.source_document,
        "source_page": q.source_page,
        "source_present": source_present,
        "retrieval_status": None,
        "hit": None,
        "retrieved_pairs": [],
        "answer_status": None,
        "answer_text": None,
        "fact_present": None,
        "fact_total": len(q.essential_facts),
        "citation_valid": None,
        "retrieval_error_reason": None,
        "answer_error_reason": None,
    }

    # --- Retrieval step ---------------------------------------------------
    retrieved: list[RetrievedChunk] = []
    try:
        retrieved = retrieval_service.retrieve(q.question)
        row["retrieval_status"] = "evaluated"
        row["hit"] = retrieval_hit(q.source_document, q.source_page, retrieved)
        row["retrieved_pairs"] = [
            [item.chunk.document_name, item.chunk.page_number] for item in retrieved
        ]
    except Exception as exc:  # includes ReembeddingRequiredError
        row["retrieval_status"] = "error"
        row["hit"] = None
        row["retrieved_pairs"] = []
        row["retrieval_error_reason"] = _safe_reason(exc)

    # --- Answer step ------------------------------------------------------
    try:
        resp = query_service.answer(q.question)
        answer_text = resp.answer
        citations = resp.citations
        row["answer_status"] = "evaluated"
        row["answer_text"] = answer_text
        present, total = fact_coverage(answer_text, q.essential_facts)
        row["fact_present"] = present
        row["fact_total"] = total
        # Citation validity needs the retrieved set; only compute it when
        # retrieval was evaluated. Otherwise leave it None (not_evaluated).
        if row["retrieval_status"] == "evaluated":
            row["citation_valid"] = citation_validity(
                answer_text, citations, retrieved, no_answer_sentinel
            )
        else:
            row["citation_valid"] = None
    except Exception as exc:
        row["answer_status"] = "error"
        row["answer_text"] = None
        row["fact_present"] = None  # NOT 0 — not evaluated
        row["citation_valid"] = None  # NOT fabricated
        row["answer_error_reason"] = _safe_reason(exc)

    return row


# ===========================================================================
# OUTPUT WRITERS.
# ===========================================================================


def _fmt_score(score: float | None) -> str:
    """Format a score for the markdown table (``NA`` when ``None``)."""
    return "NA" if score is None else f"{score:.3f}"


def _build_report(
    rows: list[dict[str, Any]],
    run_meta: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    """Build the markdown report text and the machine-readable JSON object.

    Args:
        rows: The per-question result dicts.
        run_meta: Run metadata (timestamp, provider/model names, top_k, counts).

    Returns:
        ``(markdown_text, json_object)``.
    """
    retrieval = aggregate_retrieval(rows)
    coverage, coverage_extra = aggregate_fact_coverage(rows)
    citations = aggregate_citation_validity(rows)

    # --- JSON (machine-readable, enables cross-run comparison) ------------
    json_obj: dict[str, Any] = {
        "run": run_meta,
        "aggregate": {
            "Retrieval_Hit_Rate": retrieval.as_dict(),
            "Answer_Key_Fact_Coverage": {**coverage.as_dict(), **coverage_extra},
            "Citation_Validity": citations.as_dict(),
        },
        "questions": rows,
    }

    # --- Markdown (human-readable) ----------------------------------------
    lines: list[str] = []
    lines.append("# RAG Evaluation Results")
    lines.append("")
    lines.append(
        f"- Run (UTC): {run_meta['timestamp_utc']}  "
        f"| embedding: {run_meta['embedding_provider']}/{run_meta['embedding_model']}  "
        f"| llm: {run_meta['llm_provider']}/{run_meta['llm_model']}  "
        f"| top_k: {run_meta['top_k']}  "
        f"| dataset size: {run_meta['dataset_size']}"
    )
    lines.append("")
    lines.append("## Aggregate")
    lines.append("")
    lines.append("| Metric | Numerator | Denominator | Score | Evaluated | Errors/NA |")
    lines.append("|---|---|---|---|---|---|")
    lines.append(
        f"| Retrieval_Hit_Rate | {retrieval.numerator:g} | {retrieval.denominator:g} "
        f"| {_fmt_score(retrieval.score)} | {retrieval.evaluated} | {retrieval.errors} |"
    )
    lines.append(
        f"| Answer_Key_Fact_Coverage (mean of fractions) | {coverage.numerator:g} "
        f"| {coverage.denominator:g} | {_fmt_score(coverage.score)} "
        f"| {coverage.evaluated} | {coverage.errors} |"
    )
    lines.append(
        f"| Citation_Validity | {citations.numerator:g} | {citations.denominator:g} "
        f"| {_fmt_score(citations.score)} | {citations.evaluated} | {citations.errors} |"
    )
    lines.append("")
    lines.append(
        "> Answer_Key_Fact_Coverage headline is the **mean across questions** of "
        "per-question (facts present / total). Micro-average (raw sums): "
        f"{coverage_extra['micro_average_present_sum']}/"
        f"{coverage_extra['micro_average_total_sum']} = "
        f"{_fmt_score(coverage_extra['micro_average_score'])}."
    )
    lines.append("")
    lines.append(
        "> A retrieval/LLM **service exception** is counted as an error and "
        "excluded from the metric denominator (never a fabricated hit/miss or "
        "0 coverage). A plain retrieval miss (expected doc/page not in top-k, or "
        "document not indexed) is a real miss counted in the denominator."
    )
    lines.append("")
    lines.append("## Per-question")
    lines.append("")
    lines.append(
        "| ID | Source doc | Page | Present? | Retrieval | Hit | Coverage | "
        "Citations | Failure reason |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        # Retrieval + hit cell.
        if r["retrieval_status"] == "evaluated":
            retrieval_cell = "evaluated"
            hit_cell = "yes" if r["hit"] else "no"
        else:
            retrieval_cell = "error"
            hit_cell = "NA"
        # Coverage cell.
        if r["answer_status"] == "evaluated" and r["fact_present"] is not None:
            coverage_cell = f"{r['fact_present']}/{r['fact_total']}"
        else:
            coverage_cell = "error"
        # Citation cell.
        if isinstance(r["citation_valid"], bool):
            citation_cell = "valid" if r["citation_valid"] else "invalid"
        else:
            citation_cell = "not_evaluated"
        present_cell = (
            "unknown" if r["source_present"] is None else ("yes" if r["source_present"] else "no")
        )
        reason = r.get("retrieval_error_reason") or r.get("answer_error_reason") or ""
        reason = reason.replace("|", "\\|")
        lines.append(
            f"| {r['id']} | {r['source_document']} | {r['source_page']} | "
            f"{present_cell} | {retrieval_cell} | {hit_cell} | {coverage_cell} | "
            f"{citation_cell} | {reason} |"
        )
    lines.append("")

    return "\n".join(lines), json_obj


def _write_results(markdown_text: str, json_obj: dict[str, Any]) -> tuple[str, str]:
    """Write timestamped ``.md`` and ``.json`` results into ``eval/results/``.

    Creates the (git-ignored) ``eval/results/`` directory if missing.

    Args:
        markdown_text: The human-readable report.
        json_obj: The machine-readable results object.

    Returns:
        ``(md_path, json_path)`` of the written files.
    """
    os.makedirs(_RESULTS_DIR, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    md_path = os.path.join(_RESULTS_DIR, f"results_{stamp}.md")
    json_path = os.path.join(_RESULTS_DIR, f"results_{stamp}.json")
    with open(md_path, "w", encoding="utf-8") as handle:
        handle.write(markdown_text)
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(json_obj, handle, indent=2, ensure_ascii=False)
    return md_path, json_path


# ===========================================================================
# LIVE ENTRY POINT (NOTHING heavy runs at import time).
# ===========================================================================


def main(argv: list[str] | None = None) -> int:
    """Build the production stack, evaluate the Eval_Set, write results.

    Constructs the stack EXACTLY as the API does (one ``Config``, one
    ``VectorStore``, one embedding provider, one ``Retrieval_Service``, and a
    ``Query_Service`` wired to that same retrieval service and a fresh LLM
    provider). Makes live embedding + Gemini calls — this is why it lives behind
    ``main()`` and is never triggered by an import or the test suite.

    Returns:
        Process exit code: ``0`` on success, non-zero on a fatal setup error
        (empty store, load failure).
    """
    # Local imports so that importing this module never drags in the heavy
    # provider SDKs / vector store (import safety for the test suite).
    from src.config import Config
    from src.embeddings.base import build_embedding_provider
    from src.llm.base import build_llm_provider
    from src.query import NO_ANSWER_RESPONSE, Query_Service
    from src.retrieval import Retrieval_Service
    from src.vectorstore import VectorStore

    # --- Load the Eval_Set first (clear error + non-zero exit on failure). --
    try:
        questions = load_eval_set(_DEFAULT_EVAL_SET)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: could not load Eval_Set: {_safe_reason(exc)}", file=sys.stderr)
        return 1

    if not questions:
        print("ERROR: the Eval_Set is empty; nothing to evaluate.", file=sys.stderr)
        return 1

    # --- Build the production stack (mirrors src/api/app.py wiring). --------
    config = Config.from_env()
    store = VectorStore()
    embedding_provider = build_embedding_provider(config)
    retrieval_service = Retrieval_Service(config, store, embedding_provider)
    query_service = Query_Service(
        config,
        retrieval_service=retrieval_service,
        llm_provider=build_llm_provider(config),
    )

    # --- Empty-store guard (clear message + non-zero exit). -----------------
    try:
        indexed_records = store.list_documents()
    except Exception as exc:
        print(f"ERROR: could not read the vector store: {_safe_reason(exc)}", file=sys.stderr)
        return 1

    if not indexed_records:
        print(
            "ERROR: the vector store is empty — ingest the Eval_Set source "
            "documents before running the evaluation.",
            file=sys.stderr,
        )
        return 1

    indexed_names = {record.document_name for record in indexed_records}

    # --- Evaluate every question. ------------------------------------------
    rows: list[dict[str, Any]] = []
    for q in questions:
        source_present = q.source_document in indexed_names
        rows.append(
            evaluate_question(
                q,
                retrieval_service,
                query_service,
                NO_ANSWER_RESPONSE,
                source_present=source_present,
            )
        )

    # --- Build + write the report. -----------------------------------------
    run_meta = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "embedding_provider": embedding_provider.provider_name,
        "embedding_model": embedding_provider.model_name,
        "llm_provider": config.llm_provider,
        "llm_model": config.llm_model,
        "top_k": config.top_k,
        "dataset_size": len(questions),
    }
    markdown_text, json_obj = _build_report(rows, run_meta)
    md_path, json_path = _write_results(markdown_text, json_obj)

    # --- Concise stdout summary (no secrets). ------------------------------
    retrieval = aggregate_retrieval(rows)
    coverage, _ = aggregate_fact_coverage(rows)
    citations = aggregate_citation_validity(rows)
    print(f"Evaluated {len(questions)} questions.")
    print(
        f"  Retrieval_Hit_Rate      = {_fmt_score(retrieval.score)} "
        f"({retrieval.numerator:g}/{retrieval.denominator:g}, "
        f"errors={retrieval.errors})"
    )
    print(
        f"  Answer_Key_Fact_Coverage = {_fmt_score(coverage.score)} "
        f"(mean over {coverage.evaluated}, errors={coverage.errors})"
    )
    print(
        f"  Citation_Validity        = {_fmt_score(citations.score)} "
        f"({citations.numerator:g}/{citations.denominator:g}, "
        f"not_evaluated={citations.errors})"
    )
    print(f"Wrote: {md_path}")
    print(f"Wrote: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

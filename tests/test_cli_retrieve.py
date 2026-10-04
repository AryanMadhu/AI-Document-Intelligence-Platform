"""Focused, isolated tests for the retrieval CLI (Task 11, decision D6).

These tests never load the real embedding model or touch the real Chroma store.
They inject a fake :class:`Retrieval_Service` into :func:`run` and capture
stdout/stderr with pytest's ``capsys``. The one test that exercises
:func:`main` monkeypatches ``build_service`` so argparse wiring is verified
without constructing a real store or provider.
"""

from __future__ import annotations

import pytest

from src.cli import retrieve as cli
from src.embeddings.base import EmbeddingError
from src.models import Chunk, RetrievedChunk
from src.retrieval import ReembeddingRequiredError


def _make_retrieved_chunk(
    document_name: str,
    page_number: int,
    chunk_text: str,
) -> RetrievedChunk:
    """Build a RetrievedChunk with harmless provenance for tests."""
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


class _FakeService:
    """A stand-in Retrieval_Service that returns a preset result or raises.

    Records whether ``retrieve`` was called so tests can assert the service is
    never queried for an invalid question.
    """

    def __init__(self, result=None, error: Exception | None = None) -> None:
        self._result = result if result is not None else []
        self._error = error
        self.called = False

    def retrieve(self, question: str):
        self.called = True
        if self._error is not None:
            raise self._error
        return self._result


def test_successful_retrieval_prints_blocks(capsys):
    """Two retrieved chunks print as numbered blocks with all required fields."""
    results = [
        _make_retrieved_chunk("report.pdf", 4, "First chunk body text."),
        _make_retrieved_chunk("report.pdf", 7, "Second chunk body text."),
    ]
    fake = _FakeService(result=results)

    code = cli.run("what is the summary", fake)

    out = capsys.readouterr().out
    assert code == 0
    assert fake.called is True
    # Block markers and header format.
    assert "[1]" in out
    assert "[2]" in out
    assert "(page 4)" in out
    assert "(page 7)" in out
    # Required fields: document_name, page_number, chunk_text for each result.
    assert "report.pdf" in out
    assert "First chunk body text." in out
    assert "Second chunk body text." in out


def test_empty_retrieval_informational_message(capsys):
    """An empty result set is an informational stdout message and exit 0."""
    fake = _FakeService(result=[])

    code = cli.run("anything", fake)

    captured = capsys.readouterr()
    assert code == 0
    assert "No results" in captured.out
    assert captured.err == ""


def test_reembedding_required_error(capsys):
    """ReembeddingRequiredError reports affected ids to stderr and exits 1."""
    fake = _FakeService(
        error=ReembeddingRequiredError(["doc-1", "doc-2"], "mismatch details")
    )

    code = cli.run("question", fake)

    captured = capsys.readouterr()
    assert code == 1
    assert "doc-1" in captured.err
    assert "doc-2" in captured.err
    assert "re-embed" in captured.err.lower()
    # No results leaked to stdout.
    assert "[1]" not in captured.out


def test_empty_whitespace_question_rejected(capsys):
    """A whitespace-only question is rejected without querying the service."""
    fake = _FakeService(result=[_make_retrieved_chunk("a.pdf", 1, "x")])

    code = cli.run("   ", fake)

    captured = capsys.readouterr()
    assert code == 2
    assert "a question is required" in captured.err.lower()
    assert fake.called is False


def test_embedding_error_secret_safe(capsys):
    """An EmbeddingError yields a concise, value-free stderr message, exit 1."""
    fake = _FakeService(error=EmbeddingError("provider blew up with secret=abc"))

    code = cli.run("question", fake)

    captured = capsys.readouterr()
    assert code == 1
    assert "embedding failed" in captured.err.lower()
    # The underlying exception detail (and any secret) is not echoed.
    assert "secret=abc" not in captured.err


def test_format_results_pure_function():
    """format_results renders required fields and block markers."""
    results = [
        _make_retrieved_chunk("doc.pdf", 2, "Alpha text."),
        _make_retrieved_chunk("doc.pdf", 5, "Beta text."),
    ]

    out = cli.format_results(results)

    assert "[1] doc.pdf (page 2)" in out
    assert "[2] doc.pdf (page 5)" in out
    assert "Alpha text." in out
    assert "Beta text." in out
    # Blocks separated by a blank line.
    assert "\n\n" in out


def test_format_results_empty_is_empty_string():
    """No results renders as an empty string."""
    assert cli.format_results([]) == ""


def test_main_wiring_uses_build_service(capsys, monkeypatch):
    """main() parses the positional question and runs without a real store."""
    results = [_make_retrieved_chunk("wired.pdf", 3, "Wired body.")]
    fake = _FakeService(result=results)

    # Replace build_service so no real VectorStore/provider is constructed.
    monkeypatch.setattr(cli, "build_service", lambda: fake)

    code = cli.main(["my question"])

    out = capsys.readouterr().out
    assert code == 0
    assert fake.called is True
    assert "wired.pdf" in out
    assert "Wired body." in out

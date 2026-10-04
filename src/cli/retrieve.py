"""Retrieval CLI: query the document store from the command line (R7.1, R7.2).

Run as a module::

    python -m src.cli.retrieve "your question"

The CLI reuses :class:`~src.retrieval.Retrieval_Service` as the *sole* retrieval
abstraction — it never duplicates embedding or vector-store query logic. For
each retrieved chunk it prints a readable plain-text block carrying the chunk's
``document_name``, ``page_number``, and ``chunk_text`` (no LLM, no answer
generation; that is a later task).

Design for testability:
    * :func:`format_results` is a pure function (no I/O) that turns a list of
      :class:`~src.models.RetrievedChunk` into the plain-text block output.
    * :func:`run` is the testable core: it takes an *already-constructed*
      :class:`~src.retrieval.Retrieval_Service` (dependency injection) so tests
      can pass a fake, and it returns a process exit code.
    * :func:`build_service` performs the real wiring and is called only inside
      :func:`main` — never at import time — so importing this module (or running
      ``--help``) never constructs a :class:`VectorStore` or an embedding
      provider.

Error handling and exit codes (D4):
    * Empty/whitespace question -> error to stderr, exit ``2``.
    * Empty store / no retrieved chunks -> informational message to stdout,
      exit ``0``.
    * :class:`~src.retrieval.ReembeddingRequiredError` -> message naming the
      affected document ids to stderr, exit ``1``.
    * :class:`~src.embeddings.base.EmbeddingError` -> concise, secret-safe
      message to stderr, exit ``1``.

No broad ``except Exception`` is used: unexpected (programming) errors propagate
and produce a normal traceback. No credential value is ever printed.
"""

from __future__ import annotations

import argparse
import sys

from src.config import Config
from src.embeddings.base import EmbeddingError
from src.models import RetrievedChunk
from src.retrieval import ReembeddingRequiredError, Retrieval_Service
from src.vectorstore import VectorStore


def format_results(results: list[RetrievedChunk]) -> str:
    """Render retrieved chunks as readable plain-text blocks (pure, no I/O).

    Each result becomes a header line of the form
    ``[<n>] <document_name> (page <page_number>)`` followed by the chunk's text
    on the next line(s). Blocks are separated by a single blank line. Only the
    ``document_name``, ``page_number``, and ``chunk_text`` are shown — no
    similarity score or other fields (D1).

    Args:
        results: The retrieved chunks to render, in rank order.

    Returns:
        The assembled plain-text output (no trailing newline). An empty string
        when ``results`` is empty.
    """
    blocks: list[str] = []
    for index, result in enumerate(results, start=1):
        chunk = result.chunk
        header = f"[{index}] {chunk.document_name} (page {chunk.page_number})"
        blocks.append(f"{header}\n{chunk.chunk_text}")
    # A blank line between blocks: join with a double newline.
    return "\n\n".join(blocks)


def run(question: str, service: Retrieval_Service) -> int:
    """Execute a single retrieval and print the result; return an exit code.

    This is the testable core. It takes an already-constructed
    :class:`~src.retrieval.Retrieval_Service` so tests can inject a fake and no
    heavy dependency is loaded.

    Args:
        question: The user's question. ``None`` or whitespace-only is rejected.
        service: The retrieval service to query (dependency-injected).

    Returns:
        A process exit code:
            * ``0`` — results printed, or the store had no matching chunks.
            * ``1`` — re-embedding required, or an embedding failure occurred.
            * ``2`` — the question was empty/whitespace.
    """
    # Guard empty/whitespace question (argparse usually requires the positional,
    # but a whitespace-only value still reaches here). Report and bail early;
    # the service is never queried.
    if question is None or question.strip() == "":
        print("Error: a question is required.", file=sys.stderr)
        return 2

    try:
        results = service.retrieve(question)
    except ReembeddingRequiredError as err:
        affected = ", ".join(err.document_ids)
        print(
            "Error: re-embedding is required before querying. The following "
            f"document(s) were indexed with a different embedding "
            f"provider/model than the active one: {affected}. Re-embed them "
            "with the active provider/model, then try again.",
            file=sys.stderr,
        )
        return 1
    except EmbeddingError:
        # Concise, secret-safe: never echo the exception detail, which could in
        # principle carry provider internals. The message is value-free.
        print(
            "Error: embedding failed; the embedding provider could not process "
            "the question. Check the provider configuration and try again.",
            file=sys.stderr,
        )
        return 1

    if results == []:
        print(
            "No results found. (The document store may be empty or contain no "
            "matching chunks.)"
        )
        return 0

    print(format_results(results))
    return 0


def build_service() -> Retrieval_Service:
    """Construct the real retrieval wiring (called only from :func:`main`).

    Builds a :class:`~src.config.Config` from the environment, opens the
    :class:`~src.vectorstore.VectorStore` at its existing default persistence
    path (``chroma_db/`` — the same store ingestion writes to; D3), and returns
    a :class:`~src.retrieval.Retrieval_Service` wired to both. The service
    builds the active embedding provider from the config lazily, only when
    :meth:`Retrieval_Service.retrieve` runs — so merely constructing the service
    here does not load the heavy embedding backend.

    Returns:
        A fully-wired :class:`~src.retrieval.Retrieval_Service`.
    """
    config = Config.from_env()
    store = VectorStore()  # Existing default persist path: chroma_db/ (D3).
    return Retrieval_Service(config, store)


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and run the retrieval CLI.

    Defines a single positional ``question`` argument and the standard
    ``--help``. The real service is built via :func:`build_service` *after*
    argument parsing, so running ``--help`` (which makes argparse exit before
    this point) never constructs a vector store or embedding provider.

    Args:
        argv: Argument vector to parse; defaults to ``sys.argv[1:]`` when
            ``None``.

    Returns:
        The process exit code produced by :func:`run`.
    """
    parser = argparse.ArgumentParser(
        prog="python -m src.cli.retrieve",
        description=(
            "Retrieve the most relevant document chunks for a question and "
            "print each chunk's document name, page number, and text."
        ),
    )
    parser.add_argument(
        "question",
        help="The natural-language question to search the document store with.",
    )
    args = parser.parse_args(argv)

    return run(args.question, build_service())


if __name__ == "__main__":
    sys.exit(main())

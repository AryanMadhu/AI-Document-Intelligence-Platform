# Implementation Plan: AI Document Intelligence Platform

## Overview

This plan converts the approved design into small, ordered, independently testable coding tasks in **Python**. Work proceeds bottom-up: scaffolding → config → data models → provider interfaces/factories → concrete providers → vector store → ingestion (parse/chunk/hash/limits) → retrieval + CLI → query/citations → re-embedding → API → UI → eval corpus/script → integration test → containerization/deployment → documentation.

Every task is labeled **MVP** or **Stretch**, consistent with the design's "MVP vs Stretch Scope" subsection. Each task references the requirement IDs and/or design elements it implements. Tests are woven into the task that introduces the logic (property tests via Hypothesis at ≥100 iterations plus example/edge-case unit tests), following the design's Testing Strategy. Correctness properties P1–P11 are cited in the relevant test-bearing tasks.

Property-test tag format (from design): `# Feature: ai-document-intelligence-platform, Property {number}: {property_text}`

Sub-tasks marked with `*` are optional (test/eval-only) and can be skipped for a faster MVP; core implementation sub-tasks are never optional.

## Tasks

- [x] 1. MVP — Scaffold the project structure and reproducible environment
  - [x] 1.1 MVP — Create the repository layout and dependency/ignore files: create `src/` (with `ingestion/`, `embeddings/`, `vectorstore/`, `retrieval/`, `llm/`, `query/`, `api/`, `cli/` packages + `__init__.py`), `ui/`, `tests/`, `data/`, `eval/`, `notebooks/`; add a pinned `requirements.txt` (fastapi, uvicorn, streamlit, pymupdf, chromadb, sentence-transformers, `google-generativeai` (pinned, for the Gemini MVP provider), openai, pydantic, python-dotenv, pytest, hypothesis — all version-pinned); add `.gitignore` excluding `.env`, virtualenvs, `data/` local artifacts, and Chroma persistence; add `.env.example` listing required variable names with no values, including `GEMINI_API_KEY=your_key_here` (placeholder only) alongside the other env var names. Never commit a real key — `.gitignore` already excludes `.env`.
    - References R1.1, R1.2, R1.4, R1.5, R17.3, R15.2, R16.4; Design: Repository layout, Configuration approach, Secret management

- [ ] 2. MVP — Configuration module
  - [ ] 2.1 MVP — Implement `src/config.py`: typed `Config` loading `EMBEDDING_PROVIDER`, `EMBEDDING_MODEL`, `LLM_PROVIDER`, `LLM_MODEL`, `TOP_K` (5), `CHUNK_SIZE_TOKENS` (700), `CHUNK_OVERLAP_TOKENS` (90), `MAX_PDF_SIZE_MB` (5), `MAX_PDF_PAGES` (50), and per-provider credential keys (`GEMINI_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`) from environment/`.env` with the design's default values; provide a helper that reports a missing required credential **by name only** (never its value).
    - References R5.4, R6.3, R8.5, R11.4, R16.4, R17.3; Design: Config, Finalized configuration defaults, Secret management
  - [ ]* 2.2 MVP — Write unit tests for `Config`: defaults resolve to the documented values; overrides from env are applied; missing-credential lookup raises a named, value-free error.
    - References R16.4; Design: Testing Strategy (provider factories / missing credential)

- [ ] 3. MVP — Core data models
  - [ ] 3.1 MVP — Implement `src/models.py`: `Chunk` (dataclass), `DocumentRecord` (dataclass w/ a stable `document_id` (UUID string) that is distinct from `document_hash`, plus `document_hash`, provenance `embedding_provider`/`embedding_model`, counts, `ingested_at`), `RetrievedChunk` (dataclass), `Citation` (Pydantic), API models `QueryRequest`, `QueryResponse`, `UploadResponse` (carries `document_id`), `DocumentListResponse`, `ReembedResponse` (echoes `document_id`), `ErrorResponse` (Pydantic), and `EvalQuestion` (dataclass). Include type hints and docstrings. Keep the distinction explicit: `document_id` = stable UUID used to ADDRESS a document; `document_hash` = SHA-256 used ONLY for duplicate detection.
    - References R4.3, R5.2, R5.5, R9.2, R10, R11, R12, R15.7, R2.3, R17.1, R17.2; Design: Data Models (DocumentRecord — document_id vs document_hash)

- [ ] 4. MVP — Embedding provider interface, factory, and default implementation
  - [ ] 4.1 MVP — Implement `src/embeddings/base.py`: `EmbeddingProvider` ABC (`provider_name`, `model_name`, `dimension`, `embed_texts`, `embed_query`) plus an `EmbeddingError`, and the `build_embedding_provider(config)` factory that maps the config string to a concrete class and raises a named, value-free error on missing credentials.
    - References R5.1, R5.4, R15.1, R15.3, R16.4; Design: EmbeddingProvider interface, Provider selection
  - [ ] 4.2 MVP — Implement `src/embeddings/sentence_transformers_provider.py`: `SentenceTransformersEmbeddingProvider` (`all-MiniLM-L6-v2`, 384-dim) satisfying the ABC and exposing provider/model identity for provenance.
    - References R5.1, R15.1; Design: EmbeddingProvider (SentenceTransformers impl, MVP default)
  - [ ]* 4.3 MVP — Write unit tests for the embedding factory: each config value maps to the expected concrete provider; missing credential raises a named, value-free error.
    - References R5.4, R15.3, R16.4; Design: Testing Strategy (provider factories)
  - [ ]* 4.4 Stretch — Implement `src/embeddings/openai_provider.py`: `OpenAIEmbeddingProvider` (`text-embedding-3-small`, 1536-dim) behind the same interface; register it in the factory.
    - References R5.1, R15.1, R15.3; Design: MVP vs Stretch Scope (OpenAIEmbeddingProvider — Stretch)

- [ ] 5. MVP — LLM provider interface, factory, and default implementation
  - [ ] 5.1 MVP — Implement `src/llm/base.py`: `LLMProvider` ABC (`provider_name`, `model_name`, `generate`) plus an `LLMError`, and the `build_llm_provider(config)` factory that maps `gemini` → `GeminiLLMProvider` (the MVP default), and `openai` → `OpenAILLMProvider`, `anthropic` → `AnthropicLLMProvider`, `ollama` → `OllamaLLMProvider` (their Stretch impls) behind the same interface; a missing `GEMINI_API_KEY` (when `gemini` is selected) raises a named, value-free error.
    - References R8.5, R15.2, R15.4, R16.4; Design: LLMProvider interface, Provider selection
  - [ ] 5.2 MVP — Implement `src/llm/gemini_provider.py`: `GeminiLLMProvider` (Google Gemini, model `gemini-flash-latest`) satisfying the ABC; read `GEMINI_API_KEY` from Config/env only (never hard-coded); raise `LLMError` on provider failure; no domain-specific logic. Depends on the pinned `google-generativeai` client.
    - References R8.5, R15.2, R15.5; Design: LLMProvider (Gemini impl, MVP default)
  - [ ]* 5.3 MVP — Write unit tests for the LLM factory: each config value maps to the expected concrete provider (assert `gemini` maps to `GeminiLLMProvider`); a missing `GEMINI_API_KEY` raises a named, value-free error.
    - References R8.5, R15.4, R16.4; Design: Testing Strategy (provider factories)
  - [ ]* 5.4 Stretch — Implement `src/llm/openai_provider.py`: `OpenAILLMProvider` (`gpt-4o-mini`) satisfying the ABC; raise `LLMError` on provider failure. No domain-specific logic. (Stretch/Future; interface stays MVP.)
    - References R8.5, R15.2, R15.5; Design: MVP vs Stretch Scope (OpenAILLMProvider — Stretch)
  - [ ]* 5.5 Stretch — Implement `src/llm/anthropic_provider.py` (`AnthropicLLMProvider`) behind the same interface and register it in the factory.
    - References R15.2, R15.4; Design: MVP vs Stretch Scope (AnthropicLLMProvider — Stretch)
  - [ ]* 5.6 Stretch — Implement `src/llm/ollama_provider.py` (`OllamaLLMProvider`, local dev only) behind the same interface and register it in the factory.
    - References R15.2, R15.4; Design: MVP vs Stretch Scope (OllamaLLMProvider — Stretch)

- [ ] 6. MVP — Vector store wrapper (Chroma)
  - [ ] 6.1 MVP — Implement `src/vectorstore/chroma_store.py`: `VectorStore` wrapping a persistent Chroma client — `add_chunks(document_id, chunks, embeddings)` (persists chunks + vectors under the owning `document_id`), `query(query_embedding, top_k)` returning `RetrievedChunk` ranked by similarity, `list_documents`, `document_exists(document_hash)` (dedupe check — stays keyed on the SHA-256 `document_hash`), `count`; persist chunk metadata `{document_name, page_number, section, chunk_text}` plus provenance `{embedding_provider, embedding_model}`, the owning `document_id`, and `document_hash`, so chunks are addressable by `document_id`. Keep Chroma types from leaking out.
    - References R5.2, R6.2, R6.4, R6.5, R11.6, R12.1, R12.2; Design: Vector_Store wrapper (keying convention — addressing by document_id, dedupe by document_hash)
  - [ ]* 6.2 MVP — Write property test for storage round-trip.
    - **Property 4: Storage and retrieval preserve chunk metadata (round trip)**
    - **Validates: Requirements 5.2, 6.4**
    - Hypothesis ≥100 iterations, stub embeddings; assert stored `{document_name, page_number, section, chunk_text}` is returned unchanged and each stored chunk remains addressable by its owning `document_id`.

- [ ] 7. MVP — PDF parsing with page preservation
  - [ ] 7.1 MVP — Implement `src/ingestion/pdf_parser.py`: `PDF_Parser` (PyMuPDF) returning per-page `PageText{page_number, text}`; record empty-text pages and continue; raise `PDFParseError(file, reason)` on non-PDF/corrupt input.
    - References R3.1, R3.2, R3.3, R3.4; Design: PDF_Parser
  - [ ]* 7.2 MVP — Write unit tests for the parser: empty-text pages are recorded and processing continues (R3.4); a corrupt file raises `PDFParseError` with file + reason (R3.3).
    - References R3.3, R3.4; Design: Testing Strategy (parser edge cases)

- [ ] 8. MVP — Chunking module
  - [ ] 8.1 MVP — Implement `src/ingestion/chunking.py`: `Chunking_Module` token-based splitter (`CHUNK_SIZE_TOKENS=700`, `CHUNK_OVERLAP_TOKENS=90`) that preserves page boundaries (each chunk maps to one page) and attaches `{document_name, page_number, section, chunk_text}`, with `section=None` when unavailable.
    - References R4.1, R4.2, R4.3, R4.4; Design: Chunking_Module, page-boundary-preserving tradeoff
  - [ ]* 8.2 MVP — Write property test for chunk size bounds.
    - **Property 1: Chunk size stays within configured bounds**
    - **Validates: Requirements 4.1**
    - Hypothesis ≥100 iterations over arbitrary text; every non-final chunk within 500–800 tokens.
  - [ ]* 8.3 MVP — Write property test for consecutive-chunk overlap.
    - **Property 2: Consecutive chunks overlap within the configured range**
    - **Validates: Requirements 4.2**
    - Hypothesis ≥100 iterations over text yielding ≥2 chunks; overlap within 10–15% of chunk size.
  - [ ]* 8.4 MVP — Write property test for chunk metadata completeness.
    - **Property 3: Every chunk carries complete metadata**
    - **Validates: Requirements 4.3, 4.4**
    - Hypothesis ≥100 iterations with/without section info; assert non-empty `document_name`, in-range `page_number`, matching `chunk_text`, valid-or-null `section`.
  - [ ]* 8.5 MVP — Write chunking edge-case unit tests: empty text, single short chunk, exact-boundary lengths, section-less input (`section is None`).
    - References R4.4; Design: Testing Strategy (chunking edge cases)

- [ ] 9. MVP — Ingestion service (limits, hashing, dedupe, provenance, orchestration)
  - [ ] 9.1 MVP — Implement PDF validation and SHA-256 document hashing in `src/ingestion/service.py`: enforce `MAX_PDF_SIZE_MB` (5MB) before parse, enforce `MAX_PDF_PAGES` (50) after parse, reject invalid PDFs (surface `PDFParseError`), compute the SHA-256 `Document_Hash` of file bytes; add unit tests.
    - References R11.3, R11.4, R11.5, R11.6, R16.2; Design: Ingestion request flow (steps 1–4), Error Handling table
  - [ ] 9.2 MVP — Implement the ingestion orchestration in `src/ingestion/service.py`: dedupe via `Document_Hash` (on match return "already ingested" referencing the existing document's `document_id`, write nothing and generate no new id), otherwise generate a fresh UUID `document_id` for the new non-duplicate document, then parse → chunk → embed via the active `EmbeddingProvider` → store in `VectorStore` under that `document_id`; on embedding failure report and identify the document while preserving prior data; record provenance (`embedding_provider`, `embedding_model`, `document_hash`) together with the `document_id` on the `DocumentRecord` and every chunk; return an `UploadResponse` carrying the `document_id`.
    - References R5.1, R5.2, R5.3, R5.5, R11.1, R11.2, R11.7, R11.8, R16.3; Design: Ingestion request flow (steps 2–8, document_id assignment)
  - [ ]* 9.3 MVP — Write property test for embedding provenance recording.
    - **Property 6: Embedding provenance is recorded for every indexed unit**
    - **Validates: Requirements 5.5**
    - Hypothesis ≥100 iterations under a known provider/model; every stored chunk and the document record carry that provider/model.
  - [ ]* 9.4 MVP — Write property test for idempotent (duplicate) ingestion.
    - **Property 7: Ingestion is idempotent under identical content (duplicate detection)**
    - **Validates: Requirements 11.6, 11.7, 11.8**
    - Hypothesis ≥100 iterations ingesting identical bytes twice; store state (chunk count, document set) unchanged and second upload reported as duplicate.
  - [ ]* 9.5 MVP — Write property test for document listing reflecting the ingested set.
    - **Property 8: Document listing reflects exactly the ingested set**
    - **Validates: Requirements 12.1, 12.2**
    - Hypothesis ≥100 iterations over sets of distinct documents; listing returns exactly the ingested set (empty when none).
  - [ ]* 9.6 MVP — Write ingestion boundary unit tests: files just over `MAX_PDF_SIZE_MB` / `MAX_PDF_PAGES` rejected with nothing stored.
    - References R11.5; Design: Testing Strategy (ingestion boundaries)

- [ ] 10. MVP — Retrieval service with embedding-space compatibility gating
  - [ ] 10.1 MVP — Implement `src/retrieval/service.py`: `Retrieval_Service` embeds the question via the active `EmbeddingProvider`, runs the compatibility check (block documents whose recorded provider/model differ from active, raising `ReembeddingRequiredError` listing affected documents), then returns top-k (`min(TOP_K, store_size)`) `RetrievedChunk`s ranked by similarity with full metadata; empty store → empty list; support `TOP_K` config default and per-query override.
    - References R6.1, R6.2, R6.3, R6.4, R6.5, R15.6, R15.7; Design: Retrieval_Service
  - [ ]* 10.2 MVP — Write property test for top-k ranking.
    - **Property 5: Retrieval returns Top_K results ranked by similarity**
    - **Validates: Requirements 6.2, 6.3**
    - Hypothesis ≥100 iterations, stub embeddings, varied k; exactly `min(Top_K, store_size)` chunks in non-increasing score order.
  - [ ]* 10.3 MVP — Write property test for embedding-space compatibility gating.
    - **Property 11: Retrieval is gated by embedding-space compatibility**
    - **Validates: Requirements 15.6, 15.7**
    - Hypothesis ≥100 iterations over documents with assorted provenance vs active provider/model; retrieval permitted for matches, blocked (re-embedding required) for mismatches.
  - [ ]* 10.4 MVP — Write retrieval edge-case unit tests: empty store returns `[]` (R6.5); default k=5 and config override (R6.3).
    - References R6.3, R6.5; Design: Testing Strategy (retrieval edge cases)

- [ ] 11. MVP — Retrieval CLI
  - [ ] 11.1 MVP — Implement `src/cli/retrieve.py` (`python -m src.cli.retrieve "question"`): reuse `Retrieval_Service`, print per retrieved chunk the `document_name`, `page_number`, and `chunk_text` (no LLM).
    - References R7.1, R7.2; Design: Retrieval CLI

- [ ] 12. MVP — Query service, grounded prompt, and citation formatter
  - [ ] 12.1 MVP — Implement the grounded prompt builder in `src/query/service.py`: construct the design's grounded template embedding every retrieved chunk (tagged with document + page) and the instruction to answer only from context and return the `No_Answer_Response` otherwise.
    - References R8.1, R8.2; Design: Grounded prompt template
  - [ ]* 12.2 MVP — Write property test for grounded-prompt content.
    - **Property 9: The grounded prompt contains all context and grounding instructions**
    - **Validates: Requirements 8.1, 8.2**
    - Hypothesis ≥100 iterations over retrieved chunk sets; prompt contains every chunk's text and both grounding instructions.
  - [ ] 12.3 MVP — Implement `src/query/formatter.py`: `Response_Formatter` builds `{answer, citations[]}` where each `Citation{document, page, excerpt}` is derived only from an in-context retrieved chunk; empty citations for `No_Answer_Response`.
    - References R9.1, R9.2, R9.3, R9.4; Design: Response_Formatter
  - [ ] 12.4 MVP — Implement `Query_Service.answer(question)` orchestration in `src/query/service.py`: reject empty/whitespace question without invoking the LLM; retrieve top-k; empty chunks → `No_Answer_Response` with empty citations; build prompt; call `LLMProvider.generate` (on failure return "answer generation failed", preserve prior data); detect the no-answer sentinel → empty citations; use `Response_Formatter` (task 12.3) to assemble the final `{answer, citations}`.
    - References R8.3, R8.4, R10.4, R16.1, R16.3; Design: Query_Service flow, Error Handling table
  - [ ]* 12.5 MVP — Write property test for citation validity.
    - **Property 10: Every citation is derived from an in-context chunk**
    - **Validates: Requirements 9.1, 9.2, 9.3, 9.4, 10.2**
    - Hypothesis ≥100 iterations over retrieved chunk sets + grounded answers; every citation traces to an in-context chunk; empty on `No_Answer_Response`.
  - [ ]* 12.6 MVP — Write query edge-case unit tests: empty/whitespace question rejected and LLM stub never called (R16.1); `No_Answer_Response` yields empty citations (R8.4, R9.3).
    - References R8.4, R9.3, R16.1; Design: Testing Strategy (query edge cases)

- [ ] 13. MVP — Re-embedding flow
  - [ ] 13.1 MVP — Extend `VectorStore` with `get_document_chunks(document_id)` and scoped, all-or-nothing `replace_document_embeddings(document_id, chunks, embeddings, embedding_provider, embedding_model)` returning the updated `DocumentRecord`; both operations key off the stable `document_id` (never `document_hash`) so they address exactly the intended document; guarantee a document is never left partially re-embedded.
    - References R5.5, R15.7; Design: Re-embedding flow (VectorStore ops keyed on document_id)
  - [ ] 13.2 MVP — Implement the re-embedding flow in `src/ingestion/service.py`: load stored chunks by `document_id` (raise `DocumentNotFoundError` for an unknown `document_id`), re-embed `chunk_text` with the active provider, replace vectors within that `document_id`'s scope, update provenance on the record and every chunk; on embedding failure leave the document on its prior embedding space.
    - References R5.3, R5.5, R15.7, R16.3; Design: Re-embedding flow (steps 1–5, load/replace by document_id)
  - [ ]* 13.3 MVP — Write re-embedding example/edge-case unit tests: after switching active provider/model and re-embedding, provenance equals active provider/model, chunk texts unchanged, and the compatibility check re-admits the document (R15.7); unknown `document_id` raises `DocumentNotFoundError` with store unchanged; mid-re-embed embedding failure leaves the document on its prior space.
    - References R5.3, R15.7, R16.3; Design: Testing Strategy (re-embedding flow)

- [ ] 14. MVP — FastAPI backend and error-to-HTTP mapping
  - [x] 14.1 MVP — Implement `src/api/app.py` endpoints wired to the services: `POST /documents/upload` (Ingestion_Service; success/duplicate/invalid/oversized/too-many-pages; `UploadResponse` includes the `document_id`), `GET /documents` (list; empty list when none), `POST /query` (validate non-empty question → `{answer, citations}`), `POST /documents/{document_id}/reembed` where the `{document_id}` path param is the stable `document_id` → `ReembedResponse` echoing that `document_id`.
    - References R10.1, R10.2, R10.3, R11.1, R11.2, R11.3, R12.1, R12.2, R15.7; Design: API_Backend, reembed contract (document_id path param)
  - [x] 14.2 MVP — Implement the full error-to-HTTP mapping from the Error Handling table: `EMPTY_QUESTION` 400, `INVALID_PDF` 400, `FILE_TOO_LARGE` 413, `TOO_MANY_PAGES` 413, duplicate 200, `EMBEDDING_FAILED` 502, `LLM_FAILED` 502, `MISSING_CREDENTIAL` 500, `REEMBEDDING_REQUIRED` 409, `DOCUMENT_NOT_FOUND` 404 — all returning credential-safe `ErrorResponse` bodies.
    - References R10.3, R10.4, R11.3, R11.5, R15.6, R15.7, R16.2, R16.3, R16.4; Design: Error Handling table
  - [x]* 14.3 MVP — Write the end-to-end integration test.
    - Ingest a small fixture PDF with stub providers → `POST /query` → assert `{answer, citations}` where every citation traces to an ingested chunk.
    - References R18.3; Design: Testing Strategy (integration test)
    - **Task 14 implementation notes:**
      - **Dependency added**: `python-multipart==0.0.32` (required by FastAPI/Starlette to parse the `multipart/form-data` body of `POST /documents/upload`).
      - **D1** — a successful (or duplicate) upload returns HTTP 200.
      - **D2** — `TOO_MANY_PAGES` → 413; `EMPTY_DOCUMENT` → 422.
      - **D3** — `QueryRequest.top_k` is accepted but intentionally unused; `/query` calls `service.answer(question)` with the question only.
      - **D4** — a custom `RequestValidationError` handler returns the project's `ErrorResponse` envelope (422, `VALIDATION_ERROR`) with a fixed `"The request was invalid."` detail; the raw pydantic error tree / request input is never serialized into the body.
      - **D5** — a *missing* `question` field is a 422 validation error; an *empty/whitespace* question value is owned by `Query_Service` → 400 `EMPTY_QUESTION`.
      - **D7** — lazy, cached DI via `functools.lru_cache` `_get_*` builders; importing `src.api.app` and calling `create_app()` constructs nothing heavy (no provider/store/model/network/Chroma dir). Verified by two import-safety tests (subprocess + in-process) and an autouse test guard that fails any test in which a real `_get_*` builder runs.
      - **D10** — no `/retrieve` HTTP endpoint.
      - **Upload size guard** — a *before-read* size guard runs in a module-level `_handle_upload` helper: when `UploadFile.size` is known it rejects over-limit uploads without reading the body; when `size` is `None` (missing/unreliable `Content-Length`) it does a **bounded** read of at most `max_bytes + 1` bytes and rejects if over — never pulling an unbounded body into memory. Task 9's `len(file_bytes)` check remains the authoritative backstop.
      - **MISSING_CREDENTIAL** — the handler is registered and returns a name-only (value-free) 500, but is **currently unreachable in the MVP**: a missing `GEMINI_API_KEY` is caught in the Gemini provider and re-raised as `LLMError` → 502 `LLM_FAILED`, and the MVP sentence-transformers embedding provider needs no credential. The handler is retained for a future credential-propagating provider.

- [ ] 15. MVP — Streamlit UI
  - [ ] 15.1 MVP — Implement `ui/app.py`: PDF upload control, chat-style question input, display answer text with citations as expandable cards showing `document`/`page`/`excerpt`, and an error message on answer-generation failure. Stateless per query (no conversation memory).
    - References R13.1, R13.2, R13.3, R13.4, R13.5; Design: UI component

- [ ] 16. MVP — Evaluation corpus and Eval_Set
  - [ ] 16.1 MVP — Add 3–5 real domain PDFs to `data/` and author `eval/eval_set.json` with 8–10 questions, each recording `essential_facts`, `source_document`, and `source_page`.
    - References R2.1, R2.2, R2.3, R2.4; Design: Data Models (EvalQuestion), Project Structure

- [ ] 17. MVP — Evaluation script (three metrics)
  - [ ] 17.1 MVP — Implement `eval/run_eval.py`: run every Eval_Set question through the query pipeline and compute Retrieval_Hit_Rate (source chunk within top-k), Answer_Key_Fact_Coverage (deterministic normalized fact matching), and Citation_Validity (every citation traces to an in-context chunk); write aggregate + per-question results to `eval/results.md`.
    - References R14.1, R14.2, R14.3, R14.4, R14.5; Design: Evaluation Design, Results file
  - [ ]* 17.2 MVP — Write evaluation-metric unit tests: each metric equals a hand-computed value on fixed inputs and is bounded in [0,1].
    - References R14.2, R14.3, R14.4; Design: Testing Strategy (evaluation metrics)
  - [ ]* 17.3 Stretch — Add an LLM-as-judge scorer option for Answer_Key_Fact_Coverage to catch semantic paraphrases.
    - References R14.3; Design: Evaluation Design (Metric 2 limitation — Stretch)

- [ ] 18. MVP — Checkpoint
  - [ ] 18.1 MVP — Ensure all tests pass, ask the user if questions arise.

- [ ] 19. MVP — Containerization
  - [ ] 19.1 MVP — Write a `Dockerfile` (`python:3.11-slim`, install pinned `requirements.txt`, run FastAPI + Streamlit) that builds a runnable image.
    - References R19.1; Design: Dockerfile & deployment

- [ ] 20. MVP — Deployment to a live URL
  - [ ] 20.1 MVP — Deploy the container to HuggingFace Spaces (Docker) exposing a live public URL; verify the deployed UI loads and answers a query. Document free-tier fallbacks (switch embedding/LLM provider via Config) applied if needed.
    - References R19.2, R19.3; Design: Dockerfile & deployment, free-tier conflicts

- [ ] 21. MVP — Documentation and decision records
  - [ ] 21.1 MVP — Write `README.md`: problem statement, architecture diagram, setup/run instructions, live demo link, screenshots/GIF, and known limitations; include the "run natively for local dev / measure memory before assuming fit" note and the OpenAI-embedding fallback. Note the MVP LLM is Google Gemini free tier (`gemini-flash-latest`) selected for the ₹0 budget, with the key loaded only via `GEMINI_API_KEY` env/`.env` and never committed; OpenAI/Anthropic/Ollama are recorded as future options behind the `LLMProvider` interface.
    - References R20.1, R19.3, R15.2, R16.4; Design: Documentation deliverables, Local development environment note
  - [ ] 21.2 MVP — Write `CASE_STUDY.md`: design decisions and rationale, approaches that did not work initially, and changes at larger scale.
    - References R20.2; Design: Documentation deliverables
  - [ ] 21.3 MVP — Write `DECISIONS.md`: seed with the five open-technology comparisons (vector store, embedding, LLM, UI, hosting), the finalized config defaults, the page-boundary chunking tradeoff, the MVP-interfaces/Stretch-implementations rationale, and the local-dev-native/measure-memory note — each with decision, alternatives, and rationale. For the LLM decision, record that the MVP provider is Google Gemini free tier (`gemini-flash-latest`) chosen for the ₹0 budget, with the key loaded only via `GEMINI_API_KEY` env/`.env` and never committed; OpenAI (`gpt-4o-mini`), Anthropic, and Ollama are recorded as future options retained behind the `LLMProvider` interface.
    - References R20.3, R20.4, R15.2, R16.4; Design: Open technology decisions, Documentation deliverables, chunking tradeoff, MVP vs Stretch Scope

- [ ] 22. MVP — Final checkpoint
  - [ ] 22.1 MVP — Ensure all tests pass and the deployed demo works, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional (tests/eval extras) and can be skipped for a faster MVP; core implementation sub-tasks are never optional.
- Every task references specific requirements and design elements for traceability, and is labeled MVP or Stretch consistent with the design's "MVP vs Stretch Scope".
- Property tests use Hypothesis at ≥100 iterations and carry the design's tag format; example/edge-case unit tests and one integration test cover boundaries and wiring.
- Checkpoints (tasks 18, 22) provide incremental validation points.
- The MVP LLM provider is Google Gemini free tier (`GeminiLLMProvider`, `gemini-flash-latest`); the OpenAI/Anthropic/Ollama providers are Stretch and plug into the same `LLMProvider` interface later with no architectural change.
- Stretch tasks (4.4, 5.4, 5.5, 5.6, 17.3) plug into the MVP provider seams and evaluation with no architectural change.

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1"] },
    { "id": 1, "tasks": ["2.1", "3.1"] },
    { "id": 2, "tasks": ["2.2", "4.1", "5.1", "7.1"] },
    { "id": 3, "tasks": ["4.2", "4.3", "4.4", "5.2", "5.4", "5.5", "5.6", "7.2", "8.1", "6.1"] },
    { "id": 4, "tasks": ["5.3", "8.2", "8.3", "8.4", "8.5", "6.2", "9.1"] },
    { "id": 5, "tasks": ["9.2"] },
    { "id": 6, "tasks": ["9.3", "9.4", "9.5", "9.6", "10.1"] },
    { "id": 7, "tasks": ["10.2", "10.3", "10.4", "11.1", "12.1", "13.1"] },
    { "id": 8, "tasks": ["12.2", "12.3", "13.2"] },
    { "id": 9, "tasks": ["12.4", "13.3"] },
    { "id": 10, "tasks": ["12.5", "12.6", "14.1"] },
    { "id": 11, "tasks": ["14.2", "15.1", "16.1"] },
    { "id": 12, "tasks": ["14.3", "17.1"] },
    { "id": 13, "tasks": ["17.2", "17.3", "19.1"] },
    { "id": 14, "tasks": ["20.1"] },
    { "id": 15, "tasks": ["21.1", "21.2", "21.3"] }
  ]
}
```

# Project Brief: AI Document Intelligence Platform (RAG-based Q&A System)

> Use this document as the build spec. Follow the phases in order — do not skip ahead to advanced features before the MVP (Phase 1-4) is fully working and tested.

## 1. Project Overview

**What we're building:** An enterprise-style document intelligence system that lets a user upload financial/regulatory documents (e.g. RBI circulars, loan policy documents, or annual reports) and ask natural-language questions about them. The system retrieves the most relevant passages and generates a grounded answer with citations back to the exact source document and section.

**Who it's for:** This is a portfolio project for a Software Engineer (Java/Spring Boot backend, fintech domain) leveling up into AI/ML engineering. It should demonstrate real retrieval-augmented generation (RAG) engineering, not a toy "chat with PDF" demo.

**What makes this different from a basic RAG tutorial:**
- Citations are mandatory, not optional — every answer must reference the source document, page, and section.
- Retrieval quality is evaluated, not assumed.
- The system is built with production habits: clean repo structure, tests, documentation, and a deployed live demo — not just a notebook.

**Domain note:** Default domain is banking/financial regulatory documents (RBI circulars, loan eligibility policies, annual reports) since it ties into the developer's fintech background. This is swappable — the architecture should not hard-code anything domain-specific.

## 2. Success Criteria (Definition of Done for MVP)

- [ ] A user can upload one or more PDF documents through a UI
- [ ] The system extracts, chunks, and embeds the document content
- [ ] A user can ask a natural-language question and receive an answer grounded in the documents
- [ ] Every answer includes at least one citation (source file + page/section)
- [ ] The system is deployed with a live, shareable demo link
- [ ] The repo has a professional README, architecture diagram, and setup instructions
- [ ] There is at least basic automated testing (ingestion pipeline + retrieval accuracy on a small eval set)

## 3. Tech Stack

| Layer | Choice | Notes |
|---|---|---|
| Backend | Python 3.11+, FastAPI | New skill area for the developer — keep Java for day job, build Python depth here |
| Document parsing | PyMuPDF or pdfplumber | Must preserve page numbers for citation |
| Chunking | LangChain or custom recursive splitter | Chunk size ~500-800 tokens, with overlap |
| Embeddings | OpenAI `text-embedding-3-small` or a local model (e.g. `sentence-transformers`) | Prefer a free/local option if API cost is a concern |
| Vector store | Chroma (local, free) or PostgreSQL + pgvector | Chroma is faster to stand up for an MVP |
| LLM | OpenAI GPT-4o-mini, Anthropic Claude, or a local model via Ollama | Keep provider abstracted behind an interface — don't hard-code one vendor |
| Frontend | Streamlit or Gradio | Fast to build, sufficient for a demo |
| Deployment | HuggingFace Spaces, Render, or Railway (free tiers) | Must be a live, clickable URL |
| Testing | pytest | Cover ingestion, chunking, and retrieval |
| Version control | Git/GitHub | Clean commit history, not one giant commit |

## 4. Architecture

```
User (Browser)
     |
     v
Streamlit/Gradio UI
     |
     v
FastAPI Backend
     |
     +--> Document Service   -> parses PDFs, extracts text + page metadata
     |         |
     |         v
     |    Chunking Module     -> splits text into overlapping chunks, retains page refs
     |         |
     |         v
     |    Embedding Module    -> generates vector embeddings per chunk
     |         |
     |         v
     |    Vector Store (Chroma / pgvector)
     |
     +--> Query Service       -> embeds user question, retrieves top-k relevant chunks
     |         |
     |         v
     |    LLM Service         -> generates answer grounded in retrieved chunks
     |         |
     |         v
     |    Response Formatter  -> returns answer + citations (doc name, page, excerpt)
     |
     v
UI displays answer with expandable source citations
```

## 5. Functional Requirements (by phase)

### Phase 1 — Data & Environment Setup
- Set up Python project structure (`src/`, `tests/`, `data/`, `notebooks/` for exploration only — no production logic in notebooks)
- Set up virtual environment and dependency management (`requirements.txt` or `poetry`)
- Collect 3-5 real documents in the chosen domain
- Define 8-10 realistic test questions with expected answer content, to be used later for evaluation

### Phase 2 — Ingestion & Retrieval Pipeline
- Build a PDF parser that extracts text while preserving page numbers
- Implement a chunking strategy (recursive character/token splitter, ~500-800 tokens, ~10-15% overlap)
- Generate embeddings for each chunk and store them in the vector database, along with metadata: `{document_name, page_number, section (if available), chunk_text}`
- Implement a retrieval function: given a query, embed it and return the top-k (start with k=5) most similar chunks
- Write a script/CLI to test retrieval manually before wiring up the LLM

### Phase 3 — RAG Answer Generation
- Build a prompt template that instructs the LLM to answer ONLY using the provided context chunks, and to say "I don't know based on the provided documents" if the answer isn't present
- Pass retrieved chunks + user question to the LLM and generate an answer
- Parse the LLM response and attach citations: which chunk(s) were used, with document name + page number
- Handle the "no relevant answer found" case gracefully — do not let the model hallucinate

### Phase 4 — API & UI
- Expose FastAPI endpoints:
  - `POST /documents/upload` — upload and ingest a document
  - `GET /documents` — list ingested documents
  - `POST /query` — ask a question, returns `{answer, citations: [{document, page, excerpt}]}`
- Build a simple Streamlit/Gradio UI:
  - Upload documents
  - Ask questions in a chat-style interface
  - Display the answer with expandable citation cards (document, page, excerpt)

### Phase 5 — Evaluation
- Using the 8-10 test questions from Phase 1, build a small evaluation script that checks:
  - Retrieval: is the correct source chunk in the top-k results?
  - Answer quality: does the generated answer contain the expected key facts? (manual review is fine for this scale, or use an LLM-as-judge approach)
- Log results in a simple table/markdown file in the repo (`eval/results.md`)

### Phase 6 — Deployment & Documentation
- Containerize with a `Dockerfile` (even if deploying to a platform that doesn't strictly require it — shows the skill)
- Deploy to HuggingFace Spaces / Render / Railway with a live public URL
- Write a professional `README.md` including:
  - Problem statement
  - Architecture diagram (the one above, or an improved version)
  - Setup/run instructions
  - Live demo link
  - Screenshots or a short GIF/video demo
  - Known limitations and possible future improvements
- Write a short `CASE_STUDY.md`: what design decisions were made and why, what didn't work initially, what you'd change at larger scale

## 6. Non-Functional Requirements

- **Code quality:** type hints throughout, docstrings on public functions, no secrets committed (use `.env` + `.gitignore`)
- **Modularity:** LLM provider and embedding provider should be swappable via a config/interface, not hard-coded
- **Error handling:** graceful handling of unparseable PDFs, empty queries, and LLM/API failures
- **Testing:** unit tests for chunking and retrieval logic at minimum; integration test for the full query flow
- **Commit hygiene:** incremental, meaningful commits — not one large "initial commit"

## 7. Explicitly Out of Scope for MVP (do not build these yet)

- Multi-tenancy / multiple isolated user accounts
- Hybrid search or reranking models
- Document comparison across versions
- Query analytics dashboards
- Authentication/authorization systems

These are good "Phase 2 of the portfolio" extensions once the MVP is live and working — not part of this build.

## 8. Instructions for the Build Agent

- Build and verify each phase before moving to the next; do not generate the entire codebase in one pass without checkpoints.
- After each phase, summarize what was built and any decisions made (e.g. chosen chunk size, k value) so they can be recorded in the case study later.
- Prefer well-known, well-documented libraries over custom implementations for parsing/chunking/embeddings.
- Flag clearly if a requirement conflicts with a free-tier limitation (e.g. API rate limits, hosting memory limits) and propose the best free-tier-compatible alternative.
- Optimize for a working, demonstrable end-to-end system over premature polish — get Phases 1-4 functional first, then improve.
- For any open technology choice in this brief (vector store, UI framework, LLM provider, hosting platform), do not decide silently. Compare the realistic options against: learning value, portfolio value, implementation complexity, free-tier cost, deployment compatibility, and future scalability — then recommend one and wait for approval before proceeding with it.

## 9. Additional Build-Agent Rules

### 9.1 Do Not Code Immediately
Before writing any implementation code, analyze the requirements and propose: final project structure, architecture, technology/library choices, data flow, API contracts, development sequence, and risks/free-tier constraints. Wait for approval before beginning implementation.

### 9.2 Teach While Building
This project is being used to build professional software engineering and AI/ML skill, not just to produce a working app. For every major implementation step: explain what is being built, why this approach was chosen over alternatives, the important technical concepts involved, and how to manually test/verify it. Do not assume prior familiarity with RAG/LLM concepts.

### 9.3 Do Not Hide Complexity
Do not ship code that works without explaining its important architectural decisions. Avoid unnecessary abstraction and over-engineering, but use clean, professional patterns where they provide genuine value.

### 9.4 Checkpoint After Every Phase
Do not automatically continue to the next phase. At the end of each phase: run the relevant tests, verify the implementation, summarize what was completed, list files created/modified, list important technical decisions, list known issues, explain how to manually verify the result, suggest a Git commit message, and wait for approval before proceeding.

### 9.5 Maintain a Decision Log
Keep a running `DECISIONS.md` in the repo. After each phase, append: the decision made, the alternatives considered, and why this option was chosen. This becomes the raw material for interview answers and the case study — don't leave it until the end.

### 9.6 Portfolio Quality
This should be something to confidently discuss in a software engineering / AI-ML interview — not just "it works." The final implementation should demonstrate: clean architecture, maintainable code, meaningful testing, proper error handling, reproducibility, documentation, measurable retrieval/evaluation results, a professional Git history, and a working deployed demo. If a requirement can't reasonably be implemented within free-tier resources, explain the limitation before changing the approach.
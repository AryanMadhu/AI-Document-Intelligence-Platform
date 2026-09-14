# Requirements Document

## Introduction

The AI Document Intelligence Platform is an enterprise-style, retrieval-augmented generation (RAG) system that lets a user upload financial or regulatory PDF documents (for example RBI circulars, loan policy documents, or annual reports) and ask natural-language questions about their content. The system extracts and indexes document text, retrieves the most relevant passages for a question, and generates an answer that is grounded strictly in the retrieved source material.

The defining characteristic of the platform is mandatory citation: every generated answer must reference the exact source document, page or section, and a supporting excerpt. Retrieval quality is measured against a defined evaluation set rather than assumed. The architecture is domain-agnostic — no component encodes banking-specific logic — and both the embedding provider and the language-model provider are swappable through configuration.

This document specifies the MVP scope, organized to align with the six delivery phases in the Project Brief (data/environment setup, ingestion and retrieval, RAG answer generation, API and UI, evaluation, and deployment and documentation), plus non-functional and quality requirements. Explicit non-goals are recorded to bound the MVP.

## Glossary

- **Platform**: The complete AI Document Intelligence Platform, including backend services, retrieval pipeline, and user interface.
- **Ingestion_Service**: The component that accepts uploaded PDF files, extracts text with page metadata, splits text into chunks, generates embeddings, and stores them in the Vector_Store.
- **PDF_Parser**: The component that extracts text content from a PDF file while preserving the originating page number for each unit of text.
- **Chunking_Module**: The component that splits extracted document text into overlapping segments (chunks) suitable for embedding and retrieval.
- **Embedding_Provider**: The pluggable component that converts a text string into a numeric vector embedding. Concrete implementations include a hosted embedding API and a local embedding model.
- **Vector_Store**: The persistence component that stores chunk embeddings together with chunk metadata and supports similarity search.
- **Retrieval_Service**: The component that embeds a user question and returns the top-k most similar chunks from the Vector_Store.
- **LLM_Provider**: The pluggable component that generates a natural-language answer from a prompt. Concrete implementations include hosted language-model APIs and a local language model.
- **Query_Service**: The component that orchestrates retrieval, prompt construction, LLM invocation, and citation assembly for a single question.
- **Response_Formatter**: The component that assembles the final response object containing the answer text and its list of citations.
- **Citation**: A structured reference attached to an answer, containing at minimum the source document name, the page number (or section identifier), and a supporting text excerpt.
- **Chunk**: A single segment of document text stored in the Vector_Store, with metadata `{document_name, page_number, section, chunk_text}`.
- **Top_K**: The configurable number of most-similar chunks returned by the Retrieval_Service for a query. The default value is 5.
- **Grounded_Answer**: An answer whose content is derived only from retrieved chunks and not from the LLM's general knowledge.
- **No_Answer_Response**: The fixed response "I don't know based on the provided documents", returned when the retrieved context does not support an answer.
- **Eval_Set**: A defined set of 8 to 10 test questions, each paired with expected answer content and the source location of the correct answer, used to measure retrieval and answer quality.
- **Evaluation_Script**: The component that runs the Eval_Set against the Platform and records retrieval hit-rate and answer-quality results.
- **Retrieval_Hit_Rate**: The fraction of Eval_Set questions for which the correct source chunk appears within the Top_K retrieved chunks.
- **Answer_Key_Fact_Coverage**: The fraction of predefined essential facts for an Eval_Set question that are present in the generated answer for that question. Each Eval_Set question records a set of predefined essential facts against which the generated answer is checked.
- **Citation_Validity**: The property that every Citation returned for an answer corresponds to a Chunk that was actually included in the context provided to the LLM_Provider for that answer.
- **Embedding_Model**: The specific model, identified by name or version, used by an Embedding_Provider to convert text into a vector embedding. Embeddings produced by different Embedding_Model values occupy different vector spaces and are not directly comparable.
- **Document_Hash**: A content-derived identifier computed from the bytes of an uploaded PDF file, used to detect whether an identical document has already been ingested.
- **API_Backend**: The FastAPI service exposing HTTP endpoints for document upload, document listing, and querying.
- **UI**: The Streamlit or Gradio web interface through which a user uploads documents and asks questions.
- **User**: A person interacting with the Platform through the UI to upload documents and ask questions.
- **Config**: The configuration source (environment variables and/or configuration files) that selects the active Embedding_Provider, LLM_Provider, and related settings.

## Requirements

### Requirement 1: Project Structure and Environment Setup

**User Story:** As a developer, I want a clean, reproducible Python project structure and dependency setup, so that the codebase is professional, maintainable, and easy for others to run.

#### Acceptance Criteria

1. THE Platform SHALL organize source code, tests, and data into separate top-level directories for application code, automated tests, and document data.
2. THE Platform SHALL declare all runtime and development dependencies in a dependency-management file that pins package versions.
3. WHERE Jupyter notebooks are included, THE Platform SHALL restrict notebook content to exploration and SHALL NOT place production logic in notebooks.
4. THE Platform SHALL provide a reproducible environment setup procedure that installs all declared dependencies from a clean checkout.
5. THE Platform SHALL include a `.gitignore` file that excludes environment files, secrets, virtual environments, and local data artifacts from version control.

### Requirement 2: Evaluation Corpus and Test Questions

**User Story:** As a developer, I want a small corpus of real domain documents and a defined set of test questions with expected answers, so that retrieval and answer quality can be measured objectively.

#### Acceptance Criteria

1. THE Platform SHALL include between 3 and 5 real domain documents in PDF format within the data directory for ingestion and evaluation.
2. THE Platform SHALL define an Eval_Set containing between 8 and 10 natural-language test questions.
3. THE Platform SHALL record, for each question in the Eval_Set, the predefined essential facts that a correct answer must contain and the source document and page where the answer is located.
4. THE Platform SHALL store the Eval_Set in a version-controlled file within the repository.

### Requirement 3: PDF Parsing with Page Preservation

**User Story:** As a User, I want the system to read my PDF documents accurately while tracking page numbers, so that answers can cite the exact page a fact came from.

#### Acceptance Criteria

1. WHEN a PDF file is submitted for ingestion, THE PDF_Parser SHALL extract the text content of the PDF file.
2. WHEN the PDF_Parser extracts text, THE PDF_Parser SHALL associate each extracted text unit with the page number of the page from which the text unit originated.
3. IF a submitted file cannot be parsed as a valid PDF, THEN THE Ingestion_Service SHALL reject the file and return an error message identifying the file and the reason.
4. IF a PDF file contains a page with no extractable text, THEN THE PDF_Parser SHALL record the page as empty and continue processing the remaining pages.

### Requirement 4: Document Chunking

**User Story:** As a developer, I want document text split into overlapping chunks of a controlled size, so that retrieval returns focused, relevant passages while preserving context across boundaries.

#### Acceptance Criteria

1. WHEN extracted document text is provided to the Chunking_Module, THE Chunking_Module SHALL split the text into chunks with a target size between 500 and 800 tokens.
2. WHEN the Chunking_Module produces chunks, THE Chunking_Module SHALL create an overlap between 10 and 15 percent of the chunk size between consecutive chunks.
3. WHEN the Chunking_Module produces a chunk, THE Chunking_Module SHALL attach to the chunk the metadata fields `document_name`, `page_number`, `section`, and `chunk_text`.
4. WHERE a source section identifier is not available for a chunk, THE Chunking_Module SHALL set the `section` metadata field to a defined empty or null value.

### Requirement 5: Embedding Generation and Storage

**User Story:** As a developer, I want each chunk embedded and stored with its metadata in a vector store, so that questions can be matched to relevant passages by semantic similarity.

#### Acceptance Criteria

1. WHEN a chunk is produced during ingestion, THE Embedding_Provider SHALL generate a vector embedding for the `chunk_text` of the chunk.
2. WHEN an embedding is generated for a chunk, THE Vector_Store SHALL persist the embedding together with the chunk metadata `{document_name, page_number, section, chunk_text}`.
3. IF the Embedding_Provider fails to generate an embedding for a chunk, THEN THE Ingestion_Service SHALL report the failure and identify the affected document.
4. THE Platform SHALL select the active Embedding_Provider from Config without requiring source-code changes.
5. WHEN a document is indexed, THE Platform SHALL record the identity of the Embedding_Provider and the Embedding_Model used to index that document as metadata associated with the document and its chunks.

### Requirement 6: Top-K Retrieval

**User Story:** As a User, I want the system to find the passages most relevant to my question, so that answers are based on the right parts of my documents.

#### Acceptance Criteria

1. WHEN a question is submitted to the Retrieval_Service, THE Retrieval_Service SHALL generate a vector embedding of the question using the active Embedding_Provider.
2. WHEN the Retrieval_Service has embedded a question, THE Retrieval_Service SHALL return the Top_K chunks from the Vector_Store ranked by similarity to the question embedding.
3. THE Retrieval_Service SHALL use a default Top_K value of 5 and SHALL allow the Top_K value to be set through Config.
4. WHEN the Retrieval_Service returns a chunk, THE Retrieval_Service SHALL include the chunk metadata `{document_name, page_number, section, chunk_text}` with the returned chunk.
5. IF the Vector_Store contains no chunks, THEN THE Retrieval_Service SHALL return an empty result set.

### Requirement 7: Retrieval Testing Interface

**User Story:** As a developer, I want a command-line way to test retrieval before the LLM is wired up, so that I can verify retrieval quality in isolation.

#### Acceptance Criteria

1. THE Platform SHALL provide a command-line interface that accepts a text question and returns the Top_K retrieved chunks with their metadata.
2. WHEN the command-line retrieval interface returns chunks, THE Platform SHALL display for each chunk the `document_name`, `page_number`, and `chunk_text`.

### Requirement 8: Grounded Answer Generation

**User Story:** As a User, I want answers based only on my uploaded documents, so that I can trust the response is grounded in my source material rather than invented.

#### Acceptance Criteria

1. WHEN the Query_Service constructs a prompt for the LLM_Provider, THE Query_Service SHALL include the retrieved chunks as the context and SHALL instruct the LLM_Provider to answer using only the provided context.
2. WHEN the Query_Service constructs a prompt for the LLM_Provider, THE Query_Service SHALL instruct the LLM_Provider to return the No_Answer_Response when the provided context does not contain the answer.
3. WHEN retrieved context is available for a question, THE Query_Service SHALL pass the question and the retrieved chunks to the LLM_Provider and SHALL obtain a Grounded_Answer.
4. IF the retrieved context does not support an answer, THEN THE Query_Service SHALL return the No_Answer_Response.
5. THE Platform SHALL select the active LLM_Provider from Config without requiring source-code changes.

### Requirement 9: Mandatory Citations

**User Story:** As a User, I want every answer to cite its sources, so that I can verify each claim against the original document, page, and excerpt.

#### Acceptance Criteria

1. WHEN the Query_Service returns a Grounded_Answer, THE Response_Formatter SHALL attach at least one Citation to the answer.
2. WHEN the Response_Formatter attaches a Citation, THE Response_Formatter SHALL include in the Citation the `document` name, the `page` number or section identifier, and a supporting `excerpt` drawn from the retrieved chunk.
3. WHEN the Query_Service returns the No_Answer_Response, THE Response_Formatter SHALL return an empty citation list.
4. THE Response_Formatter SHALL derive every Citation from a chunk that was included in the context provided to the LLM_Provider for that answer.

### Requirement 10: Query API Endpoint

**User Story:** As a client application, I want an HTTP endpoint to submit a question and receive a structured answer with citations, so that the Platform can be integrated with any front end.

#### Acceptance Criteria

1. WHEN a client sends a `POST /query` request containing a question, THE API_Backend SHALL return a response object containing an `answer` field and a `citations` field.
2. WHEN the API_Backend returns a `citations` field, THE API_Backend SHALL format each citation as an object containing `document`, `page`, and `excerpt`.
3. IF a `POST /query` request contains an empty or missing question, THEN THE API_Backend SHALL reject the request and return an error response indicating that a question is required.
4. IF the LLM_Provider fails to return a response, THEN THE API_Backend SHALL return an error response indicating that answer generation failed.

### Requirement 11: Document Upload API Endpoint

**User Story:** As a User, I want to upload PDF documents through an HTTP endpoint, so that the Platform can ingest and index them for questioning.

#### Acceptance Criteria

1. WHEN a client sends a `POST /documents/upload` request containing a PDF file, THE API_Backend SHALL pass the file to the Ingestion_Service for extraction, chunking, embedding, and storage.
2. WHEN the Ingestion_Service completes ingestion of an uploaded document, THE API_Backend SHALL return a success response identifying the ingested document.
3. IF a `POST /documents/upload` request contains a file that cannot be parsed as a valid PDF, THEN THE API_Backend SHALL return an error response identifying the reason.
4. THE Platform SHALL enforce a configurable maximum PDF file size limit and a configurable maximum page-count limit, with default values documented in Config.
5. IF an uploaded PDF exceeds the configured maximum file size limit or the configured maximum page-count limit, THEN THE API_Backend SHALL reject the file and return an error response identifying the file and the limit that was exceeded, and THE Platform SHALL NOT store any Chunk from that file in the Vector_Store.
6. WHEN a PDF file is uploaded, THE Ingestion_Service SHALL compute the Document_Hash of the file and SHALL compare the Document_Hash against the Document_Hash values of already-ingested documents to detect an identical document.
7. IF an uploaded PDF is identical to an already-ingested document, THEN THE Ingestion_Service SHALL treat the upload as a duplicate and SHALL NOT create duplicate Chunk entries in the Vector_Store.
8. WHEN a duplicate document is detected, THE API_Backend SHALL return a response indicating that the document was already ingested and identifying the existing document.

### Requirement 12: Document Listing API Endpoint

**User Story:** As a User, I want to see which documents have been ingested, so that I know what content my questions can be answered from.

#### Acceptance Criteria

1. WHEN a client sends a `GET /documents` request, THE API_Backend SHALL return the list of documents that have been ingested into the Vector_Store.
2. WHILE no documents have been ingested, THE API_Backend SHALL return an empty list in response to a `GET /documents` request.

### Requirement 13: User Interface

**User Story:** As a User, I want a simple web interface to upload documents and ask questions in a chat style, so that I can use the Platform without writing code.

#### Acceptance Criteria

1. THE UI SHALL provide a control for uploading one or more PDF documents to the Platform.
2. THE UI SHALL provide a chat-style input through which a User submits a natural-language question.
3. WHEN the Platform returns an answer, THE UI SHALL display the answer text together with its citations.
4. WHEN the UI displays citations, THE UI SHALL present each citation as an expandable element showing the `document`, `page`, and `excerpt`.
5. IF answer generation fails, THEN THE UI SHALL display an error message to the User.

### Requirement 14: Evaluation of Retrieval and Answer Quality

**User Story:** As a developer, I want an automated evaluation of retrieval and answer quality against the Eval_Set, so that I can demonstrate measurable performance rather than assumed quality.

#### Acceptance Criteria

1. WHEN the Evaluation_Script runs, THE Evaluation_Script SHALL execute every question in the Eval_Set against the Platform.
2. WHEN the Evaluation_Script processes a question, THE Evaluation_Script SHALL determine whether the correct source chunk appears within the Top_K retrieved chunks and SHALL compute the Retrieval_Hit_Rate across the Eval_Set.
3. WHEN the Evaluation_Script processes a question, THE Evaluation_Script SHALL compute the Answer_Key_Fact_Coverage by determining which of the predefined essential facts recorded for that question are present in the generated answer.
4. WHEN the Evaluation_Script processes a question, THE Evaluation_Script SHALL compute the Citation_Validity by determining whether every Citation returned for the answer corresponds to a Chunk that was included in the context provided to the LLM_Provider for that answer.
5. WHEN the Evaluation_Script completes, THE Evaluation_Script SHALL write the Retrieval_Hit_Rate, the Answer_Key_Fact_Coverage, and the Citation_Validity results, as aggregate values across the Eval_Set and as per-question values, to a version-controlled results file within the repository.

### Requirement 15: Provider Modularity

**User Story:** As a developer, I want the embedding and LLM providers to be swappable through configuration, so that the Platform is not locked to a single vendor and can adapt to cost or availability constraints.

#### Acceptance Criteria

1. THE Platform SHALL define a common interface for the Embedding_Provider that concrete embedding implementations satisfy.
2. THE Platform SHALL define a common interface for the LLM_Provider that concrete language-model implementations satisfy.
3. WHEN the active Embedding_Provider is changed through Config, THE Platform SHALL use the newly selected Embedding_Provider without source-code changes.
4. WHEN the active LLM_Provider is changed through Config, THE Platform SHALL use the newly selected LLM_Provider without source-code changes.
5. THE Platform SHALL keep all components free of banking-domain-specific logic so that the architecture applies to any document domain.
6. WHEN the active Embedding_Provider or the active Embedding_Model changes, THE Platform SHALL treat every previously indexed document whose recorded Embedding_Provider or Embedding_Model differs from the active Embedding_Provider or active Embedding_Model as incompatible with the active embedding space.
7. WHILE a previously indexed document is incompatible with the active embedding space, THE Platform SHALL require that document to be re-embedded with the active Embedding_Provider and active Embedding_Model before the Retrieval_Service performs retrieval against that document, because vector embeddings produced by different Embedding_Model values occupy different vector spaces and produce meaningless similarity scores when compared.

### Requirement 16: Error Handling and Robustness

**User Story:** As a User, I want the Platform to handle bad input and provider failures gracefully, so that failures produce clear messages instead of crashes.

#### Acceptance Criteria

1. IF a User submits an empty question, THEN THE Platform SHALL return a message indicating that a question is required and SHALL NOT invoke the LLM_Provider.
2. IF an uploaded PDF cannot be parsed, THEN THE Platform SHALL return a message identifying the file and the reason and SHALL continue to serve previously ingested documents.
3. IF the Embedding_Provider or the LLM_Provider returns an error, THEN THE Platform SHALL return a message indicating the failure and SHALL preserve the previously ingested document data.
4. WHERE a required credential for a configured provider is absent, THE Platform SHALL report the missing credential by name and SHALL NOT expose the credential value.

### Requirement 17: Code Quality and Secret Management

**User Story:** As a developer, I want type hints, docstrings, and disciplined secret handling, so that the codebase is professional and safe to publish.

#### Acceptance Criteria

1. THE Platform SHALL include type hints on function signatures across the application source code.
2. THE Platform SHALL include docstrings on public functions and public classes.
3. THE Platform SHALL load secrets from environment variables or an environment file and SHALL exclude that environment file from version control.
4. THE Platform SHALL NOT commit secret values to the version-control repository.

### Requirement 18: Automated Testing

**User Story:** As a developer, I want automated tests covering chunking, retrieval, and the full query flow, so that regressions are caught and the code is trustworthy.

#### Acceptance Criteria

1. THE Platform SHALL include unit tests that verify the chunking behavior of the Chunking_Module, including chunk size and overlap.
2. THE Platform SHALL include unit tests that verify the retrieval behavior of the Retrieval_Service.
3. THE Platform SHALL include an integration test that exercises the full query flow from question submission to an answer with citations.
4. WHEN the automated test suite is executed, THE Platform SHALL report a pass or fail result for each test.

### Requirement 19: Containerization and Deployment

**User Story:** As a reviewer, I want the Platform packaged in a container and deployed to a live public URL, so that I can try the demo without setting it up locally.

#### Acceptance Criteria

1. THE Platform SHALL include a `Dockerfile` that builds a runnable container image of the Platform.
2. THE Platform SHALL be deployed to a publicly reachable host and SHALL expose a live URL through which a User can access the UI.
3. IF a Platform requirement conflicts with a free-tier hosting or provider limitation, THEN THE Platform SHALL document the conflict and SHALL record a free-tier-compatible alternative.

### Requirement 20: Documentation and Decision Records

**User Story:** As a reviewer, I want professional documentation and a record of design decisions, so that I can understand the system, run it, and evaluate the engineering behind it.

#### Acceptance Criteria

1. THE Platform SHALL include a `README.md` containing the problem statement, an architecture diagram, setup and run instructions, the live demo link, screenshots or a demo recording, and known limitations.
2. THE Platform SHALL include a `CASE_STUDY.md` describing the design decisions made, the approaches that did not work initially, and the changes that would be made at larger scale.
3. THE Platform SHALL maintain a `DECISIONS.md` that records, for each significant decision, the decision made, the alternatives considered, and the rationale for the chosen option.
4. WHERE an open technology choice exists for the Vector_Store, the Embedding_Provider, the LLM_Provider, the UI framework, or the hosting platform, THE Platform SHALL resolve the choice with a documented comparison of the realistic options rather than an unrecorded selection.

## Non-Goals (Out of Scope for MVP)

The following are explicitly excluded from the MVP and are not required by this document:

- **Multi-tenancy and multi-user accounts**: The MVP does not isolate documents or data per user account.
- **Hybrid search and reranking models**: The MVP uses vector similarity retrieval only, without keyword hybrid search or a dedicated reranking model.
- **Document version comparison**: The MVP does not compare documents across versions.
- **Query analytics dashboards**: The MVP does not provide usage or query analytics dashboards.
- **Authentication and authorization**: The MVP does not implement login, user identity, or access control.
- **Conversation memory and multi-turn contextual queries**: The MVP treats each query independently and does not retain previous questions or answers as conversational context. The chat-style UI is a presentation choice only and does not imply retained conversational state.

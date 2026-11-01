# Task 16 — Evaluation Corpus

This is an original, public-safe starter corpus for the AI Document Intelligence Platform.

## Contents
- `data/machine_learning_fundamentals.pdf`
- `data/neural_networks_primer.pdf`
- `data/natural_language_processing_basics.pdf`
- `eval/eval_set.json`

Each PDF has exactly three deliberately separated pages. The JSON contains nine answerable questions (three per document). Source page numbers are 1-based PDF page numbers. Essential facts are short literal phrases from the corresponding source page to support the project's deterministic fact-coverage metric.

## Use
1. Copy the three PDFs into the project's `data/` directory.
2. Upload/index them through the application's normal ingestion workflow.
3. Preserve filenames exactly; `eval_set.json` references those names.
4. Confirm all three appear in the indexed-document list.
5. Run the evaluation script only after Task 17 implements it.

No personal resume or private document is included. No unanswerable questions are included in this initial baseline; add those only if the evaluation schema and metrics explicitly support them.

This is a small baseline corpus, not a comprehensive benchmark. The PDFs are original educational material created for this project.

## Evaluation metrics (Task 17)

The evaluation harness (`eval/run_eval.py`) runs every question through the real RAG pipeline and reports three metrics: **Retrieval Hit Rate**, **Answer Fact Coverage**, and **Citation Validity**. Fact coverage supports two matching modes.

### Answer Fact Coverage — matching modes

**Substring mode (default).** Exact, normalized substring matching: both the answer and each `essential_fact` are whitespace-normalized (runs of whitespace collapsed to a single space), stripped, and lowercased, then each fact must appear as a literal substring of the answer. This is the conservative baseline — it never credits a fact unless its exact wording is present, so it undercounts correct answers that paraphrase or reformat the expected phrase.

**Overlap mode (opt-in).** Deterministic content-token overlap. Each fact and the answer are tokenized; stopwords are removed; a light plural normalization (a heuristic, not a true lemmatizer) maps common plurals to a shared stem (for example `categories` → `category`, `values` → `value`). A fact is counted as covered when the fraction of its distinct content tokens that also appear in the answer's content tokens is at least the threshold (default **0.85**). This credits correct answers that reword or reorder the expected phrasing.

### Running the evaluation

Run from the project root. These make live Gemini calls, so trigger them manually:

```
# Substring mode (default, conservative baseline)
.venv\Scripts\python.exe eval\run_eval.py

# Token-overlap mode (opt-in), with the default 0.85 threshold
.venv\Scripts\python.exe eval\run_eval.py --coverage-mode overlap

# Token-overlap mode with an explicit threshold
.venv\Scripts\python.exe eval\run_eval.py --coverage-mode overlap --overlap-threshold 0.85
```

Results are written (git-ignored) to `eval/results/` as timestamped `.md` and `.json` files. The chosen mode and threshold are recorded in each result file so runs are self-describing and reproducible.

### Important limitations

- **Overlap mode can over-credit.** It is bag-of-words set overlap: it ignores word order and does not understand meaning, so an answer that merely contains a fact's content words — in a different arrangement or a different sense — can be scored as covered even when it does not actually state the fact. The plural heuristic can also over-stem (for example `values` → `valu`); this is harmless for matching because the same stemmer is applied to both sides, but it is not a real lemmatizer and will not handle irregular plurals.
- **Neither mode is a true semantic correctness evaluator.** Substring is a conservative lower bound; overlap is a looser upper-leaning estimate. Both are deterministic, offline, and cheap, but neither verifies that an answer is factually correct or faithful to the source — only whether expected wording (substring) or expected content tokens (overlap) are present. A semantic LLM-as-judge scorer is a documented Stretch improvement (Task 17.3), intentionally not implemented here.

### Current baseline

Recorded from a run against the three-document, nine-question corpus:

| Metric | Value |
|---|---|
| Retrieval Hit Rate | 100% (9/9) |
| Citation Validity | 100% (9/9) |
| Answer Fact Coverage — substring (default) | 27.8% |
| Answer Fact Coverage — token-overlap @ 0.85 | 77.8% |

Run configuration: LLM model `gemini-3.5-flash-lite`, embedding model `all-MiniLM-L6-v2`, `top_k` = 5.

The large gap between substring (27.8%) and overlap (77.8%) coverage reflects that the generated answers are largely correct but paraphrase or reformat the benchmark's exact wording — the substring metric undercounts them. Per project policy, the benchmark questions and expected facts are NOT edited to raise the score; the two modes are reported side by side so the baseline stays honest.

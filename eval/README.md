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

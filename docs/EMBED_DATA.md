# Embed data guide

The Embed capability uses three retrieval objects per split:

- `queries.jsonl`: `query_id` and legal question text.
- `corpus.jsonl`: `passage_id`, source passage text, and `document_id`.
- `qrels.tsv`: positive `query_id` to `passage_id` relevance judgments.

The persistent directory-level explanation is in `data/processed/embed/README.md`. The detailed inspection, checksums, duplicate counts, token-length statistics, split sizes, licence, and provenance are in `data/processed/embed/dataset_report.json` after preparation.

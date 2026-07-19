# Embed models

## Candidates

- BM25
- `BAAI/bge-small-en-v1.5`
- `Alibaba-NLP/gte-modernbert-base`
- `Qwen/Qwen3-Embedding-0.6B`

## Initial decision

Use `Alibaba-NLP/gte-modernbert-base` for the first fine-tuning baseline. Record NDCG@10, Recall@1/5/10, MRR, latency, dimension, and memory for every model.

## Unresolved questions

- Exact target hardware and memory budget.
- Whether Granite adds enough value to justify implementation complexity.

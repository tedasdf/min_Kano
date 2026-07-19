# Embed training objectives

## Baseline

Train normalized query and positive-passage embeddings using dot-product similarity and `CachedMultipleNegativesRankingLoss`.

## Stages

1. In-batch negatives.
2. BM25 hard negatives.
3. Dense-model hard negatives.
4. Optional teacher distillation.

Checkpoint selection and negative-mining decisions use the internal validation/test layers. MLEB is used only before fine-tuning and after the final selected run.

# Embed training objectives

## Baseline

Train normalized query and positive-passage embeddings using dot-product similarity and `CachedMultipleNegativesRankingLoss`.

The Embed v1 implementation samples one known positive per query per epoch so another known positive for the same query is not explicitly presented as its negative. Multi-positive queries rotate deterministically between epochs. Other queries' positives provide the cached in-batch negatives.

## Stages

1. In-batch negatives.
2. BM25 hard negatives.
3. Dense-model hard negatives.
4. Optional teacher distillation.

Checkpoint selection and negative-mining decisions use the internal validation/test layers. MLEB is used only before fine-tuning and after the final selected run.

BM25 mining uses only the training corpus. It excludes all known positive passages and, by default, every passage sharing a positive's source document. This reduces the chance that an unlabeled passage from the same legal document is incorrectly treated as negative supervision.

Embed v2 consumes the BM25 mining artifact as explicit n-tuple negatives while retaining other examples in the logical batch as in-batch negatives. The v2 run uses separate checkpoints and reports, so its result remains directly comparable with v1.

Embed v3 mines nearest non-positive training passages with the completed v2 checkpoint, then trains a fresh copy of the original base model with four dense hard negatives per query. Training fresh rather than continuing v2 isolates the supervision change.

Embed v4 scores one known positive and eight v3 candidates with a cross-encoder. Its student objective blends teacher-distribution KL divergence with a known-positive cross-entropy target. It has separate candidate artifacts, checkpoints, and reports.

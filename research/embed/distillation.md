# M3-style distillation

Run only after the ordinary contrastive baseline works. Candidate teacher signals are BM25, a dense teacher, and a cross-encoder relevance score. Compare the combined teacher relevance distribution against ordinary hard-negative training.

## Implemented v4 baseline

The first executable version uses the v3 dense-negative pool as its candidate set and `cross-encoder/ms-marco-MiniLM-L-6-v2` as a straightforward teacher. Each record contains one known positive followed by eight dense hard negatives and the teacher's score for every candidate.

The student minimizes a weighted combination of:

- KL divergence from the teacher relevance distribution (`distillation_weight: 0.7`); and
- cross-entropy with the known positive in position zero (`0.3`).

The second term prevents noisy teacher rankings from erasing the known positive label. Teacher choice, temperature, candidate count, and blend weight remain ablations rather than fixed conclusions.

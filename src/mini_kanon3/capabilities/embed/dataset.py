"""Training-pair construction for the Embed capability."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

from .io import load_retrieval_split


@dataclass(frozen=True)
class PositivePair:
    query_id: str
    passage_id: str
    query: str
    passage: str


def load_positive_groups(split_dir: str | Path):
    """Return query text, corpus text, and sorted positives per query."""
    queries, corpus, qrels = load_retrieval_split(split_dir)
    positives = {query_id: sorted(passage_ids) for query_id, passage_ids in qrels.items()}
    return queries, corpus, positives


def sample_one_positive_per_query(queries, corpus, positives, seed: int, epoch: int):
    """Avoid treating another known positive for the same query as an in-batch negative.

    Multi-positive queries rotate deterministically across epochs. Query order is
    shuffled reproducibly; passages are not duplicated for a single query batch item.
    """
    rng = random.Random(f"{seed}:{epoch}")
    query_ids = sorted(positives)
    rng.shuffle(query_ids)
    pairs = []
    for query_id in query_ids:
        passage_ids = positives[query_id]
        passage_id = passage_ids[(seed + epoch) % len(passage_ids)]
        pairs.append(PositivePair(query_id, passage_id, queries[query_id], corpus[passage_id]))
    return pairs


def arrange_no_duplicate_batches(pairs, batch_size: int):
    """Reorder pairs so query/passage text is not duplicated within a batch."""
    remaining = list(pairs)
    arranged = []
    while remaining:
        batch, deferred = [], []
        query_texts, passage_texts = set(), set()
        for pair in remaining:
            if (len(batch) < batch_size and pair.query not in query_texts
                    and pair.passage not in passage_texts):
                batch.append(pair)
                query_texts.add(pair.query)
                passage_texts.add(pair.passage)
            else:
                deferred.append(pair)
        if not batch:
            raise ValueError("Unable to construct a no-duplicates training batch")
        arranged.extend(batch)
        remaining = deferred
    return arranged

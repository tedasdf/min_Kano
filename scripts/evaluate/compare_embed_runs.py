#!/usr/bin/env python3
"""Compare two retrieval runs query by query at a chosen rank cutoff."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mini_kanon3.capabilities.embed.io import load_retrieval_split


def load_result(path: Path) -> dict:
    result = json.loads(path.read_text(encoding="utf-8"))
    if "per_query" not in result:
        raise ValueError(
            f"{path} has no per_query results; rerun evaluation with the "
            "updated evaluator"
        )
    return result


def succeeded(row: dict, cutoff: int) -> bool:
    rank = row["best_positive_rank"]
    return rank is not None and rank <= cutoff


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--left-label", default="v2")
    parser.add_argument("--right-label", default="v3")
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--cutoff", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.cutoff < 1:
        parser.error("--cutoff must be positive")

    left = load_result(args.left)
    right = load_result(args.right)
    queries, corpus, qrels = load_retrieval_split(args.split)
    query_ids = set(left["per_query"]) | set(right["per_query"])
    if query_ids != set(qrels):
        raise ValueError("Run query IDs do not match the supplied split")

    rows = []
    counts = {
        "both_succeed": 0,
        "left_only": 0,
        "right_only": 0,
        "neither_succeeds": 0,
    }
    for query_id in sorted(query_ids):
        left_row = left["per_query"][query_id]
        right_row = right["per_query"][query_id]
        left_ok = succeeded(left_row, args.cutoff)
        right_ok = succeeded(right_row, args.cutoff)
        category = (
            "both_succeed"
            if left_ok and right_ok
            else "left_only"
            if left_ok
            else "right_only"
            if right_ok
            else "neither_succeeds"
        )
        counts[category] += 1
        if category not in {"left_only", "right_only"}:
            continue

        left_top = left_row["top_10_passage_ids"][0]
        right_top = right_row["top_10_passage_ids"][0]
        positive_ids = sorted(qrels[query_id])
        rows.append(
            {
                "category": (
                    f"{args.left_label}_succeeds_{args.right_label}_fails"
                    if category == "left_only"
                    else f"{args.right_label}_succeeds_{args.left_label}_fails"
                ),
                "cutoff": args.cutoff,
                "query_id": query_id,
                "query": queries[query_id],
                "positive_passages": [
                    {"passage_id": pid, "text": corpus[pid]}
                    for pid in positive_ids
                ],
                args.left_label: {
                    "best_positive_rank": left_row["best_positive_rank"],
                    "top_passage_id": left_top,
                    "top_passage_text": corpus[left_top],
                },
                args.right_label: {
                    "best_positive_rank": right_row["best_positive_rank"],
                    "top_passage_id": right_top,
                    "top_passage_text": corpus[right_top],
                },
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(
        json.dumps(
            {
                "cutoff": args.cutoff,
                "left_model": left["model"],
                "right_model": right["model"],
                "counts": counts,
                "disagreement_rows_written": len(rows),
                "output": str(args.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

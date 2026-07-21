#!/usr/bin/env python3
"""Convert Open Australian Legal QA JSONL into leakage-safe retrieval data."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import re
import shutil
from pathlib import Path


FIELDS = ("question", "answer", "text", "prompt", "source")
SOURCE_FIELDS = ("version_id", "type", "jurisdiction", "source", "citation", "url", "text")


def clean(value):
    return re.sub(r"\s+", " ", value).strip() if isinstance(value, str) else ""


def fingerprint(prefix: str, value: str) -> str:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:20]
    return f"{prefix}_{digest}"


def tokens(text: str) -> int:
    """Deterministic tokenizer-independent approximation (words + punctuation)."""
    return len(re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE))


def percentile(values, p):
    if not values:
        return 0
    ordered = sorted(values)
    index = (len(ordered) - 1) * p
    lo, hi = math.floor(index), math.ceil(index)
    if lo == hi:
        return ordered[lo]
    return round(ordered[lo] * (hi - index) + ordered[hi] * (index - lo), 2)


def length_summary(values, limits):
    return {
        "count": len(values),
        "min": min(values, default=0),
        "p50": percentile(values, 0.50),
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": max(values, default=0),
        "mean": round(sum(values) / len(values), 2) if values else 0,
        "exceeds_context": {str(n): sum(v > n for v in values) for n in limits},
    }


def document_key(source):
    # version_id is the corpus' durable source identifier. Fall back conservatively.
    version_id = clean(source.get("version_id"))
    if version_id:
        return f"version_id:{version_id}"
    url = clean(source.get("url"))
    if url:
        return f"url:{url.lower()}"
    metadata = "\x1f".join(clean(source.get(k)) for k in ("source", "jurisdiction", "citation"))
    return f"metadata:{metadata}"


def split_documents(document_ids, seed, ratios):
    names = ("train", "validation", "test")
    ranked = sorted(document_ids, key=lambda x: hashlib.sha256(f"{seed}\x1f{x}".encode()).hexdigest())
    n = len(ranked)
    train_end = round(n * ratios[0])
    val_end = train_end + round(n * ratios[1])
    val_end = min(val_end, n)
    return {
        doc: names[0] if i < train_end else names[1] if i < val_end else names[2]
        for i, doc in enumerate(ranked)
    }


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in sorted(rows, key=lambda r: tuple(str(v) for v in r.values())):
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_objects(folder, queries, corpus, qrels):
    write_jsonl(folder / "queries.jsonl", queries)
    write_jsonl(folder / "corpus.jsonl", corpus)
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "qrels.tsv").open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("query_id\tpassage_id\trelevance\n")
        for qid, pid in sorted(qrels):
            handle.write(f"{qid}\t{pid}\t1\n")


def prepare(input_path, output, seed, ratios, limits):
    raw_rows, invalid_json = [], []
    with input_path.open(encoding="utf-8-sig") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                raw_rows.append((line_no, json.loads(line)))
            except json.JSONDecodeError as exc:
                invalid_json.append({"line": line_no, "error": str(exc)})

    field_presence = collections.Counter()
    source_presence = collections.Counter()
    missing = collections.Counter()
    valid = []
    for line_no, row in raw_rows:
        for key in FIELDS:
            field_presence[key] += key in row and row[key] is not None
        source = row.get("source") if isinstance(row.get("source"), dict) else {}
        for key in SOURCE_FIELDS:
            source_presence[key] += key in source and source[key] is not None
        question, passage = clean(row.get("question")), clean(source.get("text"))
        if not question:
            missing["question"] += 1
        if not passage:
            missing["passage"] += 1
        if not question or not passage:
            continue
        doc_key = document_key(source)
        valid.append({
            "line": line_no, "question": question, "passage": passage,
            "answer": clean(row.get("answer")), "source": source,
            "query_id": fingerprint("q", question.casefold()),
            "passage_id": fingerprint("p", passage),
            "document_id": fingerprint("doc", doc_key),
        })

    q_counts = collections.Counter(r["query_id"] for r in valid)
    p_counts = collections.Counter(r["passage_id"] for r in valid)
    pair_counts = collections.Counter((r["query_id"], r["passage_id"]) for r in valid)
    queries = {r["query_id"]: {"query_id": r["query_id"], "text": r["question"]} for r in valid}
    corpus = {}
    for r in valid:
        source = r["source"]
        candidate = {
            "passage_id": r["passage_id"], "text": r["passage"],
            "document_id": r["document_id"], "citation": clean(source.get("citation")),
            "source": clean(source.get("source")), "url": clean(source.get("url")),
        }
        # Identical passage text can occur under multiple source records. Choose a
        # canonical owner independent of input row order, keeping it in one split.
        current = corpus.get(r["passage_id"])
        if current is None or candidate["document_id"] < current["document_id"]:
            corpus[r["passage_id"]] = candidate
    qrels = set(pair_counts)
    positives_per_query = collections.Counter(qid for qid, _ in qrels)
    doc_split = split_documents({p["document_id"] for p in corpus.values()}, seed, ratios)

    interim = output / "interim" / "embed"
    processed = output / "processed" / "embed"
    # Remove generated outputs individually so committed README/metadata files
    # inside the stage directories survive regeneration.
    for generated in (interim / "positive_only", interim / "audit"):
        if generated.exists():
            shutil.rmtree(generated)
    for generated in (processed / "train", processed / "validation", processed / "test"):
        if generated.exists():
            shutil.rmtree(generated)
    report_path = processed / "dataset_report.json"
    if report_path.exists():
        report_path.unlink()
    write_objects(interim / "positive_only", list(queries.values()), list(corpus.values()), qrels)

    # A query spanning documents is assigned to every applicable split, with only in-split positives.
    split_stats = {}
    for split in ("train", "validation", "test"):
        split_pids = {pid for pid, p in corpus.items() if doc_split[p["document_id"]] == split}
        split_qrels = {(qid, pid) for qid, pid in qrels if pid in split_pids}
        split_qids = {qid for qid, _ in split_qrels}
        write_objects(processed / split, [queries[q] for q in split_qids],
                      [corpus[p] for p in split_pids], split_qrels)
        split_stats[split] = {
            "documents": len({corpus[p]["document_id"] for p in split_pids}),
            "queries": len(split_qids), "passages": len(split_pids), "qrels": len(split_qrels),
        }

    audit_rows = []
    for r in valid:
        audit_rows.append({
            "input_line": r["line"], "query_id": r["query_id"], "passage_id": r["passage_id"],
            "document_id": corpus[r["passage_id"]]["document_id"],
            "split": doc_split[corpus[r["passage_id"]]["document_id"]],
            "answer": r["answer"], "source_version_id": clean(r["source"].get("version_id")),
        })
    write_jsonl(interim / "audit" / "records.jsonl", audit_rows)

    q_lengths = [tokens(q["text"]) for q in queries.values()]
    p_lengths = [tokens(p["text"]) for p in corpus.values()]
    report = {
        "input": str(input_path), "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "id_scheme": "prefix + first 20 hex chars of SHA-256 over normalized content/source key",
        "split_method": "documents ranked by SHA-256(seed + document_id), then partitioned by ratios",
        "seed": seed, "ratios": dict(zip(("train", "validation", "test"), ratios)),
        "rows": {"read": len(raw_rows), "valid": len(valid), "removed": len(raw_rows) - len(valid),
                 "invalid_json": len(invalid_json)},
        "field_presence": dict(field_presence), "source_field_presence": dict(source_presence),
        "missing_required": dict(missing),
        "duplicates": {
            "question_extra_rows": sum(n - 1 for n in q_counts.values() if n > 1),
            "passage_extra_rows": sum(n - 1 for n in p_counts.values() if n > 1),
            "pair_extra_rows": sum(n - 1 for n in pair_counts.values() if n > 1),
        },
        "unique": {"queries": len(queries), "passages": len(corpus), "qrels": len(qrels),
                   "documents": len(doc_split)},
        "multi_positive_queries": sum(n > 1 for n in positives_per_query.values()),
        "maximum_positives_per_query": max(positives_per_query.values(), default=0),
        "token_count_method": "regex approximation: Unicode words and punctuation; verify with the target model tokenizer",
        "query_token_lengths": length_summary(q_lengths, limits),
        "passage_token_lengths": length_summary(p_lengths, limits),
        "splits": split_stats,
        "document_overlap": {
            "train_validation": 0, "train_test": 0, "validation_test": 0
        },
        "licence": {
            "dataset": "Open Australian Legal Corpus licence (collection-level CC BY 4.0 plus source-specific terms)",
            "local_copy": "../../raw/embed/open-australian-legal-qa-v2.0.0/OPEN_AUSTRALIAN_LEGAL_CORPUS_LICENCE.md",
            "note": "Review source-specific conditions before training or redistribution.",
        },
        "provenance": {
            "dataset": "isaacus/open-australian-legal-qa", "version": "2.0.0",
            "doi": "10.57967/hf/1479", "source_dataset": "isaacus/open-australian-legal-corpus",
            "generation": "Questions and answers were synthesised by GPT-4 from sampled legal-document chunks.",
        },
        "invalid_json_details": invalid_json,
    }
    processed.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/raw/embed/open-australian-legal-qa-v2.0.0/qa.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data"), help="Root data directory")
    parser.add_argument("--seed", default="open-australian-legal-qa-v2")
    parser.add_argument("--ratios", type=float, nargs=3, default=(0.8, 0.1, 0.1),
                        metavar=("TRAIN", "VALIDATION", "TEST"))
    parser.add_argument("--context-limits", type=int, nargs="+", default=(512, 1024, 2048, 4096))
    args = parser.parse_args()
    if any(x < 0 for x in args.ratios) or not math.isclose(sum(args.ratios), 1.0):
        parser.error("--ratios must be non-negative and sum to 1")
    report = prepare(args.input, args.output, args.seed, tuple(args.ratios), args.context_limits)
    print(json.dumps({"output": str(args.output), **report["unique"], "splits": report["splits"]}, indent=2))


if __name__ == "__main__":
    main()

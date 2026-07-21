"""Teacher scoring and relevance-distribution distillation for Embed v4."""

from __future__ import annotations

import json
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path

from .io import load_retrieval_split
from .metrics import evaluate_rankings
from mini_kanon3.tracking import WandbTrainingCallback


def score_teacher_candidates(config: dict) -> dict:
    """Score one positive plus mined candidate negatives with a cross-encoder."""
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as exc:
        raise RuntimeError("Teacher scoring requires sentence-transformers") from exc
    split_dir, mined_path = Path(config["input_split"]), Path(config["candidate_negatives_path"])
    queries, corpus, qrels = load_retrieval_split(split_dir)
    mined = _load_jsonl_by_key(mined_path, "query_id")
    negatives_per_query = int(config["negatives_per_query"])
    records, pairs, spans = [], [], []
    for query_id in sorted(queries):
        negative_ids = [item["passage_id"] for item in mined[query_id]["negative_passages"]]
        positive_id = sorted(qrels[query_id])[0]
        candidate_ids = [positive_id, *negative_ids[:negatives_per_query]]
        if len(candidate_ids) != negatives_per_query + 1:
            raise ValueError(f"Insufficient teacher candidates for {query_id}")
        start = len(pairs)
        pairs.extend((queries[query_id], corpus[pid]) for pid in candidate_ids)
        spans.append((start, len(pairs)))
        records.append({"query_id": query_id, "positive_passage_id": positive_id,
                        "candidate_passage_ids": candidate_ids})
    teacher = CrossEncoder(config["teacher_model"], device=config.get("device"),
                           trust_remote_code=bool(config.get("trust_remote_code", True)))
    scores = teacher.predict(pairs, batch_size=int(config.get("batch_size", 32)),
                             show_progress_bar=True, convert_to_numpy=True)
    output_rows = []
    for record, (start, end) in zip(records, spans):
        candidates = [{"passage_id": pid, "teacher_score": round(float(score), 8),
                       "is_positive": index == 0}
                      for index, (pid, score) in enumerate(zip(record["candidate_passage_ids"], scores[start:end]))]
        output_rows.append({"query_id": record["query_id"], "candidates": candidates})
    output_path, report_path = Path(config["output_path"]), Path(config["report_path"])
    _write_jsonl(output_path, output_rows)
    report = {"schema_version": 1, "method": "cross_encoder_teacher_scoring",
              "teacher_model": config["teacher_model"], "input_split": str(split_dir),
              "candidate_negatives": str(mined_path), "output": str(output_path),
              "queries": len(output_rows), "candidates_per_query": negatives_per_query + 1,
              "positive_position": 0}
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


class DistillationTrainer:
    def __init__(self, config: dict):
        self.config = config
        self.seed = int(config["random_seed"])
        self.output_dir = Path(config["output_directory"])
        self.report_path = Path(config["report_path"])

    def train(self):
        try:
            import numpy as np
            import torch
            import torch.nn.functional as functional
            from sentence_transformers import SentenceTransformer, models, util
            from transformers import get_linear_schedule_with_warmup
        except ImportError as exc:
            raise RuntimeError("Distillation requires the project training dependencies") from exc
        random.seed(self.seed); np.random.seed(self.seed); torch.manual_seed(self.seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(self.seed)
        device = "cuda" if self.config.get("device", "auto") == "auto" and torch.cuda.is_available() else self.config.get("device", "cpu")
        model = SentenceTransformer(self.config["model_name"], device=device, trust_remote_code=True)
        model.max_seq_length = int(self.config["sequence_length"])
        if not any(isinstance(module, models.Normalize) for module in model._modules.values()):
            model.add_module("normalize", models.Normalize())
        queries, corpus, _ = load_retrieval_split(Path(self.config["train_queries"]).parent)
        records = list(_read_jsonl(Path(self.config["teacher_scores_path"])))
        _validate_teacher_records(records, queries, corpus)
        batch_size, epochs = int(self.config["batch_size"]), int(self.config["epochs"])
        total_steps = ((len(records) + batch_size - 1) // batch_size) * epochs
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(self.config["learning_rate"]),
                                      weight_decay=float(self.config["weight_decay"]))
        scheduler = get_linear_schedule_with_warmup(
            optimizer, round(total_steps * float(self.config["warmup_ratio"])), total_steps)
        teacher_temperature = float(self.config["teacher_temperature"])
        student_scale = float(self.config["student_scale"])
        alpha = float(self.config["distillation_weight"])
        history, started = [], time.perf_counter(); self.output_dir.mkdir(parents=True, exist_ok=True)
        tracker = WandbTrainingCallback(self.config.get("tracking", {}), self.config)
        global_step = 0
        for epoch in range(epochs):
            model.train(); random.Random(f"{self.seed}:{epoch}").shuffle(records); epoch_losses = []
            for batch_index, offset in enumerate(range(0, len(records), batch_size), 1):
                batch = records[offset:offset + batch_size]
                candidate_count = len(batch[0]["candidates"])
                query_texts = [queries[row["query_id"]] for row in batch]
                document_texts = [corpus[item["passage_id"]] for row in batch for item in row["candidates"]]
                qfeatures = util.batch_to_device(model.tokenize(query_texts), model.device)
                dfeatures = util.batch_to_device(model.tokenize(document_texts), model.device)
                qemb = model(qfeatures)["sentence_embedding"]
                demb = model(dfeatures)["sentence_embedding"].reshape(len(batch), candidate_count, -1)
                student_logits = torch.einsum("bd,bcd->bc", qemb, demb) * student_scale
                teacher_logits = torch.tensor([[item["teacher_score"] for item in row["candidates"]]
                                               for row in batch], device=model.device, dtype=student_logits.dtype)
                teacher_distribution = functional.softmax(teacher_logits / teacher_temperature, dim=1)
                distill_loss = functional.kl_div(functional.log_softmax(student_logits, dim=1),
                                                  teacher_distribution, reduction="batchmean")
                positive_targets = torch.zeros(len(batch), dtype=torch.long, device=model.device)
                contrastive_loss = functional.cross_entropy(student_logits, positive_targets)
                loss = alpha * distill_loss + (1.0 - alpha) * contrastive_loss
                optimizer.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(self.config["max_grad_norm"]))
                optimizer.step(); scheduler.step(); loss_value = float(loss.detach().cpu())
                epoch_losses.append(loss_value); global_step += 1
                tracker.log_train_step(global_step, epoch + 1, batch_index, loss_value,
                                       optimizer.param_groups[0]["lr"], torch)
            validation = _evaluate_model(model, Path(self.config["validation_queries"]).parent,
                                         int(self.config["validation_batch_size"]))
            history.append({"epoch": epoch + 1, "mean_training_loss": sum(epoch_losses) / len(epoch_losses),
                            "validation": validation})
            tracker.log_validation(global_step, epoch + 1, validation,
                                   history[-1]["mean_training_loss"])
            model.save_pretrained(str(self.output_dir / f"checkpoint-epoch-{epoch + 1}"))
            self._report(history, device, time.perf_counter() - started, False)
        model.save_pretrained(str(self.output_dir / "final"))
        report = self._report(history, device, time.perf_counter() - started, True)
        tracker.finish()
        return report

    def _report(self, history, device, elapsed, complete):
        report = {"schema_version": 1, "run": "embed_v4_distillation", "complete": complete,
                  "timestamp_utc": datetime.now(timezone.utc).isoformat(), "model": self.config["model_name"],
                  "device": device, "training_supervision": "teacher distribution + positive contrastive target",
                  "config": self.config, "history": history, "elapsed_seconds": round(elapsed, 3)}
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.report_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, self.report_path)
        return report


def _evaluate_model(model, split_dir, batch_size):
    import numpy as np
    queries, corpus, qrels = load_retrieval_split(split_dir)
    qids, pids = list(queries), list(corpus)
    qvec = getattr(model, "encode_query", model.encode)([queries[q] for q in qids], batch_size=batch_size,
                                                         normalize_embeddings=True, convert_to_numpy=True)
    pvec = getattr(model, "encode_document", model.encode)([corpus[p] for p in pids], batch_size=batch_size,
                                                            normalize_embeddings=True, convert_to_numpy=True)
    order = np.argsort(-(qvec @ pvec.T), axis=1)
    return evaluate_rankings({qid: [pids[index] for index in order[row]] for row, qid in enumerate(qids)}, qrels)


def _validate_teacher_records(records, queries, corpus):
    for row in records:
        if row["query_id"] not in queries: raise ValueError(f"Unknown teacher query: {row['query_id']}")
        if not row["candidates"] or not row["candidates"][0].get("is_positive"):
            raise ValueError(f"Positive candidate must be first for {row['query_id']}")
        if any(item["passage_id"] not in corpus for item in row["candidates"]):
            raise ValueError(f"Unknown teacher candidate for {row['query_id']}")
    sizes = {len(row["candidates"]) for row in records}
    if len(sizes) != 1: raise ValueError("Every teacher record must have the same candidate count")


def _load_jsonl_by_key(path, key): return {row[key]: row for row in _read_jsonl(path)}
def _read_jsonl(path):
    with path.open(encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip(): yield json.loads(line)
def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows: handle.write(json.dumps(row, sort_keys=True) + "\n")

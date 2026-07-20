"""Embed v1 training with cached in-batch negatives."""

from __future__ import annotations

import json
import math
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path

from .dataset import arrange_no_duplicate_batches, load_positive_groups, sample_one_positive_per_query
from .io import load_retrieval_split
from .metrics import evaluate_rankings


class EmbedTrainer:
    def __init__(self, config: dict):
        self.config = config
        self.seed = int(config["random_seed"])
        self.output_dir = Path(config["output_directory"])
        self.report_path = Path(config["report_path"])

    def train(self):
        try:
            import numpy as np
            import torch
            from sentence_transformers import InputExample, SentenceTransformer, losses, models, util
            from torch.utils.data import DataLoader
            from transformers import get_linear_schedule_with_warmup
        except ImportError as exc:
            raise RuntimeError("Install the project training dependencies with: pip install -e '.[train]'") from exc

        self._seed_everything(torch, np)
        device = self._resolve_device(torch)
        model = SentenceTransformer(self.config["model_name"], device=device, trust_remote_code=True)
        model.max_seq_length = int(self.config["sequence_length"])
        if not any(isinstance(module, models.Normalize) for module in model._modules.values()):
            model.add_module("normalize", models.Normalize())
        queries, corpus, positives = load_positive_groups(self._path("train_queries").parent)
        epochs = int(self.config["epochs"])
        batch_size = int(self.config["batch_size"])
        steps_per_epoch = math.ceil(len(queries) / batch_size)
        total_steps = steps_per_epoch * epochs
        warmup_steps = round(total_steps * float(self.config["warmup_ratio"]))
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(self.config["learning_rate"]),
                                      weight_decay=float(self.config["weight_decay"]))
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
        loss_model = losses.CachedMultipleNegativesRankingLoss(
            model=model, mini_batch_size=int(self.config["mini_batch_size"]),
            similarity_fct=util.dot_score, scale=float(self.config["loss_scale"])
        )
        use_amp = bool(self.config.get("mixed_precision", True)) and device.startswith("cuda")
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        history, started = [], time.perf_counter()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(epochs):
            model.train()
            pairs = sample_one_positive_per_query(queries, corpus, positives, self.seed, epoch)
            if self.config.get("sampling", {}).get("no_duplicates_per_batch", True):
                pairs = arrange_no_duplicate_batches(pairs, batch_size)
            examples = [InputExample(texts=[pair.query, pair.passage]) for pair in pairs]
            loader = DataLoader(examples, batch_size=batch_size, shuffle=False,
                                num_workers=int(self.config.get("num_workers", 0)),
                                collate_fn=model.smart_batching_collate, drop_last=False)
            losses_this_epoch = []
            for features, labels in loader:
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                    loss = loss_model(features, labels)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(self.config["max_grad_norm"]))
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                losses_this_epoch.append(float(loss.detach().cpu()))

            validation = self._evaluate(model, Path(self.config["validation_queries"]).parent,
                                        int(self.config["validation_batch_size"]))
            epoch_record = {"epoch": epoch + 1,
                            "mean_training_loss": sum(losses_this_epoch) / len(losses_this_epoch),
                            "validation": validation}
            history.append(epoch_record)
            if (epoch + 1) % int(self.config["checkpoint_every_epochs"]) == 0:
                model.save_pretrained(str(self.output_dir / f"checkpoint-epoch-{epoch + 1}"))
            self._write_report(history, device, time.perf_counter() - started, complete=False)

        model.save_pretrained(str(self.output_dir / "final"))
        return self._write_report(history, device, time.perf_counter() - started, complete=True)

    def _evaluate(self, model, split_dir: Path, batch_size: int):
        import numpy as np
        queries, corpus, qrels = load_retrieval_split(split_dir)
        query_ids, passage_ids = list(queries), list(corpus)
        encode_query = getattr(model, "encode_query", model.encode)
        encode_document = getattr(model, "encode_document", model.encode)
        qvec = encode_query([queries[key] for key in query_ids], batch_size=batch_size,
                            normalize_embeddings=True, convert_to_numpy=True)
        pvec = encode_document([corpus[key] for key in passage_ids], batch_size=batch_size,
                               normalize_embeddings=True, convert_to_numpy=True)
        order = np.argsort(-(qvec @ pvec.T), axis=1)
        rankings = {qid: [passage_ids[index] for index in order[row]]
                    for row, qid in enumerate(query_ids)}
        return evaluate_rankings(rankings, qrels)

    def _seed_everything(self, torch, np):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        torch.use_deterministic_algorithms(True, warn_only=True)

    def _resolve_device(self, torch):
        configured = str(self.config.get("device", "auto"))
        return ("cuda" if torch.cuda.is_available() else "cpu") if configured == "auto" else configured

    def _path(self, key):
        path = Path(self.config[key])
        if not path.exists():
            raise FileNotFoundError(f"Configured path does not exist: {path}")
        return path

    def _write_report(self, history, device, elapsed, complete):
        report = {"schema_version": 1, "run": "embed_v1", "complete": complete,
                  "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                  "model": self.config["model_name"], "device": device,
                  "training_supervision": "positive pairs + cached in-batch negatives",
                  "config": self.config, "history": history,
                  "elapsed_seconds": round(elapsed, 3)}
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.report_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, self.report_path)
        return report

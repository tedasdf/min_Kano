"""Optional experiment tracking with a no-op path when disabled."""

from __future__ import annotations

import os


class WandbTrainingCallback:
    """Log comparable training and validation metrics across Embed runs."""

    def __init__(self, config: dict, full_config: dict):
        tracking = config or {}
        self.enabled = bool(tracking.get("enabled", False))
        self.run = None
        if not self.enabled:
            return
        try:
            import wandb
        except ImportError as exc:
            raise RuntimeError("W&B tracking is enabled; install with pip install -e '.[tracking]'") from exc
        self.wandb = wandb
        self.run = wandb.init(
            project=tracking.get("project", "mini-kanon3"),
            entity=tracking.get("entity") or None,
            group=tracking.get("group", "embed-comparison"),
            name=tracking.get("run_name") or full_config.get("run_name"),
            job_type=tracking.get("job_type", "training"),
            tags=list(tracking.get("tags", [])),
            config=full_config,
            resume=tracking.get("resume", "allow"),
            id=tracking.get("run_id") or None,
        )

    def log_train_step(self, step: int, epoch: int, batch: int, loss: float, learning_rate: float,
                       torch_module=None):
        if not self.run:
            return
        metrics = {"train/loss": loss, "train/learning_rate": learning_rate,
                   "train/epoch": epoch, "train/batch": batch}
        if torch_module is not None and torch_module.cuda.is_available():
            metrics.update({
                "system/gpu_memory_allocated_mb": torch_module.cuda.memory_allocated() / 1024**2,
                "system/gpu_memory_reserved_mb": torch_module.cuda.memory_reserved() / 1024**2,
                "system/gpu_peak_memory_mb": torch_module.cuda.max_memory_allocated() / 1024**2,
            })
        self.wandb.log(metrics, step=step)

    def log_validation(self, step: int, epoch: int, metrics: dict, mean_loss: float):
        if not self.run:
            return
        payload = {f"validation/{name}": value for name, value in metrics.items()}
        payload.update({"epoch/mean_training_loss": mean_loss, "epoch/index": epoch})
        self.wandb.log(payload, step=step)
        for name, value in metrics.items():
            summary_name = f"best_validation/{name}"
            previous = self.run.summary.get(summary_name)
            if previous is None or value > previous:
                self.run.summary[summary_name] = value

    def finish(self):
        if self.run:
            self.run.finish()

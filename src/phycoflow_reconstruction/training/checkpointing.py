"""Atomic periodic `last`/`best` checkpoint management for every trainer."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from .preview import TrainingReconstructionPreview
from .run_store import RunStore, file_sha256


class PeriodicCheckpointManager:
    """Save resumable state and select best weights on a fixed preview sample."""

    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        store: RunStore,
        steps_per_epoch: int,
    ) -> None:
        settings = config.get("checkpointing", {})
        self.enabled = bool(settings.get("enabled", True))
        self.every_epochs = int(settings.get("every_epochs", 10))
        self.save_epoch_one = bool(settings.get("save_epoch_one", True))
        configured_epochs = settings.get("epochs")
        self.checkpoint_epochs = (
            frozenset(int(epoch) for epoch in configured_epochs)
            if configured_epochs is not None
            else None
        )
        self.steps_per_epoch = max(1, int(steps_per_epoch))
        self.store = store
        self.best_value = self._existing_best_value()

    def _existing_best_value(self) -> float:
        manifest_path = self.store.run_dir / "run_manifest.json"
        if not manifest_path.is_file():
            return math.inf
        value = json.loads(manifest_path.read_text()).get("best_metric", {}).get("value")
        return float(value) if value is not None else math.inf

    def due(self, global_step: int) -> bool:
        if not self.enabled or global_step % self.steps_per_epoch:
            return False
        epoch = global_step // self.steps_per_epoch
        if self.checkpoint_epochs is not None:
            return epoch in self.checkpoint_epochs
        return (self.save_epoch_one and epoch == 1) or epoch % self.every_epochs == 0

    def due_for_preview_or_checkpoint(
        self,
        global_step: int,
        preview: TrainingReconstructionPreview,
    ) -> bool:
        """Return whether either independent epoch cadence needs saved weights."""
        return self.due(global_step) or preview.due(global_step)

    def save(
        self,
        payload: Mapping[str, Any],
        *,
        model: torch.nn.Module,
        preview: TrainingReconstructionPreview,
        global_step: int,
        fallback_metric: float,
        force: bool = False,
    ) -> tuple[Path, Path | None] | None:
        """Atomically refresh last/latest and best when due or explicitly forced."""
        # Disabling periodic saves must never suppress the terminal recovery
        # checkpoint. ``force`` is used by every trainer at normal/truncated
        # termination so a completed run is always evaluable and resumable.
        milestone_due = self.due(global_step)
        if not force and not self.due_for_preview_or_checkpoint(global_step, preview):
            return None

        checkpoint = dict(payload)
        checkpoint["global_step"] = int(global_step)
        # Save first, then deliberately reload these exact bytes for the fixed
        # validation preview. This validates checkpoint readability while the
        # originating process and model are still available.
        last_path = self.store.save_checkpoint("last", checkpoint)
        milestone_path = None
        if milestone_due:
            epoch = global_step // self.steps_per_epoch
            milestone_path = self.store.save_checkpoint(f"epoch_{epoch:03d}", checkpoint)
        preview_report = preview.update(
            model,
            global_step=global_step,
            force=force,
            checkpoint_path=last_path,
        )
        metric_name = "training_loss"
        metric_value = float(fallback_metric)
        if preview_report is not None:
            preview_metric = preview_report.get("metrics", {}).get("mse_normalized")
            if preview_metric is not None:
                metric_name = "preview_mse_normalized"
                metric_value = float(preview_metric)

        # When qualitative validation is enabled, `best.pt` is selected only
        # at its configured validation cadence—not from incomparable training
        # mini-batch losses between previews.
        eligible_for_best = preview_report is not None or not preview.enabled
        improved = (
            eligible_for_best
            and math.isfinite(metric_value)
            and metric_value < self.best_value
        )
        if improved:
            self.best_value = metric_value
        checkpoint["checkpoint_metric"] = {
            "name": metric_name,
            "value": metric_value,
        }
        checkpoint["best_metric_value"] = self.best_value
        checkpoint["best_metric_name"] = metric_name
        # Keep the already validated `last.pt` bytes untouched so the preview
        # report's checkpoint hash remains exact. Selection metadata lives in
        # the run manifest and latest-checkpoint report; an improved best file
        # also carries it in its payload.
        best_path = self.store.save_checkpoint("best", checkpoint) if improved else None

        checkpoint_hashes = {
            "last": file_sha256(last_path),
            "latest": file_sha256(last_path),
        }
        if milestone_path is not None:
            checkpoint_hashes[f"epoch_{global_step // self.steps_per_epoch:03d}"] = file_sha256(
                milestone_path
            )
        existing_best = self.store.run_dir / "checkpoints" / "best.pt"
        if existing_best.is_file():
            checkpoint_hashes["best"] = file_sha256(existing_best)
        manifest_path = self.store.run_dir / "run_manifest.json"
        existing_hashes = json.loads(manifest_path.read_text(encoding="utf-8")).get(
            "checkpoint_hashes", {}
        )
        if isinstance(existing_hashes, Mapping):
            checkpoint_hashes = {**existing_hashes, **checkpoint_hashes}
        self.store.update_manifest(
            checkpoint_hashes=checkpoint_hashes,
            latest_checkpoint_step=int(global_step),
            latest_checkpoint_epoch=global_step / self.steps_per_epoch,
            best_metric={"name": metric_name, "value": self.best_value},
        )
        self.store.write_json(
            "evaluation/latest_checkpoint.json",
            {
                "global_step": int(global_step),
                "training_epoch": global_step / self.steps_per_epoch,
                "last": str(last_path.relative_to(self.store.run_dir)),
                "latest": "checkpoints/latest.pt",
                "milestone": (
                    str(milestone_path.relative_to(self.store.run_dir))
                    if milestone_path is not None
                    else None
                ),
                "best": (
                    str(existing_best.relative_to(self.store.run_dir))
                    if existing_best.is_file()
                    else None
                ),
                "selection_metric": {"name": metric_name, "value": metric_value},
                "best_metric_value": self.best_value,
                "best_updated": improved,
            },
        )
        self.store.set_status(
            "running",
            global_step=int(global_step),
            checkpoint_epoch=global_step / self.steps_per_epoch,
            latest_checkpoint=str(last_path.relative_to(self.store.run_dir)),
            best_metric={"name": metric_name, "value": self.best_value},
        )
        return last_path, best_path

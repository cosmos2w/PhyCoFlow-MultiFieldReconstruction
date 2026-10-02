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
    """Save periodic recovery/milestone state and validation-selected best weights."""

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
        self.every_steps = settings.get("every_steps")
        self.validation_every_epochs = settings.get("validation_every_epochs")
        self.save_epoch_one = bool(settings.get("save_epoch_one", False))
        configured_epochs = settings.get("epochs")
        self.checkpoint_epochs = (
            frozenset(int(epoch) for epoch in configured_epochs)
            if configured_epochs is not None
            else None
        )
        self.steps_per_epoch = max(1, int(steps_per_epoch))
        self.store = store
        self.selection_metric = str(
            settings.get("selection_metric", "native_validation_loss")
        )
        self.panel_selection = self.selection_metric in {
            "topology_with_fidelity", "coherence_with_fidelity"
        }
        self.best_name, self.best_value = self._existing_best_metric()
        self.last_validation_report: dict[str, Any] | None = None
        self.last_best_checked = False
        self.panel_evaluator = None
        manifest_path = self.store.run_dir / "run_manifest.json"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
        self.best_fidelity_value = float(manifest.get("best_panel_mse", math.inf))
        self.feasible_archive = list(manifest.get("coherence_feasible_archive", []))

    def restore_coherence_selector(self, checkpoint: Mapping[str, Any]) -> None:
        """Recover selection from the committed checkpoint, not a newer manifest."""
        if self.selection_metric != "coherence_with_fidelity":
            return
        state = checkpoint.get("coherence_selector_state")
        if not isinstance(state, Mapping):
            raise TypeError("coherence recovery checkpoint is missing selector state")
        archive = list(state["feasible_archive"])
        step = int(checkpoint["global_step"])
        if len(archive) > 3 or any(int(item["step"]) > step for item in archive):
            raise ValueError("invalid recovered coherence feasible archive")
        for item in archive:
            path = (self.store.run_dir / item["checkpoint"]).resolve()
            if path.parent != (self.store.run_dir / "checkpoints").resolve() or not path.is_file():
                raise ValueError("recovered coherence archive checkpoint is unavailable")
        self.feasible_archive = archive
        self.best_value = float(state["best_value"])
        self.best_fidelity_value = float(state["best_fidelity_value"])
        self.best_name = str(checkpoint["best_metric_name"])
        if archive:
            selected = min(archive, key=lambda item: (item["metric"], item["step"]))
            selected_checkpoint = self.store.load_checkpoint(Path(selected["checkpoint"]).stem)
            best_path = self.store.save_checkpoint("best", selected_checkpoint)
            selected_report = selected_checkpoint["coherence_selection_report"]
            self.store.write_json("evaluation/selected.json", {
                "global_step": int(selected["step"]),
                "is_source_baseline": int(selected["step"]) == 0,
                **selected_report,
            })
            manifest_path = self.store.run_dir / "run_manifest.json"
            hashes = json.loads(manifest_path.read_text()).get("checkpoint_hashes", {})
            self.store.update_manifest(checkpoint_hashes={**hashes, "best": file_sha256(best_path)})
        self.store.update_manifest(
            best_metric={"name": self.best_name, "value": self.best_value},
            best_panel_mse=self.best_fidelity_value,
            coherence_feasible_archive=archive,
        )

    def _existing_best_metric(self) -> tuple[str, float]:
        manifest_path = self.store.run_dir / "run_manifest.json"
        if not manifest_path.is_file():
            return "validation_loss", math.inf
        metric = json.loads(manifest_path.read_text()).get("best_metric", {})
        value = metric.get("value")
        return str(metric.get("name", "validation_loss")), (
            float(value) if value is not None else math.inf
        )

    def _epoch(self, global_step: int) -> int | None:
        if global_step % self.steps_per_epoch:
            return None
        return global_step // self.steps_per_epoch

    def last_due(self, global_step: int) -> bool:
        """Return whether the rolling recovery checkpoint should be refreshed."""
        epoch = self._epoch(global_step)
        if not self.enabled:
            return False
        if self.every_steps is not None and global_step > 0 and global_step % self.every_steps == 0:
            return True
        if epoch is None:
            return False
        return (self.save_epoch_one and epoch == 1) or epoch % self.every_epochs == 0

    def milestone_due(self, global_step: int) -> bool:
        """Return whether an immutable requested epoch checkpoint is due."""
        epoch = self._epoch(global_step)
        return (
            self.enabled
            and epoch is not None
            and self.checkpoint_epochs is not None
            and epoch in self.checkpoint_epochs
        )

    def due(self, global_step: int) -> bool:
        """Return whether any checkpoint file independent of validation is due."""
        if not self.enabled:
            return False
        return self.last_due(global_step) or self.milestone_due(global_step)

    def due_for_preview_or_checkpoint(
        self,
        global_step: int,
        preview: TrainingReconstructionPreview,
    ) -> bool:
        """Return whether either independent epoch cadence needs saved weights."""
        epoch = self._epoch(global_step)
        panel_due = (self.enabled and self.validation_every_epochs is not None and epoch is not None
                     and epoch % int(self.validation_every_epochs) == 0)
        return self.due(global_step) or preview.due(global_step) or panel_due

    def save(
        self,
        payload: Mapping[str, Any],
        *,
        model: torch.nn.Module,
        preview: TrainingReconstructionPreview,
        global_step: int,
        fallback_metric: float,
        force: bool = False,
    ) -> tuple[Path | None, Path | None] | None:
        """Refresh independently scheduled last, best, and milestone checkpoints."""
        # Disabling periodic saves must never suppress the terminal recovery
        # checkpoint. ``force`` is used by every trainer at normal/truncated
        # termination so a completed run is always evaluable and resumable.
        last_due = force or self.last_due(global_step)
        milestone_due = self.milestone_due(global_step)
        if not force and not self.due_for_preview_or_checkpoint(global_step, preview):
            return None

        checkpoint = dict(payload)
        checkpoint["global_step"] = int(global_step)
        preview_report = preview.update(
            model,
            global_step=global_step,
            force=force,
        )
        validation_report = (
            preview_report.get("validation") if preview_report is not None else None
        )
        reconstruction_report = (
            preview_report.get("reconstruction") if preview_report is not None else None
        )
        self.last_validation_report = validation_report
        metric_name = "training_loss"
        metric_value = float(fallback_metric)
        panel_report = None
        fidelity_improved = False
        epoch = self._epoch(global_step)
        panel_due = (force or self.validation_every_epochs is None or
                     (epoch is not None and epoch % int(self.validation_every_epochs) == 0))
        if self.panel_selection and not panel_due:
            eligible_for_best = False
        elif self.panel_selection:
            if self.panel_evaluator is None:
                raise ValueError("topology checkpoint selection requires a fixed-panel evaluator")
            panel_report = self.panel_evaluator(global_step)
            metric_name = f"fixed_panel_{self.selection_metric}"
            metric_value = float(panel_report["metric"])
            eligible_for_best = bool(panel_report["eligible"])
            mse = float(panel_report["mse"])
            fidelity_improved = math.isfinite(mse) and mse < self.best_fidelity_value
            if fidelity_improved:
                self.best_fidelity_value = mse
            self.last_validation_report = {"global_step": global_step, "loss": metric_value,
                "training_epoch": global_step / self.steps_per_epoch,
                "topology_selection_eligible": eligible_for_best}
            history_name = ("coherence_validation" if self.selection_metric == "coherence_with_fidelity"
                            else "topology_validation")
            with (self.store.run_dir / "metrics" / f"{history_name}.jsonl").open("a") as handle:
                handle.write(json.dumps({"step": global_step, **panel_report}) + "\n")
        elif self.selection_metric == "reconstruction_mse":
            if reconstruction_report is not None:
                metric_name = "fixed_validation_reconstruction_mse"
                metric_value = float(reconstruction_report["metrics"]["mse_normalized"])
            eligible_for_best = reconstruction_report is not None
        else:
            if validation_report is not None:
                metric_name = "native_validation_loss"
                metric_value = float(validation_report["loss"])
            # The training-loss fallback remains solely for old configurations
            # that explicitly disabled previews.
            eligible_for_best = validation_report is not None or not preview.enabled

        self.last_best_checked = panel_report is not None or eligible_for_best
        improved = (
            eligible_for_best
            and math.isfinite(metric_value)
            and metric_value < self.best_value
        )
        if improved:
            self.best_name = metric_name
            self.best_value = metric_value
        checkpoint["checkpoint_metric"] = {
            "name": metric_name,
            "value": metric_value,
        }
        checkpoint["best_metric_value"] = self.best_value
        checkpoint["best_metric_name"] = self.best_name
        if panel_report is not None:
            checkpoint["topology_selection"] = {key: value for key, value in panel_report.items()
                                                if key != "metrics"}
            if self.selection_metric == "coherence_with_fidelity":
                checkpoint["coherence_selection"] = checkpoint["topology_selection"]
                # The archived payload can restore the complete selected
                # report after a crash leaves a newer selected.json behind.
                checkpoint["coherence_selection_report"] = panel_report
                if bool(panel_report["eligible"]) and math.isfinite(metric_value):
                    entry = {"step": global_step, "metric": metric_value,
                             "mse": float(panel_report["mse"]),
                             "family_scores": panel_report["family_source_normalized_scores"],
                             "checkpoint": f"checkpoints/feasible_{global_step:07d}.pt"}
                    previous = list(self.feasible_archive)
                    candidates = [item for item in previous if item["step"] != global_step] + [entry]
                    self.feasible_archive = sorted(candidates, key=lambda item: (item["metric"], item["step"]))[:3]
                checkpoint["coherence_selector_state"] = {"feasible_archive": self.feasible_archive,
                                                          "best_value": self.best_value,
                                                          "best_fidelity_value": self.best_fidelity_value}
                if (bool(panel_report["eligible"]) and math.isfinite(metric_value)
                        and entry in self.feasible_archive):
                    self.store.save_checkpoint(f"feasible_{global_step:07d}", checkpoint)
        last_path = self.store.save_checkpoint("last", checkpoint) if last_due else None
        if last_path is not None and self.selection_metric == "coherence_with_fidelity":
            # Retain the previous committed archive until last.pt is atomic.
            keep = {item["checkpoint"] for item in self.feasible_archive}
            for path in (self.store.run_dir / "checkpoints").glob("feasible_*.pt"):
                if str(path.relative_to(self.store.run_dir)) not in keep:
                    path.unlink()
        milestone_path = None
        if milestone_due:
            epoch = global_step // self.steps_per_epoch
            milestone_path = self.store.save_checkpoint(f"epoch_{epoch:03d}", checkpoint)
        best_path = self.store.save_checkpoint("best", checkpoint) if improved else None
        best_fidelity_path = (
            self.store.save_checkpoint("best_fidelity", checkpoint)
            if fidelity_improved or (improved and self.selection_metric == "reconstruction_mse")
            else None
        )
        if improved and panel_report is not None:
            self.store.write_json("evaluation/selected.json", {
                "global_step": global_step, "is_source_baseline": global_step == 0,
                **panel_report,
            })

        checkpoint_hashes = {}
        if last_path is not None:
            checkpoint_hashes["last"] = file_sha256(last_path)
        if milestone_path is not None:
            checkpoint_hashes[f"epoch_{global_step // self.steps_per_epoch:03d}"] = file_sha256(
                milestone_path
            )
        existing_best = self.store.run_dir / "checkpoints" / "best.pt"
        if existing_best.is_file():
            checkpoint_hashes["best"] = file_sha256(existing_best)
        existing_best_fidelity = self.store.run_dir / "checkpoints" / "best_fidelity.pt"
        if existing_best_fidelity.is_file():
            checkpoint_hashes["best_fidelity"] = file_sha256(existing_best_fidelity)
        manifest_path = self.store.run_dir / "run_manifest.json"
        existing_hashes = json.loads(manifest_path.read_text(encoding="utf-8")).get(
            "checkpoint_hashes", {}
        )
        if isinstance(existing_hashes, Mapping):
            checkpoint_hashes = {**existing_hashes, **checkpoint_hashes}
        manifest_details: dict[str, Any] = {
            "checkpoint_hashes": checkpoint_hashes,
            "best_metric": {"name": self.best_name, "value": self.best_value},
        }
        if panel_report is not None:
            manifest_details["best_panel_mse"] = self.best_fidelity_value
            if self.selection_metric == "coherence_with_fidelity":
                manifest_details["coherence_feasible_archive"] = self.feasible_archive
        if last_path is not None:
            manifest_details.update(
                last_checkpoint_step=int(global_step),
                last_checkpoint_epoch=global_step / self.steps_per_epoch,
            )
        self.store.update_manifest(**manifest_details)
        existing_last = self.store.run_dir / "checkpoints" / "last.pt"
        self.store.write_json(
            "evaluation/checkpoint_status.json",
            {
                "global_step": int(global_step),
                "training_epoch": global_step / self.steps_per_epoch,
                "last": (
                    str(existing_last.relative_to(self.store.run_dir))
                    if existing_last.is_file()
                    else None
                ),
                "last_updated": last_path is not None,
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
                "best_fidelity": (
                    str(existing_best_fidelity.relative_to(self.store.run_dir))
                    if existing_best_fidelity.is_file()
                    else None
                ),
                "selection_metric": {"name": metric_name, "value": metric_value},
                "best_metric_name": self.best_name,
                "best_metric_value": self.best_value,
                "best_updated": improved,
                "best_fidelity_updated": best_fidelity_path is not None,
            },
        )
        self.store.set_status(
            "running",
            global_step=int(global_step),
            checkpoint_epoch=global_step / self.steps_per_epoch,
            last_checkpoint=(
                str(existing_last.relative_to(self.store.run_dir))
                if existing_last.is_file()
                else None
            ),
            best_metric={"name": self.best_name, "value": self.best_value},
        )
        return last_path, best_path

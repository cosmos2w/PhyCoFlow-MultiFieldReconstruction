"""Periodic checkpoint-backed sparse-reconstruction previews during training."""

from __future__ import annotations

import json
import math
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..contracts import ObservationBatch
from ..data.factory import open_field_dataset
from ..data.sensor_protocols import build_observation_batch
from ..evaluation import reconstruction_metrics
from .common import sensor_protocol_from_config
from .model_lifecycle import (
    add_training_aux_state,
    evaluation_weight_context,
    load_training_aux_state,
)
from .run_store import (
    RunStore,
    checkpoint_model_state,
    file_sha256,
    load_model_state_strict,
    load_project_checkpoint,
)


def _physical_observations(batch: ObservationBatch, normalizer) -> torch.Tensor:
    field_ids = batch.obs_field_ids[0].cpu()
    values = batch.obs_values[0, :, 0].cpu()
    return values * normalizer.scale[field_ids] + normalizer.offset[field_ids]


def _relative_l2_error(estimate: np.ndarray, truth: np.ndarray) -> float | None:
    """Return ||estimate - truth||_2 / ||truth||_2, or None for a zero reference."""
    estimate64 = np.asarray(estimate, dtype=np.float64)
    truth64 = np.asarray(truth, dtype=np.float64)
    denominator = float(np.linalg.norm(truth64.ravel()))
    if denominator == 0.0 or not np.isfinite(denominator):
        return None
    value = float(np.linalg.norm((estimate64 - truth64).ravel()) / denominator)
    return value if np.isfinite(value) else None


def _absolute_error_title(relative_l2: float | None) -> str:
    metric = "N/A" if relative_l2 is None else f"{relative_l2:.3e}"
    return f"Absolute error\nRelative $L_2$ = {metric}"


def _plot_preview(
    path_stem: Path,
    *,
    prediction: np.ndarray,
    target: np.ndarray,
    query_coords: np.ndarray,
    obs_coords: np.ndarray,
    obs_values: np.ndarray,
    obs_fields: np.ndarray,
    obs_valid: np.ndarray,
    field_names: tuple[str, ...],
    logical_shape: tuple[int, ...],
    epoch: float,
) -> tuple[Path, ...]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["svg.fonttype"] = "none"
    fields = len(field_names)
    figure, axes = plt.subplots(
        fields,
        3,
        figsize=(12.0, max(3.2, 2.8 * fields)),
        squeeze=False,
        constrained_layout=True,
    )
    complete_grid = prediction.shape[0] == math.prod(logical_shape)

    for field_index, field_name in enumerate(field_names):
        truth = target[:, field_index]
        estimate = prediction[:, field_index]
        error = np.abs(estimate - truth)
        error_title = _absolute_error_title(_relative_l2_error(estimate, truth))
        field_sensor_mask = obs_valid & (obs_fields == field_index)

        if len(logical_shape) == 1 and complete_grid:
            x = query_coords[:, 0]
            order = np.argsort(x)
            panels = ((truth, "Target"), (estimate, "Reconstruction"), (error, error_title))
            for column, (values, title) in enumerate(panels):
                axes[field_index, column].plot(x[order], values[order], linewidth=1.4)
                axes[field_index, column].set_title(title)
            axes[field_index, 1].scatter(
                obs_coords[field_sensor_mask, 0],
                obs_values[field_sensor_mask],
                s=15,
                facecolors="none",
                edgecolors="black",
                linewidths=0.8,
                label="sensors",
                zorder=3,
            )
        elif len(logical_shape) == 2 and complete_grid:
            truth_grid = truth.reshape(logical_shape)
            estimate_grid = estimate.reshape(logical_shape)
            error_grid = error.reshape(logical_shape)
            low = float(min(truth.min(), estimate.min()))
            high = float(max(truth.max(), estimate.max()))
            panels = (
                (truth_grid, "Target", "viridis", low, high),
                (estimate_grid, "Reconstruction", "viridis", low, high),
                (error_grid, error_title, "magma", 0.0, None),
            )
            for column, (values, title, cmap, vmin, vmax) in enumerate(panels):
                image = axes[field_index, column].imshow(
                    values,
                    origin="lower",
                    aspect="auto",
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                )
                axes[field_index, column].set_title(title)
                figure.colorbar(image, ax=axes[field_index, column], fraction=0.046, pad=0.03)
            if field_sensor_mask.any():
                rows, columns = logical_shape
                axes[field_index, 1].scatter(
                    obs_coords[field_sensor_mask, 0] * max(columns - 1, 1),
                    obs_coords[field_sensor_mask, 1] * max(rows - 1, 1),
                    s=8,
                    facecolors="none",
                    edgecolors="white",
                    linewidths=0.5,
                    label="sensors",
                )
        else:
            x = query_coords[:, 0]
            y = query_coords[:, 1] if query_coords.shape[1] > 1 else np.zeros_like(x)
            panels = (
                (truth, "Target", "viridis"),
                (estimate, "Reconstruction", "viridis"),
                (error, error_title, "magma"),
            )
            for column, (values, title, cmap) in enumerate(panels):
                points = axes[field_index, column].scatter(x, y, c=values, s=8, cmap=cmap)
                axes[field_index, column].set_title(title)
                figure.colorbar(points, ax=axes[field_index, column], fraction=0.046, pad=0.03)

        axes[field_index, 0].set_ylabel(field_name)
        for axis in axes[field_index]:
            axis.tick_params(labelsize=8)
        if field_sensor_mask.any():
            axes[field_index, 1].legend(loc="best", frameon=False, fontsize=7)

    figure.suptitle(f"Sparse reconstruction preview — epoch {epoch:.3f}")
    outputs = tuple(path_stem.with_suffix(suffix) for suffix in (".png", ".svg", ".pdf"))
    for output in outputs:
        figure.savefig(output, dpi=160)
    plt.close(figure)
    return outputs


def render_preview_payload(
    payload_path: str | Path,
    *,
    output_stem: str | Path | None = None,
    epoch: float = 0.0,
) -> tuple[Path, ...]:
    """Re-render a saved training preview without model inference."""
    payload_path = Path(payload_path)
    with np.load(payload_path, allow_pickle=False) as payload:
        return _plot_preview(
            Path(output_stem) if output_stem is not None else payload_path.with_suffix(""),
            prediction=payload["prediction_physical"],
            target=payload["target_physical"],
            query_coords=payload["query_coords"],
            obs_coords=payload["obs_coords"],
            obs_values=payload["obs_values_physical"],
            obs_fields=payload["obs_field_ids"],
            obs_valid=payload["obs_valid_mask"].astype(bool),
            field_names=tuple(str(value) for value in payload["field_names"]),
            logical_shape=tuple(int(value) for value in payload["logical_shape"]),
            epoch=float(epoch),
        )


class TrainingReconstructionPreview:
    """Reload a periodic checkpoint and refresh one fixed validation preview."""

    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        store: RunStore,
        steps_per_epoch: int,
        device: torch.device,
    ) -> None:
        settings = config.get("evaluation", {}).get("preview", {})
        self.enabled = bool(settings.get("enabled", False))
        self.store = store
        self.steps_per_epoch = max(1, int(steps_per_epoch))
        self.device = device
        self.settings = settings
        self.dataset = None
        self.batch = None
        self.output_dir = store.run_dir / "evaluation" / "training_preview"
        if not self.enabled:
            return

        split = str(settings.get("split", "validation"))
        self.dataset = open_field_dataset(config["dataset"], split=split)
        sample_index = int(settings.get("sample_index", 0))
        if not 0 <= sample_index < len(self.dataset):
            raise IndexError(
                f"evaluation.preview.sample_index={sample_index} is outside {split} split"
            )
        query_points = settings.get("query_points")
        sample = self.dataset[sample_index]
        self.batch = build_observation_batch(
            [sample],
            sensor_protocol_from_config(
                config, seed_offset=int(settings.get("seed", 2027))
            ),
            query_points=None if query_points is None else int(query_points),
        ).to(device)
        # The preview batch is now self-contained. Close the lazy HDF5 handle
        # before DataLoader workers may fork so no unrelated descriptor is
        # inherited by the asynchronous training path.
        self.dataset.close()
        self.generation_steps = int(
            settings.get(
                "generation_steps",
                config.get("evaluation", {}).get("generation_steps", 2),
            )
        )
        self.every_epochs = int(settings.get("every_epochs", 10))
        self.keep_history = bool(settings.get("keep_history", False))
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def due(self, global_step: int) -> bool:
        if not self.enabled or global_step % self.steps_per_epoch:
            return False
        epoch = global_step // self.steps_per_epoch
        return epoch == 1 or epoch % self.every_epochs == 0

    def update(
        self,
        model: torch.nn.Module,
        *,
        global_step: int,
        force: bool = False,
        checkpoint_path: Path | None = None,
    ) -> dict[str, Any] | None:
        if not self.enabled or (not force and not self.due(global_step)):
            return None
        assert self.dataset is not None and self.batch is not None
        if checkpoint_path is None:
            checkpoint_payload = {
                "model": checkpoint_model_state(model),
                "global_step": int(global_step),
                "config_sha256": self.store.config_hash,
                "purpose": "qualitative_training_preview",
            }
            checkpoint_path = self.store.save_checkpoint(
                "preview_latest",
                add_training_aux_state(checkpoint_payload, model),
            )
        checkpoint = load_project_checkpoint(checkpoint_path)
        load_model_state_strict(model, checkpoint["model"])
        load_training_aux_state(model, checkpoint)
        was_training = model.training
        model.eval()
        seed = int(self.settings.get("seed", 2027)) + int(global_step)
        with evaluation_weight_context(model), torch.no_grad():
            generator = torch.Generator(device=self.device).manual_seed(seed)
            reconstruction = model.reconstruct(
                self.batch,
                steps=self.generation_steps,
                generator=generator,
            )
        if was_training:
            model.train()

        target = self.batch.target_fields
        if target is None:
            raise ValueError("training preview requires dense validation targets")
        prediction_physical = self.dataset.normalizer.decode(
            reconstruction.prediction[0]
        ).detach().cpu()
        target_physical = self.dataset.normalizer.decode(target[0]).detach().cpu()
        epoch = global_step / self.steps_per_epoch
        npz_path = self.output_dir / "latest_reconstruction.npz"
        np.savez_compressed(
            npz_path,
            prediction_physical=prediction_physical.numpy(),
            target_physical=target_physical.numpy(),
            query_coords=self.batch.query_coords[0].detach().cpu().numpy(),
            obs_coords=self.batch.obs_coords[0].detach().cpu().numpy(),
            obs_values_physical=_physical_observations(
                self.batch, self.dataset.normalizer
            ).numpy(),
            obs_field_ids=self.batch.obs_field_ids[0].detach().cpu().numpy(),
            obs_valid_mask=self.batch.obs_valid_mask[0].detach().cpu().numpy(),
            logical_shape=np.asarray(self.dataset.data_spec.logical_shape),
            field_names=np.asarray(self.dataset.field_names),
        )
        figure_paths = render_preview_payload(
            npz_path,
            output_stem=self.output_dir / "latest_reconstruction",
            epoch=epoch,
        )
        metrics = reconstruction_metrics(
            reconstruction.prediction,
            target,
            self.batch,
            self.dataset.field_names,
        )
        physical_squared_error = (prediction_physical - target_physical).square()
        metrics["mse_physical"] = float(physical_squared_error.mean())
        metrics["per_field_mse_physical"] = {
            name: float(physical_squared_error[:, field_index].mean())
            for field_index, name in enumerate(self.dataset.field_names)
        }
        metrics["per_field_relative_l2_physical"] = {
            name: _relative_l2_error(
                prediction_physical[:, field_index].numpy(),
                target_physical[:, field_index].numpy(),
            )
            for field_index, name in enumerate(self.dataset.field_names)
        }
        report = {
            "global_step": int(global_step),
            "training_epoch": epoch,
            "sample_id": self.batch.sample_ids[0],
            "checkpoint": str(checkpoint_path.relative_to(self.store.run_dir)),
            "checkpoint_sha256": file_sha256(checkpoint_path),
            "generation_steps": self.generation_steps,
            "metrics": metrics,
            "figures": {
                path.suffix.removeprefix("."): str(path.relative_to(self.store.run_dir))
                for path in figure_paths
            },
            "payload": str(npz_path.relative_to(self.store.run_dir)),
        }
        report_path = self.output_dir / "latest_metrics.json"
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        contract = self.output_dir / "figure_contract.md"
        contract.write_text(
            "# Training reconstruction preview\n\n"
            "- **Claim:** qualitative sparse-reconstruction quality of the latest saved "
            "training checkpoint.\n"
            f"- **Checkpoint:** `{report['checkpoint']}`\n"
            f"- **Sample:** `{report['sample_id']}` from the configured preview split.\n"
            "- **Panels:** physical target, checkpoint reconstruction, and absolute error; "
            "each error panel reports its field-wise relative L2 error, and white circles "
            "mark conditioned sensors.\n"
            f"- **Metrics:** `{report_path.name}`; reusable arrays: `{npz_path.name}`.\n"
            "- **Caveat:** this fixed-sample diagnostic is not an aggregate benchmark.\n",
            encoding="utf-8",
        )
        if self.keep_history:
            history = self.output_dir / "history" / f"epoch_{epoch:010.3f}"
            history.mkdir(parents=True, exist_ok=True)
            for path in (*figure_paths, npz_path, report_path):
                shutil.copy2(path, history / path.name)
        self.store.update_manifest(training_preview=report)
        return report

    def close(self) -> None:
        if self.dataset is not None:
            self.dataset.close()

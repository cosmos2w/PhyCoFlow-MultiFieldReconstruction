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
from .model_lifecycle import evaluation_weight_context
from .run_store import RunStore


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


def _coordinates_in_dataset_units(
    coordinates: np.ndarray,
    normalized_reference: np.ndarray,
    physical_reference: np.ndarray,
) -> np.ndarray:
    """Undo the dataset's per-axis min-max coordinate scaling for display."""
    coordinates64 = np.asarray(coordinates, dtype=np.float64)
    normalized64 = np.asarray(normalized_reference, dtype=np.float64)
    physical64 = np.asarray(physical_reference, dtype=np.float64)
    if normalized64.ndim != 2 or physical64.shape != normalized64.shape:
        raise ValueError("reference coordinate arrays must align as [points, dimensions]")
    if coordinates64.ndim != 2 or coordinates64.shape[1] != normalized64.shape[1]:
        raise ValueError("display coordinates must align with the dataset coordinate dimension")
    result = np.empty_like(coordinates64)
    for dimension in range(coordinates64.shape[1]):
        normalized_min = float(normalized64[:, dimension].min())
        normalized_span = float(np.ptp(normalized64[:, dimension]))
        physical_min = float(physical64[:, dimension].min())
        physical_span = float(np.ptp(physical64[:, dimension]))
        if normalized_span <= np.finfo(np.float64).eps:
            result[:, dimension] = physical_min
        else:
            result[:, dimension] = physical_min + (
                (coordinates64[:, dimension] - normalized_min)
                * physical_span
                / normalized_span
            )
    return result.astype(np.float32)


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
    field_units: tuple[str, ...] | None = None,
    coordinate_space: str = "normalized",
    sample_id: str | None = None,
) -> tuple[Path, ...]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    plt.rcParams["svg.fonttype"] = "none"
    fields = len(field_names)
    if prediction.shape != target.shape or prediction.ndim != 2:
        raise ValueError("preview fields must align as [query_points, fields]")
    if prediction.shape[1] != fields or len(logical_shape) not in {1, 2}:
        raise ValueError("preview fields and logical grid metadata do not align")
    if query_coords.shape[0] != prediction.shape[0] or obs_coords.shape[1] != query_coords.shape[1]:
        raise ValueError("preview query and observation coordinates do not align")
    if obs_values.shape != obs_fields.shape or obs_fields.shape != obs_valid.shape:
        raise ValueError("preview observations do not align")
    units = field_units or ("unknown",) * fields
    if len(units) != fields:
        raise ValueError("preview field units do not align with field names")

    coordinate_dim = query_coords.shape[1]
    if coordinate_dim == 1:
        coordinate_names = ("coordinate",)
    else:
        coordinate_names = ("x coordinate", "y coordinate")
    suffix = (
        " (normalized)"
        if coordinate_space == "normalized"
        else " (dataset units)"
    )
    axis_labels = tuple(f"{name}{suffix}" for name in coordinate_names)

    figure, axes = plt.subplots(
        fields,
        3,
        figsize=(12.6, max(2.8, 2.55 * fields)),
        squeeze=False,
        layout="constrained",
        gridspec_kw={"wspace": 0.18, "hspace": 0.16},
    )
    complete_grid = prediction.shape[0] == math.prod(logical_shape)

    if len(logical_shape) == 2 and coordinate_dim >= 2 and complete_grid:
        coordinate_grid = query_coords.reshape(*logical_shape, coordinate_dim)
        x_grid = coordinate_grid[..., 0]
        y_grid = coordinate_grid[..., 1]
        rectilinear = np.allclose(x_grid, x_grid[:1, :], rtol=1e-5, atol=1e-7) and np.allclose(
            y_grid, y_grid[:, :1], rtol=1e-5, atol=1e-7
        )
    else:
        coordinate_grid = None
        rectilinear = False

    for field_index, field_name in enumerate(field_names):
        truth = target[:, field_index]
        estimate = prediction[:, field_index]
        error = np.abs(estimate - truth)
        error_title = _absolute_error_title(_relative_l2_error(estimate, truth))
        field_sensor_mask = obs_valid & (obs_fields == field_index)
        field_low = float(min(np.min(truth), np.min(estimate)))
        field_high = float(max(np.max(truth), np.max(estimate)))
        if field_high <= field_low:
            field_padding = max(abs(field_low), 1.0) * 1e-6
            field_low -= field_padding
            field_high += field_padding
        error_high = max(float(np.max(error)), np.finfo(np.float64).eps)
        unit = units[field_index]
        unit_label = (
            f" [{unit}]" if unit and unit.lower() != "unknown" else " [units not specified]"
        )
        error_colorbar = ScalarMappable(
            norm=Normalize(vmin=0.0, vmax=error_high), cmap="YlOrRd"
        )

        if len(logical_shape) == 1:
            x = query_coords[:, 0]
            order = np.argsort(x)
            panels = ((truth, "Target"), (estimate, "Reconstruction"), (error, error_title))
            for column, (values, title) in enumerate(panels):
                axis = axes[field_index, column]
                if complete_grid:
                    axis.plot(
                        x[order],
                        values[order],
                        color="#2878A5" if column < 2 else "#9C3D36",
                        linewidth=1.1,
                    )
                else:
                    axis.scatter(
                        x,
                        values,
                        s=10,
                        alpha=0.75,
                        color="#2878A5" if column < 2 else "#9C3D36",
                        edgecolors="none",
                        rasterized=True,
                    )
                if column < 2 and field_index == 0:
                    axis.set_title(title)
            if field_sensor_mask.any():
                axes[field_index, 1].scatter(
                    obs_coords[field_sensor_mask, 0],
                    obs_values[field_sensor_mask],
                    s=20,
                    marker="o",
                    facecolors="white",
                    edgecolors="#1D2933",
                    linewidths=0.65,
                    label="conditioned observations",
                    zorder=3,
                )
            axes[field_index, 0].set_ylim(field_low, field_high)
            axes[field_index, 1].set_ylim(field_low, field_high)
            axes[field_index, 2].set_ylim(0.0, error_high)
            axes[field_index, 2].set_title(error_title)
            axes[field_index, 0].set_xlabel(axis_labels[0])
            axes[field_index, 1].set_xlabel(axis_labels[0])
            axes[field_index, 2].set_xlabel(axis_labels[0])
            axes[field_index, 0].set_ylabel(f"{field_name}{unit_label}\nfield value")
            axes[field_index, 2].set_title(error_title)
            if field_sensor_mask.any() and field_index == 0:
                axes[field_index, 1].legend(frameon=False, fontsize=7.5, loc="best")
        else:
            is_spatial = len(logical_shape) == 2 and coordinate_dim >= 2
            x = coordinate_grid[..., 0] if rectilinear and coordinate_grid is not None else query_coords[:, 0]
            y = coordinate_grid[..., 1] if rectilinear and coordinate_grid is not None else (
                query_coords[:, 1] if coordinate_dim >= 2 else np.zeros_like(x)
            )
            panel_values = (truth, estimate, error)
            panel_titles = ("Target", "Reconstruction", error_title)
            for column, (values, panel_title) in enumerate(zip(panel_values, panel_titles)):
                axis = axes[field_index, column]
                if rectilinear:
                    image = axis.pcolormesh(
                        x,
                        y,
                        values.reshape(logical_shape),
                        shading="auto",
                        cmap="viridis" if column < 2 else "YlOrRd",
                        vmin=field_low if column < 2 else 0.0,
                        vmax=field_high if column < 2 else error_high,
                        rasterized=True,
                    )
                else:
                    image = axis.scatter(
                        x,
                        y,
                        c=values,
                        s=5.0 if not complete_grid else 8.0,
                        alpha=0.82 if column < 2 else 0.76,
                        cmap="viridis" if column < 2 else "YlOrRd",
                        vmin=field_low if column < 2 else 0.0,
                        vmax=field_high if column < 2 else error_high,
                        linewidths=0.0,
                        rasterized=True,
                    )
                if is_spatial:
                    if field_index == len(field_names) - 1:
                        axis.set_xlabel(axis_labels[0])
                    axis.set_ylabel(axis_labels[1] if column == 0 else "")
                    if not rectilinear:
                        axis.set_xlim(float(np.min(x)), float(np.max(x)))
                        axis.set_ylim(float(np.min(y)), float(np.max(y)))
                    axis.set_aspect("equal", adjustable="box")
                if column < 2 and field_index == 0:
                    axis.set_title(panel_title)
                elif column == 2:
                    axis.set_title(error_title)
            if is_spatial:
                figure.colorbar(
                    ScalarMappable(norm=Normalize(vmin=field_low, vmax=field_high), cmap="viridis"),
                    ax=axes[field_index, :2],
                    fraction=0.035,
                    pad=0.025,
                    label=f"Field value{unit_label}",
                )
                figure.colorbar(
                    error_colorbar,
                    ax=axes[field_index, 2],
                    fraction=0.055,
                    pad=0.03,
                    label=f"Absolute error{unit_label}",
                )
                if field_sensor_mask.any():
                    axes[field_index, 1].scatter(
                        obs_coords[field_sensor_mask, 0],
                        obs_coords[field_sensor_mask, 1],
                        s=8,
                        marker="o",
                        facecolors="none",
                        edgecolors="#1D2933",
                        linewidths=0.35,
                        alpha=0.62,
                        label="conditioned observations",
                        zorder=3,
                    )
                    if field_index == 0:
                        axes[field_index, 1].legend(
                            frameon=True,
                            facecolor="white",
                            edgecolor="none",
                            framealpha=0.88,
                            fontsize=7.5,
                            loc="upper right",
                        )
            else:
                figure.colorbar(image, ax=axes[field_index, 2], fraction=0.055, pad=0.03)
        for axis in axes[field_index]:
            axis.tick_params(labelsize=8.2, width=0.75, length=3, pad=2)
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
        if len(logical_shape) == 2:
            axes[field_index, 0].set_ylabel(f"{field_name}{unit_label}\n{axis_labels[1]}")

    sample_label = f" — {sample_id}" if sample_id else ""
    figure.suptitle(f"Reconstruction preview — epoch {epoch:g}{sample_label}", fontsize=11.5)
    outputs = tuple(path_stem.with_suffix(suffix) for suffix in (".png", ".svg", ".pdf"))
    path_stem.parent.mkdir(parents=True, exist_ok=True)
    for output in outputs:
        figure.savefig(output, dpi=300, bbox_inches="tight")
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
        query_coords = np.asarray(payload["query_coords"])
        obs_coords = np.asarray(payload["obs_coords"])
        coordinate_space = "normalized"
        if "query_coords_physical" in payload.files:
            query_coords = np.asarray(payload["query_coords_physical"])
            coordinate_space = "dataset"
        if "obs_coords_physical" in payload.files:
            obs_coords = np.asarray(payload["obs_coords_physical"])
        field_units = (
            tuple(str(value) for value in payload["field_units"])
            if "field_units" in payload.files
            else None
        )
        sample_id = str(payload["sample_id"]) if "sample_id" in payload.files else None
        return _plot_preview(
            Path(output_stem) if output_stem is not None else payload_path.with_suffix(""),
            prediction=payload["prediction_physical"],
            target=payload["target_physical"],
            query_coords=query_coords,
            obs_coords=obs_coords,
            obs_values=payload["obs_values_physical"],
            obs_fields=payload["obs_field_ids"],
            obs_valid=payload["obs_valid_mask"].astype(bool),
            field_names=tuple(str(value) for value in payload["field_names"]),
            logical_shape=tuple(int(value) for value in payload["logical_shape"]),
            epoch=float(epoch),
            field_units=field_units,
            coordinate_space=coordinate_space,
            sample_id=sample_id,
        )


class TrainingReconstructionPreview:
    """Run cheap validation loss and qualitative reconstruction independently."""

    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        store: RunStore,
        steps_per_epoch: int,
        device: torch.device,
    ) -> None:
        settings = config.get("evaluation", {}).get("preview", {})
        self.enabled = bool(settings.get("enabled", True))
        self.store = store
        self.steps_per_epoch = max(1, int(steps_per_epoch))
        self.device = device
        self.settings = settings
        self.dataset = None
        self.batch = None
        self.field_units: tuple[str, ...] = ()
        self.query_coords_physical: np.ndarray | None = None
        self.obs_coords_physical: np.ndarray | None = None
        self.output_dir = store.run_dir / "evaluation" / "training_preview"
        self.generation_steps = int(
            settings.get(
                "generation_steps",
                config.get("evaluation", {}).get("generation_steps", 2),
            )
        )
        legacy_every = int(settings.get("every_epochs", 10))
        self.loss_every_epochs = int(settings.get("loss_every_epochs", legacy_every))
        self.reconstruct_every_epochs = int(
            settings.get("reconstruct_every_epochs", legacy_every)
        )
        self.keep_history = bool(settings.get("keep_history", False))
        self.last_validation_report: dict[str, Any] | None = None
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
        self.field_units = tuple(self.dataset.data_spec.field_units)
        normalized_reference = sample.coordinates.detach().cpu().numpy()
        physical_reference = sample.coordinates_raw.detach().cpu().numpy()
        self.query_coords_physical = _coordinates_in_dataset_units(
            self.batch.query_coords[0].detach().cpu().numpy(),
            normalized_reference,
            physical_reference,
        )
        self.obs_coords_physical = _coordinates_in_dataset_units(
            self.batch.obs_coords[0].detach().cpu().numpy(),
            normalized_reference,
            physical_reference,
        )
        # The preview batch is now self-contained. Close the lazy HDF5 handle
        # before DataLoader workers may fork so no unrelated descriptor is
        # inherited by the asynchronous training path.
        self.dataset.close()
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _epoch_due(self, global_step: int, every_epochs: int) -> bool:
        if not self.enabled or global_step % self.steps_per_epoch:
            return False
        epoch = global_step // self.steps_per_epoch
        return epoch % every_epochs == 0

    def due_loss(self, global_step: int) -> bool:
        """Return whether the fixed validation loss is due."""
        return self._epoch_due(global_step, self.loss_every_epochs)

    def due_reconstruction(self, global_step: int) -> bool:
        """Return whether the qualitative reconstruction is due."""
        return self._epoch_due(global_step, self.reconstruct_every_epochs)

    def due(self, global_step: int) -> bool:
        """Return whether either independent validation action is due."""
        return self.due_loss(global_step) or self.due_reconstruction(global_step)

    def _validation_loss(
        self,
        model: torch.nn.Module,
        *,
        global_step: int,
    ) -> dict[str, Any]:
        assert self.batch is not None
        was_training = model.training
        model.eval()
        seed = int(self.settings.get("seed", 2027))
        device_indices = [self.device.index or 0] if self.device.type == "cuda" else []
        with (
            torch.random.fork_rng(devices=device_indices),
            evaluation_weight_context(model),
            # Physics-informed losses may require coordinate derivatives even
            # during evaluation. No backward pass is performed, so parameter
            # gradients remain untouched while the temporary graph is freed.
            torch.enable_grad(),
        ):
            torch.manual_seed(seed)
            losses = model.training_loss(self.batch)
        if was_training:
            model.train()
        total = float(losses.total.detach().cpu())
        components = {
            name: float(value.detach().cpu())
            for name, value in losses.components.items()
        }
        report = {
            "global_step": int(global_step),
            "training_epoch": global_step / self.steps_per_epoch,
            "sample_id": self.batch.sample_ids[0],
            "seed": seed,
            "loss": total,
            "components": components,
        }
        self.store.write_json(
            "evaluation/training_preview/latest_validation.json",
            report,
        )
        self.store.update_manifest(training_validation=report)
        return report

    def _reconstruct(
        self,
        model: torch.nn.Module,
        *,
        global_step: int,
    ) -> dict[str, Any]:
        assert self.dataset is not None and self.batch is not None
        was_training = model.training
        model.eval()
        seed = int(self.settings.get("seed", 2027))
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
            query_coords_physical=self.query_coords_physical,
            obs_coords=self.batch.obs_coords[0].detach().cpu().numpy(),
            obs_coords_physical=self.obs_coords_physical,
            obs_values_physical=_physical_observations(
                self.batch, self.dataset.normalizer
            ).numpy(),
            obs_field_ids=self.batch.obs_field_ids[0].detach().cpu().numpy(),
            obs_valid_mask=self.batch.obs_valid_mask[0].detach().cpu().numpy(),
            logical_shape=np.asarray(self.dataset.data_spec.logical_shape),
            field_names=np.asarray(self.dataset.field_names),
            field_units=np.asarray(self.field_units),
            coordinate_space=np.asarray("dataset"),
            sample_id=np.asarray(self.batch.sample_ids[0]),
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
            "weight_source": "configured_evaluation_weights",
            "generation_steps": self.generation_steps,
            "seed": seed,
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
            "- **Claim:** qualitative sparse-reconstruction quality of the configured "
            "evaluation weights.\n"
            f"- **Training epoch:** `{epoch:.3f}`\n"
            f"- **Sample:** `{report['sample_id']}` from the configured preview split.\n"
            "- **Panels:** physical target, reconstruction, and absolute error; "
            "target and reconstruction share a field scale, while absolute error starts at zero. "
            "Each error panel reports its field-wise relative L2 error, and open circles "
            "mark conditioned sensors.\n"
            "- **Axes and units:** coordinates use raw dataset coordinates; field values use "
            "physical units listed in the saved payload when declared.\n"
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

    def update(
        self,
        model: torch.nn.Module,
        *,
        global_step: int,
        force: bool = False,
        checkpoint_path: Path | None = None,
    ) -> dict[str, Any] | None:
        del checkpoint_path  # retained as a source-compatible keyword
        self.last_validation_report = None
        if not self.enabled or (not force and not self.due(global_step)):
            return None
        assert self.dataset is not None and self.batch is not None
        validation = (
            self._validation_loss(model, global_step=global_step)
            if force or self.due_loss(global_step)
            else None
        )
        self.last_validation_report = validation
        reconstruction = (
            self._reconstruct(model, global_step=global_step)
            if force or self.due_reconstruction(global_step)
            else None
        )
        return {"validation": validation, "reconstruction": reconstruction}

    def close(self) -> None:
        if self.dataset is not None:
            self.dataset.close()

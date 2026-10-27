"""Fixed validation reconstructions across selected frames and random seeds."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from ..data.factory import open_field_dataset
from ..data.manifest import SensorManifest, build_batch_from_manifest, manifest_from_batch
from ..data.sensor_protocols import build_observation_batch
from ..evaluation.metrics import reconstruction_metrics
from .common import sensor_protocol_from_config
from .history_plotting import HISTORY_FAMILY_COLORS, style_history_axis
from .model_lifecycle import evaluation_weight_context
from .run_store import RunStore, file_sha256


def _integer_list(value: Any, name: str) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value:
        raise ValueError(f"evaluation.reconstruction_panel.{name} must be a non-empty list")
    if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        raise TypeError(f"evaluation.reconstruction_panel.{name} must contain integers")
    result = tuple(int(item) for item in value)
    if len(set(result)) != len(result):
        raise ValueError(f"evaluation.reconstruction_panel.{name} must contain unique values")
    return result


def _upsert_jsonl(path: Path, rows: Sequence[Mapping[str, Any]], key_fields: tuple[str, ...]) -> list[dict[str, Any]]:
    """Replace rows with matching keys and retain the rest of the run history."""
    indexed: dict[tuple[Any, ...], dict[str, Any]] = {}
    if path.is_file():
        lines = path.read_text(encoding="utf-8").splitlines()
        for line_number, line in enumerate(lines, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                if line_number == len(lines):
                    break
                raise ValueError(f"invalid JSON in {path}:{line_number}") from error
            key = tuple(row[field] for field in key_fields)
            indexed[key] = row
    for row in rows:
        payload = dict(row)
        indexed[tuple(payload[field] for field in key_fields)] = payload
    ordered = sorted(indexed.values(), key=lambda row: tuple(row[field] for field in key_fields))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in ordered),
        encoding="utf-8",
    )
    os.replace(temporary, path)
    return ordered


def _render_mse_history(path: Path, rows: Sequence[Mapping[str, Any]]) -> tuple[Path, Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [float(row["training_epoch"]) for row in rows]
    values = [float(row["mse_normalized"]) for row in rows]
    figure, axis = plt.subplots(figsize=(6.2, 3.8), layout="constrained")
    figure.patch.set_facecolor("white")
    axis.set_facecolor("white")
    axis.plot(
        epochs,
        values,
        marker="o",
        markersize=3.5,
        linewidth=1.5,
        color=HISTORY_FAMILY_COLORS[0],
        label="fixed validation reconstruction panel",
    )
    axis.set_title("Fixed validation reconstruction MSE")
    axis.set_xlabel("Training epoch")
    axis.set_ylabel("Normalized full-field MSE")
    style_history_axis(axis, values, x_max=max(epochs, default=1.0))
    axis.legend(frameon=False, fontsize=8)
    png_path = path.with_suffix(".png")
    pdf_path = path.with_suffix(".pdf")
    figure.savefig(png_path, dpi=200, bbox_inches="tight")
    figure.savefig(pdf_path, bbox_inches="tight")
    plt.close(figure)
    return png_path, pdf_path


class TrainingReconstructionPanel:
    """Evaluate a fixed validation-frame/seed panel for checkpoint selection.

    The callable result follows ``PeriodicCheckpointManager``'s
    ``reconstruction_mse`` contract. Inputs to ``model.reconstruct`` always
    have ``target_fields=None``; targets are retained only for scoring.
    """

    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        model: torch.nn.Module,
        store: RunStore,
        steps_per_epoch: int,
        device: torch.device,
    ) -> None:
        evaluation = config.get("evaluation", {})
        settings = evaluation.get("reconstruction_panel") if isinstance(evaluation, Mapping) else None
        self.enabled = settings is not None
        self.model = model
        self.store = store
        self.dataset_config = config.get("dataset", {})
        self.device = torch.device(device)
        self.steps_per_epoch = max(1, int(steps_per_epoch))
        self.output_dir = store.run_dir / "evaluation" / "reconstruction_panel"
        self.manifest_dir = self.output_dir / "sensor_manifests"
        self.manifest_paths: dict[int, Path] = {}
        self.normalizer = None
        self.normalization_digest: str | None = None
        self.field_names: tuple[str, ...] = ()
        self.sample_indices: tuple[int, ...] = ()
        self.seeds: tuple[int, ...] = ()
        self.generation_steps = 0
        self.sample_ids: dict[int, str] = {}
        if not self.enabled:
            return
        if not isinstance(settings, Mapping):
            raise TypeError("evaluation.reconstruction_panel must be a mapping")
        if not callable(getattr(model, "reconstruct", None)):
            raise TypeError("reconstruction panel model must expose reconstruct()")
        self.sample_indices = _integer_list(settings.get("sample_indices"), "sample_indices")
        if any(index < 0 for index in self.sample_indices):
            raise ValueError("reconstruction panel sample_indices must be non-negative")
        self.seeds = _integer_list(settings.get("seeds"), "seeds")
        self.generation_steps = int(settings.get("generation_steps", 32))
        if self.generation_steps < 1:
            raise ValueError("reconstruction panel generation_steps must be positive")

        training_dataset = open_field_dataset(config["dataset"], split="train")
        try:
            self.normalizer = training_dataset.normalizer
            self.field_names = tuple(training_dataset.field_names)
        finally:
            training_dataset.close()
        self.normalization_digest = self.normalizer.digest()

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_dir.mkdir(parents=True, exist_ok=True)
        validation_dataset = open_field_dataset(
            config["dataset"], split="validation", normalizer=self.normalizer
        )
        try:
            if tuple(validation_dataset.field_names) != self.field_names:
                raise ValueError("validation field order differs from the training split")
            if any(index >= len(validation_dataset) for index in self.sample_indices):
                raise IndexError("reconstruction panel sample index is outside validation split")
            for index in self.sample_indices:
                sample = validation_dataset[index]
                batch = build_observation_batch(
                    [sample],
                    sensor_protocol_from_config(config, seed_offset=index),
                    query_points=None,
                )
                manifest = manifest_from_batch(
                    batch, validation_dataset.path, validation_dataset.split_name
                )
                manifest_path = self.manifest_dir / f"frame_{index:06d}.json"
                if manifest_path.is_file():
                    existing_manifest = SensorManifest.load(manifest_path)
                    if existing_manifest.digest() != manifest.digest():
                        raise ValueError(
                            f"existing reconstruction panel sensor manifest differs: {manifest_path}"
                        )
                    manifest = existing_manifest
                else:
                    manifest.save(manifest_path)
                self.manifest_paths[index] = manifest_path
                self.sample_ids[index] = batch.sample_ids[0]
        finally:
            validation_dataset.close()

        model_settings = config.get("model", {})
        model_settings = model_settings if isinstance(model_settings, Mapping) else {}
        panel_identity = {
            "schema": "phycoflow_fixed_reconstruction_panel.v1",
            "split": "validation",
            "sample_indices": list(self.sample_indices),
            "sample_ids": [self.sample_ids[index] for index in self.sample_indices],
            "seeds": list(self.seeds),
            "generation_steps": self.generation_steps,
            "normalization_digest": self.normalization_digest,
            "sensor_seed_policy": "observation_protocol_seed_plus_split_relative_sample_index",
            "generator_seed_policy": "configured_seed_per_frame_and_reconstruction",
            "weight_policy": {
                "context": "configured_evaluation_weight_context",
                "model_ema_enabled": bool(model_settings.get("model_ema_enabled", False)),
                "model_ema_eval": bool(model_settings.get("model_ema_eval", False)),
            },
            "sensor_manifests": {
                str(index): {
                    "path": str(path.relative_to(self.store.run_dir)),
                    "sha256": file_sha256(path),
                }
                for index, path in self.manifest_paths.items()
            },
        }
        panel_identity_path = self.output_dir / "panel.json"
        if panel_identity_path.is_file():
            existing_identity = json.loads(panel_identity_path.read_text(encoding="utf-8"))
            if existing_identity != panel_identity:
                raise ValueError(
                    "existing reconstruction panel identity differs from the configured panel"
                )
        else:
            self.store.write_json(
                "evaluation/reconstruction_panel/panel.json", panel_identity
            )

    @staticmethod
    def _weight_source(model: torch.nn.Module) -> str:
        if getattr(model, "_ema", None) is not None and bool(
            getattr(model, "_ema_eval", False)
        ):
            return "ema"
        return "live"

    def __call__(self, global_step: int) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        assert self.normalizer is not None
        weight_source = self._weight_source(self.model)
        dataset = open_field_dataset(
            self.dataset_config, split="validation", normalizer=self.normalizer
        )
        squared_error_total = 0.0
        normalized_entry_count = 0
        physical_squared_error_total = 0.0
        physical_entry_count = 0
        per_field_normalized: dict[str, float] = {name: 0.0 for name in self.field_names}
        per_field_physical: dict[str, float] = {name: 0.0 for name in self.field_names}
        per_field_entry_count: dict[str, int] = {name: 0 for name in self.field_names}
        examples: list[dict[str, Any]] = []
        module_modes = [(module, module.training) for module in self.model.modules()]
        cuda_devices = []
        if self.device.type == "cuda":
            cuda_devices = [
                self.device.index if self.device.index is not None else torch.cuda.current_device()
            ]
        self.model.eval()
        try:
            with (
                torch.random.fork_rng(devices=cuda_devices),
                evaluation_weight_context(self.model),
                torch.no_grad(),
            ):
                for sample_index in self.sample_indices:
                    sample = dataset[sample_index]
                    manifest_path = self.manifest_paths[sample_index]
                    manifest = SensorManifest.load(manifest_path)
                    batch = build_batch_from_manifest(
                        [sample], manifest, dataset.path, query_points=None
                    ).to(self.device)
                    target = batch.target_fields
                    if target is None:
                        raise ValueError("fixed validation panel requires dense validation targets")
                    inference_batch = replace(batch, target_fields=None)
                    valid_query_count = int(batch.query_valid_mask.sum().item())
                    sensor_seed = int(manifest.protocol["seed"])
                    for seed in self.seeds:
                        generator = torch.Generator(device=self.device).manual_seed(seed)
                        reconstruction = self.model.reconstruct(
                            inference_batch,
                            steps=self.generation_steps,
                            generator=generator,
                        )
                        prediction = reconstruction.prediction
                        if prediction.shape != target.shape:
                            raise ValueError(
                                "reconstruction panel prediction shape does not match validation target"
                            )
                        metrics = reconstruction_metrics(
                            prediction,
                            target,
                            batch,
                            self.field_names,
                            normalizer=self.normalizer,
                        )
                        entry_count = valid_query_count * len(self.field_names)
                        squared_error_total += float(metrics["mse_normalized"]) * entry_count
                        normalized_entry_count += entry_count
                        field_entries = valid_query_count
                        normalized_field_metrics = metrics["per_field_mse_normalized"]
                        assert isinstance(normalized_field_metrics, Mapping)
                        for name in self.field_names:
                            per_field_normalized[name] += float(normalized_field_metrics[name]) * field_entries
                            per_field_entry_count[name] += field_entries
                        if "mse_physical" in metrics:
                            physical_squared_error_total += float(metrics["mse_physical"]) * entry_count
                            physical_entry_count += entry_count
                            physical_field_metrics = metrics["per_field_mse_physical"]
                            assert isinstance(physical_field_metrics, Mapping)
                            for name in self.field_names:
                                per_field_physical[name] += float(physical_field_metrics[name]) * field_entries
                        valid_observations = batch.obs_valid_mask[0]
                        examples.append(
                            {
                                "sample_index": sample_index,
                                "sample_id": batch.sample_ids[0],
                                "sensor_seed": sensor_seed,
                                "sensor_manifest": str(
                                    manifest_path.relative_to(self.store.run_dir)
                                ),
                                "sensor_manifest_sha256": file_sha256(manifest_path),
                                "observed_point_field_indices": [
                                    [int(point), int(field)]
                                    for point, field in zip(
                                        batch.obs_indices[0, valid_observations].tolist(),
                                        batch.obs_field_ids[0, valid_observations].tolist(),
                                    )
                                ],
                                "seed": seed,
                                "weight_source": weight_source,
                                "generation_steps": self.generation_steps,
                                "metrics": metrics,
                            }
                        )
                        del prediction, reconstruction, metrics
                    del inference_batch, target, batch, sample
        finally:
            dataset.close()
            for module, was_training in module_modes:
                module.training = was_training

        if normalized_entry_count == 0:
            raise ValueError("reconstruction panel did not evaluate any valid target entries")
        metrics: dict[str, Any] = {
            "mse_normalized": squared_error_total / normalized_entry_count,
            "per_field_mse_normalized": {
                name: per_field_normalized[name] / per_field_entry_count[name]
                for name in self.field_names
            },
        }
        if physical_entry_count:
            metrics["mse_physical"] = physical_squared_error_total / physical_entry_count
            metrics["per_field_mse_physical"] = {
                name: per_field_physical[name] / per_field_entry_count[name]
                for name in self.field_names
            }
        epoch = int(global_step) / self.steps_per_epoch
        report = {
            "schema": "phycoflow_fixed_reconstruction_panel_result.v1",
            "global_step": int(global_step),
            "training_epoch": epoch,
            "split": "validation",
            "sample_count": len(self.sample_indices),
            "seed_count": len(self.seeds),
            "example_seed_count": len(examples),
            "sample_indices": list(self.sample_indices),
            "sample_ids": [self.sample_ids[index] for index in self.sample_indices],
            "seeds": list(self.seeds),
            "generation_steps": self.generation_steps,
            "normalization_digest": self.normalization_digest,
            "weight_source": weight_source,
            "weight_policy": "configured_evaluation_weight_context",
            "sensor_seed_policy": "observation_protocol_seed_plus_split_relative_sample_index",
            "generator_seed_policy": "configured_seed_per_frame_and_reconstruction",
            "metrics": metrics,
            "examples": examples,
        }
        self.store.write_json("evaluation/reconstruction_panel/latest.json", report)
        history_row = {
            "global_step": int(global_step),
            "training_epoch": epoch,
            "mse_normalized": metrics["mse_normalized"],
            "sample_count": len(self.sample_indices),
            "seed_count": len(self.seeds),
            "weight_source": weight_source,
        }
        history_path = self.output_dir / "history.jsonl"
        history = _upsert_jsonl(history_path, [history_row], ("global_step",))
        _upsert_jsonl(
            self.output_dir / "examples.jsonl",
            [
                {"global_step": int(global_step), "training_epoch": epoch, **example}
                for example in examples
            ],
            ("global_step", "sample_index", "seed"),
        )
        figure_paths = _render_mse_history(self.output_dir / "reconstruction_mse", history)
        report["history"] = str(history_path.relative_to(self.store.run_dir))
        report["figures"] = {
            path.suffix.removeprefix("."): str(path.relative_to(self.store.run_dir))
            for path in figure_paths
        }
        self.store.write_json("evaluation/reconstruction_panel/latest.json", report)
        return report

    def close(self) -> None:
        """Provided for symmetry with other training evaluators; no handle is retained."""

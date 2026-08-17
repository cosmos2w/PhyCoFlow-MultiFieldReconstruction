"""Common differentiable data-coherence post-training.

The trainer always creates an immutable child run. Rectified-flow sources use
public flow hooks; diffusion, latent, and deterministic adapters use their
native differentiable reconstruction hook. Target use and provenance remain
explicit for every route.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any

import torch

from ..coherence import ReferenceBank, build_enabled_families, fit_reference_bank
from ..contracts import FamilyResult, ObservationBatch
from ..data.factory import FieldDataset, open_field_dataset
from ..data.manifest import dataset_fingerprint, manifest_from_batch
from ..data.sensor_protocols import SensorProtocol, build_observation_batch
from ..data.training_batches import (
    build_training_batch_source,
    dataset_field_bytes,
    fixed_query_indices,
)
from ..evaluation import reconstruction_metrics
from ..utils.reproducibility import seed_everything
from .checkpointing import PeriodicCheckpointManager
from .common import (
    iter_unique_batch_indices,
    sensor_protocol_from_config,
)
from .gradient_balance import data_only_update, two_objective_update
from .monitoring import TrainingMonitor
from .preview import TrainingReconstructionPreview
from .rollout import differentiable_reconstruction, subset_query_batch
from .run_store import (
    RunStore,
    checkpoint_model_state,
    file_sha256,
    load_model_state_strict,
)
from .source import (
    load_source_model,
    set_trainable_scope,
    source_checkpoint_path,
    source_hashes,
)


def _slice_batch(batch: ObservationBatch, count: int) -> ObservationBatch:
    count = min(int(count), batch.obs_coords.shape[0])
    metadata = dict(batch.metadata)
    if isinstance(metadata.get("query_indices"), torch.Tensor):
        metadata["query_indices"] = metadata["query_indices"][:count]
    context = metadata.get("sample_context")
    if isinstance(context, dict):
        context = dict(context)
        for key, value in context.items():
            if isinstance(value, torch.Tensor) and value.shape[0] == batch.obs_coords.shape[0]:
                context[key] = value[:count]
            elif isinstance(value, dict):
                context[key] = {
                    name: (
                        item[:count]
                        if isinstance(item, torch.Tensor)
                        and item.shape[0] == batch.obs_coords.shape[0]
                        else item
                    )
                    for name, item in value.items()
                }
        metadata["sample_context"] = context
    return ObservationBatch(
        obs_coords=batch.obs_coords[:count],
        obs_values=batch.obs_values[:count],
        obs_field_ids=batch.obs_field_ids[:count],
        obs_valid_mask=batch.obs_valid_mask[:count],
        query_coords=batch.query_coords[:count],
        query_valid_mask=batch.query_valid_mask[:count],
        target_fields=None if batch.target_fields is None else batch.target_fields[:count],
        sample_ids=batch.sample_ids[:count],
        obs_indices=None if batch.obs_indices is None else batch.obs_indices[:count],
        logical_shapes=batch.logical_shapes[:count],
        metadata=metadata,
    )


def _without_target(batch: ObservationBatch) -> ObservationBatch:
    return ObservationBatch(
        obs_coords=batch.obs_coords,
        obs_values=batch.obs_values,
        obs_field_ids=batch.obs_field_ids,
        obs_valid_mask=batch.obs_valid_mask,
        query_coords=batch.query_coords,
        query_valid_mask=batch.query_valid_mask,
        target_fields=None,
        sample_ids=batch.sample_ids,
        obs_indices=batch.obs_indices,
        logical_shapes=batch.logical_shapes,
        metadata=batch.metadata,
    )


def _gather_prediction(
    prediction: torch.Tensor,
    source_batch: ObservationBatch,
    selected_batch: ObservationBatch,
) -> torch.Tensor:
    """Gather common comparison points from a complete structured prediction."""
    source_ids = source_batch.metadata.get("query_indices")
    selected_ids = selected_batch.metadata.get("query_indices")
    if not isinstance(source_ids, torch.Tensor) or not isinstance(selected_ids, torch.Tensor):
        raise TypeError("structured comparison requires serialized query indices")
    if prediction.shape[1] != source_ids.shape[1]:
        raise ValueError("structured prediction does not align with the complete query grid")
    gathered = []
    for batch_index in range(prediction.shape[0]):
        valid_source = source_ids[batch_index] >= 0
        valid_selected = selected_ids[batch_index] >= 0
        ids = source_ids[batch_index, valid_source]
        requested = selected_ids[batch_index, valid_selected]
        if ids.numel() != torch.unique(ids).numel():
            raise ValueError("source query indices must be unique")
        lookup = torch.full(
            (int(ids.max().item()) + 1,),
            -1,
            device=prediction.device,
            dtype=torch.long,
        )
        lookup[ids.to(prediction.device)] = torch.arange(ids.numel(), device=prediction.device)
        requested = requested.to(prediction.device)
        if torch.any(requested >= lookup.numel()):
            raise ValueError("selected query index is absent from structured prediction")
        if torch.any(lookup[requested] < 0):
            raise ValueError("selected query index is absent from structured prediction")
        gathered.append(prediction[batch_index, lookup[requested]])
    if len({item.shape[0] for item in gathered}) != 1:
        raise ValueError("comparison query counts must agree within a batch")
    return torch.stack(gathered)


def _requires_fixed_geometry(family) -> bool:
    return any(component.required_geometry != "none" for component in family.spec.components)


def _validate_reference_geometry(
    family_name: str,
    family,
    bank: ReferenceBank,
    query_ids: torch.Tensor | None,
    *,
    step: int,
    batch_size: int,
) -> None:
    if not _requires_fixed_geometry(family):
        return
    if query_ids is None:
        raise TypeError(f"{family_name} requires serialized query indices")
    bank_rows = [
        (int(step) * int(batch_size) + offset) % bank.values.shape[0]
        for offset in range(batch_size)
    ]
    expected = bank.point_indices[bank_rows]
    if not torch.equal(expected, query_ids.detach().cpu()):
        raise ValueError(f"{family_name} training references do not share prediction geometry")


def _coherence_objective(
    model,
    batch: ObservationBatch,
    family,
    bank: ReferenceBank | Mapping[str, ReferenceBank | None] | None,
    config: Mapping[str, Any],
    *,
    step: int,
    generator: torch.Generator,
) -> tuple[FamilyResult, tuple[str, ...]]:
    compute = config["coherence"]["compute_budget"]
    selected = _slice_batch(batch, int(compute["batch_size"]))
    complete = selected
    selected = subset_query_batch(complete, int(compute["point_count"]), generator=generator)
    # Dense targets are reference payloads only. They never enter a coherence
    # rollout, including explicitly paired-supervised structural losses.
    model_batch = complete if model.capabilities.structured_grid_required else selected
    model_batch = _without_target(model_batch)
    prediction = differentiable_reconstruction(
        model,
        model_batch,
        steps=int(config["rollout"]["steps"]),
        solver=config["rollout"]["solver"],
        generator=generator,
        observation_config=dict(config["observation_consistency"]),
    )
    if model.capabilities.structured_grid_required:
        prediction = _gather_prediction(prediction, complete, selected)
    families = dict(family) if isinstance(family, Mapping) else {family.family_name: family}
    banks = dict(bank) if isinstance(bank, Mapping) else {
        name: bank for name in families
    }
    component_results = {}
    family_results: dict[str, FamilyResult] = {}
    reference_ids_all: list[str] = []
    scalar_loss = prediction.sum() * 0.0
    per_sample_parts = []
    for family_name, family_module in families.items():
        family_bank = banks.get(family_name)
        if family_module.target_use == "paired_supervised":
            if selected.target_fields is None:
                raise ValueError("paired_supervised coherence requires the dense training target")
            reference, reference_ids = selected.target_fields, selected.sample_ids
        else:
            if family_bank is None:
                raise ValueError(
                    f"training_reference coherence family {family_name} requires a reference bank"
                )
            reference, reference_ids = family_bank.select(
                prediction.shape[0], step=step, device=prediction.device, dtype=prediction.dtype
            )
            if reference.shape[1] != prediction.shape[1]:
                raise ValueError("reference-bank and coherence point counts differ")
            query_ids = selected.metadata.get("query_indices")
            _validate_reference_geometry(
                family_name,
                family_module,
                family_bank,
                query_ids if isinstance(query_ids, torch.Tensor) else None,
                step=step,
                batch_size=prediction.shape[0],
            )
        result = family_module(
            prediction,
            reference,
            coordinates=selected.query_coords,
            context={"sample_ids": selected.sample_ids, "reference_ids": reference_ids},
        )
        family_results[family_name] = result
        component_results.update(result.component_results)
        weight = float(getattr(family_module, "family_weight", 1.0))
        scalar_loss = scalar_loss + weight * result.scalar_loss
        if result.per_sample_cost is not None:
            per_sample_parts.append(weight * result.per_sample_cost)
        reference_ids_all.extend(reference_ids)
    per_sample = (
        torch.stack(per_sample_parts).sum(dim=0)
        if len(per_sample_parts) == len(families)
        else None
    )
    combined = FamilyResult(
        component_results=component_results,
        per_sample_cost=per_sample,
        scalar_loss=scalar_loss,
        diagnostics={
            "family": "combined" if len(families) > 1 else next(iter(families)),
            "families": {
                name: {
                    "weight": float(getattr(families[name], "family_weight", 1.0)),
                    "scalar_loss": result.scalar_loss.detach(),
                    **result.diagnostics,
                }
                for name, result in family_results.items()
            },
        },
    )
    return combined, tuple(dict.fromkeys(reference_ids_all))


def _component_scalars(result: FamilyResult) -> dict[str, float]:
    return {
        path: float(component.scalar_loss.detach().cpu())
        for path, component in result.component_results.items()
    }


def _coherence_weight(config: Mapping[str, Any], epoch: int) -> float:
    if not bool(config["objectives"]["coherence"].get("enabled", True)):
        return 0.0
    schedule = config["coherence"]["schedule"]
    start = int(schedule.get("start_epoch", 1))
    if epoch < start:
        return 0.0
    weight = float(config["objectives"]["coherence"]["weight"])
    warmup = int(schedule.get("weight_warmup_epochs", 0))
    return weight if warmup <= 0 else weight * min(1.0, (epoch - start + 1) / warmup)


def _data_weight(config: Mapping[str, Any]) -> float:
    settings = config["objectives"]["data_retention"]
    return float(settings["weight"]) if bool(settings.get("enabled", True)) else 0.0


def _balanced_weights(config: Mapping[str, Any], coherence_weight: float) -> tuple[float, float]:
    data_weight = _data_weight(config)
    if config["optimization"].get("gradient_balance", "weighted_sum") == "config":
        data_weight *= float(config["optimization"].get("config_data_grad_scale", 1.0))
        coherence_weight *= float(config["optimization"].get("config_coherence_grad_scale", 1.0))
    return data_weight, coherence_weight


def _build_evaluation_batch(
    dataset: FieldDataset,
    config: Mapping[str, Any],
    protocol: SensorProtocol,
    device: torch.device,
) -> ObservationBatch:
    count = min(int(config.get("evaluation", {}).get("max_samples", 1)), len(dataset))
    if count < 1:
        raise ValueError(f"evaluation split {dataset.split_name!r} contains no samples")
    samples = [dataset[index] for index in range(count)]
    return build_observation_batch(samples, protocol, query_points=None).to(device)


def _build_comparison_batch(
    batch: ObservationBatch,
    config: Mapping[str, Any],
) -> ObservationBatch:
    point_count = int(
        config.get("evaluation", {}).get(
            "query_points", config["coherence"]["compute_budget"]["point_count"]
        )
    )
    evaluation = config.get("evaluation", {})
    compute = config["coherence"]["compute_budget"]
    explicit_indices = None
    if compute.get("query_policy", "random_per_sample") == "fixed_shared":
        explicit_indices = fixed_query_indices(
            batch.query_coords.shape[1],
            point_count,
            seed=int(
                compute.get(
                    "query_seed",
                    config["observations"].get("seed", config["runtime"].get("seed", 42))
                    + 100_003,
                )
            ),
        )
    generator = torch.Generator(device=batch.query_coords.device).manual_seed(
        int(evaluation.get("seed", 2027)) + 100_003
    )
    return subset_query_batch(
        batch,
        point_count,
        generator=generator,
        indices=explicit_indices,
        shared=explicit_indices is not None,
    )


def _evaluate(
    model,
    complete_batch: ObservationBatch,
    comparison_batch: ObservationBatch,
    family,
    bank: ReferenceBank | Mapping[str, ReferenceBank | None] | None,
    field_names: tuple[str, ...],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    target = comparison_batch.target_fields
    if target is None:
        raise ValueError("evaluation requires dense targets for metrics")
    model_batch = (
        complete_batch if model.capabilities.structured_grid_required else comparison_batch
    )
    inference_batch = _without_target(model_batch)
    evaluation = config.get("evaluation", {})
    evaluation_seed = int(evaluation.get("seed", 2027))
    generator = torch.Generator(device=complete_batch.query_coords.device).manual_seed(
        evaluation_seed
    )
    model.eval()
    with torch.no_grad():
        warmup_generator = torch.Generator(device=complete_batch.query_coords.device).manual_seed(
            evaluation_seed + 1
        )
        model.reconstruct(
            inference_batch,
            steps=int(evaluation.get("generation_steps", 2)),
            generator=warmup_generator,
        )
    if complete_batch.query_coords.device.type == "cuda":
        torch.cuda.synchronize(complete_batch.query_coords.device)
    inference_started = perf_counter()
    with torch.no_grad():
        prediction = model.reconstruct(
            inference_batch,
            steps=int(evaluation.get("generation_steps", 2)),
            generator=generator,
        ).prediction
        if model.capabilities.structured_grid_required:
            prediction = _gather_prediction(prediction, complete_batch, comparison_batch)
        if complete_batch.query_coords.device.type == "cuda":
            torch.cuda.synchronize(complete_batch.query_coords.device)
        inference_seconds = perf_counter() - inference_started
        metrics = reconstruction_metrics(prediction, target, comparison_batch, field_names)
        families = dict(family) if isinstance(family, Mapping) else {family.family_name: family}
        banks = dict(bank) if isinstance(bank, Mapping) else {name: bank for name in families}
        family_payloads = {}
        combined_components = {}
        combined_total = prediction.sum() * 0.0
        all_reference_ids: list[str] = []
        for family_name, family_module in families.items():
            if family_module.target_use == "paired_supervised":
                reference, reference_ids = target, comparison_batch.sample_ids
            else:
                family_bank = banks.get(family_name)
                if family_bank is None:
                    raise ValueError(
                        f"target-free evaluation for {family_name} requires a training bank"
                    )
                reference, reference_ids = family_bank.select(
                    prediction.shape[0],
                    step=0,
                    device=prediction.device,
                    dtype=prediction.dtype,
                )
                if reference.shape[1] != prediction.shape[1]:
                    raise ValueError(
                        "evaluation.query_points must equal reference_bank.points_per_sample"
                    )
                query_ids = comparison_batch.metadata.get("query_indices")
                _validate_reference_geometry(
                    family_name,
                    family_module,
                    family_bank,
                    query_ids if isinstance(query_ids, torch.Tensor) else None,
                    step=0,
                    batch_size=prediction.shape[0],
                )
            family_result = family_module(
                prediction,
                reference,
                coordinates=comparison_batch.query_coords,
                context={
                    "sample_ids": comparison_batch.sample_ids,
                    "reference_ids": reference_ids,
                },
            )
            weight = float(getattr(family_module, "family_weight", 1.0))
            combined_total = combined_total + weight * family_result.scalar_loss
            combined_components.update(_component_scalars(family_result))
            all_reference_ids.extend(reference_ids)
            family_payloads[family_name] = {
                "target_use": family_module.target_use,
                "units": family_module.units,
                "weight": weight,
                "total": float(family_result.scalar_loss.cpu()),
                "weighted_total": float((weight * family_result.scalar_loss).cpu()),
                "components": _component_scalars(family_result),
                "reference_ids": list(reference_ids),
                "diagnostics": family_result.diagnostics,
            }
        target_uses = {module.target_use for module in families.values()}
        units = {module.units for module in families.values()}
        coherence_payload = {
            "family": "combined" if len(families) > 1 else next(iter(families)),
            "target_use": next(iter(target_uses)) if len(target_uses) == 1 else "mixed",
            "units": next(iter(units)) if len(units) == 1 else "mixed",
            "total": float(combined_total.cpu()),
            "components": combined_components,
            "reference_ids": list(dict.fromkeys(all_reference_ids)),
            "families": family_payloads,
        }
    return {
        **metrics,
        "inference": {
            "seconds": inference_seconds,
            "samples": prediction.shape[0],
            "points_per_sample": prediction.shape[1],
            "generation_steps": int(evaluation.get("generation_steps", 2)),
        },
        "coherence": coherence_payload,
        "sample_ids": list(comparison_batch.sample_ids),
    }


def _reference_bank(
    config: Mapping[str, Any],
    dataset: FieldDataset,
    *,
    existing_path: Path | None = None,
    family_name: str = "global_distribution",
) -> ReferenceBank | None:
    family = config["coherence"]["families"][family_name]
    if family.get("target_use") != "training_reference":
        return None
    settings = family["reference_bank"]
    if existing_path is not None and existing_path.is_file():
        bank = ReferenceBank.load(existing_path)
    elif settings.get("path"):
        source = Path(settings["path"])
        if not source.is_absolute():
            raise ValueError("reference_bank.path must be resolved by the case launcher")
        bank = ReferenceBank.load(source)
    else:
        compute = config["coherence"]["compute_budget"]
        fixed_points = None
        if compute.get("query_policy", "random_per_sample") == "fixed_shared":
            fixed_points = fixed_query_indices(
                math.prod(dataset.data_spec.logical_shape),
                int(settings["points_per_sample"]),
                seed=int(
                    compute.get(
                        "query_seed",
                        config["observations"].get(
                            "seed", config["runtime"].get("seed", 42)
                        )
                        + 100_003,
                    )
                ),
            )
        bank = fit_reference_bank(
            dataset,
            max_samples=int(settings.get("max_samples", 64)),
            points_per_sample=int(settings["points_per_sample"]),
            seed=int(settings.get("seed", 1234)),
            fixed_point_indices=fixed_points,
        )
    if bank.metadata["dataset_fingerprint"] != dataset_fingerprint(dataset.path):
        raise ValueError("reference bank belongs to another dataset payload")
    if tuple(bank.metadata["field_names"]) != tuple(dataset.field_names):
        raise ValueError("reference bank field order disagrees with the dataset")
    bank_spec = bank.metadata.get("data_spec", {})
    if (
        tuple(bank_spec.get("logical_shape", ())) != tuple(dataset.data_spec.logical_shape)
        or int(bank_spec.get("coordinate_dim", -1)) != dataset.data_spec.coordinate_dim
    ):
        raise ValueError("reference bank geometry disagrees with the dataset")
    if bank.values.shape[1] != int(config["coherence"]["compute_budget"]["point_count"]):
        raise ValueError("reference bank point count disagrees with coherence compute budget")
    return bank


def run_post_training(
    config: Mapping[str, Any],
    *,
    case_dir: str | Path,
    max_steps: int | None = None,
    resume: str | Path | None = None,
) -> Path:
    """Create or resume one immutable child post-training run."""
    if config["stage"] != "post_training":
        raise ValueError("run_post_training accepts only stage=post_training")
    seed = int(config["runtime"].get("seed", 42))
    seed_everything(seed, bool(config["runtime"].get("deterministic", True)))
    device = torch.device(config["runtime"].get("device", "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    source_hashes_before = source_hashes(config)
    model, train_dataset, source_metadata = load_source_model(config, device)
    if not len(train_dataset):
        raise ValueError(f"training split is empty for {config['dataset']['path']}")
    if not model.capabilities.differentiable_rollout:
        raise ValueError("source model does not support differentiable rollout post-training")
    trainable_names = set_trainable_scope(model, config["trainable"])
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameter_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(config["optimization"].get("lr", 5e-5)),
        weight_decay=float(config["optimization"].get("weight_decay", 0.0)),
    )

    if resume is None:
        store = RunStore.create(
            case_dir,
            config["output"].get("experiment_name", "global_distribution_posttrain"),
            config,
            parent_run=str(config["source_run"]),
        )
        start_step = 0
        checkpoint = None
    else:
        store = RunStore.resume(resume, config)
        checkpoint = store.load_checkpoint("last")
        load_model_state_strict(model, checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = int(checkpoint["global_step"])
        torch.set_rng_state(checkpoint["rng_state"]["torch_cpu"])
        if device.type == "cuda" and checkpoint["rng_state"].get("torch_cuda"):
            torch.cuda.set_rng_state_all(checkpoint["rng_state"]["torch_cuda"])

    families = build_enabled_families(
        config["coherence"], train_dataset.data_spec, train_dataset.normalizer
    )
    families = {name: family.to(device) for name, family in families.items()}
    banks: dict[str, ReferenceBank | None] = {}
    for family_name in families:
        artifact_name = (
            "coherence_reference.pt"
            if family_name == "global_distribution"
            else f"coherence_reference_{family_name}.pt"
        )
        bank_path = store.run_dir / "artifacts" / artifact_name
        banks[family_name] = _reference_bank(
            config,
            train_dataset,
            existing_path=bank_path if resume else None,
            family_name=family_name,
        )
        if banks[family_name] is not None and not bank_path.exists():
            banks[family_name].save(bank_path)
    family_paths = {
        name: store.run_dir / "artifacts" / f"{name}_family.pt" for name in families
    }
    if resume is not None:
        manifest = json.loads((store.run_dir / "run_manifest.json").read_text())
        expected_hashes = manifest.get("coherence_family_state_sha256s", {})
        for name, family in families.items():
            path = family_paths[name]
            if not path.is_file():
                raise FileNotFoundError(f"resume is missing coherence family artifact: {path}")
            if expected_hashes.get(name) and file_sha256(path) != expected_hashes[name]:
                raise ValueError(f"resume coherence family artifact hash mismatch: {name}")
            artifact = torch.load(path, map_location=device, weights_only=True)
            family.load_state_artifact(artifact)
            if checkpoint is not None and name in checkpoint.get("family_states", {}):
                family.load_state_dict(checkpoint["family_states"][name])
    else:
        family_paths = {
            name: store.save_artifact(f"{name}_family.pt", family.state_artifact())
            for name, family in families.items()
        }
    store.save_artifact("normalization.pt", train_dataset.normalizer.state_dict())
    store.write_json("artifacts/trainable_parameters.json", {"names": list(trainable_names)})
    store.write_json(
        "artifacts/split_manifest.json",
        {
            "split": train_dataset.selection.split,
            "strategy": train_dataset.selection.strategy,
            "trajectory_indices": list(train_dataset.selection.trajectory_indices),
            "frame_indices": list(train_dataset.selection.frame_indices),
        },
    )
    store.update_manifest(
        source_kind=source_metadata["kind"],
        source_checkpoint=str(source_checkpoint_path(config)),
        source_hashes=source_hashes_before,
        source_metadata=source_metadata,
        inherited_base_keys=config.get("source", {}).get("inherited_base_keys", []),
        config_origins=config.get("source", {}).get("config_origins", {}),
        trainable_scope=config["trainable"],
        trainable_parameter_names=list(trainable_names),
        parameter_count=parameter_count,
        trainable_parameter_count=trainable_parameter_count,
        differentiable_adapter=(
            "rectified_flow"
            if hasattr(model, "sample_source") and hasattr(model, "velocity")
            else "native_reconstruction"
        ),
        dataset_path=str(train_dataset.path),
        dataset_fingerprint=dataset_fingerprint(train_dataset.path),
        reference_bank_sha256=(
            None
            if banks.get("global_distribution") is None
            else banks["global_distribution"].digest()
        ),
        reference_bank_sha256s={
            name: None if bank is None else bank.digest() for name, bank in banks.items()
        },
        coherence_family_state_sha256=(
            file_sha256(family_paths["global_distribution"])
            if set(family_paths) == {"global_distribution"}
            else None
        ),
        coherence_family_state_sha256s={
            name: file_sha256(path) for name, path in family_paths.items()
        },
        coherence_families=list(families),
    )

    evaluation_split = config.get("evaluation", {}).get("split", "validation")
    evaluation_dataset = open_field_dataset(
        config["dataset"], split=evaluation_split, normalizer=train_dataset.normalizer
    )
    protocol = sensor_protocol_from_config(config)
    evaluation_batch = _build_evaluation_batch(evaluation_dataset, config, protocol, device)
    comparison_batch = _build_comparison_batch(evaluation_batch, config)
    evaluation_manifest = manifest_from_batch(
        evaluation_batch, evaluation_dataset.path, evaluation_split
    )
    evaluation_manifest.save(store.run_dir / "artifacts" / "evaluation_sensor_manifest.json")
    query_path = store.save_artifact(
        "evaluation_query_indices.pt",
        {"query_indices": comparison_batch.metadata.get("query_indices")},
    )
    store.update_manifest(evaluation_query_indices_sha256=file_sha256(query_path))
    before_metrics = None
    if resume is None:
        before_metrics = _evaluate(
            model,
            evaluation_batch,
            comparison_batch,
            families,
            banks,
            train_dataset.field_names,
            config,
        )
        before_metrics["sensor_manifest_sha256"] = evaluation_manifest.digest()
        store.write_json("evaluation/before.json", before_metrics)
    # Evaluation initializes geometry-dependent fixed artifacts. Persist those
    # exact tensors rather than the empty lazy-construction placeholders.
    family_paths = {
        name: store.save_artifact(f"{name}_family.pt", family.state_artifact())
        for name, family in families.items()
    }
    store.update_manifest(
        coherence_family_state_sha256s={
            name: file_sha256(path) for name, path in family_paths.items()
        }
    )

    batch_size = int(config["optimization"].get("batch_size", 1))
    epochs = int(config["optimization"].get("epochs", 1))
    fraction = float(config["optimization"].get("train_fraction", 1.0))
    if not 0.0 < fraction <= 1.0:
        raise ValueError("optimization.train_fraction must lie in (0,1]")
    steps_per_epoch = max(1, math.ceil(len(train_dataset) * fraction / batch_size))
    configured_steps = epochs * steps_per_epoch
    final_step = (
        min(configured_steps, start_step + max_steps) if max_steps is not None else configured_steps
    )
    if start_step >= final_step:
        raise ValueError("post-training would perform no optimizer steps")
    index_generator = torch.Generator(device="cpu").manual_seed(seed + 17)
    if checkpoint is not None:
        index_generator.set_state(checkpoint["rng_state"]["index_generator"])
    sampled_indices = iter_unique_batch_indices(
        len(train_dataset),
        final_step - start_step,
        batch_size,
        generator=index_generator,
    )
    query_points = (
        None
        if model.capabilities.structured_grid_required
        else int(config["coherence"]["compute_budget"]["point_count"])
    )
    batch_source = build_training_batch_source(
        train_dataset,
        sampled_indices,
        config,
        query_points=query_points,
        device=device,
        start_step=start_step,
    )
    store.update_manifest(
        training_data_strategy=batch_source.strategy,
        training_dataset_logical_bytes=dataset_field_bytes(train_dataset),
    )
    store.set_status("running", start_step=start_step, final_step=final_step)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    training_started = perf_counter()
    model.train()
    monitor = TrainingMonitor(
        store.run_dir,
        start_step=start_step,
        final_step=final_step,
        configured_steps=configured_steps,
        steps_per_epoch=steps_per_epoch,
        description=f"post:{config['model']['name']}",
        enabled=bool(config["runtime"].get("progress", True)),
        plot_every_steps=int(config["runtime"].get("plot_every_steps", 10)),
    )
    preview = TrainingReconstructionPreview(
        config,
        store=store,
        steps_per_epoch=steps_per_epoch,
        device=device,
    )
    checkpoint_manager = PeriodicCheckpointManager(
        config,
        store=store,
        steps_per_epoch=steps_per_epoch,
    )
    last_result = None
    for offset, batch in enumerate(batch_source):
        global_step = start_step + offset
        epoch = global_step // steps_per_epoch + 1
        model.train()
        data_loss = model.training_loss(batch).total
        every = int(config["coherence"]["schedule"].get("every_n_steps", 1))
        coherence_weight = _coherence_weight(config, epoch)
        coherence_active = coherence_weight > 0 and global_step % every == 0
        row: dict[str, Any] = {
            "step": global_step + 1,
            "epoch": epoch,
            "data_loss": float(data_loss.detach().cpu()),
            "coherence_applied": coherence_active,
            "coherence_weight": coherence_weight,
        }
        if coherence_active:
            rollout_generator = torch.Generator(device=device).manual_seed(
                seed + 1_000_003 + global_step
            )
            result, reference_ids = _coherence_objective(
                model,
                batch,
                families,
                banks,
                config,
                step=global_step,
                generator=rollout_generator,
            )
            coherence_loss = result.scalar_loss
            if bool(config["coherence"]["schedule"].get("interval_rescale", False)):
                coherence_loss = coherence_loss * every
            data_update_weight, coherence_update_weight = _balanced_weights(
                config, coherence_weight
            )
            gradient = two_objective_update(
                model,
                optimizer,
                data_loss,
                coherence_loss,
                mode=config["optimization"].get("gradient_balance", "weighted_sum"),
                data_weight=data_update_weight,
                coherence_weight=coherence_update_weight,
                grad_clip=config["optimization"].get("grad_clip"),
                config_missing_behavior=config["optimization"].get(
                    "config_missing_behavior", "error"
                ),
            )
            row.update(
                coherence_loss=float(result.scalar_loss.detach().cpu()),
                coherence_reference_ids=list(reference_ids),
                **_component_scalars(result),
                **gradient,
            )
            last_result = result
        else:
            row.update(
                data_only_update(
                    model,
                    optimizer,
                    data_loss,
                    weight=_data_weight(config),
                    grad_clip=config["optimization"].get("grad_clip"),
                )
            )
        store.append_history(row)
        monitor.record(row, lr=optimizer.param_groups[0]["lr"])
        if checkpoint_manager.due_for_preview_or_checkpoint(global_step + 1, preview):
            # Keep the case artifacts and checkpoint family state at the same
            # recovery boundary.
            family_paths = {
                name: store.save_artifact(f"{name}_family.pt", family.state_artifact())
                for name, family in families.items()
            }
            checkpoint_manager.save(
                _post_checkpoint_payload(
                    model,
                    optimizer,
                    global_step=global_step + 1,
                    config=config,
                    train_dataset=train_dataset,
                    families=families,
                    source_hashes_before=source_hashes_before,
                    index_generator=index_generator,
                    device=device,
                    config_sha256=store.config_hash,
                ),
                model=model,
                preview=preview,
                global_step=global_step + 1,
                fallback_metric=row["data_loss"],
            )

    monitor.close()
    batch_source.close()

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    post_training_seconds = perf_counter() - training_started
    after_metrics = _evaluate(
        model,
        evaluation_batch,
        comparison_batch,
        families,
        banks,
        train_dataset.field_names,
        config,
    )
    after_metrics["sensor_manifest_sha256"] = evaluation_manifest.digest()
    store.write_json("evaluation/after.json", after_metrics)
    family_paths = {
        name: store.save_artifact(f"{name}_family.pt", family.state_artifact())
        for name, family in families.items()
    }
    checkpoint_payload = _post_checkpoint_payload(
        model,
        optimizer,
        global_step=final_step,
        config=config,
        train_dataset=train_dataset,
        families=families,
        source_hashes_before=source_hashes_before,
        index_generator=index_generator,
        device=device,
        config_sha256=store.config_hash,
    )
    saved = checkpoint_manager.save(
        checkpoint_payload,
        model=model,
        preview=preview,
        global_step=final_step,
        fallback_metric=float(row["data_loss"]),
        force=True,
    )
    if saved is None:
        raise RuntimeError("final checkpoint save was unexpectedly skipped")
    last_path, _ = saved
    best_path = store.run_dir / "checkpoints" / "best.pt"
    if not best_path.is_file():
        best_path = store.save_checkpoint("best", checkpoint_payload)
    preview.close()
    source_hashes_after = source_hashes(config)
    if source_hashes_after != source_hashes_before:
        raise RuntimeError("source run changed during child post-training")
    status = "completed" if final_step >= configured_steps else "integration_truncated"
    store.update_manifest(
        checkpoint_hashes={
            "last": file_sha256(last_path),
            "latest": file_sha256(last_path),
            "best": file_sha256(best_path),
        },
        source_hashes_after=source_hashes_after,
        source_immutable_verified=True,
        evaluation_sensor_manifest_sha256=evaluation_manifest.digest(),
        before_metric=None if before_metrics is None else before_metrics.get("mse_normalized"),
        after_metric=after_metrics.get("mse_normalized"),
        coherence_family_state_sha256s={
            name: file_sha256(path) for name, path in family_paths.items()
        },
    )
    store.set_status(
        status,
        global_step=final_step,
        configured_steps=configured_steps,
        source_immutable_verified=True,
        coherence_target_use=(
            next(iter({family.target_use for family in families.values()}))
            if len({family.target_use for family in families.values()}) == 1
            else "mixed"
        ),
        coherence_target_uses={name: family.target_use for name, family in families.items()},
        final_coherence_loss=(
            None if last_result is None else float(last_result.scalar_loss.detach().cpu())
        ),
        post_training_seconds=post_training_seconds,
        seconds_per_step=post_training_seconds / max(final_step - start_step, 1),
        peak_cuda_memory_bytes=(
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
        ),
    )
    train_dataset.close()
    evaluation_dataset.close()
    return store.run_dir


def _post_checkpoint_payload(
    model,
    optimizer,
    *,
    global_step: int,
    config: Mapping[str, Any],
    train_dataset,
    families: Mapping[str, Any],
    source_hashes_before: Mapping[str, Any],
    index_generator: torch.Generator,
    device: torch.device,
    config_sha256: str,
) -> dict[str, Any]:
    """Build an internally consistent post-training recovery checkpoint."""
    return {
        "model": checkpoint_model_state(model),
        "optimizer": optimizer.state_dict(),
        "global_step": int(global_step),
        "source_run": str(config["source_run"]),
        "source_checkpoint": str(source_checkpoint_path(config)),
        "source_hashes": dict(source_hashes_before),
        "data_spec": asdict(train_dataset.data_spec),
        "normalization": train_dataset.normalizer.state_dict(),
        "family_states": {name: family.state_dict() for name, family in families.items()},
        "family_configs": {name: family.config for name, family in families.items()},
        "config_sha256": config_sha256,
        "rng_state": {
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
            "index_generator": index_generator.get_state(),
        },
    }

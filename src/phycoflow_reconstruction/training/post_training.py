"""Common differentiable data-coherence post-training.

The trainer always creates an immutable child run. Rectified-flow sources use
public flow hooks; diffusion, latent, and deterministic adapters use their
native differentiable reconstruction hook. Target use and provenance remain
explicit for every route.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import statistics
import warnings
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch

from ..coherence import ReferenceBank, build_enabled_families, fit_reference_bank
from ..contracts import DataSpec, FamilyResult, ObservationBatch
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
from .coherence_calibration import (
    gradient_diagnostics,
    gradient_norm,
    loss_gradients,
    resolve_family_scales,
)
from .common import (
    iter_unique_batch_indices,
    sensor_protocol_from_config,
)
from .fidelity_controller import (
    EPOCH_NATIVE_VERSION,
    NATIVE_VERSION,
    FidelityController,
    coherence_selection_report,
    endpoint_risks,
    fidelity_settings,
    matched_native_losses,
    r3_coherence_selection_report,
    unobserved_endpoint_risks,
)
from .gradient_balance import (
    calibrate_coherence_direction,
    coherence_primal_dual_update,
    combine_coherence_gradients,
    data_only_update,
    two_objective_update,
)
from .model_lifecycle import (
    add_training_aux_state,
    after_optimizer_step,
    evaluation_weight_context,
    load_training_aux_state,
)
from .monitoring import TrainingMonitor
from .parameter_interpolation import require_training_checkpoint
from .parameter_retention import (
    RANDOM_POLICY,
    SourceParameterRetention,
    private_call,
    scalar_update,
    trainable_parameter_identity,
)
from .preview import TrainingReconstructionPreview
from .retention import endpoint_retention, post_training_mode
from .rollout import differentiable_reconstruction, subset_query_batch
from .rollout_contract import verify_rollout_contract
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
from .topology_constraints import (
    ComponentConstrainedAdam,
    RegularizedTopologyAdam,
    calibrated_score,
    constraint_settings,
    fidelity_components,
    leaf_components,
    make_calibration,
    optimizer_source_digest,
    optimizer_source_snapshot,
    regularized_fidelity_loss,
    restore_constraint_audit,
    scalar_values,
    topology_constraint_components,
)
from .topology_selection import fidelity_eligibility, topology_selection_report
from .training_policy import persist_training_policy, resolve_training_policy
from .update_budget import post_training_final_step, post_training_steps_per_epoch, sample_exposure


def _configure_persistence_workers(config: Mapping[str, Any]) -> None:
    """Parallelize independent GUDHI pairings without changing their results.

    Keep this execution setting outside the versioned persistence definition so
    an existing run can resume after changing only the worker count. An explicit
    environment setting remains authoritative.
    """
    topology = config.get("coherence", {}).get("families", {}).get("topology", {})
    if topology.get("enabled") and topology.get("strategy") == "cubical_persistence":
        os.environ.setdefault("PHYCOFLOW_TOPOLOGY_WORKERS", str(min(4, os.cpu_count() or 1)))


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
    bank_rows: list[int],
) -> None:
    if not _requires_fixed_geometry(family):
        return
    if query_ids is None:
        raise TypeError(f"{family_name} requires serialized query indices")
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
    family_scales: Mapping[str, float] | None = None,
    retention_losses: dict[str, torch.Tensor] | None = None,
    source_anchor_model=None,
    step_context: dict | None = None,
    phase: str = "train",
    committed_step: int | None = None,
    require_source_prediction: bool = True,
) -> tuple[FamilyResult, tuple[str, ...]]:
    execution_mode = config.get("runtime", {}).get("execution_mode", "legacy")
    profile_timings = (execution_mode == "legacy"
                       or config.get("runtime", {}).get("profile_timings", False))
    compute = config["coherence"]["compute_budget"]
    selected = _slice_batch(batch, int(compute["batch_size"]))
    complete = selected
    selected = subset_query_batch(complete, int(compute["point_count"]), generator=generator)
    if not bool(selected.query_valid_mask.all()):
        raise ValueError("coherence requires every selected query_valid_mask entry to be true")
    # Dense targets are reference payloads only. They never enter a coherence
    # rollout, including explicitly paired-supervised structural losses.
    model_batch = complete if model.capabilities.structured_grid_required else selected
    model_batch = _without_target(model_batch)
    initial_generator_state = generator.get_state()
    rollout_started = perf_counter()
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
    if step_context is not None:
        if profile_timings and prediction.device.type == "cuda":
            torch.cuda.synchronize(prediction.device)
        rollout_seconds = perf_counter() - rollout_started

        def reconstruct(candidate):
            candidate_generator = torch.Generator(device=prediction.device)
            candidate_generator.set_state(initial_generator_state)
            value = differentiable_reconstruction(
                candidate,
                model_batch,
                steps=int(config["rollout"]["steps"]),
                solver=config["rollout"]["solver"],
                generator=candidate_generator,
                observation_config=dict(config["observation_consistency"]),
            )
            return (
                _gather_prediction(value, complete, selected)
                if model.capabilities.structured_grid_required
                else value
            )

        source_started = perf_counter()
        source_prediction = None
        if require_source_prediction:
            if source_anchor_model is None:
                raise ValueError("matched endpoint requires a frozen source model")
            with torch.no_grad():
                source_prediction = reconstruct(source_anchor_model)
        if profile_timings and prediction.device.type == "cuda":
            torch.cuda.synchronize(prediction.device)
        step_context.update(
            prediction=prediction,
            reference=selected.target_fields,
            coordinates=selected.query_coords,
            reconstruct=reconstruct,
            source_prediction=source_prediction,
            persistence_cache={},
            batch=selected,
            rollout_seconds=rollout_seconds,
            source_forward_seconds=perf_counter() - source_started,
            timing_scope="synchronized" if profile_timings else "host_enqueue_includes_required_transfers",
        )
    if retention_losses is not None:
        settings = config.get("objectives", {})
        for name in ("endpoint", "source_anchor"):
            term = settings.get(name, {})
            if not term.get("enabled", False):
                continue
            if selected.target_fields is None:
                raise ValueError("endpoint retention requires dense training targets")
            target = selected.target_fields
            if name == "source_anchor":
                if source_anchor_model is None:
                    raise ValueError("source_anchor requires the frozen source model")
                anchor_generator = torch.Generator(device=prediction.device)
                anchor_generator.set_state(initial_generator_state)
                with torch.no_grad():
                    target = differentiable_reconstruction(
                        source_anchor_model,
                        model_batch,
                        steps=int(config["rollout"]["steps"]),
                        solver=config["rollout"]["solver"],
                        generator=anchor_generator,
                        observation_config=dict(config["observation_consistency"]),
                    )
                if model.capabilities.structured_grid_required:
                    target = _gather_prediction(target, complete, selected)
            retention_losses[name] = endpoint_retention(
                prediction,
                target,
                scale_reference=selected.target_fields,
                variance_floor=float(term.get("variance_floor", 1e-8)),
            )
    families = dict(family) if isinstance(family, Mapping) else {family.family_name: family}
    banks = dict(bank) if isinstance(bank, Mapping) else {name: bank for name in families}
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
            bank_rows = family_bank.selection_indices(
                prediction.shape[0],
                step=step,
                current_sample_ids=selected.sample_ids,
                strict_distinct=True,
            )
            reference, reference_ids = family_bank.select(
                prediction.shape[0],
                step=step,
                device=prediction.device,
                dtype=prediction.dtype,
                current_sample_ids=selected.sample_ids,
                strict_distinct=True,
            )
            if reference.shape[1] != prediction.shape[1]:
                raise ValueError("reference-bank and coherence point counts differ")
            query_ids = selected.metadata.get("query_indices")
            _validate_reference_geometry(
                family_name,
                family_module,
                family_bank,
                query_ids if isinstance(query_ids, torch.Tensor) else None,
                bank_rows=bank_rows,
            )
        family_started = perf_counter()
        family_context = {
            "sample_ids": selected.sample_ids,
            "reference_ids": reference_ids,
            "global_step": step if committed_step is None else committed_step,
            "phase": phase,
            "execution_mode": execution_mode,
            "diagnostics": phase != "train" or (step_context or {}).get("diagnostics", True),
            **({"persistence_cache": step_context["persistence_cache"]}
               if step_context is not None else {}),
        }
        if step_context is not None:
            step_context.setdefault("family_contexts", {})[family_name] = family_context
        result = family_module(
            prediction,
            reference,
            coordinates=selected.query_coords,
            context=family_context,
        )
        if step_context is not None:
            if profile_timings and prediction.device.type == "cuda":
                torch.cuda.synchronize(prediction.device)
            step_context.setdefault("family_seconds", {})[family_name] = perf_counter() - family_started
        family_results[family_name] = result
        component_results.update(result.component_results)
        weight = float(getattr(family_module, "family_weight", 1.0))
        calibration_scale = float((family_scales or {}).get(family_name, 1.0))
        scalar_loss = scalar_loss + weight * calibration_scale * result.scalar_loss
        if result.per_sample_cost is not None:
            per_sample_parts.append(weight * calibration_scale * result.per_sample_cost)
        reference_ids_all.extend(reference_ids)
    per_sample = (
        torch.stack(per_sample_parts).sum(dim=0) if len(per_sample_parts) == len(families) else None
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
                    "calibration_scale": float((family_scales or {}).get(name, 1.0)),
                    "scalar_loss": result.scalar_loss.detach(),
                    "weighted_contribution": (
                        float(getattr(families[name], "family_weight", 1.0))
                        * float((family_scales or {}).get(name, 1.0))
                        * result.scalar_loss.detach()
                    ),
                    **result.diagnostics,
                }
                for name, result in family_results.items()
            },
        },
    )
    if step_context is not None:
        step_context["family_results"] = family_results
    return combined, tuple(dict.fromkeys(reference_ids_all))


def _calibrate_endpoint_fidelity(model, train_dataset, config, device, source_hash):
    """Fit source risk scales on fixed TRAIN batches and preserve caller RNG."""
    settings = fidelity_settings(config)
    count = int(settings["calibration_batches"])
    calibration_seed = int(settings.get("calibration_seed", int(config["runtime"].get("seed", 42)) + 700_061))
    rng = (torch.get_rng_state(), random.getstate(), np.random.get_state())
    cuda_rng = torch.cuda.get_rng_state_all() if device.type == "cuda" else []
    batches = None
    risks, records, native_losses = [], [], []
    native_constrained = settings.get("version") in {NATIVE_VERSION, EPOCH_NATIVE_VERSION}
    try:
        indices = iter_unique_batch_indices(
            len(train_dataset), count, int(config["optimization"]["batch_size"]),
            generator=torch.Generator().manual_seed(calibration_seed),
        )
        batches = build_training_batch_source(
            train_dataset, indices, config,
            query_points=None if model.capabilities.structured_grid_required
            else int(config["coherence"]["compute_budget"]["point_count"]),
            device=device, start_step=0,
        )
        with torch.no_grad():
            for index, batch in enumerate(batches):
                native_record = {}
                if native_constrained:
                    # Frozen source native calibration uses the ordinary full
                    # TRAIN batch, independently of endpoint subset scales.
                    native_seed = calibration_seed + 2_000_003 + index
                    torch.manual_seed(native_seed)
                    if device.type == "cuda":
                        torch.cuda.manual_seed_all(native_seed)
                    random.seed(native_seed)
                    np.random.seed(native_seed % 2**32)
                    native = model.training_loss(batch).total.detach()
                    if native.ndim != 0 or not torch.isfinite(native) or native < 0:
                        raise FloatingPointError("invalid TRAIN source native calibration loss")
                    native_losses.append(float(native))
                    native_record = {"native_seed": native_seed,
                                     "native_sample_ids": list(batch.sample_ids),
                                     "native_query_sha256": _query_digest(batch.metadata["query_indices"]),
                                     "source_native_loss": float(native)}
                generator = torch.Generator(device=device).manual_seed(calibration_seed + 1_000_003 + index)
                complete = _slice_batch(batch, int(config["coherence"]["compute_budget"]["batch_size"]))
                selected = subset_query_batch(complete, int(config["coherence"]["compute_budget"]["point_count"]), generator=generator)
                prediction = differentiable_reconstruction(
                    model, _without_target(complete if model.capabilities.structured_grid_required else selected),
                    steps=int(config["rollout"]["steps"]), solver=config["rollout"]["solver"],
                    generator=generator, observation_config=dict(config["observation_consistency"]),
                )
                if model.capabilities.structured_grid_required:
                    prediction = _gather_prediction(prediction, complete, selected)
                risk = endpoint_risks(prediction, selected.target_fields, selected.query_valid_mask)
                risks.append(risk.double().cpu())
                records.append({"sample_ids": list(selected.sample_ids),
                                "query_sha256": _query_digest(selected.metadata["query_indices"]),
                                "noise_seed": calibration_seed + 1_000_003 + index,
                                "source_field_risks": risk.cpu().tolist(), **native_record})
    finally:
        if batches is not None:
            batches.close()
        torch.set_rng_state(rng[0])
        random.setstate(rng[1])
        np.random.set_state(rng[2])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(cuda_rng)
    fields = torch.stack(risks).mean(dim=0)
    risk_vector = torch.cat((fields.mean()[None], fields)).clamp_min(float(settings["absolute_floor"]))
    artifact = {"version": "endpoint_model_mse_source_calibration_v1", "split": "train",
            "source_checkpoint_sha256": source_hash, "source_weight_selection": "live",
            "field_names": list(train_dataset.field_names), "source_risks": risk_vector.tolist(),
            "unfloored_source_field_risks": fields.tolist(), "settings": settings,
            "calibration_seed": calibration_seed, "batches": records}
    if native_constrained:
        mean_native = statistics.mean(native_losses)
        artifact["version"] = "endpoint_native_source_calibration_v2"
        artifact["native"] = {"version": "native_source_calibration_v2", "split": "train",
                              "source_scale": max(mean_native, float(settings["absolute_floor"])),
                              "unfloored_source_loss": mean_native,
                              "source_checkpoint_sha256": source_hash,
                              "source_weight_selection": "live",
                              "random_policy": "global_rng_same_native_draws_restore_post_live_v2"}
    return artifact


def _calibrate_topology_components(
    model, train_dataset, families, banks, config, device, source_hash
):
    settings = constraint_settings(config)
    calibration_seed = int(config["runtime"].get("seed", 42)) + 700_019
    torch_rng, python_rng, numpy_rng = (
        torch.get_rng_state(),
        random.getstate(),
        np.random.get_state(),
    )
    cuda_rng = torch.cuda.get_rng_state_all() if device.type == "cuda" else []
    source = None
    records = []
    try:
        indices = iter_unique_batch_indices(
            len(train_dataset),
            settings["calibration_batches"],
            int(config["optimization"]["batch_size"]),
            generator=torch.Generator().manual_seed(calibration_seed),
        )
        source = build_training_batch_source(
            train_dataset,
            indices,
            config,
            query_points=None
            if model.capabilities.structured_grid_required
            else int(config["coherence"]["compute_budget"]["point_count"]),
            device=device,
            start_step=0,
        )
        model.eval()
        with torch.no_grad():
            for i, batch in enumerate(source):
                result, ids = _coherence_objective(
                    model,
                    batch,
                    families,
                    banks,
                    config,
                    step=i,
                    generator=torch.Generator(device=device).manual_seed(calibration_seed + i),
                )
                records.append(
                    {
                        "sample_ids": list(ids),
                        "raw_total": float(result.scalar_loss),
                        "components": {k: float(v) for k, v in leaf_components(result).items()},
                    }
                )
        return make_calibration(
            records,
            floor=settings["scale_floor"],
            weights=families["topology"].spatial_objective.weights,
            source_hash=source_hash,
            objective_weight=float(config["objectives"]["coherence"]["weight"]),
        )
    finally:
        if source is not None:
            source.close()
        torch.set_rng_state(torch_rng)
        if cuda_rng:
            torch.cuda.set_rng_state_all(cuda_rng)
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)


def _constrained_topology_update(controller, result, context, family, config, field_names):
    calibration = family.component_calibration
    settings = constraint_settings(config)
    components = leaf_components(result)
    score = calibrated_score(components, calibration)
    constraints = topology_constraint_components(result, settings["topology_reduction"])
    topology_constraint_count = len(constraints)
    bounds = scalar_values(constraints)
    tolerances = {
        name: settings["numerical_tolerance"] * calibration["scales"][name.rsplit("#", 1)[0]]
        for name in constraints
    }
    fidelity_settings = {
        **settings,
        "max_relative_field_mse_increase": config["posttrain_fidelity"][
            "max_relative_field_mse_increase"
        ],
    }
    fidelity, fidelity_bounds = fidelity_components(
        context["prediction"],
        context["reference"],
        context["source_prediction"],
        field_names,
        fidelity_settings,
    )
    constraints.update(fidelity)
    bounds.update(fidelity_bounds)
    tolerances.update({name: settings["fidelity_numerical_tolerance"] for name in fidelity})

    def evaluate(candidate):
        # Match the baseline's attention/rollout kernels. No-grad inference can
        # select different fused kernels; a tiny bias matters to tight guards.
        # Detach immediately: only the live baseline needs backward passes.
        with torch.enable_grad():
            prediction = context["reconstruct"](candidate).detach()
        candidate_result = family(
            prediction,
            context["reference"],
            coordinates=context["coordinates"],
            context=context.get("family_contexts", {}).get(
                "topology", {"persistence_cache": context["persistence_cache"]}
            ),
        )
        candidate_components = leaf_components(candidate_result)
        candidate_fidelity, _ = fidelity_components(
            prediction,
            context["reference"],
            context["source_prediction"],
            field_names,
            fidelity_settings,
        )
        candidate_topology = topology_constraint_components(
            candidate_result, settings["topology_reduction"]
        )
        return {**candidate_topology, **candidate_fidelity}, calibrated_score(
            candidate_components, calibration
        )

    endpoint_loss = torch.stack(
        [v for k, v in fidelity.items() if k.startswith("fidelity.endpoint.")]
    ).mean()
    regularized = config["optimization"].get("gradient_balance") == "topology_regularized"
    proposal_loss = endpoint_loss if settings["proposal_objective"] == "endpoint" else None
    penalty = None
    if regularized:
        # Fixed calibration defines units; no batch-dependent loss rescaling.
        source_scale = sum(
            calibration["scales"][name] * calibration["coefficients"][name]
            for name in calibration["scales"]
        )
        penalty = (
            source_scale
            * settings["fidelity_penalty_weight"]
            * regularized_fidelity_loss(fidelity, fidelity_bounds, settings)
        )
        proposal_loss = score + penalty
    report = controller.step(
        score,
        constraints,
        bounds,
        tolerances,
        evaluate,
        grad_clip=config["optimization"].get("grad_clip"),
        proposal_loss=proposal_loss,
    )
    report["proposal_objective"] = (
        "topology_with_fidelity_regularization" if regularized else settings["proposal_objective"]
    )
    if penalty is not None:
        report["fidelity_penalty_before"] = float(penalty.detach())
    report["topology_reduction"] = settings["topology_reduction"]
    report["topology_constraint_count"] = topology_constraint_count
    report["fidelity_constraint_count"] = len(fidelity)
    report["constraint_batch_size"] = int(context["prediction"].shape[0])
    report["endpoint_nmse_before"] = float(
        torch.stack(
            [v.detach() for k, v in fidelity.items() if k.startswith("fidelity.endpoint.")]
        ).mean()
    )
    report["source_anchor_nmse_before"] = float(
        torch.stack(
            [v.detach() for k, v in fidelity.items() if k.startswith("fidelity.anchor.")]
        ).mean()
    )
    return report


def _component_scalars(result: FamilyResult) -> dict[str, float]:
    return {
        path: float(component.scalar_loss.detach().cpu())
        for path, component in result.component_results.items()
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu()
        return float(tensor) if tensor.ndim == 0 else tensor.tolist()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _component_reports(family, result: FamilyResult) -> dict[str, dict[str, Any]]:
    reported = result.diagnostics.get("components", {})
    output = {}
    for path, component in result.component_results.items():
        if component.diagnostics.get("evaluation_only", False):
            inner_weight = 0.0
        elif path in reported:
            inner_weight = float(reported[path]["weight"])
        else:
            matches = [
                float(weight)
                for key, weight in getattr(family, "component_weights", {}).items()
                if f".{key}." in f".{path}." or path.startswith(f"{family.family_name}.{key}.")
            ]
            inner_weight = matches[0] if len(matches) == 1 else 1.0
        raw = float(component.scalar_loss.detach().cpu())
        output[path] = {
            "raw_scalar": raw,
            "inner_weight": inner_weight,
            "weighted_contribution": inner_weight * raw,
            "diagnostics": _json_safe(component.diagnostics),
        }
    return output


def _family_scalar_report(result: FamilyResult) -> dict[str, float]:
    payload: dict[str, float] = {}
    for name, diagnostics in result.diagnostics.get("families", {}).items():
        raw = float(diagnostics["scalar_loss"])
        scale = float(diagnostics.get("calibration_scale", 1.0))
        outer = float(diagnostics.get("weight", 1.0))
        payload[f"coherence_family/{name}/raw"] = raw
        payload[f"coherence_family/{name}/outer_weight"] = outer
        payload[f"coherence_family/{name}/calibration_scale"] = scale
        payload[f"coherence_family/{name}/weighted_contribution"] = outer * scale * raw
    return payload


def _component_history_report(
    result: FamilyResult, families: Mapping[str, Any] | None = None
) -> dict[str, float]:
    """Return generic raw and additive component metrics for history rendering."""
    payload: dict[str, float] = {}
    family_diagnostics_by_name = result.diagnostics.get("families", {})
    paths = set(result.component_results)
    paths.update(
        path
        for diagnostics in family_diagnostics_by_name.values()
        for path, component in diagnostics.get("components", {}).items()
        if component.get("executed", component.get("raw_scalar_loss") is not None)
    )
    for path in sorted(paths):
        family_name = path.split(".", 1)[0]
        family_diagnostics = family_diagnostics_by_name.get(family_name, {})
        outer = float(family_diagnostics.get("weight", 1.0))
        calibration = float(family_diagnostics.get("calibration_scale", 1.0))
        diagnostics = family_diagnostics.get("components", {}).get(path, {})
        raw_value = diagnostics.get("raw_scalar_loss")
        if raw_value is None and path in result.component_results:
            raw_value = float(result.component_results[path].scalar_loss.detach().cpu())
        if raw_value is None:
            continue
        term = result.component_results.get(path)
        if term is not None and term.diagnostics.get("evaluation_only", False):
            inner = 0.0
        elif "weight" in diagnostics:
            inner = float(diagnostics["weight"])
        else:
            family = (families or {}).get(family_name)
            matches = [
                float(weight)
                for key, weight in getattr(family, "component_weights", {}).items()
                if f".{key}." in f".{path}." or path.startswith(f"{family_name}.{key}.")
            ]
            inner = matches[0] if len(matches) == 1 else 1.0
        component = str(path).removeprefix(f"{family_name}.")
        raw = float(raw_value)
        prefix = f"coherence_component/{family_name}/{component}"
        payload[f"{prefix}/role"] = ("evaluation_only" if term is not None and
            term.diagnostics.get("evaluation_only", False) else "training_component")
        payload[f"{prefix}/raw"] = raw
        payload[f"{prefix}/weighted_contribution"] = raw * inner * outer * calibration
        payload[f"{prefix}/inner_weight"] = inner
        payload[f"{prefix}/outer_weight"] = outer
        payload[f"{prefix}/calibration_scale"] = calibration
    return payload


def _query_digest(query_ids: torch.Tensor) -> str:
    payload = query_ids.detach().to(device="cpu", dtype=torch.long).contiguous().numpy()
    return hashlib.sha256(payload.tobytes()).hexdigest()


def _median_nested_cosines(records: list[Mapping[str, Any]], key: str) -> dict[str, Any]:
    first = records[0][key]
    if key == "data_family_cosines":
        return {
            name: float(statistics.median(record[key][name] for record in records))
            for name in first
        }
    return {
        left: {
            right: float(statistics.median(record[key][left][right] for record in records))
            for right in first[left]
        }
        for left in first
    }


def _calibrate_finite_primary(
    model, train_dataset, families, banks, config, device, source_hash,
):
    """Fit finite/essential units on the fixed TRAIN source panel, before gradients.

    The source predictions use the same panel and rollout seeds as family balance.
    Calibration evaluates the full master bank; training still samples four lines.
    These units are internal to C, while family balance supplies one outer scalar.
    """
    topology = families.get("topology")
    objective = getattr(topology, "spatial_objective", None)
    r3_reporting = config.get("checkpointing", {}).get("exploratory_policy", {}).get("version") == "r3_endpoint_corridor_v1"
    if objective is None or (getattr(objective, "aggregation", None) != "finite_primary_v1"
                             and not r3_reporting):
        return None
    settings = config["coherence"].get("family_balance", {})
    count = int(settings.get("calibration_batches", 2))
    if not 1 <= count <= 8:
        raise ValueError("finite-primary TRAIN calibration requires one to eight batches")
    seed = int(settings.get("seed", config["runtime"].get("seed", 42) + 700_001))
    indices = iter_unique_batch_indices(
        len(train_dataset), count, int(config["optimization"].get("batch_size", 1)),
        generator=torch.Generator().manual_seed(seed),
    )
    source = build_training_batch_source(
        train_dataset, indices, config,
        query_points=(None if model.capabilities.structured_grid_required
                      else int(config["coherence"]["compute_budget"]["point_count"])),
        device=device, start_step=0,
    )
    records, panels = [], []
    cpu_rng, python_rng, numpy_rng = torch.get_rng_state(), random.getstate(), np.random.get_state()
    cuda_rng = torch.cuda.get_rng_state_all() if device.type == "cuda" else []
    try:
        post_training_mode(model, config)
        with torch.no_grad():
            for index, batch in enumerate(source):
                rollout_seed = seed + 1_000_003 + index
                result, _ = _coherence_objective(
                    model, batch, {"topology": topology}, {"topology": banks["topology"]},
                    config, step=index, generator=torch.Generator(device=device).manual_seed(rollout_seed),
                    phase="calibration",
                )
                records.append(objective.collect_source_components(result))
                selected = subset_query_batch(
                    _slice_batch(batch, int(config["coherence"]["compute_budget"]["batch_size"])),
                    int(config["coherence"]["compute_budget"]["point_count"]),
                    generator=torch.Generator(device=device).manual_seed(rollout_seed),
                )
                panels.append({"sample_ids": list(selected.sample_ids),
                               "query_sha256": _query_digest(selected.metadata["query_indices"]),
                               "rollout_seed": rollout_seed})
    finally:
        source.close()
        torch.set_rng_state(cpu_rng)
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(cuda_rng)
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
    return objective.freeze_source_calibration(records, {
        "split": "train", "source_checkpoint_sha256": source_hash,
        "weight_identity": "live_training_weights", "calibration_seed": seed,
        "panels": panels, "raw_source_components": records,
    }, representation="training_raster")


def _calibrate_family_balance(
    model,
    train_dataset: FieldDataset,
    families: Mapping[str, Any],
    banks: Mapping[str, ReferenceBank | None],
    config: Mapping[str, Any],
    *,
    device: torch.device,
    source_hashes_before: Mapping[str, str],
) -> dict[str, Any]:
    """Calibrate fixed family scales at the untouched source checkpoint."""
    settings = dict(config["coherence"].get("family_balance", {}))
    mode = str(settings.get("mode", "none"))
    if mode == "none":
        return {
            "version": 1,
            "mode": mode,
            "settings": settings,
            "source_checkpoint_sha256": source_hashes_before["checkpoint"],
            "weight_identity": "live_training_weights",
            "resolved_scales": {name: 1.0 for name in families},
            "outer_weights": {
                name: float(getattr(family, "family_weight", 1.0))
                for name, family in families.items()
            },
            "calibration_batches": [],
        }
    if mode != "initial_grad_norm":
        raise ValueError(f"unsupported family balance mode: {mode}")

    count = int(settings.get("calibration_batches", 2))
    calibration_seed = int(settings.get("seed", config["runtime"].get("seed", 42) + 700_001))
    batch_size = int(config["optimization"].get("batch_size", 1))
    index_generator = torch.Generator(device="cpu").manual_seed(calibration_seed)
    sampled_indices = iter_unique_batch_indices(
        len(train_dataset), count, batch_size, generator=index_generator
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
        start_step=0,
    )
    parameters = tuple(parameter for parameter in model.parameters() if parameter.requires_grad)
    if not parameters:
        raise ValueError("family calibration selected no trainable parameters")
    epsilon = float(settings.get("epsilon", 1.0e-12))
    batch_records: list[dict[str, Any]] = []
    started = perf_counter()
    peak_before = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    torch_cpu_rng = torch.get_rng_state()
    torch_cuda_rng = torch.cuda.get_rng_state_all() if device.type == "cuda" else []
    python_rng = random.getstate()
    numpy_rng = np.random.get_state()
    direction_grams = []
    calibrate_direction = config["optimization"].get("coherence_direction_calibration", "none") != "none"
    r3_diagnostics = config.get("checkpointing", {}).get("exploratory_policy", {}).get("version") == "r3_endpoint_corridor_v1"
    if r3_diagnostics and not 1 <= count <= 8:
        raise ValueError("R3 TRAIN component diagnosis requires at most eight batches")
    try:
        post_training_mode(model, config)
        for batch_index, batch in enumerate(batch_source):
            spectrum = families.get("cross_spectrum")
            if batch_index == 0 and getattr(spectrum, "definition", None) == "second_order_blocks_v4":
                # Only training targets establish the frozen spectral masks
                # and scales, before source gradient calibration or selection.
                selected_reference = subset_query_batch(
                    _slice_batch(batch, int(config["coherence"]["compute_budget"]["batch_size"])),
                    int(config["coherence"]["compute_budget"]["point_count"]),
                    generator=torch.Generator(device=device).manual_seed(calibration_seed + 1_000_003),
                )
                spectrum.freeze_reference_calibration(
                    selected_reference.target_fields, selected_reference.query_coords
                )
            data_loss = model.training_loss(batch).total
            data_gradients = loss_gradients(data_loss, parameters, retain_graph=r3_diagnostics)
            family_gradients = {}
            family_losses = {}
            component_diagnostics = {}
            rollout_seed = calibration_seed + 1_000_003 + batch_index
            for name, family in families.items():
                context = {} if r3_diagnostics else None
                result, _ = _coherence_objective(
                    model,
                    batch,
                    {name: family},
                    {name: banks[name]},
                    config,
                    step=batch_index,
                    generator=torch.Generator(device=device).manual_seed(rollout_seed),
                    phase="calibration",
                    **({"step_context": context, "source_anchor_model": model}
                       if r3_diagnostics else {}),
                )
                if r3_diagnostics:
                    from .coherence_diagnostics import component_gradient_diagnostics

                    endpoint = endpoint_risks(context["prediction"], context["reference"]).mean()
                    component_diagnostics[name] = component_gradient_diagnostics(
                        context["family_results"], {name: family}, parameters,
                        batch_index=batch_index, sample_ids=context["batch"].sample_ids,
                        source_checkpoint_sha256=source_hashes_before["checkpoint"],
                        native_loss=data_loss if name == "topology" else None,
                        endpoint_loss=endpoint if name == "topology" else None,
                        native_pressure=data_loss * 0 if name == "topology" else None,
                        endpoint_pressure=endpoint * 0 if name == "topology" else None,
                    )
                    component_diagnostics[name]["pressure_context"] = "SOURCE initialization; zero multipliers and feasible slack; pressure gradient is zero"
                outer_weight = float(getattr(family, "family_weight", 1.0))
                raw_loss = result.scalar_loss / outer_weight
                family_losses[name] = float(raw_loss.detach())
                family_gradients[name] = loss_gradients(raw_loss, parameters)
            diagnostics = gradient_diagnostics(data_gradients, family_gradients, epsilon=epsilon)
            if calibrate_direction or r3_diagnostics:
                matrix = torch.stack([
                    torch.cat([(torch.view_as_real(g) if g.is_complex() else g).reshape(-1).double()
                               if g is not None else
                               torch.zeros(p.numel() * (2 if p.is_complex() else 1), device=p.device, dtype=torch.float64)
                               for p, g in zip(parameters, family_gradients[name])])
                    for name in families])
                direction_grams.append((matrix @ matrix.T).cpu().tolist())
                del matrix
            selected = subset_query_batch(
                _slice_batch(batch, int(config["coherence"]["compute_budget"]["batch_size"])),
                int(config["coherence"]["compute_budget"]["point_count"]),
                generator=torch.Generator(device=device).manual_seed(rollout_seed),
            )
            query_ids = selected.metadata.get("query_indices")
            if not isinstance(query_ids, torch.Tensor):
                raise TypeError("family calibration requires serialized query indices")
            batch_records.append(
                {
                    "batch": batch_index,
                    "sample_ids": list(selected.sample_ids),
                    "query_digest": _query_digest(query_ids),
                    "native_data_loss": float(data_loss.detach()),
                    "family_losses": family_losses,
                    **({"R3_component_diagnostics": component_diagnostics} if r3_diagnostics else {}),
                    **diagnostics,
                }
            )
    finally:
        batch_source.close()
        torch.set_rng_state(torch_cpu_rng)
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(torch_cuda_rng)
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
    aggregate_norms, scales = resolve_family_scales(
        [record["family_gradient_norms"] for record in batch_records], settings
    )
    aggregate_losses = {
        name: float(statistics.median(record["family_losses"][name] for record in batch_records))
        for name in families
    }
    outer_weights = {
        name: float(getattr(family, "family_weight", 1.0)) for name, family in families.items()
    }
    peak_after = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    reference_norm = float(statistics.median(aggregate_norms.values()))
    epsilon = float(settings.get("epsilon", 1.0e-12))
    unclipped_scales = {
        name: reference_norm / max(norm, epsilon) for name, norm in aggregate_norms.items()
    }
    clipped_families = [
        name
        for name in families
        if not math.isclose(scales[name], unclipped_scales[name], rel_tol=1.0e-12)
    ]
    artifact = {
        "version": 1,
        "mode": mode,
        "settings": settings,
        "source_checkpoint_sha256": source_hashes_before["checkpoint"],
        "weight_identity": "live_training_weights",
        "configured_evaluation_weight_identity": (
            "ema"
            if getattr(model, "_ema", None) is not None and getattr(model, "_ema_eval", False)
            else "live"
        ),
        "sample_ids": [record["sample_ids"] for record in batch_records],
        "query_digests": [record["query_digest"] for record in batch_records],
        "source_family_losses": aggregate_losses,
        "raw_family_gradient_norms": aggregate_norms,
        "resolved_scales": scales,
        "unclipped_scales": unclipped_scales,
        "clipped_families": clipped_families,
        "outer_weights": outer_weights,
        "effective_weighted_gradient_norms": {
            name: outer_weights[name] * scales[name] * aggregate_norms[name] for name in families
        },
        "native_data_gradient_norm": float(
            statistics.median(record["native_data_gradient_norm"] for record in batch_records)
        ),
        "family_family_cosines": _median_nested_cosines(batch_records, "family_family_cosines"),
        "data_family_cosines": _median_nested_cosines(batch_records, "data_family_cosines"),
        "calibration_seconds": perf_counter() - started,
        "peak_cuda_memory_delta_bytes": max(0, peak_after - peak_before),
        "seeds": {
            "runtime": int(config["runtime"].get("seed", 42)),
            "calibration": calibration_seed,
            "rollout_first": calibration_seed + 1_000_003,
        },
        "calibration_batches": batch_records,
    }
    if calibrate_direction:
        direction = calibrate_coherence_direction(
            direction_grams, list(families),
            {name: outer_weights[name] * scales[name] for name in families},
            cagrad_alpha=float(config["optimization"].get("cagrad_alpha", .5)),
            cagrad_rescale=int(config["optimization"].get("cagrad_rescale", 1)),
        )
        direction.update(source_checkpoint_sha256=source_hashes_before["checkpoint"],
                         sample_ids=artifact["sample_ids"], query_digests=artifact["query_digests"],
                         seeds=artifact["seeds"], raw_source_gradient_grams=direction_grams)
        artifact["combined_direction"] = direction
    if r3_diagnostics:
        factors = torch.tensor([outer_weights[name] * scales[name] for name in families], dtype=torch.float64)
        direction_scale = artifact.get("combined_direction", {}).get("resolved_scales", {}).get("config", 1.0)
        for record, gram in zip(batch_records, direction_grams, strict=True):
            weighted = torch.as_tensor(gram, dtype=torch.float64) * factors[:, None] * factors[None, :]
            values, vectors = torch.linalg.eigh((weighted + weighted.T) / 2)
            rows = vectors * values.clamp_min(0).sqrt()[None, :]
            combined, combiner = combine_coherence_gradients(dict(zip(families, rows)), method="config")
            norm = float(torch.linalg.vector_norm(combined)) * direction_scale
            for name, diagnosis in record["R3_component_diagnostics"].items():
                diagnosis.update(ABC_norm=norm, ABC_norm_role="exact_TRAIN_ConFIG_gradient_Gram_isometry",
                                 coherence_direction_scale=direction_scale, combiner=combiner)
                for path, item in diagnosis["components"].items():
                    if path.startswith(name + ".") and not path.endswith(("finite_raw", "essential_raw")):
                        item["weight"] *= scales[name]
                        item["weighted_value"] *= scales[name]
                        if item["weighted_parameter_gradient_norm"] is not None:
                            item["weighted_parameter_gradient_norm"] *= scales[name]
                    item["norm_relative_to_ABC"] = (item["weighted_parameter_gradient_norm"] / norm
                        if norm > 0 and item["weighted_parameter_gradient_norm"] is not None else None)
    return artifact


def _sparse_family_gradient_diagnostics(
    model,
    batch: ObservationBatch,
    data_loss: torch.Tensor,
    combined_loss: torch.Tensor,
    families: Mapping[str, Any],
    banks: Mapping[str, ReferenceBank | None],
    config: Mapping[str, Any],
    *,
    step: int,
    rollout_seed: int,
    device: torch.device,
) -> dict[str, Any]:
    parameters = tuple(parameter for parameter in model.parameters() if parameter.requires_grad)
    epsilon = float(config["coherence"].get("family_balance", {}).get("epsilon", 1.0e-12))
    data_gradients = loss_gradients(data_loss, parameters, retain_graph=True)
    combined_gradients = loss_gradients(combined_loss, parameters, retain_graph=True)
    family_gradients = {}
    family_losses = {}
    for name, family in families.items():
        result, _ = _coherence_objective(
            model,
            batch,
            {name: family},
            {name: banks[name]},
            config,
            step=step,
            generator=torch.Generator(device=device).manual_seed(rollout_seed),
        )
        raw_loss = result.scalar_loss / float(getattr(family, "family_weight", 1.0))
        family_losses[name] = float(raw_loss.detach())
        family_gradients[name] = loss_gradients(raw_loss, parameters)
    return {
        "family_losses": family_losses,
        **gradient_diagnostics(data_gradients, family_gradients, epsilon=epsilon),
        "total_coherence_gradient_norm": gradient_norm(combined_gradients),
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


def _shared_scalar_family_diagnostics(model, data_loss, coherence_loss, family_results, config):
    """Sparse R5 diagnostics reuse the one ordinary live rollout graph."""
    parameters = tuple(parameter for parameter in model.parameters() if parameter.requires_grad)

    def gradients(loss):
        values = loss_gradients(loss, parameters, retain_graph=True)
        return tuple(torch.view_as_real(value.resolve_conj()) if value is not None and value.is_complex()
                     else value for value in values)

    data_gradients = gradients(data_loss)
    coherence_gradients = gradients(coherence_loss)
    family_gradients = {name: gradients(result.scalar_loss)
                        for name, result in family_results.items()}
    return {"family_losses": {name: float(result.scalar_loss.detach())
                              for name, result in family_results.items()},
            **gradient_diagnostics(data_gradients, family_gradients,
                epsilon=float(config["coherence"].get("family_balance", {}).get("epsilon", 1e-12))),
            "total_coherence_gradient_norm": gradient_norm(coherence_gradients),
            "reused_training_rollout": True}


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
    selection = config.get("evaluation", {}).get("sample_selection", "first")
    r2 = config.get("evaluation", {}).get("r2_protocol", {})
    if r2.get("enabled", False):
        from ..evaluation.upgrade_1002_r2_protocol import declare_panels

        indices = declare_panels(
            len(dataset), previously_used_indices=r2.get("previously_used_validation_indices", ())
        )["selection"]["dataset_indices"]
        return build_observation_batch([dataset[index] for index in indices], protocol, query_points=None).to(device)
    indices = (
        np.linspace(0, len(dataset) - 1, count).round().astype(int).tolist()
        if selection == "uniform"
        else list(range(count))
    )
    samples = [dataset[index] for index in indices]
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
                    config["observations"].get("seed", config["runtime"].get("seed", 42)) + 100_003,
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
    *,
    normalizer=None,
    family_scales: Mapping[str, float] | None = None,
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
    with evaluation_weight_context(model), torch.no_grad():
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
    with evaluation_weight_context(model), torch.no_grad():
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
        metrics = reconstruction_metrics(
            prediction,
            target,
            comparison_batch,
            field_names,
            normalizer=normalizer,
        )
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
                    current_sample_ids=comparison_batch.sample_ids,
                    strict_distinct=True,
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
                    bank_rows=family_bank.selection_indices(
                        prediction.shape[0],
                        step=0,
                        current_sample_ids=comparison_batch.sample_ids,
                        strict_distinct=True,
                    ),
                )
            family_result = family_module(
                prediction,
                reference,
                coordinates=comparison_batch.query_coords,
                context={
                    "sample_ids": comparison_batch.sample_ids,
                    "reference_ids": reference_ids,
                    "phase": "evaluation",
                    "global_step": 0,
                },
            )
            weight = float(getattr(family_module, "family_weight", 1.0))
            calibration_scale = float((family_scales or {}).get(family_name, 1.0))
            combined_total = combined_total + weight * calibration_scale * family_result.scalar_loss
            combined_components.update(_component_scalars(family_result))
            all_reference_ids.extend(reference_ids)
            family_payloads[family_name] = {
                "target_use": family_module.target_use,
                "units": family_module.units,
                "weight": weight,
                "calibration_scale": calibration_scale,
                "total": float(family_result.scalar_loss.cpu()),
                "weighted_total": float(
                    (weight * calibration_scale * family_result.scalar_loss).cpu()
                ),
                "components": _component_reports(family_module, family_result),
                "component_scalars": _component_scalars(family_result),
                "reference_ids": list(reference_ids),
                "diagnostics": _json_safe(family_result.diagnostics),
            }
            component_calibration = getattr(family_module, "component_calibration", None)
            if component_calibration is not None:
                family_payloads[family_name]["selection_score"] = float(
                    calibrated_score(leaf_components(family_result), component_calibration)
                )
                family_payloads[family_name]["selection_score_definition"] = (
                    "fixed_train_source_component_scales"
                )
                family_payloads[family_name]["selection_components"] = {
                    name: {
                        "raw": float(value),
                        "source_scale": component_calibration["scales"][name],
                        "coefficient": component_calibration["coefficients"][name],
                        "weighted_contribution": float(value)
                        * component_calibration["coefficients"][name],
                    }
                    for name, value in leaf_components(family_result).items()
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
        native_topology = None
        if evaluation.get("native_topology", False):
            topology = families["topology"]
            native_config = deepcopy(topology.config)
            native_config["geometry"]["grid_shape"] = list(config["dataset"]["grid_shape"])
            # At equal resolution the area map is an exact permutation. Avoid
            # small IDW interpolation errors at floating-point grid coordinates.
            native_config["geometry"]["antialias_downsample"] = True
            native_config["filtration"]["smoothing_sigma"] = 0.0
            native_family = type(topology)(
                native_config,
                DataSpec(
                    field_names,
                    tuple(config["dataset"]["field_units"]),
                    comparison_batch.query_coords.shape[-1],
                    tuple(config["dataset"]["grid_shape"]),
                ),
                normalizer,
            ).to(prediction.device)
            reuse_native = (
                set(families) == {"topology"}
                and getattr(topology, "component_calibration", None) is not None
                and tuple(topology.grid_shape) == tuple(config["dataset"]["grid_shape"])
                and topology.smoothing_sigma == 0.0
                and topology.config["geometry"].get("antialias_downsample", False)
            )
            native = (
                family_result
                if reuse_native
                else native_family(prediction, target, coordinates=comparison_batch.query_coords)
            )
            native_topology = {
                "total": float(native.scalar_loss.cpu()),
                "components": _component_scalars(native),
                "grid_shape": list(config["dataset"]["grid_shape"]),
            }
    return {
        **metrics,
        "native_topology": native_topology,
        "inference": {
            "seconds": inference_seconds,
            "samples": prediction.shape[0],
            "points_per_sample": prediction.shape[1],
            "generation_steps": int(evaluation.get("generation_steps", 2)),
        },
        "coherence": coherence_payload,
        "sample_ids": list(comparison_batch.sample_ids),
        "query_ids": (
            comparison_batch.metadata["query_indices"].detach().cpu().tolist()
            if isinstance(comparison_batch.metadata.get("query_indices"), torch.Tensor)
            else None
        ),
        "generation_seed": evaluation_seed,
        "generation_steps": int(evaluation.get("generation_steps", 2)),
        "evaluation_weight_source": "configured_evaluation_weights",
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
                        config["observations"].get("seed", config["runtime"].get("seed", 42))
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


def _add_source_relative_coherence(before: Mapping[str, Any], after: dict[str, Any]) -> None:
    source_families = before.get("coherence", {}).get("families", {})
    for name, payload in after.get("coherence", {}).get("families", {}).items():
        source = source_families.get(name, {}).get("total")
        current = payload.get("total")
        payload["source_total"] = source
        if (
            source is None
            or current is None
            or not math.isfinite(float(source))
            or float(source) == 0
        ):
            payload["source_normalized_ratio"] = None
            payload["relative_reduction"] = None
        else:
            ratio = float(current) / float(source)
            payload["source_normalized_ratio"] = ratio
            payload["relative_reduction"] = 1.0 - ratio


def _fidelity_guard_report(
    before: Mapping[str, Any], after: Mapping[str, Any], config: Mapping[str, Any]
) -> dict[str, Any]:
    settings = config.get("posttrain_fidelity", {})
    threshold = settings.get("max_relative_mse_increase")
    before_mse = float(before["mse_normalized"])
    after_mse = float(after["mse_normalized"])
    relative_increase = (
        math.inf
        if before_mse == 0 and after_mse > 0
        else 0.0
        if before_mse == 0
        else (after_mse - before_mse) / before_mse
    )
    exceeded = threshold is not None and relative_increase > float(threshold)
    per_field = None
    if threshold is not None and settings.get("max_relative_field_mse_increase") is not None:
        per_field = fidelity_eligibility(before, after, settings)
        exceeded = not per_field["eligible"]
    return {
        "source_mse_normalized": before_mse,
        "posttrain_mse_normalized": after_mse,
        "relative_mse_increase": relative_increase,
        "max_relative_mse_increase": None if threshold is None else float(threshold),
        "behavior": settings.get("behavior", "report"),
        "status": "exceeded" if exceeded else "passed" if threshold is not None else "reported",
        "exceeded": exceeded,
        "per_field_guard": per_field,
    }


def run_post_training(
    config: Mapping[str, Any],
    *,
    case_dir: str | Path,
    max_steps: int | None = None,
    resume: str | Path | None = None,
    until_epoch: int | None = None,
    additional_epochs: int | None = None,
) -> Path:
    """Create or resume one immutable child post-training run."""
    if sum(value is not None for value in (max_steps, until_epoch, additional_epochs)) > 1:
        raise ValueError("choose only one of max_steps, until_epoch, additional_epochs")
    evaluation_settings = config.get("evaluation", {})
    native_audit_settings = (
        evaluation_settings.get("native_topology_audit", {})
        if isinstance(evaluation_settings, Mapping)
        else {}
    )
    native_audit = (
        native_audit_settings
        if isinstance(native_audit_settings, Mapping)
        and bool(native_audit_settings.get("enabled", False))
        else None
    )
    if native_audit is not None:
        from .native_topology_audit import (
            AUDIT_SUBDIR,
            audit_due,
            build_disjoint_validation_batch,
            build_native_topology_family,
            execute_native_audit,
            initialize_native_geometry,
            verify_saved_family_hash,
            write_idempotent_report,
        )
        from .native_topology_audit import contract_payload as native_audit_contract_payload

    adaptive = config["optimization"].get("update_policy", "legacy") == "coherence_primal_dual"
    matched_streams = config["runtime"].get("random_stream_policy") == RANDOM_POLICY
    parameter_settings = config.get("objectives", {}).get("parameter_retention", {})
    parameter_enabled = bool(parameter_settings.get("enabled", False))
    r2_settings = config.get("evaluation", {}).get("r2_protocol", {})
    r2_enabled = bool(r2_settings.get("enabled", False))
    test_lineage = (Path(str(config.get("output", {}).get("experiment_name", ""))).parts or ("",))[0] == "Test_1002"
    if test_lineage or matched_streams or parameter_enabled or native_audit is not None or adaptive or r2_enabled or config["optimization"].get("gradient_balance") in {
        "component_constrained",
        "topology_regularized",
    }:
        # Direct Python callers must not bypass the every-update/geometry guards.
        from ..config.validate import validate_config

        validate_config(config, invocation_until_epoch=until_epoch)
    if config["stage"] != "post_training":
        raise ValueError("run_post_training accepts only stage=post_training")
    if max_steps is not None and max_steps < 0:
        raise ValueError("max_steps must be non-negative")
    _configure_persistence_workers(config)
    seed = int(config["runtime"].get("seed", 42))
    seed_everything(seed, bool(config["runtime"].get("deterministic", True)))
    device = torch.device(config["runtime"].get("device", "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    source_hashes_before = source_hashes(config)
    model, train_dataset, source_metadata = load_source_model(config, device)
    execution = config["optimization"].get("rollout_execution", {})
    for key, attribute in (
        ("context_cache", "rollout_context_cache"),
        ("checkpointing", "rollout_checkpointing"),
    ):
        if key in execution:
            if not hasattr(model, attribute):
                raise ValueError("rollout_execution requires a compatible physical CQ model")
            setattr(model, attribute, execution[key])
    subset_manifest = None
    if "training_subset" in config["optimization"]:
        from ..data.topology_subset import apply_training_subset

        subset_manifest = apply_training_subset(
            train_dataset, config["optimization"]["training_subset"]
        )
    constrained = config["optimization"].get("gradient_balance") in {
        "component_constrained",
        "topology_regularized",
    }
    source_anchor_model = None
    if (
        native_audit is not None
        or r2_enabled
        or adaptive
        or constrained
        or config.get("objectives", {}).get("source_anchor", {}).get("enabled", False)
    ):
        # Snapshot before loading a resumed child: the anchor is always the source.
        source_anchor_model = deepcopy(model).eval().requires_grad_(False)
    if not len(train_dataset):
        raise ValueError(f"training split is empty for {config['dataset']['path']}")
    if not model.capabilities.differentiable_rollout:
        raise ValueError("source model does not support differentiable rollout post-training")
    trainable_names = set_trainable_scope(model, config["trainable"])
    parameter_retention = None
    if parameter_enabled:
        # Bind original LIVE SOURCE before a resumed child is loaded. No buffer,
        # prior, optimizer or family tensor is retained by this helper.
        requested_parameter_artifact = json.loads(
            Path(parameter_settings["calibration_path"]).read_text(encoding="utf-8")
        )
        parameter_retention = SourceParameterRetention(
            model, requested_parameter_artifact,
            source_checkpoint_sha256=source_hashes_before["checkpoint"],
        )
    if matched_streams:
        source_metadata["initial_trainable_parameter_identity"] = (
            parameter_retention.identity if parameter_retention is not None
            else trainable_parameter_identity(model))
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
        require_training_checkpoint(checkpoint)
        load_model_state_strict(model, checkpoint["model"])
        load_training_aux_state(model, checkpoint)
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = int(checkpoint["global_step"])
        store.recover_metric_histories(start_step)
        torch.set_rng_state(checkpoint["rng_state"]["torch_cpu"])
        if device.type == "cuda" and checkpoint["rng_state"].get("torch_cuda"):
            torch.cuda.set_rng_state_all(checkpoint["rng_state"]["torch_cuda"])
        if matched_streams or (adaptive and fidelity_settings(config).get("version") in {NATIVE_VERSION, EPOCH_NATIVE_VERSION}):
            if not {"python", "numpy"} <= checkpoint["rng_state"].keys():
                raise ValueError("native v2 resume is missing Python/NumPy RNG state")
            random.setstate(checkpoint["rng_state"]["python"])
            numpy_state = checkpoint["rng_state"]["numpy"]
            np.random.set_state((numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32),
                                 *numpy_state[2:]))

    if parameter_retention is not None:
        parameter_path = store.run_dir / "artifacts/parameter_retention.json"
        if resume is not None:
            if not parameter_path.is_file():
                raise ValueError("resume is missing frozen parameter retention calibration")
            if json.loads(parameter_path.read_text()) != requested_parameter_artifact:
                raise ValueError("resume parameter retention calibration changed")
            parameter_retention.verify_resume(checkpoint.get("parameter_retention"))
        else:
            store.write_json("artifacts/parameter_retention.json", requested_parameter_artifact)
        store.update_manifest(parameter_retention=parameter_retention.state_dict())
    elif resume is not None and checkpoint.get("parameter_retention") is not None:
        raise ValueError("resume cannot disable saved parameter retention")
    if matched_streams:
        stream_identity = {"policy": RANDOM_POLICY, "seed": seed,
                           "native_seed_offset": 3_000_017, "coherence_seed_offset": 1_000_003}
        if resume is not None and checkpoint.get("random_stream_identity") != stream_identity:
            raise ValueError("resume matched native random stream identity mismatch")
        store.update_manifest(random_stream_identity=stream_identity)

    if subset_manifest is not None:
        subset_path = store.run_dir / "artifacts/training_subset.json"
        if resume is not None:
            if not subset_path.is_file() or json.loads(subset_path.read_text()) != subset_manifest:
                raise ValueError("resume training subset differs from the saved selection")
            if checkpoint.get("training_subset_sha256") != subset_manifest["sha256"]:
                raise ValueError("recovery checkpoint does not bind the training subset")
        else:
            store.write_json("artifacts/training_subset.json", subset_manifest)
        store.update_manifest(
            training_subset_sha256=subset_manifest["sha256"],
            training_subset_count=len(train_dataset),
        )
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
    family_paths = {name: store.run_dir / "artifacts" / f"{name}_family.pt" for name in families}
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
    calibration_path = store.run_dir / "artifacts" / "coherence_calibration.json"
    if resume is None:
        finite_primary_calibration = _calibrate_finite_primary(
            model, train_dataset, families, banks, config, device,
            source_hashes_before["checkpoint"],
        )
        calibration = _calibrate_family_balance(
            model,
            train_dataset,
            families,
            banks,
            config,
            device=device,
            source_hashes_before=source_hashes_before,
        )
        if finite_primary_calibration is not None:
            calibration["finite_primary_topology"] = finite_primary_calibration
        if adaptive:
            calibration["endpoint_fidelity"] = _calibrate_endpoint_fidelity(
                source_anchor_model, train_dataset, config, device, source_hashes_before["checkpoint"]
            )
        if constrained:
            calibration["topology_components"] = _calibrate_topology_components(
                model,
                train_dataset,
                families,
                banks,
                config,
                device,
                source_hashes_before["checkpoint"],
            )
        if constrained:
            store.write_json(
                "artifacts/topology_optimizer_source.json", optimizer_source_snapshot()
            )
        store.write_json("artifacts/coherence_calibration.json", calibration)
    else:
        if not calibration_path.is_file():
            raise FileNotFoundError("resume is missing artifacts/coherence_calibration.json")
        calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
        calibration_hash = file_sha256(calibration_path)
        manifest = json.loads((store.run_dir / "run_manifest.json").read_text(encoding="utf-8"))
        if manifest.get("coherence_calibration_sha256") != calibration_hash:
            raise ValueError("resume coherence calibration artifact hash mismatch")
        if checkpoint is None or checkpoint.get("coherence_calibration_sha256") != calibration_hash:
            raise ValueError("resume checkpoint does not bind the exact coherence calibration")
        if calibration.get("source_checkpoint_sha256") != source_hashes_before["checkpoint"]:
            raise ValueError("resume coherence calibration source checkpoint hash mismatch")
        if calibration.get("mode") != config["coherence"].get("family_balance", {}).get(
            "mode", "none"
        ):
            raise ValueError("resume coherence family-balance mode mismatch")
        if checkpoint.get("family_scales") != calibration.get("resolved_scales"):
            raise ValueError("resume checkpoint family scales differ from calibration artifact")
    finite_primary_calibration = calibration.get("finite_primary_topology")
    topology_objective = getattr(families.get("topology"), "spatial_objective", None)
    if finite_primary_calibration is not None:
        if finite_primary_calibration["provenance"]["source_checkpoint_sha256"] != source_hashes_before["checkpoint"]:
            raise ValueError("finite-primary TRAIN calibration source checkpoint mismatch")
        topology_objective.load_source_calibration(finite_primary_calibration)
    elif getattr(topology_objective, "aggregation", None) == "finite_primary_v1":
        raise ValueError("finite-primary run is missing its frozen TRAIN calibration")
    family_scales = {name: float(calibration["resolved_scales"][name]) for name in families}
    coherence_direction_scale = 1.0
    if config["optimization"].get("coherence_direction_calibration", "none") != "none":
        direction = calibration.get("combined_direction", {})
        if direction.get("split") != "train" or direction.get("version") != "train_combined_direction_calibration_v1":
            raise ValueError("missing frozen TRAIN combined-direction calibration")
        method = config["optimization"].get("coherence_gradient_method", "config")
        coherence_direction_scale = float(direction["resolved_scales"][method])
        if resume is not None and checkpoint.get("coherence_direction_scale") != coherence_direction_scale:
            raise ValueError("resume combined-direction scalar differs from frozen TRAIN calibration")
    controller = None
    if adaptive:
        if "endpoint_fidelity" not in calibration:
            raise ValueError("adaptive run is missing frozen training fidelity calibration")
        if resume is not None and "fidelity_controller" not in checkpoint:
            raise ValueError("adaptive resume is missing fidelity controller state")
        controller = FidelityController(
            train_dataset.field_names, fidelity_settings(config), calibration["endpoint_fidelity"],
            state=checkpoint["fidelity_controller"] if resume is not None else None,
            steps_per_epoch=post_training_steps_per_epoch(config, len(train_dataset)),
        )
        if controller.updates != start_step:
            raise ValueError("fidelity controller accepted-update count differs from recovery step")
    if constrained:
        if "topology_components" not in calibration:
            raise ValueError("constrained resume is missing component calibration")
        if (
            calibration["topology_components"]["optimizer_source_sha256"]
            != optimizer_source_digest()
        ):
            raise ValueError("constrained optimizer implementation changed; start a new experiment")
        families["topology"].component_calibration = calibration["topology_components"]
        if resume is not None and "component_controller" not in checkpoint:
            raise ValueError("constrained resume is missing optimizer controller state")
        controller_type = (
            RegularizedTopologyAdam
            if config["optimization"].get("gradient_balance") == "topology_regularized"
            else ComponentConstrainedAdam
        )
        controller = controller_type(
            model,
            optimizer,
            constraint_settings(config),
            state=checkpoint["component_controller"] if resume is not None else None,
        )
        if controller.counts["attempted"] != start_step:
            raise ValueError("constrained optimizer attempt count differs from recovery step")
        if resume is not None:
            restore_constraint_audit(
                store.run_dir / "metrics" / "constraint_updates.jsonl", start_step
            )
    calibration_hash = file_sha256(calibration_path)
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
        coherence_calibration_sha256=calibration_hash,
        coherence_family_scales=family_scales,
        coherence_family_balance_mode=calibration["mode"],
    )
    if matched_streams:
        policy = resolve_training_policy(
            config,
            model=model,
            families=families,
            family_scales=family_scales,
            calibration=calibration,
            calibration_sha256=calibration_hash,
            source_hashes=source_hashes_before,
            source_metadata=source_metadata,
            normalizer_digest=train_dataset.normalizer.digest(),
            parameter_retention=parameter_retention,
        )
        persist_training_policy(
            store,
            policy,
            is_resume=resume is not None,
            start_step=start_step,
            resume_checkpoint_sha256=(
                file_sha256(store.run_dir / "checkpoints/last.pt") if resume is not None else None
            ),
        )

    evaluation_split = config.get("evaluation", {}).get("split", "validation")
    evaluation_dataset = open_field_dataset(
        config["dataset"], split=evaluation_split, normalizer=train_dataset.normalizer
    )
    protocol = sensor_protocol_from_config(config)
    evaluation_batch = _build_evaluation_batch(evaluation_dataset, config, protocol, device)
    comparison_batch = _build_comparison_batch(evaluation_batch, config)
    evaluate_fn = _evaluate
    native_monitor_batches = {}
    if r2_enabled:
        from ..evaluation.upgrade_1002_r2_protocol import (
            build_panel_batch,
            declare_panels,
            grouped_evaluate,
            matched_native_monitor,
            save_declaration,
            selection_indices,
        )

        evaluate_fn = grouped_evaluate
        declaration = declare_panels(
            len(evaluation_dataset),
            previously_used_indices=r2_settings.get("previously_used_validation_indices", ()),
        )
        native_validation_count = int(r2_settings.get("native_validation_count", 32))
        selected_ids = declaration["selection"]["dataset_indices"]
        positions = np.linspace(0, 4, native_validation_count // 16, endpoint=False).astype(int).tolist()
        native_indices = {
            "train": selection_indices(len(train_dataset), int(r2_settings.get("native_train_count", 32)), 16),
            "validation": [selected_ids[start + position] for start in range(0, 64, 4) for position in positions],
        }
        declaration["native_monitor"] = {
            "dataset_indices": native_indices,
            "native_seeds": list(r2_settings.get("native_seeds", [2027, 3027])),
            "every_epochs": int(r2_settings.get("native_monitor_every_epochs", 5)),
            "query_policy": config["coherence"]["compute_budget"].get("query_policy", "random_per_sample"),
            "role": "monitor_only_no_calibration_or_gradient",
        }
        path = save_declaration(store.run_dir / "artifacts/r2_panel_declaration.json", declaration)
        store.update_manifest(r2_panel_declaration_sha256=file_sha256(path),
                              coherence_direction_scale=coherence_direction_scale,
                              r2_B_aggregation="equal_sample_group32_mean_with_separate_pooled_B")
        for role, dataset in (("train", train_dataset), ("validation", evaluation_dataset)):
            full = build_panel_batch(dataset, protocol, native_indices[role], device)
            batch = _build_comparison_batch(full, config)
            native_monitor_batches[role] = batch
            manifest = manifest_from_batch(batch, dataset.path, role)
            manifest_path = store.run_dir / f"artifacts/r2_native_{role}_sensor_manifest.json"
            manifest.save(manifest_path)
            store.save_artifact(f"r2_native_{role}_query_indices.pt", {"query_indices": batch.metadata["query_indices"]})
            del full

        def frozen_native_monitor():
            return matched_native_monitor(
                model, source_anchor_model, native_monitor_batches,
                seeds=r2_settings.get("native_seeds", [2027, 3027]),
            )

        if resume is None:
            store.write_json("evaluation/native_before.json", frozen_native_monitor())
    if matched_streams:
        original_evaluate_fn = evaluate_fn

        def evaluate_fn(*args, **kwargs):
            return private_call(original_evaluate_fn, *args,
                                seed=int(config.get("evaluation", {}).get("seed", 2027)),
                                device=device, **kwargs)

    def training_coherence_objective(*args, **kwargs):
        if matched_streams:
            return private_call(_coherence_objective, *args,
                                seed=seed + 1_000_003 + int(kwargs["step"]),
                                device=device, **kwargs)
        return _coherence_objective(*args, **kwargs)
    if any(
        getattr(family, "strategy", None) == "cubical_persistence" for family in families.values()
    ):
        contract_batch = (
            evaluation_batch if model.capabilities.structured_grid_required else comparison_batch
        )
        report = verify_rollout_contract(
            model, _without_target(_slice_batch(contract_batch, 1)), config
        )
        store.write_json("evaluation/rollout_contract.json", report)
    # Bind sensors and the exact fixed comparison queries into one replayable
    # evaluation contract. The complete batch remains available only for
    # structured-model reconstruction before gathering these comparison IDs.
    evaluation_manifest = manifest_from_batch(
        comparison_batch, evaluation_dataset.path, evaluation_split
    )
    evaluation_manifest.save(store.run_dir / "artifacts" / "evaluation_sensor_manifest.json")
    if config.get("checkpointing", {}).get("exploratory_policy", {}).get("version") == "r3_endpoint_corridor_v1":
        from ..data.manifest import SensorManifest

        campaign = Path(case_dir) / "runs/Test_1002/_audit/R3_campaign"
        frozen = SensorManifest.load(campaign / "selection_sensor_manifest.json")
        if evaluation_manifest.digest() != frozen.digest():
            raise ValueError("R3 actual selection sensors differ from the prospective frozen panel")
        frozen_declaration = json.loads((campaign / "panel_declaration.json").read_text())
        if declaration["selection"]["dataset_indices"] != frozen_declaration["selection"]["dataset_indices"]:
            raise ValueError("R3 actual selection IDs differ from the frozen declaration")
    query_path = store.save_artifact(
        "evaluation_query_indices.pt",
        {"query_indices": comparison_batch.metadata.get("query_indices")},
    )
    store.update_manifest(evaluation_query_indices_sha256=file_sha256(query_path))
    before_metrics = None
    r3_reporting = checkpoint_manager_policy = config.get("checkpointing", {}).get("exploratory_policy")

    def capture_r3_coefficients(role, epoch):
        def capture(generated, reference):
            store.save_artifact(f"R3_B_coefficients_{role}_epoch_{epoch:03d}.pt", {
                "generated": generated.detach().cpu(), "reference": reference.detach().cpu(),
                "epoch": epoch, "panel_role": "frozen_selection64",
                "source_checkpoint_sha256": source_hashes_before["checkpoint"],
                "sensor_manifest_sha256": evaluation_manifest.digest(),
                "representation": "fixed4096_graph_linear_coefficients",
            })
            if role == "source":
                from .coherence_diagnostics import cached_B_regrouping

                store.write_json("evaluation/R3_B_regrouping_source.json", cached_B_regrouping(
                    generated.cpu(), generated.cpu(), reference.cpu(), families["cross_spectrum"],
                ))
        return capture

    if resume is None:
        before_metrics = evaluate_fn(
            model,
            evaluation_batch,
            comparison_batch,
            families,
            banks,
            train_dataset.field_names,
            config,
            normalizer=train_dataset.normalizer,
            family_scales=family_scales,
            **({"coefficient_callback": capture_r3_coefficients("source", 0)} if r3_reporting else {}),
        )
        before_metrics["sensor_manifest_sha256"] = evaluation_manifest.digest()
        store.write_json("evaluation/before.json", before_metrics)
    source_metrics = (
        before_metrics
        if before_metrics is not None
        else json.loads((store.run_dir / "evaluation" / "before.json").read_text(encoding="utf-8"))
    )
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

    native_audit_family = None
    native_audit_batch = None
    native_audit_contract = None
    native_audit_provenance = None
    native_audit_geometry = None
    if native_audit is not None:
        if "topology" not in families or source_anchor_model is None:
            raise ValueError("native topology audit requires a frozen source anchor and topology family")
        excluded_audit_ids = list(evaluation_batch.sample_ids)
        reserved_final_audit_ids = []
        if r2_enabled:
            from .native_topology_audit import _sample_id_for_index

            reserved_final_audit_ids = [
                _sample_id_for_index(evaluation_dataset, index)
                for index in declaration["audit"]["dataset_indices"]
            ]
            excluded_audit_ids.extend(reserved_final_audit_ids)
        native_audit_batch, native_audit_batch_metadata = build_disjoint_validation_batch(
            evaluation_dataset,
            protocol,
            selector_sample_ids=excluded_audit_ids,
            max_samples=int(native_audit.get("max_samples", 4)),
            seed=int(native_audit.get("seed", 2027)),
            device=device,
        )
        if r2_enabled:
            native_audit_batch_metadata.update(
                selector_sample_ids=list(evaluation_batch.sample_ids),
                reserved_final_audit_sample_ids=reserved_final_audit_ids,
                all_excluded_sample_ids=excluded_audit_ids,
                panel_role="periodic_development_native_audit",
            )
        native_audit_family, native_audit_provenance = build_native_topology_family(
            config,
            evaluation_dataset,
            train_dataset.normalizer,
            store.run_dir,
            device,
        )
        native_audit_geometry = initialize_native_geometry(
            native_audit_family,
            native_audit_batch,
            tuple(config["dataset"]["grid_shape"]),
            excluded_sample_ids=excluded_audit_ids,
        )
        native_audit_provenance["native_geometry"] = native_audit_geometry
        native_audit_sensor_manifest = manifest_from_batch(
            native_audit_batch, evaluation_dataset.path, "validation"
        )
        native_audit_contract = native_audit_contract_payload(
            config=config,
            dataset=evaluation_dataset,
            batch=native_audit_batch,
            batch_metadata=native_audit_batch_metadata,
            sensor_manifest=native_audit_sensor_manifest,
            family_provenance=native_audit_provenance,
            config_sha256=store.config_hash,
            selector_manifest_sha256=evaluation_manifest.digest(),
        )
        if r2_enabled:
            native_audit_contract.update(
                reserved_final_audit_sample_ids=reserved_final_audit_ids,
                all_excluded_sample_ids=excluded_audit_ids,
                panel_role="periodic_development_native_audit",
            )
            native_audit_contract["identity"]["reserved_final_audit_sample_ids"] = reserved_final_audit_ids
        if "every_epochs" in native_audit:
            native_audit_contract["audit_sampling"]["every_epochs"] = int(native_audit["every_epochs"])
        write_idempotent_report(
            store,
            f"{AUDIT_SUBDIR}/contract.json",
            native_audit_contract,
        )

    batch_size = int(config["optimization"].get("batch_size", 1))
    epochs = int(config["optimization"].get("epochs", 1))
    fraction = float(config["optimization"].get("train_fraction", 1.0))
    if not 0.0 < fraction <= 1.0:
        raise ValueError("optimization.train_fraction must lie in (0,1]")
    steps_per_epoch = post_training_steps_per_epoch(config, len(train_dataset))
    if native_audit is not None and "every_epochs" in native_audit:
        native_audit = {**native_audit, "every_steps": int(native_audit["every_epochs"]) * steps_per_epoch}
        checkpoint_interval = config.get("checkpointing", {}).get("every_steps") or (
            int(config.get("checkpointing", {}).get("every_epochs", 1)) * steps_per_epoch
        )
        if native_audit["every_steps"] % int(checkpoint_interval):
            raise ValueError("native epoch audit cadence must coincide with recoverable checkpoints")
    configured_steps = epochs * steps_per_epoch
    final_step = post_training_final_step(
        configured_steps, start_step, steps_per_epoch,
        max_steps=max_steps, until_epoch=until_epoch, additional_epochs=additional_epochs,
    )
    # A preemption can leave the terminal checkpoint intact but interrupt
    # reporting. Resuming that checkpoint must finish evaluation without
    # replaying an optimizer update or changing the configured training budget.
    reporting_only = resume is not None and start_step == configured_steps
    if start_step > configured_steps or (start_step >= final_step and not reporting_only):
        raise ValueError("post-training would perform no optimizer steps")
    if native_audit is not None and resume is not None and start_step > 0 and not reporting_only:
        resumed_committed_step = (
            int(controller.counts["accepted"]) if constrained else int(start_step)
        )
        if audit_due(
            resumed_committed_step,
            int(native_audit.get("every_steps", 0)),
            durable=True,
        ):
            last_checkpoint = store.run_dir / "checkpoints" / "last.pt"
            native_audit_provenance["current_child_artifact_sha256"] = verify_saved_family_hash(
                store
            )
            execute_native_audit(
                store=store,
                config=config,
                current_model=model,
                source_model=source_anchor_model,
                native_family=native_audit_family,
                batch=native_audit_batch,
                field_names=train_dataset.field_names,
                normalizer=train_dataset.normalizer,
                family_scales=family_scales,
                evaluate_fn=_evaluate,
                contract=native_audit_contract,
                family_provenance=native_audit_provenance,
                geometry_provenance=native_audit_geometry,
                committed_step=resumed_committed_step,
                attempt_step=start_step,
                durable_references=[
                    {
                        "kind": "last",
                        "path": str(last_checkpoint.relative_to(store.run_dir)),
                        "sha256": file_sha256(last_checkpoint),
                    }
                ],
            )
    index_generator = torch.Generator(device="cpu").manual_seed(seed + 17)
    if checkpoint is not None:
        index_generator.set_state(checkpoint["rng_state"]["index_generator"])
    sampled_indices = iter_unique_batch_indices(
        len(train_dataset),
        final_step - start_step,
        batch_size,
        # Prefetch may request future batches. Keep its cursor separate from the
        # completed-update cursor serialized in recovery checkpoints.
        generator=torch.Generator(device="cpu").set_state(index_generator.get_state()),
    )
    full_pass = config["optimization"].get("sampling") == "full_pass"
    if full_pass:
        from ..data.topology_subset import iter_full_pass_indices

        sampled_indices = iter_full_pass_indices(
            len(train_dataset),
            final_step - start_step,
            batch_size,
            seed=seed + 17,
            start_step=start_step,
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
        epoch_definition=(
            "fixed_updates_continuous_subset_passes"
            if "steps_per_epoch" in config["optimization"]
            else "full_subset_pass"
            if full_pass
            else "fractional_random_batches"
        ),
        steps_per_epoch=steps_per_epoch,
        batches_per_dataset_pass=math.ceil(len(train_dataset) / batch_size),
    )
    store.set_status("running", start_step=start_step, final_step=final_step)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    training_started = perf_counter()
    post_training_mode(model, config)
    monitor = TrainingMonitor(
        store.run_dir,
        start_step=start_step,
        final_step=final_step,
        configured_steps=configured_steps,
        steps_per_epoch=steps_per_epoch,
        description=f"post:{config['model']['name']}",
        enabled=bool(config["runtime"].get("progress", True)),
        plot_every_steps=int(config["runtime"].get("plot_every_steps", 10)),
        plot_format=str(config["runtime"].get("plot_format", "png")),
        plot_every_epochs=config["runtime"].get("plot_every_epochs"),
        **({"epoch_only": True} if r3_reporting else {}),
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
    if resume is not None and (adaptive or matched_streams):
        checkpoint_manager.restore_coherence_selector(checkpoint)
    panel_metrics_cache = {}
    r4_exact = config["runtime"].get("execution_mode", "legacy") == "r4_exact"
    rebound_monitor = None
    if r4_exact:
        from .rebound_monitor import ReboundMonitor

        rebound_monitor = ReboundMonitor(source_denominators={
            name: max(float(report["total"]), 1e-12)
            for name, report in source_metrics["coherence"]["families"].items()
        })
        rebound_path = store.run_dir / "metrics/rebound_monitor.json"
        if resume is not None and rebound_path.is_file():
            recovered = json.loads(rebound_path.read_text())
            observations = recovered["observations"]
            if not observations or observations[-1]["epoch"] <= start_step / steps_per_epoch:
                rebound_monitor.load_state_dict(recovered)
            else:
                # Discard report-only observations newer than recovery weights.
                for observation in observations:
                    if observation["epoch"] <= start_step / steps_per_epoch:
                        rebound_monitor.observe(observation["epoch"], observation["score"])
    if checkpoint_manager.selection_metric in {"topology_with_fidelity", "coherence_with_fidelity"}:
        selection_report = ((r3_coherence_selection_report
                             if checkpoint_manager.exploratory_policy is not None
                             else coherence_selection_report)
                            if checkpoint_manager.selection_metric == "coherence_with_fidelity"
                            else topology_selection_report)

        def evaluate_panel(step):
            metrics = (
                source_metrics
                if step == 0
                else evaluate_fn(
                    model,
                    evaluation_batch,
                    comparison_batch,
                    families,
                    banks,
                    train_dataset.field_names,
                    config,
                    normalizer=train_dataset.normalizer,
                    family_scales=family_scales,
                    **({"coefficient_callback": capture_r3_coefficients("candidate", step // steps_per_epoch)}
                       if checkpoint_manager_policy and step >= 100 * steps_per_epoch else {}),
                )
            )
            metrics["sensor_manifest_sha256"] = evaluation_manifest.digest()
            panel_metrics_cache.clear()
            panel_metrics_cache[step] = metrics
            return selection_report(source_metrics, metrics, config["posttrain_fidelity"])

        checkpoint_manager.panel_evaluator = evaluate_panel
        if resume is None:
            # The unchanged source is a valid candidate. Never fall back to an
            # ineligible child if every post-training checkpoint degrades fidelity.
            checkpoint_manager.save(
                _post_checkpoint_payload(
                    model,
                    optimizer,
                    global_step=0,
                    config=config,
                    train_dataset=train_dataset,
                    families=families,
                    source_hashes_before=source_hashes_before,
                    index_generator=index_generator,
                    device=device,
                    config_sha256=store.config_hash,
                    coherence_calibration_sha256=calibration_hash,
                    family_scales=family_scales,
                    controller=controller,
                    parameter_retention=parameter_retention,
                    coherence_direction_scale=coherence_direction_scale,
                ),
                model=model,
                preview=preview,
                global_step=0,
                fallback_metric=0.0,
                force=True,
            )
    if native_audit is not None and start_step == 0:
        source_checkpoint = source_checkpoint_path(config)
        if file_sha256(source_checkpoint) != source_hashes_before["checkpoint"]:
            raise ValueError("native topology audit source checkpoint changed before step zero")
        if audit_due(
            0,
            int(native_audit.get("every_steps", 0)),
            durable=True,
            initial=True,
        ):
            execute_native_audit(
                store=store,
                config=config,
                current_model=model,
                source_model=source_anchor_model,
                native_family=native_audit_family,
                batch=native_audit_batch,
                field_names=train_dataset.field_names,
                normalizer=train_dataset.normalizer,
                family_scales=family_scales,
                evaluate_fn=_evaluate,
                contract=native_audit_contract,
                family_provenance=native_audit_provenance,
                geometry_provenance=native_audit_geometry,
                committed_step=0,
                attempt_step=0,
                durable_references=[
                    {
                        "kind": "immutable_source_checkpoint",
                        "path": str(source_checkpoint),
                        "sha256": source_hashes_before["checkpoint"],
                    }
                ],
            )
    last_coherence_loss = None
    for offset, batch in enumerate(batch_source):
        global_step = start_step + offset
        if not full_pass:
            torch.randperm(len(train_dataset), generator=index_generator)
        epoch = global_step // steps_per_epoch + 1
        post_training_mode(model, config)
        r4_diagnose_batch = (r4_exact and (
            (global_step % steps_per_epoch == 0 and epoch % int(
                config["runtime"].get("diagnostics_every_epochs", 25 if matched_streams else 10)) == 0)
            or (not matched_streams and rebound_monitor.consume_diagnostic_request())))
        step_started = perf_counter()
        native_started = perf_counter()
        if adaptive:
            source_native_loss = None
            if controller.native_constrained:
                data_loss, source_native_loss = matched_native_losses(model, source_anchor_model, batch)
            else:
                # Preserve v1: the native objective supplies zero gradient.
                with torch.no_grad():
                    data_loss = model.training_loss(batch).total.detach()
        else:
            data_loss = (torch.zeros((), device=device) if constrained else
                         private_call(model.training_loss, batch,
                                      seed=seed + 3_000_017 + global_step, device=device).total
                         if matched_streams else model.training_loss(batch).total)
        every = int(config["coherence"]["schedule"].get("every_n_steps", 1))
        coherence_weight = _coherence_weight(config, epoch)
        coherence_active = coherence_weight > 0 and global_step % every == 0
        retention_active = (
            any(
                config["objectives"].get(name, {}).get("enabled", False)
                for name in ("endpoint", "source_anchor")
            )
            and global_step % every == 0
        )
        row: dict[str, Any] = {
            "step": global_step + 1,
            "epoch": epoch,
            "data_loss": float(data_loss.detach().cpu()),
            "native_data_loss": float(data_loss.detach().cpu()),
            "runtime/native_pair_seconds": perf_counter() - native_started,
            "data_retention_objective": (
                "native_plus_endpoint_retention"
                if retention_active
                else "model_native_training_loss"
            ),
            "coherence_applied": coherence_active,
            "coherence_weight": coherence_weight,
        }
        if matched_streams:
            row["random_stream_policy"] = RANDOM_POLICY
            row["native_seed"] = seed + 3_000_017 + global_step
            if global_step % steps_per_epoch == 0:
                row["stream_identity"] = {
                    "sample_ids": list(batch.sample_ids),
                    "query_sha256": _query_digest(batch.metadata["query_indices"]),
                    "observation_sha256": _query_digest(batch.obs_indices),
                    "native_seed": row["native_seed"],
                }
        if coherence_active or retention_active:
            step_context = {} if (adaptive or constrained or r4_exact) else None
            if r4_exact:
                step_context["diagnostics"] = r4_diagnose_batch
            retention_losses: dict[str, torch.Tensor] = {}
            rollout_generator = torch.Generator(device=device).manual_seed(
                seed + 1_000_003 + global_step
            )
            result, reference_ids = training_coherence_objective(
                model,
                batch,
                families,
                banks,
                config,
                step=global_step,
                generator=rollout_generator,
                family_scales=family_scales,
                retention_losses=retention_losses,
                source_anchor_model=source_anchor_model,
                step_context=step_context,
                committed_step=controller.counts["accepted"] if constrained else global_step,
                require_source_prediction=adaptive or constrained,
            )
            for name, loss in retention_losses.items():
                # Retention has the same sparse cadence as the shared rollout.
                factor = (
                    every if config["coherence"]["schedule"].get("interval_rescale", False) else 1
                )
                data_loss = data_loss + factor * float(config["objectives"][name]["weight"]) * loss
                row[f"{name}_retention_loss"] = float(loss.detach().cpu())
            if retention_losses:
                del loss
            row["data_loss"] = float(data_loss.detach().cpu())
            coherence_loss = result.scalar_loss
            if bool(config["coherence"]["schedule"].get("interval_rescale", False)):
                coherence_loss = coherence_loss * every
            diagnostic_every = int(
                config["coherence"]
                .get("family_balance", {})
                .get("gradient_diagnostics_every_epochs", 0)
            )
            simple_diagnostic_callback = None
            simple_diagnostic_event = (not adaptive and ((diagnostic_every > 0
                and global_step % steps_per_epoch == 0 and epoch % diagnostic_every == 0)
                or r4_diagnose_batch))
            data_update_weight, coherence_update_weight = _balanced_weights(config, coherence_weight)
            reuse_simple_diagnostics = (not matched_streams and r4_exact and math.isfinite(data_update_weight)
                and math.isfinite(coherence_update_weight)
                and data_update_weight > 0 and coherence_update_weight > 0)
            if simple_diagnostic_event and not reuse_simple_diagnostics:
                row["family_gradient_diagnostics"] = _shared_scalar_family_diagnostics(
                    model, data_loss, coherence_loss, step_context["family_results"], config
                ) if matched_streams else _sparse_family_gradient_diagnostics(
                    model, batch, data_loss, coherence_loss, families, banks, config,
                    step=global_step, rollout_seed=seed + 1_000_003 + global_step, device=device)
            row["data_update_weight"] = data_update_weight
            row["coherence_update_weight"] = coherence_update_weight
            if simple_diagnostic_event and reuse_simple_diagnostics:
                from .gradient_balance import make_shared_family_diagnostic_callback
                raw_family_losses = {name: value.scalar_loss
                                     for name, value in step_context["family_results"].items()}
                simple_diagnostic_callback = make_shared_family_diagnostic_callback(
                    tuple(parameter for parameter in model.parameters() if parameter.requires_grad),
                    raw_family_losses, data_weight=data_update_weight,
                    coherence_weight=coherence_update_weight)
            if adaptive:
                violations, live_risks, source_risks = controller.violations(
                    step_context["prediction"], step_context["source_prediction"],
                    step_context["reference"], step_context["batch"].query_valid_mask,
                    native_loss=data_loss if controller.native_constrained else None,
                    source_native_loss=source_native_loss,
                )
                fidelity_loss, fidelity_terms, pressure = controller.primal(violations)
                family_losses = {
                    name: coherence_weight * family_scales[name] * float(families[name].family_weight)
                    * family_result.scalar_loss
                    for name, family_result in step_context["family_results"].items()
                }
                if (not r4_exact and checkpoint_manager.exploratory_policy is not None
                        and global_step % steps_per_epoch == 0 and epoch % 25 == 0):
                    from .coherence_diagnostics import component_gradient_diagnostics

                    component_diagnosis = component_gradient_diagnostics(
                        step_context["family_results"], families,
                        tuple(p for p in model.parameters() if p.requires_grad),
                        sample_ids=step_context["batch"].sample_ids,
                        source_checkpoint_sha256=source_hashes_before["checkpoint"],
                        family_scales=family_scales,
                        native_loss=data_loss, endpoint_loss=live_risks[0],
                        native_pressure=fidelity_terms["native"],
                        endpoint_pressure=sum(value for name, value in fidelity_terms.items() if name != "native"),
                        coherence_direction_scale=coherence_direction_scale,
                    )
                    component_diagnosis.update(epoch=epoch, panel_role="one_stochastic_TRAIN_batch_sparse_diagnostic")
                    store.write_json(f"evaluation/R3_component_gradients_epoch_{epoch:03d}.json", component_diagnosis)
                diagnose = (r4_diagnose_batch if r4_exact
                    else global_step % int(controller.settings["diagnostics_every_steps"]) == 0)
                backward_started = perf_counter()
                gradient = coherence_primal_dual_update(
                    model, optimizer, family_losses, fidelity_loss,
                    method=config["optimization"].get("coherence_gradient_method", "config"),
                    grad_clip=config["optimization"].get("grad_clip"), diagnostics=diagnose,
                    constraint_losses=fidelity_terms if diagnose else None,
                    cagrad_alpha=float(config["optimization"].get("cagrad_alpha", 0.5)),
                    cagrad_rescale=int(config["optimization"].get("cagrad_rescale", 1)),
                    native_loss=data_loss if controller.native_constrained else None,
                    coherence_direction_scale=coherence_direction_scale,
                    execution="scalar" if r4_exact else "legacy",
                    config_backend="legacy",
                    **({"diagnostic_active_constraints": tuple(name for name, coefficient in
                            zip(controller.names, pressure.detach().cpu().tolist()) if coefficient != 0.0),
                        "diagnostic_endpoint_loss": live_risks[0]}
                       if r4_exact and diagnose and controller.epoch_controlled else {}),
                )
                if device.type == "cuda" and (not r4_exact or config["runtime"].get("profile_timings", False)):
                    torch.cuda.synchronize(device)
                controller.advance(violations)
                row.update(controller.telemetry(violations, live_risks, source_risks, pressure))
                row.update({"data_update_weight": 0.0, "coherence_update_weight": coherence_weight,
                            "data_retention_objective": (controller.version if controller.native_constrained
                                                         else "endpoint_primal_dual_native_monitor"),
                            "native_loss_role": controller.settings["native_loss_role"],
                            "loss/native": float(data_loss.detach()), "loss/endpoint_total": float(live_risks[0].detach()),
                            "loss/fidelity_augmented": float(fidelity_loss.detach()),
                            "runtime/rollout_seconds": step_context["rollout_seconds"],
                            "runtime/source_forward_seconds": step_context["source_forward_seconds"],
                            "runtime/backward_and_optimizer_seconds": perf_counter() - backward_started,
                            "runtime/peak_cuda_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0})
                for name, value in zip(train_dataset.field_names, live_risks[1:1 + len(train_dataset.field_names)]):
                    row[f"loss/endpoint/{name}"] = float(value.detach())
                if diagnose:
                    unobserved = unobserved_endpoint_risks(
                        step_context["prediction"], step_context["reference"], step_context["batch"]
                    )
                    if unobserved is not None:
                        for i, name in enumerate(train_dataset.field_names):
                            row[f"loss/endpoint_unobserved/{name}"] = (
                                float(unobserved["risks"][i]) if unobserved["counts"][i] else None
                            )
                            row[f"loss/endpoint_unobserved/{name}/point_count"] = int(unobserved["counts"][i])
                for name, family_result in step_context["family_results"].items():
                    row[f"loss/coherence/{name}/raw"] = float(family_result.scalar_loss.detach())
                    row[f"loss/coherence/{name}/calibrated"] = float(family_losses[name].detach())
                    row[f"runtime/{name}_seconds"] = step_context["family_seconds"][name]
                    for component, value in _component_scalars(family_result).items():
                        row[f"loss/coherence/{component}"] = value
                    if name == "topology" and "line_sampling" in family_result.diagnostics:
                        row["topology/line_sampling"] = family_result.diagnostics["line_sampling"]
                del fidelity_loss, fidelity_terms, family_losses, violations, live_risks, source_risks, pressure
                del source_native_loss
            elif constrained:
                gradient = _constrained_topology_update(
                    controller,
                    result,
                    step_context,
                    families["topology"],
                    config,
                    train_dataset.field_names,
                )
                row["data_retention_objective"] = (
                    "source_relative_fidelity_regularization"
                    if config["optimization"].get("gradient_balance") == "topology_regularized"
                    else "source_bounded_endpoint_and_anchor"
                )
                row["data_loss"] = gradient["endpoint_nmse_before"]
                row["native_data_loss"] = None
                with (store.run_dir / "metrics" / "constraint_updates.jsonl").open("a") as handle:
                    handle.write(
                        json.dumps(
                            {
                                "step": global_step + 1,
                                "sample_ids": list(reference_ids),
                                **gradient,
                                "update_seconds_before_reporting": perf_counter() - step_started,
                            },
                            allow_nan=False,
                        )
                        + "\n"
                    )
            elif matched_streams:
                gradient = scalar_update(
                    model, optimizer, data_loss, coherence_loss,
                    data_weight=data_update_weight, coherence_weight=coherence_update_weight,
                    retention=parameter_retention,
                    grad_clip=config["optimization"].get("grad_clip"),
                )
            else:
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
                    execution="scalar" if r4_exact else "legacy",
                    diagnostics=(r4_diagnose_batch or simple_diagnostic_event) if r4_exact else True,
                    config_backend="legacy",
                    **({"diagnostic_callback": simple_diagnostic_callback}
                       if simple_diagnostic_callback is not None else {}),
                )
            row.update(
                coherence_loss=float(result.scalar_loss.detach().cpu()),
                coherence_reference_ids=list(reference_ids),
                **_component_scalars(result),
                **_component_history_report(result, families),
                **_family_scalar_report(result),
                **gradient,
            )
            last_coherence_loss = row["coherence_loss"]
            # The final report needs a scalar, not a retained rollout graph.
            # Release it before the next rollout or large validation panel.
            del result, coherence_loss, retention_losses, step_context
        else:
            row["data_update_weight"] = _data_weight(config)
            row["coherence_update_weight"] = 0.0
            row.update(
                scalar_update(
                    model, optimizer, data_loss, data_weight=_data_weight(config),
                    coherence_weight=0.0, retention=parameter_retention,
                    grad_clip=config["optimization"].get("grad_clip"),
                ) if matched_streams else data_only_update(
                    model,
                    optimizer,
                    data_loss,
                    weight=_data_weight(config),
                    grad_clip=config["optimization"].get("grad_clip"),
                )
            )
        if row.get("update_accepted", True):
            after_optimizer_step(model)
        if r4_exact and device.type == "cuda" and (global_step + 1) % steps_per_epoch == 0:
            # One historical-epoch timing boundary includes the last optimizer
            # kernels. Ordinary per-family timers never synchronize for telemetry.
            torch.cuda.synchronize(device)
        row["update_seconds"] = perf_counter() - step_started
        if adaptive:
            row["runtime/step_seconds"] = row["update_seconds"]
        del data_loss
        row["total"] = row["data_update_weight"] * row["data_loss"] + row[
            "coherence_update_weight"
        ] * row.get("coherence_loss", 0.0) * (
            every if config["coherence"]["schedule"].get("interval_rescale", False) else 1
        )
        row["total"] += row.get("parameter_retention_loss", 0.0)
        if parameter_retention is not None:
            row["train/parameter_retention"] = row["parameter_retention_loss"]
        row["train/native_raw"] = row["native_data_loss"]
        row["train/coherence_weighted_score"] = row.get("coherence_loss")
        if adaptive:
            row["total_metric_name"] = "train/coherence_weighted_score"
            row["train/fidelity_penalty"] = row.get("loss/fidelity_augmented")
        else:
            row["total_metric_name"] = "weighted_update_ingredients"
        if constrained:
            row["total"] = row.get("regularized_loss_before", row["balanced_topology_before"])
        if r2_enabled and (global_step + 1) % steps_per_epoch == 0 and epoch % int(
            r2_settings.get("native_monitor_every_epochs", 5)
        ) == 0:
            monitor_started = perf_counter()
            native_reports = frozen_native_monitor()
            for role, report in native_reports.items():
                for key in ("candidate", "source", "ratio"):
                    row[f"native_monitor/{role}/{key}"] = report[key]
            row["train/native_source_matched_ratio"] = native_reports["train"]["ratio"]
            native_panel = native_reports["validation"]
            monitor.record_validation({
                "global_step": global_step + 1,
                "training_epoch": epoch,
                "metric_name": "validation/native_fixed_panel",
                "value": native_panel["candidate"],
                "source_value": native_panel["source"],
                "source_matched_ratio": native_panel["ratio"],
                "scope": "fixed_multi_snapshot_native_noise_panel",
                "panel_id": "native_validation",
                "sample_ids": list(native_monitor_batches["validation"].sample_ids),
                "normalization": "TRAIN_mean_std_model_native_objective",
                "generation_convention": "model.training_loss_not_endpoint_generation",
                "native_noise_convention": "fixed_matched_live_source_draws_restore_public_rng",
                "native_seeds": list(r2_settings.get("native_seeds", [2027, 3027])),
            })
            row["runtime/native_monitor_seconds"] = perf_counter() - monitor_started
            store.write_json(f"evaluation/native_epoch_{epoch:03d}.json", native_reports)
        monitor.record(row, lr=optimizer.param_groups[0]["lr"])
        if adaptive or r2_enabled:
            store.append_coherence_update(row)
        checkpoint_result = None
        if checkpoint_manager.due_for_preview_or_checkpoint(global_step + 1, preview):
            # Keep the case artifacts and checkpoint family state at the same
            # recovery boundary.
            family_paths = {
                name: store.save_artifact(f"{name}_family.pt", family.state_artifact())
                for name, family in families.items()
            }
            checkpoint_result = checkpoint_manager.save(
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
                    coherence_calibration_sha256=calibration_hash,
                    family_scales=family_scales,
                    controller=controller,
                    parameter_retention=parameter_retention,
                    coherence_direction_scale=coherence_direction_scale,
                ),
                model=model,
                preview=preview,
                global_step=global_step + 1,
                fallback_metric=row["data_loss"],
            )
        if checkpoint_result is not None:
            monitor.record_validation(checkpoint_manager.last_validation_report)
            if rebound_monitor is not None and epoch % 10 == 0:
                validation = checkpoint_manager.last_validation_report or {}
                for metric in validation.get("metrics", [validation]):
                    if metric.get("metric_name") == "validation/coherence_selection_score":
                        if (not rebound_monitor.observations or
                                rebound_monitor.observations[-1]["epoch"] < epoch):
                            alarm = rebound_monitor.observe(epoch, float(metric["value"]))
                            if alarm["diagnostic_triggered"]:
                                print(f"rebound alarm at epoch {epoch}: sparse TRAIN diagnostic requested")
                store.write_json("metrics/rebound_monitor.json", rebound_monitor.state_dict())
            store.write_json(
                "progress.json",
                {
                    "completed_updates": global_step + 1,
                    "reporting_epoch": (global_step + 1) / steps_per_epoch,
                    **sample_exposure(global_step + 1, len(train_dataset), batch_size),
                    "last_update_seconds": row["update_seconds"],
                    "update_mode": row.get("update_mode"),
                },
            )
        if (
            native_audit is not None
            and row.get("update_accepted", True)
            and global_step + 1 < final_step
        ):
            committed_step = (
                int(controller.counts["accepted"]) if constrained else int(global_step + 1)
            )
            durable_references = []
            if checkpoint_result is not None:
                for kind, path in zip(("last", "best"), checkpoint_result):
                    if path is not None:
                        durable_references.append(
                            {
                                "kind": kind,
                                "path": str(Path(path).relative_to(store.run_dir)),
                                "sha256": file_sha256(path),
                            }
                        )
            if audit_due(
                committed_step,
                int(native_audit.get("every_steps", 0)),
                durable=bool(durable_references),
                update_accepted=bool(row.get("update_accepted", True)),
            ):
                native_audit_provenance["current_child_artifact_sha256"] = (
                    verify_saved_family_hash(store)
                )
                execute_native_audit(
                    store=store,
                    config=config,
                    current_model=model,
                    source_model=source_anchor_model,
                    native_family=native_audit_family,
                    batch=native_audit_batch,
                    field_names=train_dataset.field_names,
                    normalizer=train_dataset.normalizer,
                    family_scales=family_scales,
                    evaluate_fn=_evaluate,
                    contract=native_audit_contract,
                    family_provenance=native_audit_provenance,
                    geometry_provenance=native_audit_geometry,
                    committed_step=committed_step,
                    attempt_step=global_step + 1,
                    durable_references=durable_references,
                )
        monitor.finish_step(
            checkpoint_checked=(
                checkpoint_result is not None and checkpoint_manager.last_best_checked
            ),
            best_checkpoint_saved=(
                checkpoint_result is not None and checkpoint_result[1] is not None
            ),
        )

    batch_source.close()

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    post_training_seconds = perf_counter() - training_started
    after_metrics = panel_metrics_cache.get(final_step)
    if after_metrics is None:
        after_metrics = evaluate_fn(
            model,
            evaluation_batch,
            comparison_batch,
            families,
            banks,
            train_dataset.field_names,
            config,
            normalizer=train_dataset.normalizer,
            family_scales=family_scales,
        )
    after_metrics["sensor_manifest_sha256"] = evaluation_manifest.digest()
    _add_source_relative_coherence(source_metrics, after_metrics)
    store.write_json("evaluation/after.json", after_metrics)
    fidelity_guard = _fidelity_guard_report(source_metrics, after_metrics, config)
    store.write_json("evaluation/fidelity_guard.json", fidelity_guard)
    if fidelity_guard["exceeded"] and fidelity_guard["behavior"] == "warn":
        warnings.warn(
            "post-training reconstruction MSE exceeded the configured fidelity guard",
            RuntimeWarning,
            stacklevel=2,
        )
    guard_failed = fidelity_guard["exceeded"] and fidelity_guard["behavior"] == "error"
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
        coherence_calibration_sha256=calibration_hash,
        family_scales=family_scales,
        controller=controller,
        parameter_retention=parameter_retention,
        coherence_direction_scale=coherence_direction_scale,
    )
    if checkpoint_manager.selection_metric in {"topology_with_fidelity", "coherence_with_fidelity"}:
        checkpoint_manager.panel_evaluator = lambda step: selection_report(
            source_metrics, after_metrics, config["posttrain_fidelity"]
        )
    saved = checkpoint_manager.save(
        checkpoint_payload,
        model=model,
        preview=preview,
        global_step=final_step,
        fallback_metric=(
            float(after_metrics["mse_normalized"]) if reporting_only else float(row["data_loss"])
        ),
        force=True,
    )
    if saved is None:
        raise RuntimeError("final checkpoint save was unexpectedly skipped")
    last_path, saved_best_path = saved
    if last_path is None:
        raise RuntimeError("forced final save did not create last.pt")
    if native_audit is not None:
        committed_final_step = (
            int(controller.counts["accepted"]) if constrained else int(final_step)
        )
        native_audit_provenance["current_child_artifact_sha256"] = verify_saved_family_hash(store)
        execute_native_audit(
            store=store,
            config=config,
            current_model=model,
            source_model=source_anchor_model,
            native_family=native_audit_family,
            batch=native_audit_batch,
            field_names=train_dataset.field_names,
            normalizer=train_dataset.normalizer,
            family_scales=family_scales,
            evaluate_fn=_evaluate,
            contract=native_audit_contract,
            family_provenance=native_audit_provenance,
            geometry_provenance=native_audit_geometry,
            committed_step=committed_final_step,
            attempt_step=int(final_step),
            durable_references=[
                {
                    "kind": "last",
                    "path": str(Path(last_path).relative_to(store.run_dir)),
                    "sha256": file_sha256(last_path),
                }
            ],
            terminal=True,
        )
    monitor.record_validation(checkpoint_manager.last_validation_report)
    best_checkpoint_saved = saved_best_path is not None
    best_path = store.run_dir / "checkpoints" / "best.pt"
    if not best_path.is_file():
        best_path = store.save_checkpoint("best", checkpoint_payload)
        best_checkpoint_saved = True
    monitor.close(
        checkpoint_checked=True,
        best_checkpoint_saved=best_checkpoint_saved,
    )
    preview.close()
    source_hashes_after = source_hashes(config)
    if source_hashes_after != source_hashes_before:
        raise RuntimeError("source run changed during child post-training")
    status = "completed" if final_step >= configured_steps else "integration_truncated"
    if guard_failed:
        status = "fidelity_guard_failed"
    final_checkpoint_hashes = {
        **json.loads((store.run_dir / "run_manifest.json").read_text()).get(
            "checkpoint_hashes", {}
        ),
        "last": file_sha256(last_path),
        "best": file_sha256(best_path),
    }
    best_fidelity_path = store.run_dir / "checkpoints" / "best_fidelity.pt"
    if best_fidelity_path.is_file():
        final_checkpoint_hashes["best_fidelity"] = file_sha256(best_fidelity_path)
    store.update_manifest(
        checkpoint_hashes=final_checkpoint_hashes,
        source_hashes_after=source_hashes_after,
        source_immutable_verified=True,
        evaluation_sensor_manifest_sha256=evaluation_manifest.digest(),
        before_metric=None if before_metrics is None else before_metrics.get("mse_normalized"),
        after_metric=after_metrics.get("mse_normalized"),
        fidelity_guard=fidelity_guard,
        coherence_family_state_sha256s={
            name: file_sha256(path) for name, path in family_paths.items()
        },
    )
    store.set_status(
        status,
        global_step=final_step,
        configured_steps=configured_steps,
        segment_start_step=start_step,
        segment_steps=final_step - start_step,
        steps_per_epoch=steps_per_epoch,
        **sample_exposure(final_step, len(train_dataset), batch_size),
        source_immutable_verified=True,
        coherence_target_use=(
            next(iter({family.target_use for family in families.values()}))
            if len({family.target_use for family in families.values()}) == 1
            else "mixed"
        ),
        coherence_target_uses={name: family.target_use for name, family in families.items()},
        final_coherence_loss=last_coherence_loss,
        post_training_seconds=post_training_seconds,
        seconds_per_step=post_training_seconds / max(final_step - start_step, 1),
        peak_cuda_memory_bytes=(
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
        ),
    )
    train_dataset.close()
    evaluation_dataset.close()
    if guard_failed:
        raise RuntimeError(
            "post-training reconstruction MSE exceeded the fidelity guard; recovery checkpoint saved"
        )
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
    coherence_calibration_sha256: str,
    family_scales: Mapping[str, float],
    controller=None,
    coherence_direction_scale=1.0,
    parameter_retention=None,
) -> dict[str, Any]:
    """Build an internally consistent post-training recovery checkpoint."""
    payload = {
        "model": checkpoint_model_state(model),
        "optimizer": optimizer.state_dict(),
        "global_step": int(global_step),
        **sample_exposure(
            global_step, len(train_dataset), int(config["optimization"]["batch_size"])
        ),
        "source_run": str(config["source_run"]),
        "source_checkpoint": str(source_checkpoint_path(config)),
        "source_hashes": dict(source_hashes_before),
        "data_spec": asdict(train_dataset.data_spec),
        "normalization": train_dataset.normalizer.state_dict(),
        "family_states": {name: family.state_dict() for name, family in families.items()},
        "family_configs": {name: family.config for name, family in families.items()},
        "config_sha256": config_sha256,
        "coherence_calibration_sha256": coherence_calibration_sha256,
        "family_scales": dict(family_scales),
        "rng_state": {
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
            "index_generator": index_generator.get_state(),
        },
    }
    if controller is not None:
        key = "fidelity_controller" if isinstance(controller, FidelityController) else "component_controller"
        payload[key] = controller.state_dict()
        if isinstance(controller, FidelityController) and controller.native_constrained:
            numpy_state = np.random.get_state()
            # Keep the existing restricted weights-only checkpoint loader:
            # NumPy reconstruct/pickle globals are never needed for RNG keys.
            payload["rng_state"].update(
                python=random.getstate(),
                numpy=(numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]),
            )
    if parameter_retention is not None:
        payload["parameter_retention"] = parameter_retention.state_dict()
    if config["runtime"].get("random_stream_policy") == RANDOM_POLICY:
        numpy_state = np.random.get_state()
        payload["rng_state"].update(
            python=random.getstate(),
            numpy=(numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]),
        )
        payload["random_stream_identity"] = {
            "policy": RANDOM_POLICY, "seed": int(config["runtime"].get("seed", 42)),
            "native_seed_offset": 3_000_017, "coherence_seed_offset": 1_000_003,
        }
    if config.get("evaluation", {}).get("r2_protocol", {}).get("enabled", False) or (
        isinstance(controller, FidelityController) and controller.native_constrained
    ):
        payload["epoch"] = int(global_step) / post_training_steps_per_epoch(config, len(train_dataset))
    if hasattr(train_dataset, "training_subset_manifest"):
        payload["training_subset_sha256"] = train_dataset.training_subset_manifest["sha256"]
    if config["optimization"].get("coherence_direction_calibration", "none") != "none":
        payload["coherence_direction_scale"] = coherence_direction_scale
    return add_training_aux_state(payload, model)

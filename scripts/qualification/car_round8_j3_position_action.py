#!/usr/bin/env python3
"""Disposable matched SOURCE J3 actions for PH-only and PH-plus-position.

Uses the real inherited model and admitted TRAIN inputs supplied by the caller.
One small batch and one fresh AdamW shadow action per package qualify execution,
not learning. The two arms reset the same SOURCE, RNG, sampler and AdamW states;
the declared nonzero multiplier is a fixture, not an inherited training policy.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

import torch

from phycoflow_reconstruction.coherence.compose import three_family_order
from phycoflow_reconstruction.coherence.shape_compute import LazyPooledShape
from phycoflow_reconstruction.data.training_batches import build_training_batch_source
from phycoflow_reconstruction.evaluation.adaptive_campaign import atomic_json, digest
from phycoflow_reconstruction.training.fidelity_budget import FidelityBudget, MeanRiskDifferenceBudget, _budget_from_state
from phycoflow_reconstruction.training.fidelity_quadrature import rotating_field_risks
from phycoflow_reconstruction.training.gradient_balance import _assign_flat_gradient
from phycoflow_reconstruction.training.gradients import stable_clip_grad_norm_
from phycoflow_reconstruction.training.multitask import EndpointBoundary, FamilyScalarizer, OuterAggregator
from phycoflow_reconstruction.training.post_training import (
    _coherence_objective, _conditional_draw_seed, _scheduled_model_lr,
)
from phycoflow_reconstruction.training.retention import post_training_mode
from phycoflow_reconstruction.training.rollout import differentiable_reconstruction
from phycoflow_reconstruction.training.scalable_fitting import _digest, validate_scalable_artifacts
from phycoflow_reconstruction.training.scalable_prefit import attach_position_artifact
from phycoflow_reconstruction.training.source import set_trainable_scope
from scripts.qualification.car_round5_panel_replay import load_context


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _norm(value):
    return float(torch.linalg.vector_norm(value.detach().double()).cpu())


def restore_scale_contract(runtime, settings):
    """The shared replay loader defaults to composite; restore saved policy."""
    objective = settings.get("scale_objective", "composite")
    kappa = float(settings.get("scale_joint_kappa", 1.0))
    if (objective not in {"composite", "joint_anchor"} or not math.isfinite(kappa) or kappa <= 0
            or (objective == "joint_anchor" and runtime.bank is None)):
        raise ValueError("invalid or unavailable saved Scale contract")
    runtime.scale_objective, runtime.scale_joint_kappa = objective, kappa


@torch.no_grad()
def _replay_field_risk(model, endpoint, dataset, config, *, seed, step, device):
    if model.capabilities.structured_grid_required:
        raise ValueError("this real-action fixture currently covers the point-cloud endpoint")
    members = []
    for index in range(len(endpoint["members"])):
        generator = torch.Generator(device=device)
        if index == 0:
            generator.set_state(endpoint["rollout_generator_state"])
        else:
            generator.manual_seed(_conditional_draw_seed(seed, index))
        members.append(differentiable_reconstruction(
            model, endpoint["rollout_model_batch"], steps=int(config["rollout"]["steps"]),
            solver=config["rollout"]["solver"], generator=generator,
            observation_config=dict(config["observation_consistency"])))
    replay = {**endpoint, "members": tuple(members), "prediction": members[0]}
    risks, complement = rotating_field_risks(model, replay, dataset, config,
        absolute_step=step, rollout_seed=seed, device=device)
    return risks.mean(0), complement


def matched_source_actions(model, dataset, batch, config, runtime, budget, position_artifact,
                           *, device, seed, field_multiplier=0.1):
    """Run the two real, same-state J3 actions; model is restored before return.

    `position_artifact` is the full frozen scalable envelope from the TRAIN fit.
    Caller supplies a disposable model/runtime and a risk reference matching the
    configured physical metric. Peak GPU allocation is measured per package.
    """
    device = torch.device(device)
    if not model.capabilities.generative or runtime.scoring_mode != "conditional":
        raise ValueError("the J3 action requires the inherited generative endpoint")
    if not math.isfinite(field_multiplier) or field_multiplier <= 0:
        raise ValueError("a finite positive fixture multiplier is required")
    metric = config["optimization"]["multitask"]["fidelity_budget"].get(
        "metric", "physical_mean_risk_model_units")
    if ((metric == "physical_member_mse_model_units" and not isinstance(budget, FidelityBudget))
            or (metric == "physical_mean_risk_model_units" and not isinstance(budget, MeanRiskDifferenceBudget))
            or metric not in {"physical_member_mse_model_units", "physical_mean_risk_model_units"}):
        raise ValueError("field reference and configured risk metric differ")
    if tuple(budget.field_names) != tuple(runtime.fields):
        raise ValueError("field reference and SOURCE field ordering differ")
    validate_scalable_artifacts(position_artifact, runtime.artifacts["provenance"])
    base_state = deepcopy(runtime.artifacts["shape_artifact"])
    location_state = deepcopy(position_artifact["shape_artifact"])
    if "position" not in location_state:
        raise ValueError("the frozen TRAIN positional observer is required")
    if _digest({k: v for k, v in location_state.items() if k != "position"}) != _digest(base_state):
        raise ValueError("positional package does not share the identical PH observer")
    state_family = runtime.families["global_distribution"]
    previous_count = state_family.conditional_draws
    state_family.conditional_draws = 3
    state_before = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    auxiliary = (deepcopy(model.training_aux_state_dict())
                 if callable(getattr(model, "training_aux_state_dict", None)) else None)
    sampler_before = deepcopy(runtime.state_dict())
    cpu_rng = torch.random.get_rng_state()
    device_rng = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
    parameters = [value for value in model.parameters() if value.requires_grad]
    if not parameters:
        raise ValueError("the real SOURCE trainable scope is empty")
    family_order = three_family_order(runtime.families)
    multipliers = torch.full((len(budget.field_names),), field_multiplier,
                             dtype=torch.float64, device=device)
    optimizer_settings = config["optimization"]
    scheduled = _scheduled_model_lr(optimizer_settings, runtime.step, runtime.steps_per_epoch)
    learning_rate = float(optimizer_settings["lr"] if scheduled is None else scheduled)
    rows = []
    try:
        for package, shape_state in (("PH_Only", base_state), ("PH_Plus_Position", location_state)):
            model.load_state_dict(state_before, strict=True)
            if auxiliary is not None:
                model.load_training_aux_state_dict(deepcopy(auxiliary))
            runtime.load_state_dict(deepcopy(sampler_before))
            runtime.adapter.shape = LazyPooledShape.from_state_dict(shape_state)
            runtime.capture_shape_components = True
            torch.random.set_rng_state(cpu_rng)
            if device_rng is not None:
                torch.cuda.set_rng_state(device_rng, device)
            optimizer = torch.optim.AdamW(parameters, lr=learning_rate,
                weight_decay=float(optimizer_settings.get("weight_decay", 0.0)),
                betas=tuple(optimizer_settings.get("adam_betas", (0.9, 0.999))))
            optimizer.zero_grad(set_to_none=True)
            runtime.begin_step(runtime.step)
            query_identity = _digest(asdict(runtime.batch))
            boundary = EndpointBoundary()
            if device.type == "cuda":
                _sync(device)
                torch.cuda.reset_peak_memory_stats(device)
                initial_allocated = torch.cuda.memory_allocated(device)
            else:
                initial_allocated = None
            started = time.monotonic()
            losses, endpoint = {}, {}
            combined, _ = _coherence_objective(model, batch, runtime.families,
                {name: None for name in runtime.families}, config, step=runtime.step,
                generator=torch.Generator(device=device).manual_seed(seed),
                family_scales={name: 1.0 for name in runtime.families}, family_losses=losses,
                endpoint_context=endpoint, scalable_runtime=runtime, scalable_probe=False,
                endpoint_boundary=boundary)
            if len(endpoint["members"]) != 3:
                raise ValueError("the affected action must execute three genuine members")
            family, _ = FamilyScalarizer(config["optimization"]["multitask"].get("family_scalarization", {}))(
                losses, scales={name: 1.0 for name in family_order},
                weights={name: float(getattr(runtime.families[name], "family_weight", 1.0))
                         for name in family_order})
            if not isinstance(family, dict):
                raise ValueError("the fixture requires independent ConFIG family rows")
            risks, complement = rotating_field_risks(model, endpoint, dataset, config,
                absolute_step=runtime.step, rollout_seed=seed, device=device, endpoint_boundary=boundary)
            risk_before = risks.mean(0)
            violations = budget.violations(risk_before)
            _sync(device)
            forward_seconds = time.monotonic() - started
            forward_peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
            started = time.monotonic()
            family_rows = boundary.rows([family[name] for name in family_order], parameters, chunk_size=1)
            field_row = boundary.rows([(multipliers.to(violations) * violations).sum()], parameters,
                                      chunk_size=1, include_complement=True)[0]
            if _norm(field_row) <= 0:
                raise ValueError("the weighted field row is inactive")
            _sync(device)
            reverse_seconds = time.monotonic() - started
            reverse_peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            started = time.monotonic()
            parts = combined.diagnostics["families"]["topology"]["shape_component_tensors"]
            shape_weight = float(getattr(runtime.families["topology"], "family_weight", 1.0))
            part_names = tuple(parts)
            part_rows = boundary.rows([parts[name] * shape_weight for name in part_names], parameters, chunk_size=1)
            torch.testing.assert_close(part_rows.sum(0), family_rows[family_order.index("topology")],
                                       rtol=3e-4, atol=3e-6)
            _sync(device)
            diagnostic_seconds = time.monotonic() - started
            diagnostic_peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
            part_row_norms = {name: _norm(row) for name, row in zip(part_names, part_rows)}
            del part_rows
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            started = time.monotonic()
            direction, _ = OuterAggregator({"method": "torchjd_config"})(family_rows.double())
            direction = direction / budget.objective_scale + field_row.double()
            before = torch.cat([value.detach().reshape(-1).cpu() for value in parameters])
            _assign_flat_gradient(parameters, direction)
            clip = stable_clip_grad_norm_(parameters, float(optimizer_settings["grad_clip"]))
            optimizer.step()
            after = torch.cat([value.detach().reshape(-1).cpu() for value in parameters])
            displacement = _norm(after - before)
            _sync(device)
            update_seconds = time.monotonic() - started
            update_peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
            peak = max(forward_peak, reverse_peak, update_peak) if device.type == "cuda" else None
            before_values = risk_before.detach().cpu()
            _sync(device)
            started = time.monotonic()
            risk_after, replay_complement = _replay_field_risk(model, endpoint, dataset, config,
                seed=seed, step=runtime.step, device=device)
            _sync(device)
            replay_seconds = time.monotonic() - started
            if not torch.equal(complement, replay_complement):
                raise ValueError("field finite-difference support changed")
            rows.append({"package": package, "members": len(endpoint["members"]),
                "sample_ids": list(endpoint["coupling_selected_batch"].sample_ids),
                "query_sha256": query_identity, "selected_sha256": _digest(endpoint["query_indices"]),
                "complement_sha256": _digest(complement),
                "first_generator_sha256": endpoint["rollout_generator_state_sha256"],
                "family_values": {name: float(losses[name].detach()) for name in family_order},
                "shape_components": {name: float(value.detach()) for name, value in parts.items()},
                "family_row_norms": {name: _norm(row) for name, row in zip(family_order, family_rows)},
                "shape_component_row_norms": part_row_norms,
                "weighted_field_row_norm": _norm(field_row), "multiplier": multipliers.cpu().tolist(),
                "field_risk_before": before_values.tolist(), "field_risk_after": risk_after.cpu().tolist(),
                "field_risk_change": (risk_after.cpu() - before_values).tolist(),
                "clip_input_norm": float(clip), "clip_limit": float(optimizer_settings["grad_clip"]),
                "adam_displacement_norm": displacement, "forward_seconds": forward_seconds,
                "parameter_rows_seconds": reverse_seconds, "component_diagnostic_seconds": diagnostic_seconds,
                "component_diagnostic_peak_allocated_bytes": diagnostic_peak, "optimizer_seconds": update_seconds,
                "finite_difference_seconds": replay_seconds, "initial_allocated_bytes": initial_allocated,
                "peak_allocated_bytes": peak,
                "peak_increment_bytes": None if peak is None else peak - initial_allocated})
            runtime.finish_step(False)
            optimizer.zero_grad(set_to_none=True)
            del (combined, losses, endpoint, family, family_rows, field_row, direction, boundary,
                 parts, risks, violations, risk_before, risk_after, complement, replay_complement,
                 before, after, optimizer, clip)
        for identity in ("sample_ids", "query_sha256", "selected_sha256", "complement_sha256",
                         "first_generator_sha256"):
            if rows[0][identity] != rows[1][identity]:
                raise ValueError(f"matched SOURCE action identity differs: {identity}")
        torch.testing.assert_close(torch.tensor(rows[0]["field_risk_before"]),
                                   torch.tensor(rows[1]["field_risk_before"]), rtol=1e-6, atol=1e-8)
        return {"schema": "car_round8_j3_position_action_v1", "status": "completed",
                "scope": "one matched SOURCE/TRAIN batch; J3 factored chunk 1; fresh AdamW; diagnostic only",
                "field_order": list(budget.field_names),
                "member_seed": seed, "trainable_parameters": len(parameters),
                "learning_rate": learning_rate, "adam_state": "identical fresh AdamW per action",
                "pi_state": "identical declared nonzero multiplier; PI feedback is not advanced",
                "scale_objective": runtime.scale_objective, "scale_joint_kappa": runtime.scale_joint_kappa,
                "metric": config["optimization"]["multitask"]["fidelity_budget"].get("metric"),
                "objective_scale": budget.objective_scale, "device": str(device),
                "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
                "actions": rows}
    finally:
        state_family.conditional_draws = previous_count
        model.load_state_dict(state_before, strict=True)
        if auxiliary is not None:
            model.load_training_aux_state_dict(deepcopy(auxiliary))
        if runtime.batch is not None:
            runtime.finish_step(False)
        runtime.load_state_dict(sampler_before)
        runtime.adapter.shape = LazyPooledShape.from_state_dict(base_state)
        runtime.capture_shape_components = False
        torch.random.set_rng_state(cpu_rng)
        if device_rng is not None:
            torch.cuda.set_rng_state(device_rng, device)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--position-artifact", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", required=True, type=torch.device)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--field-multiplier", type=float, default=0.1)
    parser.add_argument("--risk-metric", choices=("physical_member_mse_model_units", "physical_mean_risk_model_units"),
                        default="physical_member_mse_model_units")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch size must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    began = time.time()
    record = {"schema": "car_round8_j3_position_action_v1", "status": "failed"}
    source = dataset = None
    try:
        config, model, dataset, panel, runtime, _ = load_context(args.config, args.device)
        restore_scale_contract(runtime, config["coherence"]["scalable_anchored_coherence"])
        set_trainable_scope(model, config["trainable"])
        post_training_mode(model, config)
        config["coherence"]["compute_budget"]["batch_size"] = args.batch_size
        config["optimization"]["multitask"]["fidelity_budget"]["metric"] = args.risk_metric
        position = attach_position_artifact(runtime.artifacts,
            {"path": str(args.position_artifact), "sha256": digest(args.position_artifact)})
        reference = json.loads(args.reference.read_text())
        budget = _budget_from_state(reference["reference"])
        source = build_training_batch_source(dataset,
            [panel["sample_indices"][:args.batch_size]], config,
            query_points=int(config["coherence"]["compute_budget"]["point_count"]),
            device=args.device, start_step=0)
        batch = next(iter(source))
        if len(batch.sample_ids) != args.batch_size:
            raise ValueError("the action batch must match the declared qualification size")
        record = matched_source_actions(model, dataset, batch, config, runtime, budget, position,
            device=args.device, seed=int(config["runtime"]["seed"]) + 1_000_003,
            field_multiplier=args.field_multiplier)
        record.update(config_sha256=digest(args.config), position_sha256=digest(args.position_artifact),
                      reference_sha256=digest(args.reference), configured_batch_size=args.batch_size)
    except BaseException as error:
        record.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        if source is not None:
            source.close()
        if dataset is not None:
            dataset.close()
        record["wall_seconds"] = time.time() - began
        atomic_json(args.output / "receipt.json", record)


if __name__ == "__main__":
    main()

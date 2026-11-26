#!/usr/bin/env python3
"""Fit Round 7 SOURCE field reference and joint magnitude on matched TRAIN800.

Both parameter-gradient norms share each reconstructed endpoint and sampled
query batch. The field reference uses the selected-plus-complement law with
exactly the same observations and RFF draws, without extra model rollouts.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
import torch
import yaml

from phycoflow_reconstruction.coherence.compose import build_enabled_families
from phycoflow_reconstruction.coherence.physical_risk import physical_ensemble_risks
from phycoflow_reconstruction.data.physical_measure import batch_measure_context
from phycoflow_reconstruction.data.training_batches import build_training_batch_source
from phycoflow_reconstruction.evaluation.adaptive_campaign import atomic_json, digest
from phycoflow_reconstruction.training.fidelity_budget import MeanRiskDifferenceBudget
from phycoflow_reconstruction.training.fidelity_quadrature import rotating_field_risks
from phycoflow_reconstruction.training.gradient_balance import _flat_gradient
from phycoflow_reconstruction.training.post_training import _coherence_objective, _fidelity_reference_binding
from phycoflow_reconstruction.training.retention import post_training_mode
from phycoflow_reconstruction.training.scalable_fitting import _digest
from phycoflow_reconstruction.training.scalable_prefit import prepare_scalable_artifacts
from phycoflow_reconstruction.training.scalable_runtime import ScalableRuntime
from phycoflow_reconstruction.training.source import load_source_model, set_trainable_scope, source_hashes


def load_context(config_path, output, device):
    cfg = yaml.safe_load(config_path.read_text())
    model, dataset, metadata = load_source_model(cfg, device)
    set_trainable_scope(model, cfg["trainable"])
    post_training_mode(model, cfg)
    execution = cfg["optimization"].get("rollout_execution", {})
    model.rollout_checkpointing = bool(execution.get("checkpointing", True))
    model.rollout_context_cache = execution.get("context_cache", "none")
    families = build_enabled_families(cfg["coherence"], dataset.data_spec, dataset.normalizer)
    for name, family in families.items():
        declaration = cfg["coherence"]["frozen_family_artifacts"][name]
        path = Path(declaration["path"])
        if digest(path) != declaration["sha256"]:
            raise ValueError("frozen family hash differs: " + name)
        family.load_state_artifact(torch.load(path, map_location="cpu", weights_only=True))
        families[name] = family.to(device)
    source_identity = source_hashes(cfg)
    artifacts_path = output / "scalable_coherence.pt"
    core, provenance, cost = prepare_scalable_artifacts(
        model, dataset, families, cfg, source_sha256=source_identity["checkpoint"],
        device=device, artifact_path=artifacts_path)
    if not artifacts_path.exists():
        torch.save(core, artifacts_path)
    settings = cfg["coherence"]["scalable_anchored_coherence"]
    declaration = settings["common_prefit"]
    if digest(Path(declaration["path"])) != declaration["sha256"]:
        raise ValueError("frozen prefit hash differs")
    runtime = ScalableRuntime(core, families, steps_per_epoch=12, expected_provenance=provenance,
        seed=int(cfg["runtime"]["seed"])+800048, allocation_mode="uniform",
        uniform_epochs=settings.get("allocation", {}).get("uniform_epochs", 0),
        condition_query_groups=settings.get("condition_query_groups", 1))
    runtime.calibration.load_prefit(json.loads(Path(declaration["path"]).read_text()))
    panel_declaration = settings["common_core"]
    if digest(Path(panel_declaration["source_panel_path"])) != panel_declaration["source_panel_sha256"]:
        raise ValueError("frozen SOURCE panel hash differs")
    panel = torch.load(panel_declaration["source_panel_path"], map_location="cpu", weights_only=True)
    if len(panel["sample_indices"]) != 800 or len(dataset) != 8000:
        raise ValueError("matched TRAIN800 of TRAIN8000 required")
    return cfg, model, dataset, metadata, source_identity, panel, runtime


def stratified_risks(endpoint, dataset):
    """Return member MSE and signed risk under the identical frozen quadrature."""
    selected = endpoint["coupling_selected_batch"].metadata["query_indices"]
    complement = endpoint["complement_indices"]
    native = torch.as_tensor(dataset.physical_weights, device=selected.device,
                             dtype=endpoint["prediction"].dtype)
    sites = torch.cat((selected, complement.to(selected.device)), dim=-1)
    probabilities = torch.cat((torch.ones_like(selected, dtype=native.dtype),
        torch.full_like(complement.to(selected.device), complement.shape[1]/(native.numel()-selected.shape[1]),
                        dtype=native.dtype)), dim=-1)
    members = torch.cat((torch.stack(endpoint["members"], dim=1),
                         torch.stack(endpoint["complement_members"], dim=1)), dim=-2)
    target = torch.cat((endpoint["target"], endpoint["complement_target"]), dim=-2)
    return physical_ensemble_risks(members, target, native[sites], total_mass=native.sum(),
                                  sampling="without_replacement", probabilities=probabilities)


def reference_from_moments(field_names, signed_risk, member_mse, objective_scale, fraction=0.10):
    """The Round 7 scale is new M0; its floor never enlarges the allowance."""
    scales = [max(float(value), 1.0e-8) for value in member_mse]
    return MeanRiskDifferenceBudget.from_reference(field_names, signed_risk,
        [float(fraction)*float(value) for value in member_mse], scales, objective_scale)


def run(args):
    if not os.environ.get("CAR_ROUND7_LEASE_TOKEN"):
        raise ValueError("a measured Round 7 preparation lease is required")
    args.output.mkdir(parents=True, exist_ok=False)
    started, cpu_started = time.time(), time.process_time()
    result = {"schema": "car_round7_training_law_calibration_v1", "status": "failed",
              "start_unix": started, "config_sha256": digest(args.config)}
    dataset = source = None
    torch.set_num_threads(4)
    torch.manual_seed(47)
    try:
        cfg, model, dataset, metadata, identity, panel, runtime = load_context(args.config, args.output, args.device)
        budget_settings = deepcopy(cfg["optimization"]["multitask"]["fidelity_budget"])
        budget_settings.pop("reference_artifact", None)
        if budget_settings.get("reference_law") != "training_selected_complement_v1":
            raise ValueError("explicit training selected/complement reference law required")
        parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        indices = panel["sample_indices"]
        source = build_training_batch_source(dataset, [indices[i:i+64] for i in range(0, 800, 64)],
            cfg, query_points=4096, device=args.device, start_step=0)
        numerator = denominator = 0.0
        member_sum = risk_sum = None
        cursor = 0
        records = []
        with (args.output / "batches.jsonl").open("x") as stream:
            for index, batch in enumerate(source):
                count = len(batch.sample_ids)
                if list(batch.sample_ids) != panel["sample_ids"][cursor:cursor+count]:
                    raise ValueError("TRAIN800 order differs")
                if time.time() >= float(os.environ["CAR_ROUND7_LEASE_DEADLINE"]) - 180:
                    raise TimeoutError("calibration stopping before inclusive lease deadline")
                runtime.begin_step(index)
                endpoint, losses = {}, {}
                seed = int(cfg["runtime"]["seed"])+1_000_003+index
                _coherence_objective(model, batch, runtime.families,
                    {name: None for name in runtime.families}, cfg, step=index,
                    generator=torch.Generator(device=args.device).manual_seed(seed),
                    family_scales={name: 1.0 for name in runtime.families},
                    family_losses=losses, endpoint_context=endpoint,
                    scalable_runtime=runtime, scalable_probe=False)
                composite = losses["multiscale_spectrum"]
                selected = endpoint["coupling_selected_batch"]
                context = {"sample_ids": batch.sample_ids, "reference_ids": batch.sample_ids,
                           **batch_measure_context(selected), "conditional_draws": tuple(endpoint["members"][1:])}
                runtime.scale_objective, runtime.scale_joint_kappa = "joint_anchor", 1.0
                try:
                    joint, _, _ = runtime._score("multiscale_spectrum", runtime.families["multiscale_spectrum"],
                        endpoint["prediction"], endpoint["target"], selected.query_coords, context, runtime.batch)
                finally:
                    runtime.scale_objective = "composite"
                composite_norm = float(torch.linalg.vector_norm(_flat_gradient(composite, parameters).double()))
                joint_norm = float(torch.linalg.vector_norm(_flat_gradient(joint.scalar_loss, parameters).double()))
                if not math.isfinite(composite_norm) or not math.isfinite(joint_norm) or min(composite_norm, joint_norm) <= 0:
                    raise ValueError("nonpositive or nonfinite full-parameter Scale gradient norm")
                with torch.no_grad():
                    expected_risk, complement = rotating_field_risks(model, endpoint, dataset, cfg,
                        absolute_step=index, rollout_seed=seed, device=args.device, capture_endpoints=True)
                    member, risk = stratified_risks(endpoint, dataset)
                    torch.testing.assert_close(risk, expected_risk, rtol=2e-6, atol=2e-7)
                    member_total, risk_total = member.double().sum(0).cpu(), risk.double().sum(0).cpu()
                    member_sum = member_total if member_sum is None else member_sum+member_total
                    risk_sum = risk_total if risk_sum is None else risk_sum+risk_total
                numerator += count*composite_norm
                denominator += count*joint_norm
                row = {"batch": index, "sample_ids": list(batch.sample_ids), "count": count,
                       "rollout_seed": seed, "query_step": index,
                       "selected_sha256": _digest(selected.metadata["query_indices"]),
                       "complement_sha256": _digest(complement),
                       "first_member_generator_state_sha256": endpoint["rollout_generator_state_sha256"],
                       "composite_parameter_gradient_norm": composite_norm,
                       "joint_raw_parameter_gradient_norm": joint_norm,
                       "composite_value": float(composite.detach()),
                       "joint_raw_value": float(joint.scalar_loss.detach()),
                       "source_member_mse": member.detach().cpu().tolist(),
                       "source_signed_mean_risk": risk.detach().cpu().tolist()}
                records.append(row)
                stream.write(json.dumps(row, allow_nan=False)+"\n")
                stream.flush()
                cursor += count
                runtime.finish_step(True)
                del endpoint, losses, joint, composite, member, risk
        if cursor != 800 or len(records) != 13:
            raise ValueError("incomplete TRAIN800 calibration")
        kappa = numerator/denominator
        historical = json.loads(args.historical_reference.read_text())
        c0 = float(historical["reference"]["objective_scale"])
        mean_reference = (risk_sum/cursor).tolist()
        member_reference = (member_sum/cursor).tolist()
        fraction = float(budget_settings["member_allowance_fraction"])
        risk_scales = [max(value, 1.0e-8) for value in member_reference]
        budget_settings["mean_risk_scales"] = risk_scales
        cfg["optimization"]["multitask"]["fidelity_budget"]["mean_risk_scales"] = risk_scales
        reference = reference_from_moments(dataset.field_names, mean_reference, member_reference, c0, fraction)
        resolved_config = args.output/"resolved_reference_config.yaml"
        with resolved_config.open("x") as config_stream:
            yaml.safe_dump(cfg, config_stream, sort_keys=False)
        binding = _fidelity_reference_binding(cfg, dataset, identity, metadata, runtime,
                                             {name: 1.0 for name in runtime.families})
        panel_report = {"sample_count": cursor, "sample_ids": panel["sample_ids"],
            "sample_indices": panel["sample_indices"], "field_order": list(dataset.field_names),
            "source_member_mse": member_reference, "source_mean_risk": mean_reference,
            "units": "model_field_units_squared", "members": 2,
            "reference_law": "training_selected_complement_v1", "selected_points": 4096,
            "complement_points": 512, "observation_law": cfg["observations"],
            "draw_manifest": {"path": str((args.output/"batches.jsonl").resolve()),
                              "sha256": digest(args.output/"batches.jsonl")}}
        report = {"schema": "car_round7_training_law_reference_v1", "binding": binding,
            "reference": reference.state_dict(), "settings": budget_settings,
            "risk_reference_panel": panel_report, "c0_preserved_from": {
                "path": str(args.historical_reference.resolve()), "sha256": digest(args.historical_reference)},
            "test_locked": True}
        atomic_json(args.output/"reference.json", report)
        result.update(status="completed", sample_count=cursor, kappa=kappa,
            parameter_norm_numerator=numerator/cursor, parameter_norm_denominator=denominator/cursor,
            aggregation="ratio of sample-weighted mean full-parameter norms; not pooled-gradient norm",
            reference={"path": str((args.output/"reference.json").resolve()), "sha256": digest(args.output/"reference.json")},
            source_identity_before=identity, source_identity_after=source_hashes(cfg),
            resolved_reference_config={"path": str(resolved_config.resolve()), "sha256": digest(resolved_config)},
            native_training_loss=False, test_locked=True)
        if result["source_identity_before"] != result["source_identity_after"]:
            raise ValueError("SOURCE bytes changed")
    except BaseException as exc:
        result.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        if source is not None:
            source.close()
        if dataset is not None:
            dataset.close()
        result.update(end_unix=time.time(), wall_seconds=time.time()-started,
                      cpu_seconds=time.process_time()-cpu_started)
        atomic_json(args.output/"calibration.json", result)
    print(json.dumps({key: result[key] for key in ("status", "sample_count", "kappa", "wall_seconds")}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--historical-reference", type=Path, required=True)
    parser.add_argument("--device", type=torch.device, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()

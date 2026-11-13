#!/usr/bin/env python3
"""Profile a frozen TRAIN SOURCE bank with real source-parameter gradients.

This driver does not launch itself or read VALIDATION/TEST. Run it only inside
the campaign's declared GPU and wall-clock allocation.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(os.environ.get("CAR_REPO_ROOT", Path(__file__).resolve().parents[2])).resolve()
CODE_ROOT = Path(os.environ.get("CAR_CODE_ROOT", Path(__file__).resolve().parents[2])).resolve()
sys.path.insert(0, str(CODE_ROOT / "src"))


def measure_bundle_costs(timed_bundle, count, repeats):
    """Attribute each added pair to its own ID and subtract the anchor cost."""
    if count < 2 or repeats < 1:
        raise ValueError("at least two pairs and one repeat required")
    rows = []
    for index in range(count):
        anchor = (index + 1) % count
        samples = [(timed_bundle(anchor, None), timed_bundle(anchor, index))
                   for _ in range(repeats)]
        delta = sum(b - a for a, b in samples) / repeats
        if delta <= 0:
            raise ValueError("incremental timing unresolved; increase repeats or use uniform fallback")
        rows.append({"pair": index, "anchor": anchor,
                     "one_and_two_query_seconds": samples, "incremental_seconds": delta})
    fixed = sum(sum(a for a, _ in row["one_and_two_query_seconds"]) / repeats
                - rows[row["anchor"]]["incremental_seconds"] for row in rows) / count
    if fixed < 0:
        raise ValueError("fixed bundle timing unresolved; increase repeats or use uniform fallback")
    for row in rows:
        row["fixed_seconds"] = fixed
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--campaign", type=Path, default=ROOT / "cases/turbulent_combustion/runs/coherence_adaptive_repair/round_1")
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--source-family", choices=("ffm", "edm"), required=True)
    p.add_argument("--gpu-uuid", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--research-costs", type=Path, required=True)
    p.add_argument("--spill-dir", type=Path, required=True, help="existing writable scratch filesystem for temporary exact parameter gradients")
    p.add_argument("--calibration-checkpoint", type=Path)
    p.add_argument("--policy-overhead-fraction", type=float, required=True, help="measured operational sampler fraction of the complete update; exclude all research costs")
    p.add_argument("--complete-update-seconds", type=float, required=True, help="measured complete-update wall time including operational sampler, excluding research costs")
    p.add_argument("--max-seconds", type=float, required=True)
    p.add_argument("--save-margin-seconds", type=float, default=60.0)
    p.add_argument("--panel-batch-size", type=int, default=1)
    p.add_argument("--audit-conditions", type=int, default=1)
    p.add_argument("--timing-repeats", type=int, default=3)
    p.add_argument("--replicates", type=int, default=128)
    args = p.parse_args()
    from phycoflow_reconstruction.evaluation.adaptive_campaign import enforce_execution_constraints
    enforce_execution_constraints(args.campaign.resolve(), args.gpu_uuid, args.source_family)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != args.gpu_uuid or not os.environ.get("KEOPS_CACHE_FOLDER"):
        raise ValueError("bind the physical GPU UUID and a device-specific KeOps cache before import")
    if args.max_seconds <= args.save_margin_seconds or min(args.panel_batch_size, args.audit_conditions, args.timing_repeats) < 1:
        raise ValueError("positive profiling budget, margin and batch sizes required")
    if not args.spill_dir.is_dir():
        raise ValueError("spill-dir must be an existing scratch directory")
    began = time.perf_counter()
    deadline = began + args.max_seconds - args.save_margin_seconds

    def budget_check():
        if time.perf_counter() >= deadline:
            raise TimeoutError("profiling reached its save margin")

    import torch
    import yaml

    from phycoflow_reconstruction.coherence import build_enabled_families
    from phycoflow_reconstruction.coherence.adaptive_fitting import spectral_filter
    from phycoflow_reconstruction.coherence.adaptive_profile import (
        finalize_policy_overhead,
        profile_source_panel,
    )
    from phycoflow_reconstruction.coherence.query_runtime import ScalableQueryAdapter
    from phycoflow_reconstruction.coherence.query_sampling import (
        LINE_WEIGHTS,
        PAIR_SIGNS,
        Draw,
        QueryBatch,
        pair_count,
    )
    from phycoflow_reconstruction.data.physical_measure import (
        attach_batch_measure,
        batch_measure_context,
    )
    from phycoflow_reconstruction.data.sensor_protocols import build_observation_batch
    from phycoflow_reconstruction.evaluation.adaptive_campaign import atomic_json
    from phycoflow_reconstruction.training.common import sensor_protocol_from_config
    from phycoflow_reconstruction.training.post_training import _physical_endpoint_risk
    from phycoflow_reconstruction.training.rollout import differentiable_reconstruction
    from phycoflow_reconstruction.training.run_store import file_sha256
    from phycoflow_reconstruction.training.scalable_fitting import (
        _digest,
        validate_scalable_artifacts,
    )
    from phycoflow_reconstruction.training.scalable_runtime import ScalableRuntime
    from phycoflow_reconstruction.training.source import (
        load_source_model,
        set_trainable_scope,
        source_checkpoint_path,
    )

    run = args.run_dir.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    receipt = {"schema": "car_allocator_profile_v1", "status": "running", "source_family": args.source_family, "gpu_uuid": args.gpu_uuid}
    atomic_json(args.output, receipt)
    dataset = None
    try:
        config = yaml.safe_load((run / "resolved_config.yaml").read_text())
        model, dataset, identity = load_source_model(config, torch.device("cuda:0"))
        if bool(model.capabilities.structured_grid_required) != (args.source_family == "edm"):
            raise ValueError("source family does not match the loaded model contract")
        model.eval()
        set_trainable_scope(model, config["trainable"])
        execution = config["optimization"].get("rollout_execution", {})
        if execution.get("checkpointing", False):
            model.rollout_checkpointing = True
        panel_path = run / "artifacts/source_panel.pt"
        artifact_path = run / "artifacts/scalable_coherence.pt"
        panel = torch.load(panel_path, map_location="cpu", weights_only=True)
        artifacts = torch.load(artifact_path, map_location="cpu", weights_only=True)
        if panel.get("schema") != "car_source_panel_v1" or panel["provenance"] != artifacts["provenance"]:
            raise ValueError("SOURCE panel and scalable artifacts have different identities")
        validate_scalable_artifacts(artifacts, panel["provenance"])
        calibrated = {}
        coefficient_identity = "fixed_unit_internal_coefficients"
        if args.calibration_checkpoint is not None:
            checkpoint = torch.load(args.calibration_checkpoint, map_location="cpu", weights_only=True)
            runtime_state = checkpoint.get("scalable_runtime", {})
            calibrated = runtime_state.get("calibration", {}).get("coefficients")
            if runtime_state.get("artifact_sha256") != artifacts["sha256"] or runtime_state.get("provenance") != artifacts["provenance"] or not isinstance(calibrated, dict):
                raise ValueError("calibration checkpoint does not bind the exact scalable artifact")
            if any(not isinstance(v, (int, float)) or not 0 < float(v) < float("inf") for v in calibrated.values()):
                raise ValueError("invalid frozen internal calibration coefficients")
            coefficient_identity = "checkpoint_frozen_internal_coefficients"
        if panel["source_weight_view"] != identity.get("weight_view") or panel["provenance"]["normalizer_sha256"] != dataset.normalizer.digest():
            raise ValueError("SOURCE panel weight view or normalizer mismatch")
        panel_view = panel.get("parameter_buffer_sha256")
        if panel_view is not None and (panel_view != _digest(model.state_dict()) or panel.get("source_weight_view_sha256") != identity.get("materialized_view_sha256")):
            raise ValueError("SOURCE panel materialized parameter/buffer identity mismatch")
        if file_sha256(source_checkpoint_path(config)) != panel["provenance"]["source_checkpoint_sha256"]:
            raise ValueError("SOURCE checkpoint does not match the panel")
        prior = _digest({k: v for k, v in model.state_dict().items() if k in ("prior.omega", "prior.phase")})
        if prior != panel["prior_buffers_sha256"]:
            raise ValueError("SOURCE prior buffers differ from the endpoint panel")
        families = build_enabled_families(config["coherence"], dataset.data_spec, dataset.normalizer)
        manifest = json.loads((run / "run_manifest.json").read_text())
        for name, family in families.items():
            path = run / "artifacts" / f"{name}_family.pt"
            if file_sha256(path) != manifest["coherence_family_state_sha256s"][name]:
                raise ValueError(f"family artifact hash mismatch: {name}")
            family.load_state_artifact(torch.load(path, map_location="cpu", weights_only=True))
        cpu_families = families
        gpu_families = {name: copy.deepcopy(family).to("cuda:0") for name, family in families.items()}
        calibration = json.loads((run / "artifacts/coherence_calibration.json").read_text())
        weights = {name: float(calibration["resolved_scales"][name]) * float(getattr(family, "family_weight", 1.0)) for name, family in families.items()}
        inner = [(s, line, weight / 4) for s in PAIR_SIGNS for line, weight in enumerate(LINE_WEIGHTS)]
        fields = tuple(dataset.field_names)
        count = pair_count(len(fields))
        chunk = panel["coupling"][0]
        n = len(chunk["sample_ids"])
        if args.audit_conditions > n:
            raise ValueError("audit conditions exceed first recorded SOURCE generation chunk")
        indexes = panel["sample_indices"][:n]
        samples = [dataset[i] for i in indexes]
        seed = int(config["runtime"].get("seed", 42)) + 810001
        protocol = replace(sensor_protocol_from_config(config), seed=seed)
        query = None if model.capabilities.structured_grid_required else panel["query_indices"][0]
        batch = build_observation_batch(samples, protocol, query_indices=query, manifest_indices=panel["observation_manifest"])
        if list(batch.sample_ids) != chunk["sample_ids"] or _digest([batch.obs_indices, batch.obs_values, batch.obs_field_ids, batch.obs_valid_mask]) != chunk["observation_sha256"]:
            raise ValueError("reconstructed SOURCE observations do not match the recorded panel")
        batch = attach_batch_measure(dataset, batch).to("cuda:0")
        target = batch.target_fields[:args.audit_conditions]
        torch.testing.assert_close(target.cpu(), panel["reference"][:args.audit_conditions], rtol=0, atol=0)
        torch.testing.assert_close(batch.query_coords[:args.audit_conditions].cpu(), panel["coordinates"][:args.audit_conditions], rtol=0, atol=0)
        model_batch = replace(batch, target_fields=None)
        steps, solver = int(config["rollout"]["steps"]), config["rollout"]["solver"]
        adapter = ScalableQueryAdapter(gpu_families, artifacts["shape_artifact"])
        parameters = tuple(p for p in model.parameters() if p.requires_grad)
        coord = batch.query_coords[:args.audit_conditions]
        masses = batch_measure_context(batch)["generated_weights"][:args.audit_conditions]
        torch.testing.assert_close(masses.cpu(), panel["masses"][:args.audit_conditions], rtol=0, atol=0)

        def rollout():
            return [differentiable_reconstruction(model, model_batch, steps=steps, solver=solver, generator=torch.Generator(device="cuda:0").manual_seed(chunk["member_seeds"][j]), observation_config=config["observation_consistency"])[:args.audit_conditions] for j in range(2)]

        with torch.no_grad():
            for j, value in enumerate(rollout()):
                torch.testing.assert_close(value.cpu(), panel["members"][:args.audit_conditions, j], rtol=3e-4, atol=3e-4)

        timing_count = int(config["coherence"]["compute_budget"]["batch_size"])
        if timing_count > len(panel["sample_indices"]):
            raise ValueError("SOURCE panel cannot supply the declared training batch for timing")
        timing_samples = [dataset[i] for i in panel["sample_indices"][:timing_count]]
        timing_seed = seed + 999983
        timing_batch = build_observation_batch(timing_samples, replace(protocol, seed=timing_seed), query_indices=query)
        timing_batch = attach_batch_measure(dataset, timing_batch).to("cuda:0")
        timing_target = timing_batch.target_fields
        timing_model_batch = replace(timing_batch, target_fields=None)
        timing_masses = batch_measure_context(timing_batch)["generated_weights"]

        def query_batch(index, inner_index=0, extra=None, unary_field=0, unary_sign=1):
            sign, line, probability = inner[inner_index]
            indices = (index,) if extra is None else (index, extra)
            pairs = tuple(Draw(i, 1, 1 / count, count, len(indices), sign, line, probability) for i in indices)
            return QueryBatch((Draw(unary_field, 1, 1 / len(fields), len(fields), 1, (unary_sign,), None, .5),), pairs, 1)

        def scores(members, query):
            context = {"scoring_mode": "conditional", "field_order": fields, "conditional_draws": (members[1],), "generated_weights": masses, "reference_weights": masses}
            return [adapter.score(name, members[0], target, coordinates=coord, context=context, batch=query, coefficients=calibrated)[0].scalar_loss * weights[name] for name in ("multiscale_spectrum", "topology")]

        def flatten(grads):
            return torch.cat([(torch.zeros_like(p) if g is None else g).detach().reshape(-1).cpu().double() for p, g in zip(parameters, grads)])

        def parameter_gradient(coefficients):
            budget_check()
            members = rollout()
            baseline = sum(scores(members, query_batch(0)))
            base_grad = flatten(torch.autograd.grad(baseline, parameters, retain_graph=True, allow_unused=True))
            result = -float(coefficients.sum()) * base_grad
            for cell in (coefficients != 0).nonzero().flatten().tolist():
                budget_check()
                if cell < count * len(inner):
                    i, j = divmod(cell, len(inner))
                    query = query_batch(i, j)
                else:
                    field, sign_index = divmod(cell - count * len(inner), 2)
                    query = query_batch(0, unary_field=field, unary_sign=(1, -1)[sign_index])
                loss = sum(scores(members, query))
                result += float(coefficients[cell]) * flatten(torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True))
            return result

        timing = []
        def timed_bundle(index, extra):
            budget_check()
            torch.cuda.synchronize()
            started = time.perf_counter()
            members = [differentiable_reconstruction(model, timing_model_batch, steps=steps, solver=solver, generator=torch.Generator(device="cuda:0").manual_seed(timing_seed + j), observation_config=config["observation_consistency"]) for j in range(2)]
            runtime = ScalableRuntime(artifacts, gpu_families, steps_per_epoch=1, expected_provenance=artifacts["provenance"], scoring_mode="conditional")
            if calibrated:
                runtime.calibration.coefficients = calibrated
            context = {"conditional_draws": (members[1],), "generated_weights": timing_masses, "reference_weights": timing_masses}
            objectives = [runtime._score(name, family, members[0], timing_target, timing_batch.query_coords, context, query_batch(index, extra=extra))[0].scalar_loss * weights[name] for name, family in gpu_families.items()]
            endpoint = {"prediction": members[0], "members": tuple(members), "target": timing_target, "mask": timing_batch.query_valid_mask, "measure": {"generated_weights": timing_masses}}
            objectives.append(_physical_endpoint_risk(endpoint, dataset).mean())
            for objective in objectives:
                torch.autograd.grad(objective, parameters, retain_graph=True, allow_unused=True)
            torch.cuda.synchronize()
            return time.perf_counter() - started

        timing = measure_bundle_costs(timed_bundle, count, args.timing_repeats)
        def bundle_cost(index):
            return timing[index]["incremental_seconds"], timing[index]["fixed_seconds"]

        costs = json.loads(args.research_costs.read_text())
        profile = profile_source_panel(panel, families=cpu_families, artifacts=artifacts, filter_fn=lambda x: spectral_filter(cpu_families["multiscale_spectrum"], x), source_family=args.source_family, parameter_gradient_fn=parameter_gradient, bundle_cost_fn=bundle_cost, research_costs=costs, policy_overhead_fraction=args.policy_overhead_fraction, coefficients=calibrated, replicates=args.replicates, panel_batch_size=args.panel_batch_size, budget_check=budget_check, spill_directory=str(args.spill_dir), family_weights={k: weights[k] for k in ("multiscale_spectrum", "topology")})
        # Research inputs must exclude the operational sampler portion already
        # included in the measured complete-update baseline.
        measured = time.perf_counter() - began
        finalize_policy_overhead(profile, measured_profile_seconds=measured,
                                 complete_update_seconds=args.complete_update_seconds)
        receipt.update(status="passed", profile=profile, measured_profile_wall_seconds=measured, source_identity=identity, source_view_evidence="panel_parameter_buffer_digest" if panel_view is not None else "legacy_panel_label_and_strict_checkpoint_plus_endpoint_parity", panel_sha256=file_sha256(panel_path), scalable_artifact_sha256=file_sha256(artifact_path), coefficient_identity=coefficient_identity, calibration_checkpoint_sha256=file_sha256(args.calibration_checkpoint) if args.calibration_checkpoint else None, interpretation="SOURCE endpoint allocation audit; does not establish benefits after internal calibration or ConFIG", audit_sample_ids=chunk["sample_ids"][:args.audit_conditions], timing_sample_ids=list(timing_batch.sample_ids), timing_observation_seed=timing_seed, timing_observation_protocol="native_training", timing=timing, family_weights=weights, endpoint_parity_tolerance={"rtol": 3e-4, "atol": 3e-4})
    except BaseException as exc:
        receipt.update(status="failed", error_type=type(exc).__name__, error=str(exc), elapsed_seconds=time.perf_counter() - began)
        raise
    finally:
        if dataset is not None:
            dataset.close()
        atomic_json(args.output, receipt)


if __name__ == "__main__":
    main()

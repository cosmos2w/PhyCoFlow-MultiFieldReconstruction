"""Bounded TRAIN SOURCE profiling; each timed optimizer action is restored.

Six prospective plans, four batches, exact finite-support parameter-gradient
noise. This is acquisition evidence, never a scientific initializer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import signal
import subprocess
import uuid
from copy import deepcopy
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import numpy as np
import torch
import yaml


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def write_new(path, value):
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def prior_profile_seconds(root, task_id):
    """Charge retained failed/setup attempts once to the cold task decision."""
    ledger = root / "manifests/campaign_runner_ledger.jsonl"
    latest = {}
    if ledger.exists():
        for line in ledger.read_text().splitlines():
            row = json.loads(line)
            latest[row["attempt_id"]] = row
    return sum(
        float(row.get("charged_process_seconds", 0.0))
        for row in latest.values()
        if row.get("task_id") == task_id
        and row.get("action") == "profile"
        and row.get("status") != "running"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=2)
    parser.add_argument("--max-seconds", type=float, default=1800)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.gpu):
        raise ValueError("profile requires the requested physical GPU as sole visible device")
    if not 0 < args.max_seconds <= 5400:
        raise ValueError("profile duration must be within the 90-minute campaign ceiling")
    signal.alarm(math.ceil(args.max_seconds))
    root = args.root.resolve(strict=True)
    if not root.is_relative_to(Path("/data2")) or not (root / "manifests/preflight.json").is_file():
        raise ValueError("verified owned campaign required")
    attempts = root / "profiles" / f"{args.task}_attempts"
    attempts.mkdir(exist_ok=True)
    attempt = attempts / uuid.uuid4().hex
    attempt.mkdir()
    from phycoflow_reconstruction.coherence import build_enabled_families
    from phycoflow_reconstruction.coherence.families.topology import persistence
    from phycoflow_reconstruction.coherence.families.topology.filtration_sampling import (
        TopologyFiltrationSampler,
        sampled_family_forward,
    )
    from phycoflow_reconstruction.data.manifest import register_dataset_identity
    from phycoflow_reconstruction.data.training_batches import build_training_batch_source
    from phycoflow_reconstruction.training.common import iter_unique_batch_indices
    from phycoflow_reconstruction.training.execution_strategy.audit import fixed_train_probe
    from phycoflow_reconstruction.training.execution_strategy.bottleneck import profile_one_update
    from phycoflow_reconstruction.training.execution_strategy.contracts import (
        CostProfile,
        PlanMeasurement,
        TaskContract,
    )
    from phycoflow_reconstruction.training.execution_strategy.planner import select_execution_plan
    from phycoflow_reconstruction.training.execution_strategy.providers import action_registry
    from phycoflow_reconstruction.training.execution_strategy.runtime import scientific_config_hash
    from phycoflow_reconstruction.training.fidelity_budget import ExternalPolicy, field_mse
    from phycoflow_reconstruction.training.model_lifecycle import (
        add_training_aux_state,
        after_optimizer_step,
        load_training_aux_state,
    )
    from phycoflow_reconstruction.training.post_training import (
        _calibrate_fidelity_budget,
        _coherence_objective,
        _configure_persistence_workers,
    )
    from phycoflow_reconstruction.training.retention import post_training_mode
    from phycoflow_reconstruction.training.source import load_source_model, source_hashes
    from phycoflow_reconstruction.training.update_budget import post_training_steps_per_epoch
    from phycoflow_reconstruction.utils.reproducibility import seed_everything

    for record in json.loads((root / "manifests/dataset_identities.json").read_text())[
        "identities"
    ]:
        register_dataset_identity(record["path"], record["fingerprint"], record["expected_stat"])
    config = yaml.safe_load(args.config.read_text())
    if config["optimization"]["epochs"] != 100:
        raise ValueError("the bounded science profile requires its frozen 100-epoch horizon")
    if config["dataset"].get("include_temporal_derivative", False):
        raise ValueError("split-crossing derivatives prohibited")
    if config["coherence"].get("family_balance", {}).get("mode", "none") != "none":
        raise ValueError("cost-aware v1 profiling requires fixed uncalibrated family weights")
    device = torch.device("cuda:0")
    seed_everything(42, True)
    torch.set_num_threads(4)
    _configure_persistence_workers(config)
    # The admitted command clock includes parent preflight, child imports and
    # compilation. The outer controller charges this same lifetime once.
    started = float(os.environ.get("PHYCOFLOW_COMMAND_STARTED_MONOTONIC", perf_counter()))
    hardware = {
        "physical_gpu": subprocess.check_output(
            [
                "nvidia-smi",
                "-i",
                str(args.gpu),
                "--query-gpu=uuid,name,driver_version",
                "--format=csv,noheader",
            ],
            text=True,
        ).strip(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "torch_threads": torch.get_num_threads(),
        "persistence_workers": os.environ.get("PHYCOFLOW_TOPOLOGY_WORKERS", "1"),
    }
    protected = source_hashes(config)
    model, dataset, _metadata = load_source_model(config, device)
    post_training_mode(model, config)
    source_state = deepcopy(model.state_dict())
    source_aux = deepcopy(add_training_aux_state({}, model))
    families = {
        name: family.to(device)
        for name, family in build_enabled_families(
            config["coherence"],
            dataset.data_spec,
            dataset.normalizer,
        ).items()
    }
    banks = {name: None for name in families}
    scales = {name: 1.0 for name in families}
    reference, calibration = _calibrate_fidelity_budget(
        model,
        dataset,
        families,
        banks,
        config,
        scales,
        device,
    )
    write_new(attempt / "source_calibration.json", calibration)
    steps = post_training_steps_per_epoch(config, len(dataset))
    batches = build_training_batch_source(
        dataset,
        iter_unique_batch_indices(
            len(dataset),
            4,
            int(config["optimization"]["batch_size"]),
            generator=torch.Generator().manual_seed(701103),
        ),
        config,
        query_points=int(config["coherence"]["compute_budget"]["point_count"]),
        device=device,
        start_step=0,
    )
    parameters = tuple(parameter for parameter in model.parameters() if parameter.requires_grad)
    names = tuple(families)
    # At least two contributions are dense in every support point.
    within = {
        "topology": {
            "provider_id": TopologyFiltrationSampler.PROVIDER_ID,
            "estimator": "polarity",
            "conditional_inclusion_probability": 0.5,
        }
    }
    candidates = [
        ("full", {}, {}),
        ("polarity", {}, within),
        ("family_topology", {"topology": 0.5}, {}),
        ("nested_topology", {"topology": 0.5}, within),
        ("family_global_distribution", {"global_distribution": 0.5}, {}),
        ("family_cross_spectrum", {"cross_spectrum": 0.5}, {}),
    ]
    timings = {name: [] for name, _, _ in candidates}
    variances = {name: [] for name, _, _ in candidates}
    full_gradients = []
    audit = []
    bottleneck = None
    fresh_cache_records = []
    warm_full_timings = []
    fixed_probe_batches = []
    rng_snapshot = (
        random.getstate(),
        np.random.get_state(),
        torch.get_rng_state(),
        torch.cuda.get_rng_state_all(),
    )

    def restore_probe_rng():
        random.setstate(rng_snapshot[0])
        np.random.set_state(rng_snapshot[1])
        torch.set_rng_state(rng_snapshot[2])
        torch.cuda.set_rng_state_all(rng_snapshot[3])

    actions = 0
    cache_stats = {
        "reference_hits": 0,
        "reference_misses": 0,
        "generated_calls": 0,
        "evictions": 0,
        "peak_bytes": 0,
        "lookup_and_pairing_seconds": 0.0,
    }
    original_pairing = persistence._cpu_pairing

    def observed_pairing(values, periodic, cacheable):
        tick = perf_counter()
        if cacheable:
            key = (
                values.shape,
                periodic,
                hashlib.blake2b(values.tobytes(), digest_size=16).digest(),
            )
            with persistence._REFERENCE_LOCK:
                hit = key in persistence._REFERENCE_PAIRS
                before_len = len(persistence._REFERENCE_PAIRS)
            cache_stats["reference_hits" if hit else "reference_misses"] += 1
        else:
            cache_stats["generated_calls"] += 1
        result = original_pairing(values, periodic, cacheable)
        if cacheable:
            with persistence._REFERENCE_LOCK:
                used = sum(
                    array.nbytes
                    for pairs in persistence._REFERENCE_PAIRS.values()
                    for side in pairs
                    for array in side
                )
                after_len = len(persistence._REFERENCE_PAIRS)
            cache_stats["evictions"] += int(not hit and after_len <= before_len)
            cache_stats["peak_bytes"] = max(cache_stats["peak_bytes"], used)
        cache_stats["lookup_and_pairing_seconds"] += perf_counter() - tick
        return result

    persistence._cpu_pairing = observed_pairing

    def flat(loss, retain_graph=True):
        parts = torch.autograd.grad(loss, parameters, allow_unused=True, retain_graph=retain_graph)
        return torch.cat(
            [
                (torch.zeros_like(parameter) if gradient is None else gradient).reshape(-1)
                for parameter, gradient in zip(parameters, parts)
            ]
        ).detach()

    try:
        for batch_index, batch in enumerate(batches):
            restore_probe_rng()
            if batch_index < 2:
                fixed_probe_batches.append(batch)
            cache_before = dict(cache_stats)
            # Each prospective plan starts with the same realistic cache from
            # calibration and preceding TRAIN batches, before this oracle can
            # prime the current target. Snapshot/restore is acquisition work.
            with persistence._REFERENCE_LOCK:
                stream_cache_before = persistence._REFERENCE_PAIRS.copy()
            endpoint, losses = {}, {}
            result, _ = _coherence_objective(
                model,
                batch,
                families,
                banks,
                config,
                step=batch_index,
                generator=torch.Generator(device=device).manual_seed(1000045 + batch_index),
                family_losses=losses,
                endpoint_context=endpoint,
            )
            fresh_cache_records.append(
                {
                    key: cache_stats[key] - cache_before[key]
                    for key in cache_stats
                    if key != "peak_bytes"
                }
            )
            with persistence._REFERENCE_LOCK:
                stream_cache_after = persistence._REFERENCE_PAIRS.copy()
            input_norms = {
                name: float(
                    torch.autograd.grad(loss, endpoint["prediction"], retain_graph=True)[0].norm()
                )
                for name, loss in losses.items()
            }
            gradients = {
                name: flat(float(getattr(families[name], "family_weight", 1.0))
                           * loss / reference.objective_scale)
                for name, loss in losses.items()
            }
            full_gradient = sum(gradients.values())
            full_gradients.append(full_gradient.cpu())
            sampler = TopologyFiltrationSampler(
                families["topology"].spatial_objective, strategy="polarity", seed=700211
            )
            inner = []
            for choice in sampler.enumerate_choices():
                estimate = sampled_family_forward(
                    families["topology"],
                    endpoint["prediction"],
                    endpoint["target"],
                    coordinates=batch.query_coords,
                    sampler=sampler,
                    choice=choice,
                )
                inner.append(flat(float(getattr(families["topology"], "family_weight", 1.0))
                                  * estimate.scalar_loss / reference.objective_scale))
            topology = gradients["topology"]
            inner_mean = sum(inner) / len(inner)
            inner_error = float((inner_mean - topology).norm() / topology.norm().clamp_min(1e-12))
            if inner_error > 5e-4:
                raise ValueError("real parameter-gradient polarity expectation failed")
            for plan_id, probabilities, within_family in candidates:
                outer = next(iter(probabilities), None)
                if within_family:
                    outcomes = [gradient - topology for gradient in inner]
                    if outer:
                        outcomes = [-topology] + [2 * gradient - topology for gradient in inner]
                        masses = [0.5, 0.25, 0.25]
                    else:
                        masses = [0.5, 0.5]
                elif outer:
                    outcomes = [-gradients[outer], gradients[outer]]
                    masses = [0.5, 0.5]
                else:
                    outcomes, masses = [torch.zeros_like(full_gradient)], [1.0]
                variances[plan_id].append(
                    sum(
                        mass * float(delta.double().square().sum())
                        for mass, delta in zip(masses, outcomes)
                    )
                )
                if outer is None:
                    supports = [(1.0, True)]
                else:
                    supports = [(0.5, False), (0.5, True)]
                duration = 0.0
                trials = [(mass, included, False) for mass, included in supports]
                if plan_id == "full":
                    trials.append((0.0, True, True))
                for mass, included, warm_repeat in trials:
                    restore_probe_rng()
                    if not warm_repeat:
                        with persistence._REFERENCE_LOCK:
                            persistence._REFERENCE_PAIRS.clear()
                            persistence._REFERENCE_PAIRS.update(stream_cache_before)
                    optimizer = torch.optim.AdamW(
                        parameters,
                        lr=float(config["optimization"]["lr"]),
                        weight_decay=float(config["optimization"]["weight_decay"]),
                    )
                    policy = ExternalPolicy(
                        config["optimization"]["multitask"]["external_policy"], reference, device
                    )
                    execution = {
                        "family_probabilities": {
                            name: probabilities.get(name, 1.0) for name in names
                        },
                        "family_inclusions": {name: name != outer or included for name in names},
                        "within_family": {
                            name: dict(spec, seed=3700228 + batch_index)
                            for name, spec in within_family.items()
                        },
                    }
                    current_endpoint = {}
                    # Instrumentation scans/cache-byte accounting are charged
                    # acquisition, not part of a production optimizer update.
                    persistence._cpu_pairing = original_pairing
                    torch.cuda.synchronize()
                    tick = perf_counter()
                    current, _ = _coherence_objective(
                        model,
                        batch,
                        families,
                        banks,
                        config,
                        step=batch_index,
                        generator=torch.Generator(device=device).manual_seed(1000045 + batch_index),
                        endpoint_context=current_endpoint,
                        execution=execution,
                    )
                    risks = field_mse(
                        current_endpoint["prediction"],
                        current_endpoint["target"],
                        current_endpoint["mask"],
                    )
                    policy.update(
                        model,
                        optimizer,
                        current.scalar_loss,
                        risks,
                        config["optimization"].get("grad_clip"),
                    )
                    after_optimizer_step(model)
                    torch.cuda.synchronize()
                    measured = perf_counter() - tick
                    persistence._cpu_pairing = observed_pairing
                    if warm_repeat:
                        warm_full_timings.append(measured)
                    else:
                        duration += mass * measured
                    actions += 1
                    model.load_state_dict(source_state)
                    load_training_aux_state(model, source_aux)
                    restore_probe_rng()
                    optimizer.zero_grad(set_to_none=True)
                    del current, current_endpoint, optimizer, policy
                timings[plan_id].append(duration)
            with persistence._REFERENCE_LOCK:
                persistence._REFERENCE_PAIRS.clear()
                persistence._REFERENCE_PAIRS.update(stream_cache_after)
            if batch_index == 0:
                restore_probe_rng()
                optimizer = torch.optim.AdamW(
                    parameters,
                    lr=float(config["optimization"]["lr"]),
                    weight_decay=float(config["optimization"]["weight_decay"]),
                )
                policy = ExternalPolicy(
                    config["optimization"]["multitask"]["external_policy"], reference, device
                )

                def profiled_update(*, scope, batch=batch, optimizer=optimizer, policy=policy):
                    payload = {}
                    current, _ = _coherence_objective(
                        model,
                        batch,
                        families,
                        banks,
                        config,
                        step=0,
                        generator=torch.Generator(device=device).manual_seed(1000045),
                        endpoint_context=payload,
                    )
                    with scope("dense_fidelity"):
                        risks = field_mse(payload["prediction"], payload["target"], payload["mask"])
                    original_step = optimizer.step
                    original_dual_step = policy._dual_optimizer.step

                    def model_action(*a, **kw):
                        with scope("adamw_step"):
                            return original_step(*a, **kw)

                    def dual_action(*a, **kw):
                        with scope("dual_update"):
                            return original_dual_step(*a, **kw)

                    optimizer.step = model_action
                    policy._dual_optimizer.step = dual_action
                    with scope("policy_backward"):
                        diagnostics = policy.update(
                            model,
                            optimizer,
                            current.scalar_loss,
                            risks,
                            config["optimization"].get("grad_clip"),
                        )
                    with scope("model_lifecycle"):
                        after_optimizer_step(model)
                    return diagnostics

                persistence._cpu_pairing = original_pairing
                bottleneck = profile_one_update(model, families, profiled_update, device)
                persistence._cpu_pairing = observed_pairing
                actions += 1
                model.load_state_dict(source_state)
                load_training_aux_state(model, source_aux)
                restore_probe_rng()
                optimizer.zero_grad(set_to_none=True)
                del optimizer, policy, profiled_update
            with persistence._REFERENCE_LOCK:
                persistence._REFERENCE_PAIRS.clear()
                persistence._REFERENCE_PAIRS.update(stream_cache_after)
            audit.append(
                {
                    "batch": batch_index,
                    "sample_ids": list(batch.sample_ids),
                    "parameter_expectation_relative_error": inner_error,
                    "full_gradient_norm": float(full_gradient.norm()),
                    "output_space_proxy_norms": input_norms,
                    "proxy_label": "parameter gradient; finite four-batch estimate, no confidence certificate",
                }
            )
            del result, endpoint, losses, gradients, inner, full_gradient, batch
            print(
                f"completed bounded TRAIN profile {args.task} batch {batch_index + 1}/4", flush=True
            )
            with (attempt / "progress.jsonl").open("a") as handle:
                handle.write(
                    json.dumps(
                        {
                            "completed_batches": batch_index + 1,
                            "completed_restored_updates": actions,
                            "elapsed_seconds": perf_counter() - started,
                        }
                    )
                    + "\n"
                )
    finally:
        persistence._cpu_pairing = original_pairing
        batches.close()
        dataset.close()
    # Time the actual scheduled numerical audit, using retained TRAIN batches.
    # The observer/refresh stub contributes no model work and makes no decisions.
    monitor_costs = {}
    for plan_id, probabilities, within_family in candidates:
        audit_plan = SimpleNamespace(
            audit_interval_epochs=20,
            within_family=within_family,
            family_probabilities=probabilities,
            family_support=(),
            plan_id=plan_id,
        )
        probe_runtime = SimpleNamespace(
            task=SimpleNamespace(
                c0=reference.c0,
                field_bounds=tuple(
                    zip(reference.field_names, reference.upper_bounds, reference.denominators)
                ),
            ),
            plan=audit_plan,
            profile=SimpleNamespace(intrinsic_noise_variance=1.0),
            observe_audit=lambda *a, **kw: None,
            refresh=lambda epoch: None,
        )
        audit_plan.extra_variance = 0.0
        ordinary = fixed_train_probe(
            model, fixed_probe_batches, families, banks, config, probe_runtime, 10
        )
        expensive = fixed_train_probe(
            model, fixed_probe_batches, families, banks, config, probe_runtime, 20
        )
        monitor_costs[plan_id] = {
            "refresh_s": ordinary["seconds"],
            "full_audit_s": expensive["seconds"],
            "incremental_full_audit_s": max(0.0, expensive["seconds"] - ordinary["seconds"]),
        }
    restore_probe_rng()
    if source_hashes(config) != protected or any(
        not torch.equal(value, model.state_dict()[key]) for key, value in source_state.items()
    ):
        raise ValueError("SOURCE/model changed during restored profiling")
    stacked = torch.stack(full_gradients).double()
    intrinsic = float((stacked - stacked.mean(0)).square().sum(1).mean())
    current_acquisition = perf_counter() - started
    previous_acquisition = prior_profile_seconds(root, args.task)
    acquisition = current_acquisition + previous_acquisition
    ref = reference.state_dict()
    task = TaskContract(
        task_id=args.task,
        source_identity=protected["checkpoint"],
        case_id=Path(config["source_run"]).parents[2].name,
        dataset_id=digest(config["dataset"]),
        split_identity="TRAIN calibration/probes; frozen validation assessment; TEST locked",
        adapter_kind="generative" if model.capabilities.generative else "deterministic",
        reconstruction_identity=digest(
            {key: config[key] for key in ["model", "rollout", "observation_consistency"]}
        ),
        normalization_identity=digest(
            {
                key: value.tolist() if isinstance(value, torch.Tensor) else value
                for key, value in dataset.normalizer.state_dict().items()
            }
        ),
        objective_identity=digest(config["coherence"]),
        objective_rule="scalar_additive",
        component_weights=tuple(
            (name, float(getattr(family, "family_weight", 1))) for name, family in families.items()
        ),
        aggregation_rule="fixed_sum",
        source_calibration_identity=digest(calibration),
        c0=reference.objective_scale,
        field_bounds=tuple(
            zip(reference.field_names, reference.upper_bounds, reference.denominators)
        ),
        query_geometry_identity=digest(config["coherence"]["compute_budget"]),
        sampling_identity=digest(
            {"observations": config["observations"], "seed": 42, "B_group": 32}
        ),
        numerical_precision="float32 model/descriptor; float64 risk pooling",
        optimizer_identity=digest(config["optimization"]),
        epoch_definition=f"{steps} minibatches of {config['optimization']['batch_size']}",
        quality_contract_identity=digest(
            json.loads((root / "tasks/task_inventory.json").read_text())["quality_contract"]
        ),
        max_epochs=100,
        provider_ids=(TopologyFiltrationSampler.PROVIDER_ID,),
        features=(
            "scalar_optimizer",
            "dense_fidelity",
            "shared_forward",
            "complete_target_cache_identity",
            "additive_family_provider",
        ),
        metadata={
            "scientific_config_hash": scientific_config_hash(config),
            "steps_per_epoch": steps,
            "source_hashes": protected,
            "calibration_reference": ref,
        },
    )
    measurements = []
    for plan_id, probabilities, within_family in candidates:
        values = timings[plan_id]
        actions_selected = []
        if probabilities:
            actions_selected.append("whole_family_sampling")
        if within_family:
            actions_selected.append("generalized_filtration_sampling")
        measurements.append(
            PlanMeasurement(
                plan_id,
                sum(values) / len(values) * steps,
                max(values) * steps,
                actions=tuple(actions_selected),
                extra_variance=sum(variances[plan_id]) / 4,
                variance_method="exact_joint_support",
                estimator_kind="unbiased_scalar_estimator" if actions_selected else "equivalent",
                family_probabilities=(
                    {name: probabilities.get(name, 1.0) for name in names} if probabilities else {}
                ),
                within_family=within_family,
                # Budget future full TRAIN monitoring and conservative refreshes.
                refresh_overhead_s=monitor_costs[plan_id]["refresh_s"],
                audit_overhead_s=monitor_costs[plan_id]["incremental_full_audit_s"],
            )
        )
    profile = CostProfile(
        task.task_hash,
        digest(hardware),
        0,
        100,
        tuple(measurements),
        intrinsic,
        profile_overhead_s=acquisition,
    )
    plan = select_execution_plan(task, profile)
    write_new(root / f"tasks/{args.task}_task_contract.json", task.to_dict())
    write_new(root / f"profiles/{args.task}_source_calibration.json", calibration)
    write_new(root / f"profiles/{args.task}_profile.json", profile.to_dict())
    write_new(root / f"plans/{args.task}_plan_decision.json", plan.to_dict())
    registry = action_registry(task)
    write_new(
        root / f"tasks/{args.task}_action_registry.json", [item.to_dict() for item in registry]
    )
    write_new(
        root / f"profiles/{args.task}_raw_profile.json",
        {
            "timings_seconds": timings,
            "warm_full_timings_seconds": warm_full_timings,
            "reference_cache_timing_scope": "All six plans start from matched calibration/prior-TRAIN cache, before current-batch oracle preparation; full additionally repeated warm. Accounting wrapper disabled during timed updates. Four-batch finite profile, not steady-state cache certification.",
            "parameter_execution_variances": variances,
            "intrinsic_parameter_variance": intrinsic,
            "batches": audit,
            "cache": cache_stats,
            "cache_counter_scope": "Instrumented oracle/input-gradient and support-enumeration calls only; whole-update timing and bottleneck callback use the production cache function without accounting scans. Fresh TRAIN records isolate the first full oracle per batch.",
            "fresh_TRAIN_stream_cache": fresh_cache_records,
            "measured_monitor_costs": monitor_costs,
            "bottleneck": bottleneck,
            "hardware": hardware,
            "calibration_seconds": calibration["calibration_seconds"],
            "actual_optimizer_actions_restored": actions,
            "auxiliary_epoch_exposures": actions / steps,
            "inclusive_profile_seconds": perf_counter() - started,
            "prior_profile_attempt_seconds": previous_acquisition,
            "max_bytes_bound": "4096-entry primitive pairing cache; fixed 4096-vertex task; payload measured above",
            "timing_scope": "synchronized reconstruction/descriptors/dense risks/scalar backward/clipping/AdamW/dual; restoration is charged acquisition overhead",
            "limits": "four TRAIN batches; SOURCE parameters restored after each single-step child; not a convergence or latency certificate",
        },
    )
    strategy_config = deepcopy(config)
    strategy_config["output"]["experiment_name"] = (
        config["output"]["experiment_name"].rsplit("/", 1)[0] + "/strategy"
    )
    strategy_config["optimization"]["execution_strategy"] = {
        "enabled": True,
        "task": task.to_dict(),
        "profile": profile.to_dict(),
        "plan": plan.to_dict(),
        "seed": 700211,
    }
    with (root / f"configs/{args.task}_strategy.yaml").open("x") as handle:
        yaml.safe_dump(strategy_config, handle, sort_keys=False)
    print(
        json.dumps(
            {
                "task": args.task,
                "plan": plan.plan_id,
                "profile_seconds": perf_counter() - started,
                "auxiliary_exposures": actions / steps,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

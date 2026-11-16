"""Transactional actual-optimizer displacement probes for Round 2."""

import random
from copy import deepcopy
from dataclasses import replace

import numpy as np
import torch

from .internal_calibration import sample_weighted_parameter_gradients


def virtual_optimizer_step(model, optimizer, *, propose_step, assess, stateful=()):
    """Run the supplied real optimizer update and restore all affected state.

    ``assess`` must compute the common assessment and explicit fidelity/sensor
    constraints on a fixed batch and member-seed manifest. ``propose_step``
    must invoke the same ConFIG, external policy, clipping and optimizer path
    as the intended run. This function does not manufacture either operation.
    """
    original_model = deepcopy(model.state_dict())
    original_optimizer = deepcopy(optimizer.state_dict())
    original_extra = [(obj, deepcopy(obj.state_dict())) for obj in stateful]
    original_modes = {module: module.training for module in model.modules()}
    original_grads = [None if p.grad is None else p.grad.detach().clone() for p in model.parameters()]
    torch_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    py_rng, numpy_rng = random.getstate(), np.random.get_state()

    def restore_rng():
        torch.set_rng_state(torch_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
        random.setstate(py_rng)
        np.random.set_state(numpy_rng)

    try:
        before = assess()
        restore_rng()
        propose_step()
        restore_rng()
        after = assess()
        return before, after
    finally:
        model.load_state_dict(original_model, strict=True)
        optimizer.load_state_dict(original_optimizer)
        for obj, state in original_extra:
            obj.load_state_dict(state)
        for module, mode in original_modes.items():
            module.training = mode
        for parameter, gradient in zip(model.parameters(), original_grads):
            parameter.grad = gradient
        restore_rng()


def virtual_scale_response(model, optimizer, controller, aggregator, *, build_losses,
                           assess, ratio, grad_clip):
    """Run one real three-family ConFIG/fidelity/AdamW proposed displacement.

    ``build_losses(ratio)`` must produce the three ordered calibrated losses
    with only scale.joint multiplied by ratio, and the actual fidelity risk
    vector from the same member graphs. ``assess`` returns the common fixed
    target and constraints, and can include detached endpoints for atlas reuse.
    """
    if ratio not in (1, 2, 4):
        raise ValueError("Scale ratio must be one of 1, 2, or 4")
    if not isinstance(optimizer, torch.optim.AdamW):
        raise TypeError("Scale probe requires the actual AdamW optimizer")
    step_receipt = {}
    def step():
        family_losses, risks = build_losses(ratio)
        step_receipt.update(controller.update_with_coherence_aggregation(
            model, optimizer, family_losses, risks, grad_clip, aggregator
        ))
    before, after = virtual_optimizer_step(model, optimizer, propose_step=step, assess=assess,
                                           stateful=(controller,))
    if isinstance(after, dict):
        after["optimizer_diagnostics"] = step_receipt
    return before, after


def accumulated_coherence_update(model, optimizer, controller, aggregator, microbatches,
                                 *, family_names, risk_names, grad_clip):
    """Apply production policy once to sample-weighted full-batch gradients.

    Each microbatch supplies named family and individual field-risk scalar
    graphs. Their gradients are accumulated before ConFIG and external policy.
    The linear surrogate has the exact accumulated value and gradient at the
    current parameters; it performs no microbatch optimizer updates.
    """
    if len(family_names) != 3 or not risk_names or set(family_names) & set(risk_names):
        raise ValueError("three unique families and separate named field risks required")
    parameters = tuple(p for p in model.parameters() if p.requires_grad)
    gradients, values, ids = sample_weighted_parameter_gradients(microbatches, parameters)
    expected = (*family_names, *risk_names)
    if tuple(gradients) != expected:
        raise ValueError("microbatch loss order differs from declared family and risk order")
    surrogates = {}
    for name in expected:
        linear = sum((parameter * gradient).sum() for parameter, gradient in zip(parameters, gradients[name]))
        surrogates[name] = linear - linear.detach() + values[name]
    return controller.update_with_coherence_aggregation(
        model, optimizer, {name: surrogates[name] for name in family_names},
        torch.stack([surrogates[name] for name in risk_names]), grad_clip, aggregator
    ), ids


def choose_scale_ratio(records, *, source_scale, deterministic_replay_error):
    """Apply the frozen local-response decision to actual measured trial rows."""
    if source_scale < 0 or deterministic_replay_error < 0:
        raise ValueError("nonnegative SOURCE Scale and replay error required")
    records = list(records)
    if not records:
        raise ValueError("Scale probe requires measured rows")
    seeds = {row["seed"] for row in records}
    if len(seeds) != 3:
        raise ValueError("Scale probe requires exactly three fixed seed manifests")
    identities = {(row["parameterization"], row["ratio"], row["seed"]) for row in records}
    expected = {(kind, ratio, seed) for kind in ("full_model", "residual_head")
                for ratio in (1, 2, 4) for seed in seeds}
    if identities != expected:
        raise ValueError("Scale probe requires both parameterizations, all ratios and all seeds")
    threshold = max(1e-4 * source_scale, 5 * deterministic_replay_error)
    decisions = {}
    for ratio in (1, 2, 4):
        passed = True
        for kind in ("full_model", "residual_head"):
            all_rows = []
            for seed in seeds:
                rows = [r for r in records if (r["parameterization"], r["ratio"], r["seed"]) == (kind, ratio, seed)]
                if any(r["sample_count"] < 1 or not np.isfinite(r["scale_before"]) or
                       not np.isfinite(r["scale_after"]) for r in rows):
                    raise ValueError("Scale probe rows require finite values and positive sample counts")
                ids = [sample for row in rows for sample in row["sample_ids"]]
                if any(len(row["sample_ids"]) != row["sample_count"] for row in rows) or len(ids) != len(set(ids)):
                    raise ValueError("Scale probe rows require disjoint complete sample IDs")
                passed &= all(r["field_and_sensor_feasible"] is True for r in rows)
                all_rows.extend(rows)
            decrease = sum(r["sample_count"] * (r["scale_before"] - r["scale_after"]) for r in all_rows) / sum(r["sample_count"] for r in all_rows)
            improvement_fraction = sum(r["scale_after"] < r["scale_before"] for r in all_rows) / len(all_rows)
            passed &= improvement_fraction >= .75 and decrease > threshold
        decisions[ratio] = bool(passed)
    return {"chosen_ratio": next((r for r in (1, 2, 4) if decisions[r]), 1),
            "passed": decisions, "minimum_pooled_decrease": threshold,
            "fallback_used": not any(decisions.values())}


def probe_fidelity(child, source, target, masses, reference, *, generative,
                   sensor_child, sensor_source, observation_values, observation_fields,
                   observation_mask, zero_tolerance, sampling):
    """Compute declared field and per-observed-field sensor checks on matched outputs.

    Member arrays have shape [B,J,Q,C]; sensor arrays contain predictions at
    the actual observed field/site entries [B,J,M]. ``sampling`` supplies the
    native mass and inclusion probability for a selected-support proxy.
    """
    from ..coherence.physical_risk import physical_ensemble_risks, physical_field_mse
    if (child.shape != source.shape or child.ndim != 4 or target.ndim != 3 or
        child.shape[0] != target.shape[0] or child.shape[2:] != target.shape[1:]):
        raise ValueError("matched member, target and query shapes required")
    tolerances = torch.as_tensor(zero_tolerance, dtype=torch.float64, device=child.device).flatten()
    if tolerances.numel() == 1:
        tolerances = tolerances.expand(child.shape[-1])
    if (tolerances.numel() != child.shape[-1] or not torch.isfinite(tolerances).all()
        or (tolerances < 0).any()):
        raise ValueError("sensor zero-source tolerance must be finite and nonnegative")
    if generative:
        if child.shape[1] < 2:
            raise ValueError("FFM mean-risk check requires independent J2 members")
        _, risk = physical_ensemble_risks(child, target, masses, **sampling)
        risk = risk.mean(0)
        field_violation = reference.violations(risk)
        field_ok = bool((field_violation <= 0).all())
        source_risk = None
    else:
        if child.shape[1] != 1:
            raise ValueError("Senseiver fidelity requires J1")
        risk = physical_field_mse(child[:, 0], target, masses, **sampling).mean(0)
        baseline = physical_field_mse(source[:, 0], target, masses, **sampling).mean(0)
        field_ok = bool((risk <= 1.1 * baseline + (baseline == 0) * tolerances.to(risk)).all())
        source_risk = baseline.detach().cpu().tolist()
    if sensor_child.shape != sensor_source.shape or sensor_child.shape[:2] != child.shape[:2] or sensor_child.shape[2:] != observation_mask.shape[1:]:
        raise ValueError("sensor predictions must align with observed entries and members")
    errors = (sensor_child - observation_values[:, None, :, 0]).square()
    baseline_errors = (sensor_source - observation_values[:, None, :, 0]).square()
    sensor = {}
    for field in range(child.shape[-1]):
        mask = observation_mask & (observation_fields == field)
        if not mask.any():
            continue
        selected = mask[:, None, :].expand_as(errors)
        value = errors[selected].mean()
        baseline = baseline_errors[selected].mean()
        limit = 1.1 * baseline + (baseline == 0) * tolerances[field].to(baseline)
        sensor[str(field)] = {"source_mse": float(baseline.detach()), "child_mse": float(value.detach()),
                              "count": int(selected.sum()), "passed": bool(value <= limit)}
    return {"field_passed": field_ok, "sensor_passed": all(row["passed"] for row in sensor.values()),
            "sensor_by_field": sensor, "risk": risk.detach().cpu().tolist(),
            "source_risk": source_risk,
            "support": "sampled-support proxy" if sampling.get("sampling") != "full" else "native"}


def rollout_probe_members(model, batch, config, member_seeds):
    """Run the actual SOURCE or child at exactly the supplied member seeds."""
    from .rollout import differentiable_reconstruction
    expected = 2 if model.capabilities.generative else 1
    if len(member_seeds) != expected or len(set(member_seeds)) != expected:
        raise ValueError("probe needs independent J2 or genuine J1 member seeds")
    source_batch = replace(batch, target_fields=None)
    return torch.stack([differentiable_reconstruction(
        model, source_batch, steps=int(config["rollout"]["steps"]), solver=config["rollout"]["solver"],
        generator=torch.Generator(device=batch.query_coords.device).manual_seed(int(seed)),
        observation_config=dict(config["observation_consistency"])) for seed in member_seeds], dim=1)


def observed_entry_predictions(model, batch, config, member_seeds):
    """Evaluate every observed field/site, preserving its source sensor behavior."""
    if batch.obs_indices is None:
        raise ValueError("observed sensor positions require point IDs")
    for row in range(batch.obs_indices.shape[0]):
        valid_ids = batch.obs_indices[row, batch.obs_valid_mask[row]]
        if valid_ids.unique().numel() != valid_ids.numel():
            raise ValueError("sensor query rollout requires unique observed site IDs")
    sensor_batch = replace(batch, query_coords=batch.obs_coords,
                           query_valid_mask=batch.obs_valid_mask, target_fields=None,
                           metadata={**batch.metadata, "query_indices": batch.obs_indices})
    members = rollout_probe_members(model, sensor_batch, config, member_seeds)
    fields = batch.obs_field_ids.masked_fill(~batch.obs_valid_mask, 0).long()
    return torch.gather(members, -1, fields[:, None, :, None].expand(-1, members.shape[1], -1, 1)).squeeze(-1)


def probe_family_graphs(runtime, model, batch, config, member_seeds, query_batch, *, exhaustive):
    """Return actual member outputs and three differentiable family objectives."""
    from ..data.physical_measure import batch_measure_context
    members = rollout_probe_members(model, batch, config, member_seeds)
    context = {"sample_ids": batch.sample_ids, "reference_ids": batch.sample_ids,
               **batch_measure_context(batch), "conditional_draws": tuple(members[:, j] for j in range(1, members.shape[1])),
               "exhaustive_shape_queries": bool(exhaustive)}
    family_losses = {}
    for name, family in runtime.families.items():
        result, _, _ = runtime._score(name, family, members[:, 0], batch.target_fields,
                                      batch.query_coords, context, query_batch)
        family_losses[name] = result.scalar_loss * float(getattr(family, "family_weight", 1.0))
    return members, family_losses


def common_scale_score(runtime, members, batch, query_batch):
    """Complete unit-coefficient C8 Scale observer, excluding expanded leaves."""
    from ..data.physical_measure import batch_measure_context
    context = {"sample_ids": batch.sample_ids, "reference_ids": batch.sample_ids,
               **batch_measure_context(batch), "conditional_draws": tuple(members[:, j] for j in range(1, members.shape[1])),
               "scoring_mode": runtime.scoring_mode, "field_order": runtime.fields}
    result, _, _ = runtime.adapter.score("multiscale_spectrum", members[:, 0], batch.target_fields,
                                         coordinates=batch.query_coords, context=context, batch=query_batch,
                                         coefficients={})
    return result.scalar_loss


def pooled_probe_feasibility(reports, sample_counts, reference, *, generative, zero_tolerance):
    """Pool real per-batch field/sensor statistics before checking each manifest."""
    if len(reports) != len(sample_counts) or not reports or any(n < 1 for n in sample_counts):
        raise ValueError("matched nonempty reports and positive sample counts required")
    count = sum(sample_counts)
    risks = sum(torch.tensor(row["risk"], dtype=torch.float64) * n for row, n in zip(reports, sample_counts)) / count
    tolerances = torch.as_tensor(zero_tolerance, dtype=torch.float64).flatten()
    if tolerances.numel() == 1:
        tolerances = tolerances.expand(risks.numel())
    if tolerances.numel() != risks.numel() or not torch.isfinite(tolerances).all() or (tolerances < 0).any():
        raise ValueError("pooled sensor tolerances must match the field count")
    if generative:
        field_ok = bool((reference.violations(risks) <= 0).all())
    else:
        baseline = sum(torch.tensor(row["source_risk"], dtype=torch.float64) * n for row, n in zip(reports, sample_counts)) / count
        field_ok = bool((risks <= 1.1 * baseline + (baseline == 0) * tolerances).all())
    fields = set().union(*(row["sensor_by_field"] for row in reports))
    sensor_ok = True
    for field in fields:
        rows = [row["sensor_by_field"][field] for row in reports if field in row["sensor_by_field"]]
        size = sum(row["count"] for row in rows)
        value = sum(row["child_mse"] * row["count"] for row in rows) / size
        baseline = sum(row["source_mse"] * row["count"] for row in rows) / size
        sensor_ok &= value <= 1.1 * baseline + (float(tolerances[int(field)]) if baseline == 0 else 0.)
    return {"field_passed": field_ok, "sensor_passed": bool(sensor_ok),
            "field_and_sensor_feasible": bool(field_ok and sensor_ok)}


def actual_scale_panel(*, runtime, source, full_model, residual_model, dataset, config, panel,
                       reference, controller_factory, optimizer_factory, aggregator_factory,
                       output_dir, seeds, zero_tolerance, device, family_scales, deadline_unix=None):
    """Run fixed-panel actual-policy trials and cache matched endpoints for atlas reuse.

    Factories construct a fresh production controller, optimizer and aggregator
    for each virtual step. The caller supplies the frozen shared fidelity
    reference and the explicitly nominated residual optimizer schedule.
    """
    import time
    from pathlib import Path

    from ..coherence.compose import three_family_order
    from ..coherence.query_sampling import SharedQuerySampler, full_target_query_batch
    from ..data.physical_measure import batch_measure_context
    from .coherence_sampling import _conditional_draw_seed
    from .post_training import _physical_endpoint_risk
    from .round2_prefit import matched_panel_batches
    from .run_store import file_sha256

    if len(set(seeds)) != 3 or len(seeds) != 3 or runtime.calibration.coefficients is None:
        raise ValueError("frozen coefficients and exactly three distinct manifests required")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=False)
    source.eval()
    models = {"full_model": full_model, "residual_head": residual_model}
    original_coefficients = dict(runtime.calibration.coefficients)
    rows, source_scales, replay_errors = [], [], []
    try:
        for seed in seeds:
            batches = matched_panel_batches(runtime, dataset, config, panel, device=device,
                                            seed=seed, deadline_unix=deadline_unix)
            for start, batch in batches:
                member_seeds = [_conditional_draw_seed(seed + start, j)
                                for j in range(2 if source.capabilities.generative else 1)]
                population = dataset.physical_weights.numel()
                q = batch.query_coords.shape[1]
                sampling = {"total_mass": dataset.physical_weights.sum().to(device),
                            "sampling": "full" if q == population else "without_replacement",
                            "probabilities": None if q == population else q / population}
                full_queries = full_target_query_batch(len(runtime.fields))
                with torch.no_grad():
                    baseline = rollout_probe_members(source, batch, config, member_seeds)
                    baseline_sensor = observed_entry_predictions(source, batch, config, member_seeds)
                    runtime.calibration.coefficients = {key: 1.0 for key in original_coefficients}
                    source_scale = float(common_scale_score(runtime, baseline, batch, full_queries))
                for kind, model in models.items():
                    for ratio in (1, 2, 4):
                        if deadline_unix is not None and time.time() >= deadline_unix:
                            raise TimeoutError("Scale probe deadline reached at trial boundary")
                        started = time.perf_counter()
                        controller = controller_factory()
                        optimizer = optimizer_factory(kind, model)
                        aggregator = aggregator_factory()
                        sampler = SharedQuerySampler(runtime.fields, seed=int(seed + start))
                        sampled_query = sampler.draw()
                        def build_losses(chosen_ratio, model=model, batch=batch, member_seeds=member_seeds, sampled_query=sampled_query):
                            runtime.calibration.coefficients = dict(original_coefficients)
                            runtime.calibration.coefficients["scale.joint"] *= chosen_ratio
                            members, losses = probe_family_graphs(runtime, model, batch, config, member_seeds, sampled_query, exhaustive=False)
                            losses = {name: losses[name] * family_scales[name] for name in three_family_order(runtime.families)}
                            endpoint = {"prediction": members[:, 0], "members": tuple(members[:, j] for j in range(members.shape[1])),
                                        "target": batch.target_fields, "mask": batch.query_valid_mask,
                                        "measure": batch_measure_context(batch)}
                            risks = _physical_endpoint_risk(endpoint, dataset, mean_risk=source.capabilities.generative).mean(0)
                            return losses, risks
                        def assess(model=model, batch=batch, member_seeds=member_seeds, full_queries=full_queries):
                            runtime.calibration.coefficients = {key: 1.0 for key in original_coefficients}
                            with torch.no_grad():
                                members = rollout_probe_members(model, batch, config, member_seeds)
                                scale = common_scale_score(runtime, members, batch, full_queries)
                                sensors = observed_entry_predictions(model, batch, config, member_seeds)
                            return {"scale": float(scale),
                                    "members": members.detach().cpu(), "sensors": sensors.detach().cpu()}
                        before, after = virtual_scale_response(model, optimizer, controller, aggregator,
                                                               build_losses=build_losses, assess=assess,
                                                               ratio=ratio, grad_clip=config["optimization"].get("grad_clip"))
                        report = probe_fidelity(after["members"].to(device), baseline, batch.target_fields,
                                                batch_measure_context(batch)["generated_weights"], reference,
                                                generative=source.capabilities.generative,
                                                sensor_child=after["sensors"].to(device), sensor_source=baseline_sensor,
                                                observation_values=batch.obs_values, observation_fields=batch.obs_field_ids,
                                                observation_mask=batch.obs_valid_mask, zero_tolerance=zero_tolerance,
                                                sampling=sampling)
                        elapsed = time.perf_counter() - started
                        name = f"{kind}_ratio{ratio}_seed{seed}_start{start}.pt"
                        torch.save({"schema": "car_round2_virtual_endpoints_v1",
                                    "sample_ids": list(batch.sample_ids), "member_seeds": member_seeds,
                                    "query_indices": batch.metadata["query_indices"].cpu(),
                                    "obs_indices": batch.obs_indices.cpu(), "obs_fields": batch.obs_field_ids.cpu(),
                                    "obs_valid_mask": batch.obs_valid_mask.cpu(), "obs_values": batch.obs_values.cpu(),
                                    "source": baseline.cpu().movedim(1, 0),
                                    "child": after["members"].movedim(1, 0),
                                    "reference": batch.target_fields.cpu(),
                                    "masses": batch_measure_context(batch)["generated_weights"].cpu(),
                                    "sensor_source": baseline_sensor.cpu().movedim(1, 0),
                                    "sensor_child": after["sensors"].movedim(1, 0),
                                    "method": kind, "ratio": ratio, "seed": seed,
                                    "batch_size": len(batch.sample_ids),
                                    "proposed_step_receipt": {"scale_before": before["scale"],
                                                              "scale_after": after["scale"],
                                                              "optimizer_diagnostics": after["optimizer_diagnostics"],
                                                              "fidelity": report, "wall_seconds": elapsed,
                                                              "support": "sampled-support proxy"}}, out / name)
                        row = {"parameterization": kind, "ratio": ratio, "seed": seed,
                               "sample_count": len(batch.sample_ids), "sample_ids": list(batch.sample_ids),
                               "scale_before": before["scale"], "scale_after": after["scale"],
                               "fidelity": report, "cache": name, "cache_sha256": file_sha256(out / name),
                               "wall_seconds": elapsed}
                        rows.append(row)
                        source_scales.append((source_scale, len(batch.sample_ids)))
                        replay_errors.append(abs(before["scale"] - source_scale))
        for kind in models:
            for ratio in (1, 2, 4):
                for seed in seeds:
                    group = [r for r in rows if (r["parameterization"], r["ratio"], r["seed"]) == (kind, ratio, seed)]
                    pooled = pooled_probe_feasibility([r["fidelity"] for r in group], [r["sample_count"] for r in group],
                                                     reference, generative=source.capabilities.generative,
                                                     zero_tolerance=zero_tolerance)
                    for row in group:
                        row["field_and_sensor_feasible"] = pooled["field_and_sensor_feasible"]
        # Exact SOURCE parity serves as a conservative numerical replay bound.
        source_scale_mean = sum(value * count for value, count in source_scales) / sum(count for _, count in source_scales)
        decision = choose_scale_ratio(rows, source_scale=source_scale_mean,
                                      deterministic_replay_error=max(replay_errors))
        return {"schema": "car_round2_scale_probe_v1", "rows": rows, "decision": decision,
                "source_scale": source_scale_mean,
                "source_replay_max_abs": max(replay_errors), "reserved_task_used": False}
    finally:
        runtime.calibration.coefficients = original_coefficients

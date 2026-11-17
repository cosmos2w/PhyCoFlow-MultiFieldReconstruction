"""Round 2 TRAIN nomination and held-out accept/fallback rules.

Core/event observers and deployment timers are supplied from the same saved
fields. The reserved task is deliberately absent from TRAIN nomination.
"""
from __future__ import annotations

import hashlib
import json
import math

import torch

from ..coherence.physical_risk import physical_ensemble_risks, physical_field_mse

AMPLITUDES = (0., .125, .25, .375, .5, .625, .75, .875, 1.)
CORE_KEYS = ("state", "scale", "shape", "event")


def scalar_task_score(members, reference, *, model_family):
    """Fair scalar J4 CRPS or genuine J1 paired absolute task error, per frame."""
    x, y = torch.as_tensor(members), torch.as_tensor(reference)
    if x.ndim != 2 or y.shape != x.shape[1:] or not torch.isfinite(x).all() or not torch.isfinite(y).all():
        raise ValueError("finite scalar [J,B] task members and [B] references required")
    if model_family == "senseiver":
        if len(x) != 1:
            raise ValueError("Senseiver task assessment requires a genuine J1 prediction")
        return (x[0] - y).abs()
    if model_family != "ffm" or len(x) != 4:
        raise ValueError("FFM final task assessment requires independent J4 members")
    attraction = (x - y).abs().mean(0)
    separation = sum((x[j] - x[k]).abs() for j in range(4) for k in range(j + 1, 4)) / 12
    return attraction - separation


def native_risks(members, reference, masses, *, model_family):
    x, truth = torch.as_tensor(members), torch.as_tensor(reference)
    expected = 4 if model_family == "ffm" else 1 if model_family == "senseiver" else 0
    if not expected or x.ndim != 4 or len(x) != expected or x.shape[1:] != truth.shape:
        raise ValueError("matched native J4 FFM or J1 Senseiver fields required")
    if model_family == "senseiver":
        mse = physical_field_mse(x[0], truth, masses)
        return {"member_mse": mse.mean(0), "ensemble_mean_mse": mse.mean(0),
                "signed_mean_risk": None, "unbiased_member_variance": None,
                "per_condition_member_mse": mse}
    member, signed = physical_ensemble_risks(x.transpose(0, 1), truth, masses)
    mean_mse = physical_field_mse(x.mean(0), truth, masses)
    # member MSE - empirical mean MSE = (J-1)/J * unbiased member variance.
    variance = (member - mean_mse) * (4 / 3)
    return {"member_mse": member.mean(0), "ensemble_mean_mse": mean_mse.mean(0),
            "signed_mean_risk": signed.mean(0), "unbiased_member_variance": variance.mean(0),
            "per_condition_member_mse": member}


def _vector(values, name):
    result = torch.as_tensor(values, dtype=torch.float64).detach().cpu()
    if result.ndim != 1 or not result.numel() or not torch.isfinite(result).all():
        raise ValueError(f"finite ordered {name} vector required")
    return result


def field_sensor_eligibility(source, child, *, model_family, absolute_mean_allowance,
                             source_sensor_mse, child_sensor_mse, sensor_equality_tolerance):
    """Signed mean differences use absolute TRAIN allowances; never a risk ratio."""
    source_member = _vector(source["member_mse"], "SOURCE member risk")
    child_member = _vector(child["member_mse"], "child member risk")
    sensor_source = _vector(source_sensor_mse, "SOURCE sensor MSE")
    sensor_child = _vector(child_sensor_mse, "child sensor MSE")
    tolerance = _vector(sensor_equality_tolerance, "sensor equality tolerance")
    if (source_member.shape != child_member.shape or sensor_source.shape != sensor_child.shape
        or tolerance.shape != sensor_source.shape or (tolerance < 0).any()
        or (source_member < 0).any() or (sensor_source < 0).any()):
        raise ValueError("field order and nonnegative risk/tolerance definitions must match")
    # Equality tolerance is only for a true zero SOURCE risk. It must already
    # be declared, with its units, in the frozen numerical contract.
    sensor_allowance = torch.where(sensor_source == 0, tolerance, .1 * sensor_source)
    sensor = bool(torch.all(sensor_child - sensor_source <= sensor_allowance))
    legacy = bool(torch.all(child_member <= 1.1 * source_member))
    if model_family == "senseiver":
        primary = legacy
        difference = child_member - source_member
        allowance = .1 * source_member
    elif model_family == "ffm":
        allowance = _vector(absolute_mean_allowance, "absolute TRAIN mean allowance")
        difference = _vector(child["signed_mean_risk"], "child signed risk") - _vector(source["signed_mean_risk"], "SOURCE signed risk")
        if allowance.shape != difference.shape or (allowance < 0).any():
            raise ValueError("ordered nonnegative absolute mean allowances required")
        primary = bool(torch.all(difference <= allowance))
    else:
        raise ValueError("unknown model family")
    return {"primary_field_pass": primary, "sensor_pass": sensor,
            "legacy_member_pass": legacy, "field_difference": difference.tolist(),
            "field_allowance": allowance.tolist(), "sensor_allowance": sensor_allowance.tolist(),
            "primary_pass": primary and sensor}


def core_eligibility(source_scores, child_scores, normalizers):
    if set(source_scores) != set(CORE_KEYS) or set(child_scores) != set(CORE_KEYS) or set(normalizers) != set(CORE_KEYS):
        raise ValueError("complete fixed State/Scale/Shape/event observer required")
    normalized = {}
    for key in CORE_KEYS:
        values = [float(source_scores[key]), float(child_scores[key]), float(normalizers[key])]
        if any(not math.isfinite(v) for v in values) or values[2] <= 0:
            raise ValueError("finite scores and positive TRAIN-frozen normalizers required")
        normalized[key] = (values[1] - values[0]) / values[2]
    passes = (float(child_scores["scale"]) <= 1.01 * float(source_scores["scale"])
              and normalized["state"] <= .02 and normalized["shape"] <= .02
              and normalized["event"] <= .05)
    return {"pass": passes, "normalized_differences": normalized}


def nominate_train_amplitude(source, child, *, sample_ids, population, source_scores,
                              normalizers, evaluate, latency, source_identity, child_identity):
    """Freeze the one eligible minimum-utility nominee using actual blended fields.

    evaluate(fields) returns `scores` from the common fixed observer and
    `fidelity` from field_sensor_eligibility. latency(alpha) times the actual
    SOURCE/child/residual function. No reserved-task data enters this interface.
    """
    ids = list(sample_ids)
    if len(ids) != max(1, math.ceil(.1 * population)) or len(set(ids)) != len(ids):
        raise ValueError("matched approximately 10% TRAIN condition manifest required")
    if source.shape != child.shape or source.shape[1] != len(ids) or not source_identity or not child_identity:
        raise ValueError("matched SOURCE/child endpoints and literal identities required")
    rows = []
    for alpha in AMPLITUDES:
        measured = evaluate(source + alpha * (child - source))
        if set(measured) != {"scores", "fidelity"}:
            raise ValueError("nomination accepts core/event scores and primary fidelity only")
        core = core_eligibility(source_scores, measured["scores"], normalizers)
        seconds = float(latency(alpha))
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("positive measured actual deployment latency required")
        utility = sum(float(measured["scores"][key]) / normalizers[key] for key in CORE_KEYS)
        rows.append({"alpha": alpha, "utility": utility, "latency_seconds": seconds,
                     "scores": measured["scores"], "fidelity": measured["fidelity"],
                     "core": core, "eligible": core["pass"] and measured["fidelity"]["primary_pass"]})
    feasible = [r for r in rows if r["eligible"]]
    if feasible:
        best = min(r["utility"] for r in feasible)
        tied = [r for r in feasible if r["utility"] <= best + 1e-6]
        chosen = min(tied, key=lambda r: (r["latency_seconds"], r["alpha"]))
    else:
        chosen = rows[0]
    result = {"schema": "car_round2_train_nomination_v1", "split": "TRAIN",
              "sample_ids": ids, "population": population, "source_identity": source_identity,
              "child_identity": child_identity, "rows": rows, "selected_alpha": chosen["alpha"],
              "eligible": bool(feasible), "reserved_task_used": False,
              "normalizers": dict(normalizers), "source_scores": dict(source_scores)}
    result["sha256"] = hashlib.sha256(json.dumps(result, sort_keys=True, allow_nan=False).encode()).hexdigest()
    return result


def freeze_method_priority(nominees):
    """Sort TRAIN-eligible functions by latency, using 1% ties and development score."""
    candidates = [dict(row) for row in nominees if row["eligible"] and row["alpha"] > 0]
    ordered = []
    while candidates:
        fastest = min(row["latency_seconds"] for row in candidates)
        tied = [row for row in candidates if row["latency_seconds"] <= fastest * 1.01]
        chosen = min(tied, key=lambda row: (row["utility"], row["alpha"], row["method"]))
        ordered.append(chosen["method"])
        candidates.remove(chosen)
    return ordered


def deployment_disposition(priority, assessments, *, model_family, complete_costs, within_budget):
    """Apply the frozen method order without changing held-out amplitudes."""
    decisions = {}
    for method in priority:
        row = assessments[method]
        scientific = (row["fidelity"]["primary_pass"] and row["core"]["pass"]
                      and row["reserved_normalized_improvement"] >= .05)
        route = bool(scientific and row["source_latency_ratio"] <= 1.2 and within_budget) if complete_costs else None
        operational = (bool(route and (model_family == "senseiver" or row["fidelity"]["legacy_member_pass"]))
                       if route is not None else None)
        decisions[method] = {"task_useful": bool(scientific), "source_deployment_route": route,
                             "operational_eligible": operational}
    opt_in = next((name for name in priority if decisions[name]["source_deployment_route"] is True), None)
    accepted = next((name for name in priority if decisions[name]["operational_eligible"] is True), None)
    return {"schema": "car_round2_deployment_disposition_v1", "decisions": decisions,
            "opt_in_function": opt_in, "operational_function": accepted,
            "operational_recommendation": accepted or "retain SOURCE/C8",
            "cost_qualification_status": "complete" if complete_costs else "unavailable_incomplete_costs",
            "method_priority": list(priority), "held_out_amplitude_search": False}

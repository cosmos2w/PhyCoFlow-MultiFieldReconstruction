"""Frozen native relation metrics and the Round 4 prospective decision rule.

Input Gram tensors come from the same native operator and frozen calibration:
reference [condition, band, field, field], draws [member, condition, band,
field, field], scale/floor [band, field]. No inference is performed here.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch

from ..coherence.families.multiscale_spectrum.losses import band_coherence

QUERIES = ("self", "mutual", "rho")
MARGINS = {"signed": 0.012277946972195601, "state": 0.0002823454964527628,
           "shape": 0.08219151385128498}


def relation_coordinates(gram: torch.Tensor, scales: torch.Tensor,
                         power_floor: torch.Tensor) -> dict[str, torch.Tensor]:
    """Return dimensionless [condition, band, field/pair] coordinates."""
    if gram.ndim != 4 or gram.shape[-1] != gram.shape[-2]:
        raise ValueError("gram must have shape [condition, band, field, field]")
    expected = gram.shape[1:3]
    if scales.shape != expected or power_floor.shape != expected:
        raise ValueError("scale and floor must have shape [band, field]")
    if not all(torch.isfinite(v).all() for v in (gram, scales, power_floor)):
        raise ValueError("relation inputs must be finite")
    if (scales <= 0).any() or (power_floor <= 0).any():
        raise ValueError("scale and power floor must be positive")
    scales, power_floor = scales.to(gram), power_floor.to(gram)
    left, right = torch.triu_indices(gram.shape[-1], gram.shape[-1], 1, device=gram.device)
    return {
        "self": torch.diagonal(gram, dim1=-2, dim2=-1) / scales.square(),
        "mutual": gram[..., left, right] / (scales[:, left] * scales[:, right]),
        "rho": band_coherence(gram, power_floor),
    }


def train_floors(train_gram: torch.Tensor, scales: torch.Tensor,
                 power_floor: torch.Tensor) -> dict[str, float | None]:
    """Compute TRAIN-only decision floors; empty mutual coordinates are unavailable."""
    coords = relation_coordinates(train_gram, scales, power_floor)
    return {k: max(1e-12, 1e-3 * float(v.square().mean())) if v.numel() else None
            for k, v in coords.items()}


def relation_errors(draw_gram: torch.Tensor, target_gram: torch.Tensor,
                    scales: torch.Tensor, power_floor: torch.Tensor) -> dict[str, dict]:
    """Return squared errors for every member, condition, band, and field/pair."""
    if draw_gram.ndim != 5 or draw_gram.shape[0] < 1 or draw_gram.shape[1:] != target_gram.shape:
        raise ValueError("draw Gram shape must be [member, *target shape] with members")
    target = relation_coordinates(target_gram, scales, power_floor)
    draws = [relation_coordinates(g, scales, power_floor) for g in draw_gram]
    result = {}
    for name in QUERIES:
        difference = torch.stack([d[name] - target[name] for d in draws])
        squared = difference.square()
        result[name] = {"squared": squared, "mean": float(squared.mean()) if squared.numel() else None}
    return result


def bridge_comparison(source: Mapping, child: Mapping, native_source: Mapping,
                      native_child: Mapping, floors: Mapping[str, float | None]) -> dict:
    """Compare paired changes for one cohort; keep absolute per-entry arrays visible."""
    result = {}
    max_error = 0.0
    for h in QUERIES:
        arrays = [x[h]["squared"] for x in (source, child, native_source, native_child)]
        if arrays[0].numel() == 0:
            result[h] = None
            continue
        if any(x.shape != arrays[0].shape or not torch.isfinite(x).all() for x in arrays):
            raise ValueError("bridge arrays must be finite and identically shaped")
        means = [float(a.mean()) for a in arrays]
        denominator = max(means[2], float(floors[h]))
        delta, native_delta = means[1] - means[0], means[3] - means[2]
        error = abs(delta - native_delta) / denominator
        max_error = max(max_error, error)
        result[h] = {"source_squared": arrays[0], "child_squared": arrays[1],
                     "native_source_squared": arrays[2], "native_child_squared": arrays[3],
                     "source": means[0], "child": means[1], "native_source": means[2],
                     "native_child": means[3], "delta": delta, "native_delta": native_delta,
                     "denominator": denominator, "paired_change_error": error}
    return {"queries": result, "max_paired_change_error": max_error}


def assess_candidate(candidate: Mapping, source: Mapping,
                     floors: Mapping[str, float | None]) -> dict:
    """Apply the fixed physical, field and signed/State/Shape guards to one panel.

    Mappings contain self, mutual, rho, field_mse (one per field), signed,
    state and shape. Candidate also contains epoch and checkpoint identity.
    """
    values = [candidate[k] for k in QUERIES] + [source[k] for k in QUERIES]
    if any(v is None for v in values):
        raise ValueError("mutual adoption requires at least two fields")
    src_field = torch.as_tensor(source["field_mse"], dtype=torch.float64)
    child_field = torch.as_tensor(candidate["field_mse"], dtype=torch.float64)
    if (src_field.ndim != 1 or src_field.numel() == 0 or src_field.shape != child_field.shape
            or not torch.isfinite(src_field).all() or not torch.isfinite(child_field).all()
            or (src_field <= 0).any() or (child_field < 0).any()):
        raise ValueError("field MSE must be finite, matched, and SOURCE strictly positive")
    ratios = child_field / src_field
    guards = {"field": bool((ratios <= 1.10).all())}
    details = {}
    for h in QUERIES:
        base, value, floor = float(source[h]), float(candidate[h]), float(floors[h])
        if not all(torch.isfinite(torch.tensor(x)) for x in (base, value, floor)) or min(base, value) < 0 or floor <= 0:
            raise ValueError("physical scores and floors must be finite and nonnegative")
        no_headroom = base <= floor
        delta = value - base
        # Self is a preservation criterion, including when SOURCE is at its floor.
        limit = 0.02 * floor if no_headroom else (0.02 if h == "self" else -0.02) * base
        guards[h] = delta <= limit
        details[h] = {"delta": delta, "denominator": max(base, floor),
                      "normalized_delta": delta / max(base, floor),
                      "no_headroom": no_headroom, "limit": limit}
    guards["mutual_headroom"] = not (details["mutual"]["no_headroom"] and details["rho"]["no_headroom"])
    for key, margin in MARGINS.items():
        a, b = float(candidate[key]), float(source[key])
        if not torch.isfinite(torch.tensor([a, b])).all():
            raise ValueError("law/State/Shape scores must be finite")
        guards[key] = a <= b + margin
    return {"eligible": all(guards.values()), "guards": guards, "physical": details,
            "worst_field_ratio": float(ratios.max())}


def select_checkpoint(candidates: Sequence[Mapping], source: Mapping,
                      floors: Mapping[str, float | None]) -> dict:
    """Freeze the mature development selection, with no audit inputs."""
    assessed = []
    identities = set()
    for candidate in candidates:
        epoch = candidate["epoch"]
        identity = candidate["checkpoint"]
        if not isinstance(epoch, int) or epoch < 0 or not isinstance(identity, str) or not identity:
            raise ValueError("checkpoint identity and nonnegative integer epoch required")
        if identity in identities:
            raise ValueError("duplicate checkpoint identity")
        identities.add(identity)
        if epoch < 200 or epoch > 500:
            continue
        assessment = assess_candidate(candidate, source, floors)
        assessed.append((candidate, assessment))
    eligible = [(c, a) for c, a in assessed if a["eligible"]]
    selected = min(eligible, key=lambda ca: (ca[0]["mutual"], ca[0]["rho"],
                                            ca[1]["worst_field_ratio"], ca[0]["epoch"])) if eligible else None
    return {"version": "round4_selection_v1", "checkpoint": selected[0]["checkpoint"] if selected else None,
            "epoch": selected[0]["epoch"] if selected else None,
            "assessments": {c["checkpoint"]: a for c, a in assessed}}


def confirm_checkpoint(selection: Mapping, audit_candidate: Mapping, audit_source: Mapping,
                       floors: Mapping[str, float | None], final_field_mse: Sequence[float],
                       source_field_mse: Sequence[float]) -> dict:
    """Confirm the already-frozen identity once; failure returns SOURCE, never reranks."""
    if not selection.get("checkpoint") or audit_candidate.get("checkpoint") != selection["checkpoint"]:
        raise ValueError("audit candidate must match the frozen selected checkpoint")
    if audit_candidate.get("epoch") != selection.get("epoch"):
        raise ValueError("audit epoch must match the frozen selection")
    audit = assess_candidate(audit_candidate, audit_source, floors)
    final_source = torch.as_tensor(source_field_mse, dtype=torch.float64)
    final_child = torch.as_tensor(final_field_mse, dtype=torch.float64)
    if (final_source.ndim != 1 or final_source.numel() == 0 or final_source.shape != final_child.shape
            or not torch.isfinite(final_source).all() or not torch.isfinite(final_child).all()
            or (final_source <= 0).any() or (final_child < 0).any()):
        raise ValueError("final field MSE must be finite and matched with positive SOURCE")
    field_ok = bool((final_child <= 1.10 * final_source).all())
    return {"checkpoint": selection["checkpoint"], "audit": audit,
            "final_field_pass": field_ok,
            "deployment": selection["checkpoint"] if audit["eligible"] and field_ok else "SOURCE"}

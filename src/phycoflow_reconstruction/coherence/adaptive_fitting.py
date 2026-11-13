"""TRAIN-only measured discovery and a frozen expanded-query observer.

Panel factories must return fresh iterators of dictionaries containing condition
IDs, SOURCE members [J,B,N,C], and truths [B,N,C]. Filtering and C8 anchor
callbacks operate on the actual endpoint tensors and must be differentiable.
"""

from __future__ import annotations

import hashlib
import json
import math
import time

import torch

from .adaptive_discovery import candidates
from .scale_compute import third_order_scores
from .state_compute import group_projection_scores

VERSION = "adaptive_measured_train_bank_v1"


def _digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _sync(x):
    if x.device.type == "cuda":
        torch.cuda.synchronize(x.device)


def validate_bank(bank, *, field_order, support_identity, source_identity):
    payload = dict(bank)
    digest = payload.pop("sha256", None)
    if (
        payload.get("version") != VERSION
        or payload.get("split") != "TRAIN"
        or payload.get("field_order") != list(field_order)
        or payload.get("support_identity") != support_identity
        or payload.get("source_identity") != source_identity
        or digest != _digest(payload)
    ):
        raise ValueError("stale or incompatible adaptive bank")


def _geometry(row, fields, scales):
    index = {name: i for i, name in enumerate(fields)}
    if row["kind"] == "state.group_projection":
        direction = [0.0] * len(fields)
        for name, sign in zip(row["fields"], row["signs"]):
            direction[index[name]] = sign / math.sqrt(len(row["fields"])) / scales[index[name]]
        return direction
    return [row["region"], [[band, index[field]] for band, field in row["terms"]]]


def score_bank(bank, members, reference, masses, filter_fn):
    """Complete per-leaf fair scores from a frozen artifact; no target sampling.

    Caller must first validate identity with validate_bank. Returned dict contains
    [B,P] and [B,T] values; fixed additive target weights are recorded in bank.
    """
    directions = torch.as_tensor(bank["directions"], device=members.device, dtype=members.dtype)
    if not bank["directions"]:
        directions = members.new_empty((0, members.shape[-1]))
    state = group_projection_scores(members, reference, masses, masses, directions)
    if bank["triples"]:
        filtered_members = filter_fn(members)
        filtered_reference = filter_fn(reference)
        scale = third_order_scores(
            filtered_members,
            filtered_reference,
            bank["triples"],
            bank["region_masses"],
            bank["triple_scales"],
        )
    else:
        scale = members.new_empty((members.shape[1], 0))
    return {"state.group_projection": state, "scale.third_order": scale}


def fit_train_bank(
    panel_factory,
    *,
    field_order,
    masses,
    region_masses,
    filter_fn,
    anchor_fn,
    source_identity,
    support_identity,
    units,
    split="TRAIN",
    seed=0,
    floor=0.01,
    sketch_size=128,
    redundancy_limit=0.95,
):
    """Measure and freeze a bounded bank on a repeatable TRAIN/SOURCE panel.

    anchor_fn(members, reference, masses) returns the scalar C8 core on the same
    endpoints. Complements use its endpoint gradient, not the descriptor scalar
    derivative. Candidate costs include scoring, filtering and endpoint gradient,
    with CUDA synchronization at candidate boundaries. Each selected-set
    redundancy comparison is bounded by the retained maximum of 12 leaves.
    """
    if (
        split != "TRAIN"
        or not field_order
        or not source_identity
        or not support_identity
        or not units
        or floor <= 0
        or sketch_size < 1
        or not 0 <= redundancy_limit <= 1
    ):
        raise ValueError("TRAIN identities, positive floors and valid sketch settings required")
    m = torch.as_tensor(masses, dtype=torch.float64).detach().cpu()
    regions = torch.as_tensor(region_masses, dtype=torch.float64).detach().cpu()
    if (
        m.ndim != 1
        or regions.ndim != 2
        or regions.shape[1] != len(m)
        or not torch.isfinite(m).all()
        or (m < 0).any()
        or m.sum() <= 0
        or not torch.isfinite(regions).all()
        or (regions < 0).any()
        or (regions.sum(-1) <= 0).any()
        or ((regions > 0) & (m[None, :] == 0)).any()
    ):
        raise ValueError("invalid physical support masses")
    m = m / m.sum()
    regions = regions / regions.sum(-1, keepdim=True)
    ids, count, field_sum, triple_sum = [], 0, None, None
    for batch in panel_factory():
        truth = batch["reference"].detach()
        batch_ids = list(batch["ids"])
        if (
            truth.ndim != 3
            or truth.shape[1:] != (len(m), len(field_order))
            or len(batch_ids) != len(truth)
        ):
            raise ValueError("invalid TRAIN reference batch")
        filtered = filter_fn(truth).detach()
        if (
            filtered.ndim != 4
            or filtered.shape[0] != len(truth)
            or filtered.shape[2:] != truth.shape[1:]
        ):
            raise ValueError("filter_fn must return [B,S,N,C]")
        mm = m.to(truth)
        rr = regions.to(filtered)
        fs = torch.einsum("bnc,n->c", truth.square(), mm).double().cpu()
        means = torch.einsum("bsnc,rn->brsc", filtered, rr)
        centered = filtered[:, None] - means.unsqueeze(-2)
        ts = torch.einsum("brsnc,rn->rsc", centered.square(), rr).double().cpu()
        field_sum = fs if field_sum is None else field_sum + fs
        triple_sum = ts if triple_sum is None else triple_sum + ts
        ids.extend(batch_ids)
        count += len(truth)
    if not count or len(set(ids)) != len(ids):
        raise ValueError("nonempty unique TRAIN condition IDs required")
    field_scale = (field_sum / count).sqrt().clamp_min(floor)
    triple_scales = (triple_sum / count).sqrt().clamp_min(floor)
    raw = candidates(field_order, bands=triple_scales.shape[1], regions=len(regions), seed=seed)
    rows = raw["candidates"]
    acc = {
        q["id"]: {
            "sum": 0.0,
            "square": 0.0,
            "count": 0,
            "sensitivity": 0.0,
            "cost": 0.0,
            "sketch": None,
        }
        for q in rows
    }
    seen = []
    for batch in panel_factory():
        x = batch["members"].detach().requires_grad_(True)
        y = batch["reference"].detach()
        seen.extend(batch["ids"])
        if x.ndim != 4 or x.shape[1:] != y.shape or x.shape[0] < 2:
            raise ValueError("independent J>=2 SOURCE members required")
        anchor = anchor_fn(x, y, m.to(x))
        ga = torch.autograd.grad(anchor.mean(), x)[0].detach().mean(dim=1).flatten()
        rng = torch.Generator(device="cpu").manual_seed(seed)
        take = torch.randperm(ga.numel(), generator=rng)[: min(sketch_size, ga.numel())].to(
            ga.device
        )
        ga = ga[take]
        for row in rows:
            _sync(x)
            started = time.perf_counter()
            geometry = _geometry(row, field_order, field_scale.tolist())
            if row["kind"] == "state.group_projection":
                leaf = group_projection_scores(x, y, m.to(x), m.to(x), [geometry])[:, 0]
            else:
                leaf = third_order_scores(
                    filter_fn(x), filter_fn(y), [geometry], regions, triple_scales
                )[:, 0]
            grad = torch.autograd.grad(leaf.mean(), x)[0].detach().mean(dim=1).flatten()[take]
            _sync(x)
            elapsed = time.perf_counter() - started
            projection = (grad @ ga) / ga.square().sum().clamp_min(torch.finfo(grad.dtype).tiny)
            residual = grad - projection * ga
            item = acc[row["id"]]
            v = leaf.detach().double()
            item["sum"] += float(v.abs().sum())
            item["square"] += float(v.square().sum())
            item["count"] += len(v)
            item["sensitivity"] += float(residual.norm()) * len(v)
            item["cost"] += elapsed
            sketch = residual.double().cpu() * len(v)
            item["sketch"] = sketch if item["sketch"] is None else item["sketch"] + sketch
    if seen != ids:
        raise ValueError("TRAIN panel changed between discovery passes")
    measures = {}
    for ident, a in acc.items():
        mean = a["sum"] / count
        rms = math.sqrt(a["square"] / count)
        repeatability = mean / rms if rms else 0.0
        sensitivity = a["sensitivity"] / count
        cost = a["cost"]
        measures[ident] = {
            "discrepancy": mean,
            "repeatability": repeatability,
            "sensitivity": sensitivity,
            "cost": cost,
            "priority": mean * repeatability * sensitivity / math.sqrt(cost) if cost > 0 else 0.0,
        }
    chosen, sketches, selection_audit = [], [], []
    for row in sorted(rows, key=lambda r: (-measures[r["id"]]["priority"], r["id"])):
        item = measures[row["id"]]
        limit = 4 if row["kind"] == "state.group_projection" else 8
        audit = {"id": row["id"], "candidate": row, "priority": item["priority"]}
        if item["priority"] <= 0:
            audit["decision"] = "nonpositive_priority"
        elif sum(r["kind"] == row["kind"] for r in chosen) >= limit:
            audit["decision"] = "bank_limit"
        else:
            vec = acc[row["id"]]["sketch"]
            norm = vec.norm()
            if norm == 0:
                audit["decision"] = "zero_gradient_sketch"
            else:
                matches = [
                    (abs(float(vec @ previous / (norm * previous.norm()))), chosen[i]["id"])
                    for i, previous in enumerate(sketches)
                ]
                best = max(matches, default=(0.0, None))
                audit["max_selected_cosine"], audit["nearest_selected_id"] = best
                if best[0] >= redundancy_limit:
                    audit["decision"] = "redundant"
                else:
                    audit["decision"] = "selected"
                    chosen.append(row)
                    sketches.append(vec)
        selection_audit.append(audit)
    directions = [
        _geometry(q, field_order, field_scale.tolist())
        for q in chosen
        if q["kind"] == "state.group_projection"
    ]
    triples = [
        _geometry(q, field_order, field_scale.tolist())
        for q in chosen
        if q["kind"] == "scale.third_order"
    ]
    bank = {
        "version": VERSION,
        "policy_qualified": False,
        "split": split,
        "field_order": list(field_order),
        "population_ids": ids,
        "source_identity": source_identity,
        "support_identity": support_identity,
        "units": units,
        "seed": seed,
        "floor": floor,
        "field_scales": field_scale.tolist(),
        "physical_masses": m.tolist(),
        "region_masses": regions.tolist(),
        "triple_scales": triple_scales.tolist(),
        "directions": directions,
        "triples": triples,
        "selected": chosen,
        "selection_audit": selection_audit,
        "candidate_generation": {
            "version": raw["version"],
            "candidate_count": len(rows),
            "deduplication": "stable IDs and sign-equivalent group projections",
        },
        "measurements": measures,
        "redundancy_limit": redundancy_limit,
        "target_weights": {
            "state.group_projection": [1 / len(directions)] * len(directions) if directions else [],
            "scale.third_order": [1 / len(triples)] * len(triples) if triples else [],
        },
    }
    bank["block_units"] = {}
    for kind in ("state.group_projection", "scale.third_order"):
        values = [measures[q["id"]]["discrepancy"] for q in chosen if q["kind"] == kind]
        bank["block_units"][kind] = max(floor, sum(values) / len(values)) if values else 1.0
    bank["sha256"] = _digest(bank)
    return bank


def spectral_filter(family, endpoints):
    """Apply the fitted C8 spectral filters on its declared analysis support."""
    from .families.multiscale_spectrum.statistics import (
        band_fields,
        piecewise_fluctuations,
        spectral_coefficients,
    )

    if endpoints.ndim < 3:
        raise ValueError("endpoint requires batch, node and field axes")
    leading = endpoints.shape[:-2]
    flat = endpoints.reshape(-1, *endpoints.shape[-2:])
    selected = family._select(flat)
    if (
        selected.shape[1:] != flat.shape[1:]
        or list(family.field_ids) != list(range(flat.shape[-1]))
        or not torch.equal(family.analysis_index.detach().cpu(), torch.arange(flat.shape[1]))
    ):
        raise ValueError("adaptive filters require the complete declared support and fields")
    g = family._geometry_tensors(family.compute_dtype)
    g = {k: v.to(selected.device) for k, v in g.items()}
    fluct = piecewise_fluctuations(selected, g["piece_measures"], g["pieces"])
    coeff = spectral_coefficients(fluct, g["basis"] * g["mass"][:, None])
    result = band_fields(coeff, g["basis"], g["filters"])
    return result.reshape(*leading, *result.shape[1:])

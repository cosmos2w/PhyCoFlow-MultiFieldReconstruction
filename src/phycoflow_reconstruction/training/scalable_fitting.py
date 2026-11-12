"""Bounded, TRAIN-only fitting of reusable scalable coherence artifacts.

Callers select approximately 10% of TRAIN in a reproducible order and provide
matched SOURCE outputs. The default limit is 32 samples; override explicitly
when a different bounded fit budget is justified. No training RNG is consumed.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping

import torch

from ..coherence.base import projection_bank
from ..coherence.families.topology.betti_curves import gaussian_blur
from ..coherence.families.topology.geometry import rasterize_fields
from ..coherence.query_runtime import ScalableQueryAdapter, fit_pooled_shape
from ..coherence.query_sampling import SharedQuerySampler
from ..coherence.shape_compute import LazyPooledShape
from ..coherence.state_compute import fit_source_hardness, score_family, select_directions
from .internal_calibration import BLOCKS

VERSION = "scalable_fitting_v1"
IDENTITY_KEYS = frozenset(
    (
        "split",
        "field_order",
        "sample_ids",
        "source_checkpoint_sha256",
        "normalizer_sha256",
        "support_sha256",
        "config_sha256",
        "definition_version",
    )
)


def _identity(value):
    if not isinstance(value, Mapping) or set(value) != IDENTITY_KEYS:
        raise ValueError("provenance must declare exactly the required identity keys")
    try:
        result = json.loads(json.dumps(dict(value), sort_keys=True, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError("provenance must be JSON serializable") from exc
    ids, fields = result["sample_ids"], result["field_order"]
    if (
        result["split"] != "train"
        or not isinstance(ids, list)
        or not ids
        or any(not isinstance(x, str) or not x for x in ids)
        or len(set(ids)) != len(ids)
    ):
        raise ValueError("unique nonempty TRAIN sample IDs required")
    if (
        not isinstance(fields, list)
        or not fields
        or any(not isinstance(x, str) or not x for x in fields)
        or len(set(fields)) != len(fields)
    ):
        raise ValueError("unique nonempty field order required")
    if not isinstance(result["definition_version"], str) or not result["definition_version"]:
        raise ValueError("definition version required")
    for key in ("source_checkpoint_sha256", "normalizer_sha256", "support_sha256", "config_sha256"):
        digest = result[key]
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)
        ):
            raise ValueError(f"{key} must be a lowercase SHA-256")
    return result


def _digest(value):
    """Stable content hash across devices; covers all artifact tensors and metadata."""
    h = hashlib.sha256()

    def add(v):
        if isinstance(v, torch.Tensor):
            a = v.detach().cpu().contiguous()
            h.update(
                b"T" + str(a.dtype).encode() + repr(tuple(a.shape)).encode() + a.numpy().tobytes()
            )
        elif isinstance(v, Mapping):
            h.update(b"{")
            for k in sorted(v):
                add(k)
                add(v[k])
            h.update(b"}")
        elif isinstance(v, (list, tuple)):
            h.update(b"[")
            for x in v:
                add(x)
            h.update(b"]")
        else:
            h.update(json.dumps(v, sort_keys=True, allow_nan=False).encode() + b"\0")

    add(value)
    return h.hexdigest()


def validate_scalable_artifacts(artifacts, expected_provenance):
    expected = _identity(expected_provenance)
    if not isinstance(artifacts, Mapping) or set(artifacts) != {
        "version",
        "state_bank",
        "shape_artifact",
        "priorities",
        "units",
        "active",
        "provenance",
        "sha256",
    }:
        raise ValueError("invalid scalable artifact envelope")
    if artifacts["version"] != VERSION or _identity(artifacts["provenance"]) != expected:
        raise ValueError("scalable artifact provenance mismatch")
    if _digest({k: v for k, v in artifacts.items() if k != "sha256"}) != artifacts["sha256"]:
        raise ValueError("scalable artifact contents changed")
    if tuple(artifacts["state_bank"]["field_order"]) != tuple(expected["field_order"]):
        raise ValueError("State bank field order mismatch")
    shape = LazyPooledShape.from_state_dict(artifacts["shape_artifact"])
    if shape.provenance != expected or shape.mean.numel() != len(expected["field_order"]):
        raise ValueError("Shape artifact provenance mismatch")
    return artifacts


def fit_scalable_artifacts(
    *,
    families,
    source,
    reference,
    coordinates,
    provenance,
    source_weights=None,
    reference_weights=None,
    seed=0,
    priorities=None,
    units=None,
    shape_batches=8,
    max_samples=32,
    candidate_count=None,
    candidate_matrix=None,
    field_floor=1e-6,
):
    """Fit C8 from matched SOURCE [B,J,N,C] and TRAIN [B,N,C] tensors.

    By default the configured State cross component supplies its exact candidate
    matrix and seed. Explicit candidate overrides are included in the artifact.
    ``support_sha256`` binds query geometry, physical measures, masks and their
    sampling policy. ``config_sha256`` binds fitting settings and family configs.
    The caller must derive both from the actual selected data/configuration.
    """
    identity = _identity(provenance)
    if (
        source.ndim != 4
        or reference.ndim != 3
        or source.shape[0] != reference.shape[0]
        or source.shape[2:] != reference.shape[1:]
    ):
        raise ValueError("matched SOURCE and TRAIN tensors required")
    b, j, n, c = source.shape
    if (
        b < 1
        or j < 1
        or b > max_samples
        or len(identity["sample_ids"]) != b
        or len(identity["field_order"]) != c
    ):
        raise ValueError("TRAIN fit sample, draw, field count or budget mismatch")
    if (
        coordinates.ndim != 3
        or coordinates.shape[:2] != (b, n)
        or not torch.equal(coordinates, coordinates[:1].expand_as(coordinates))
    ):
        raise ValueError("fixed shared query support required")
    if (
        not torch.isfinite(source).all()
        or not torch.isfinite(reference).all()
        or not torch.isfinite(coordinates).all()
    ):
        raise ValueError("finite fit tensors required")
    if (
        shape_batches < 1
        or (
            candidate_count is not None
            and (
                isinstance(candidate_count, bool)
                or not isinstance(candidate_count, int)
                or candidate_count < 1
            )
        )
        or not math.isfinite(field_floor)
        or field_floor <= 0
    ):
        raise ValueError("invalid fitting settings")

    def masses(value):
        value = (
            torch.ones(b, n, dtype=reference.dtype, device=reference.device)
            if value is None
            else torch.as_tensor(value, device=reference.device, dtype=reference.dtype)
        )
        if (
            value.shape != (b, n)
            or not torch.isfinite(value).all()
            or (value < 0).any()
            or (value.sum(-1) <= 0).any()
        ):
            raise ValueError("fit measures must be finite nonnegative [B,N] with positive mass")
        return value

    sw, rw = masses(source_weights), masses(reference_weights)
    fields = identity["field_order"]
    keys = {f"{family}.{block}" for family, blocks in BLOCKS.items() for block in blocks}
    priorities = {k: 1.0 for k in keys} if priorities is None else dict(priorities)
    units = None if units is None else dict(units)
    if (
        set(priorities) != keys
        or (units is not None and set(units) != keys)
        or any(
            not math.isfinite(float(v)) or float(v) <= 0
            for d in (priorities, units or {})
            for v in d.values()
        )
    ):
        raise ValueError("positive finite priority and task unit for every semantic block required")
    active = {k: True for k in keys}
    if c == 1:
        for key in ("state.mixed", "shape.mutual", "scale.rho"):
            active[key] = False
    state_family = families["global_distribution"]
    if tuple(state_family.field_names) != tuple(fields):
        raise ValueError("State family field order mismatch")
    converted = [state_family._in_declared_units(source[:, i], reference)[0] for i in range(j)]
    target_state = state_family._in_declared_units(source[:, 0], reference)[1]
    state_source = torch.stack(converted, dim=1)
    cross = None
    if "cross_joint_topk_swd" in state_family.components_by_key:
        # torch.nn.ModuleDict supports indexing and membership, but not .get().
        cross = state_family.components_by_key["cross_joint_topk_swd"]

    cross_settings = state_family.config.get("components", {}).get("cross", {})
    direction_seed = int(cross_settings.get("seed", 1234))
    if candidate_matrix is not None and candidate_count is not None:
        raise ValueError("provide a candidate matrix or a candidate count override, not both")
    if c == 1:
        if candidate_matrix is not None:
            raise ValueError("one field has no mixed State directions")
        candidates = torch.empty((0, 1), dtype=torch.float64)
        hardness = torch.empty(0, dtype=torch.float64)
    else:
        if cross is None:
            raise ValueError("State family must provide its configured mixed direction bank")
        configured = cross.directions.detach().cpu()
        if candidate_matrix is not None:
            candidates = torch.as_tensor(candidate_matrix).detach().cpu()
        elif candidate_count is None or candidate_count == len(configured):
            candidates = configured.clone()
        else:
            candidates = projection_bank(c, candidate_count, seed=direction_seed, qmc=True)
        if (
            candidates.ndim != 2
            or candidates.shape[1] != c
            or len(candidates) == 0
            or not torch.isfinite(candidates).all()
            or (candidates.norm(dim=1) <= 0).any()
        ):
            raise ValueError("invalid mixed State candidate matrix")
        if not torch.allclose(
            candidates.norm(dim=1),
            torch.ones(len(candidates), dtype=candidates.dtype),
            rtol=1e-5,
            atol=1e-6,
        ):
            raise ValueError("mixed State candidates must be unit directions")
        hardness = fit_source_hardness(
            [(state_source.detach(), target_state.detach(), sw, rw)],
            candidates,
            preserve_input=True,
        )
    bank = select_directions(candidates, hardness, fields, direction_seed, preserve_input=True)
    family = families["topology"]
    if tuple(family.field_names) != tuple(fields):
        raise ValueError("Shape family field order mismatch")
    family._raster_map(coordinates)
    with torch.no_grad():
        _, target = family._units_and_fields(reference, reference)
        grid = gaussian_blur(
            rasterize_fields(
                target,
                family.neighbor_indices,
                family.neighbor_weights.to(target),
                family.grid_shape,
            ),
            family.smoothing_sigma,
            family.periodic,
            radius=family.smoothing_radius,
        ).double()
        if family.native_masked:
            mask = family.domain_mask.to(grid.device)
            values = grid[..., mask]
        else:
            values = grid.flatten(-2)
        mean = values.mean((0, 2))
        scale = values.var((0, 2), unbiased=False).sqrt().clamp_min(field_floor)
    sampler = SharedQuerySampler(fields, seed=seed, epsilon=1.0)
    batches = []
    for _ in range(shape_batches):
        batches.append(sampler.draw())
        sampler.finish()
    shape = fit_pooled_shape(
        family,
        source[:, 0].detach(),
        reference.detach(),
        coordinates,
        batches,
        field_mean=mean,
        field_scale=scale,
        provenance=identity,
        dimensions=getattr(family, "dimensions", (0, 1)),
        projections=getattr(getattr(family, "spatial_objective", None), "projections", 32),
    )
    if units is None:
        # Estimate semantic task units from paired SOURCE/reference blocks on the
        # same bounded fixed panel, with no extra model calls or random draws.
        component = state_family.components_by_key["self_marginal_w2"]
        with torch.no_grad():
            _, state_blocks = score_family(
                state_source[:, 0].unsqueeze(0),
                target_state,
                sw,
                rw,
                bank,
                channel_weights=component.channel_weights,
                marginal_weight=state_family.component_weights["self_marginal_w2"],
                mixed_weight=state_family.component_weights.get("cross_joint_topk_swd", 0.0),
                mode="paired",
            )
            adapter = ScalableQueryAdapter(families, shape)
            collected = {key: [] for key in keys}
            for key, value in state_blocks.items():
                if value is not None:
                    collected[key].append(float(value.abs()))
            for batch in batches:
                for family_name in ("multiscale_spectrum", "topology"):
                    _, blocks, _ = adapter.score(
                        family_name,
                        source[:, 0],
                        reference,
                        coordinates=coordinates,
                        context={"scoring_mode": "deterministic", "field_order": fields},
                        batch=batch,
                        coefficients={},
                    )
                    for key, value in blocks.items():
                        if value is not None:
                            collected[key].append(float(value.abs()))
        units = {
            key: max(field_floor, sum(values) / len(values)) if values else 1.0
            for key, values in collected.items()
        }
        if any(active[key] and not collected[key] for key in keys):
            raise ValueError("SOURCE semantic task unit missing for an active block")
        if any(not math.isfinite(value) or value <= 0 for value in units.values()):
            raise ValueError("SOURCE semantic task units must be finite and positive")
    result = {
        "version": VERSION,
        "state_bank": bank,
        "shape_artifact": shape,
        "priorities": priorities,
        "units": units,
        "active": active,
        "provenance": identity,
    }
    result["sha256"] = _digest(result)
    return result

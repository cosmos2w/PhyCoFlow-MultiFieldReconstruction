"""Fixed-transform conditional fair persistence, distinct from historical v4."""

from __future__ import annotations

import hashlib
from copy import deepcopy
from dataclasses import replace

import torch

from .finite_primary import FinitePrimaryTopologyObjective
from .persistence import cubical_diagrams, sliced_diagram_distances


class ConditionalFairTopologyObjective(FinitePrimaryTopologyObjective):
    definition = "conditional_fixed_transform_fair_v1"

    def __init__(self, config, field_names):
        super().__init__(config, field_names)
        self.spec = replace(self.spec, version="5")
        count = len(self.ids) + len(self.descriptors)
        self.register_buffer("descriptor_mean", torch.zeros(count, dtype=torch.float64))
        self.register_buffer("descriptor_scale", torch.ones(count, dtype=torch.float64))
        self.register_buffer("transform_frozen", torch.tensor(False))
        self.transform_record = None
        self._moments = None
        self._calibration_accumulator = None

    @torch.no_grad()
    def accumulate_transform(self, reference, *, valid_mask=None):
        if self.transform_frozen:
            raise ValueError("descriptor transform already frozen")
        y = self._descriptor_fields(reference.detach()).double()
        values = y.flatten(2) if valid_mask is None else y[..., valid_mask]
        n = values.shape[0] * values.shape[2]
        mask_id = None if valid_mask is None else valid_mask.detach().cpu().numpy().tobytes()
        sums, squares = values.sum((0, 2)), values.square().sum((0, 2))
        if not torch.isfinite(values).all() or not n:
            raise ValueError("invalid TRAIN descriptor support")
        if self._moments is None:
            self._moments = (n, sums.cpu(), squares.cpu(), values.shape[0], mask_id)
        else:
            old, s, q, visits, old_mask = self._moments
            if old_mask != mask_id:
                raise ValueError("transform raster mask changed")
            self._moments = (
                old + n,
                s + sums.cpu(),
                q + squares.cpu(),
                visits + values.shape[0],
                old_mask,
            )

    @torch.no_grad()
    def freeze_transform(self, *, floors, provenance):
        if self.transform_frozen or self._moments is None:
            raise ValueError("transform already frozen or missing TRAIN moments")
        if (
            not isinstance(provenance, dict)
            or provenance.get("split") != "train"
            or not provenance.get("sample_ids")
        ):
            raise ValueError("transform requires TRAIN provenance and sample IDs")
        n, sums, squares, visits, mask_bytes = self._moments
        ids = provenance["sample_ids"]
        if len(ids) != visits or len(set(ids)) != visits:
            raise ValueError("transform sample IDs must uniquely identify all TRAIN visits")
        floor = torch.as_tensor(floors, dtype=torch.float64).flatten()
        if floor.numel() == 1:
            floor = floor.expand_as(sums)
        if floor.shape != sums.shape or not torch.isfinite(floor).all() or (floor <= 0).any():
            raise ValueError("positive finite per-descriptor floors required")
        mean = sums / n
        scale = torch.maximum((squares / n - mean.square()).clamp_min(0).sqrt(), floor)
        self.descriptor_mean.copy_(mean)
        self.descriptor_scale.copy_(scale)
        self.transform_frozen.fill_(True)
        self.transform_record = {
            "definition": self.definition,
            "provenance": deepcopy(provenance),
            "support_count": n,
            "visits": visits,
            "raster_measure": "uniform_equal_valid_pixel_area",
            "mask_sha256": None if mask_bytes is None else hashlib.sha256(mask_bytes).hexdigest(),
            "mean": mean.tolist(),
            "scale": scale.tolist(),
            "floors": floor.tolist(),
        }
        self._moments = None
        return deepcopy(self.transform_record)

    def _diagrams(self, x, indices, valid_mask, *, cacheable=False):
        if not self.transform_frozen:
            raise ValueError("TRAIN descriptor transform must be frozen")
        mask_hash = (
            None
            if valid_mask is None
            else hashlib.sha256(valid_mask.detach().cpu().numpy().tobytes()).hexdigest()
        )
        if not self.transform_record or self.transform_record["mask_sha256"] != mask_hash:
            raise ValueError("fixed-domain raster mask differs from transform")
        y = self._descriptor_fields(x)
        mean = self.descriptor_mean.to(y)[None, :, None, None]
        scale = self.descriptor_scale.to(y)[None, :, None, None]
        bank = self._filtrations((y - mean) / scale, indices)
        diagrams = cubical_diagrams(
            bank.flatten(0, 1),
            periodic=self.periodic,
            dimensions=self.dimensions,
            valid_mask=valid_mask,
            cacheable=cacheable,
        )
        return [item[d] for item in diagrams for d in self.dimensions], bank.shape

    def _distance(self, left, right, shape, normalization):
        if any(a.essential.numel() != b.essential.numel() for a, b in zip(left, right)):
            raise ValueError("fixed-domain essential class counts differ")
        finite = sliced_diagram_distances(
            left,
            right,
            projections=self.projections,
            essential_weight=0,
            normalization=normalization,
        )
        essential = torch.stack(
            [
                (a.essential.sort().values - b.essential.sort().values).abs().sum()
                for a, b in zip(left, right)
            ]
        )
        return torch.stack((finite, essential), -1).reshape(
            shape[0], shape[1], len(self.dimensions), 2
        )

    def raw_leaves(
        self,
        generated,
        reference,
        *,
        valid_mask=None,
        reference_cache=None,
        return_counts=False,
        indices=None,
    ):
        indices = tuple(range(len(self.units))) if indices is None else tuple(indices)
        key = (
            ("conditional_fair_reference", id(self), indices)
            if reference_cache is not None and "signature" in reference_cache
            else (
                "conditional_fair_reference",
                id(self),
                indices,
                id(reference),
                reference._version,
                id(valid_mask),
                None if valid_mask is None else valid_mask._version,
            )
        )
        cached = None if reference_cache is None else reference_cache.get(key)
        if cached is None:
            target, shape = self._diagrams(reference.detach(), indices, valid_mask, cacheable=True)
            if reference_cache is not None:
                reference_cache[key] = (target, shape)
        else:
            target, shape = cached
        first, first_shape = self._diagrams(generated, indices, valid_mask)
        if first_shape != shape:
            raise ValueError("reference and prediction shapes differ")
        norm = shape[-2] * shape[-1] if valid_mask is None else int(valid_mask.sum())
        counts = torch.tensor(
            [d.essential.numel() for d in target], device=generated.device
        ).reshape(shape[0], shape[1], len(self.dimensions))
        if not torch.equal(counts, counts[:, :1, :].expand_as(counts)):
            raise ValueError("fixed-domain essential class counts differ")
        domain_counts = counts[:, 0, :]
        first_distance = self._distance(first, target, shape, norm)
        second = getattr(self, "_second_generated", None)
        if second is None:
            raw = first_distance
        else:
            second_diagrams, second_shape = self._diagrams(second, indices, valid_mask)
            if second_shape != shape:
                raise ValueError("second prediction shape differs")
            attraction = 0.5 * (
                first_distance + self._distance(second_diagrams, target, shape, norm)
            )
            separation = 0.5 * self._distance(first, second_diagrams, shape, norm)
            self._last_fair_leaves = (attraction, separation)
            raw = attraction - separation
        return raw, domain_counts if return_counts else domain_counts > 0, norm

    @torch.no_grad()
    def accumulate_calibration(self, generated, reference, *, valid_mask=None):
        if self.calibrated:
            raise ValueError("calibration already frozen")
        mask_hash = (
            None
            if valid_mask is None
            else hashlib.sha256(valid_mask.detach().cpu().numpy().tobytes()).hexdigest()
        )
        if not self.transform_record or self.transform_record["mask_sha256"] != mask_hash:
            raise ValueError("calibration raster mask differs from transform")
        raw, counts, norm = self.raw_leaves(
            generated, reference, valid_mask=valid_mask, return_counts=True
        )
        s = raw.sum(1).cpu()
        if self._calibration_accumulator is None:
            self._calibration_accumulator = (generated.shape[0], s, counts.cpu(), norm)
        else:
            n, old, old_counts, old_norm = self._calibration_accumulator
            if not torch.equal(counts.cpu(), old_counts) or norm != old_norm:
                raise ValueError("calibration domain changed")
            self._calibration_accumulator = (n + generated.shape[0], old + s, old_counts, norm)

    @torch.no_grad()
    def freeze_calibration(self, *, provenance):
        if self.calibrated or self._calibration_accumulator is None:
            raise ValueError("calibration already frozen or no paired samples")
        n, sums, counts, norm = self._calibration_accumulator
        ids = provenance.get("sample_ids", ()) if isinstance(provenance, dict) else ()
        source = (
            provenance.get("source_checkpoint_sha256", "") if isinstance(provenance, dict) else ""
        )
        if (
            not isinstance(provenance, dict)
            or provenance.get("split") != "train"
            or len(ids) != n
            or len(set(ids)) != n
            or len(source) != 64
            or any(c not in "0123456789abcdef" for c in source)
        ):
            raise ValueError("calibration requires TRAIN sample IDs and SOURCE sha256")
        if list(ids) != list(self.transform_record["provenance"]["sample_ids"]):
            raise ValueError("calibration sample IDs differ from transform cohort")
        averages = sums / n
        scales = torch.maximum(
            averages, averages.new_tensor([self.finite_floor, self.essential_floor])
        )
        if not torch.isfinite(scales).all():
            raise FloatingPointError("nonfinite calibration scales")
        self.leaf_scales.copy_(scales)
        self.essential_counts.copy_(counts)
        self.essential_present.copy_(counts > 0)
        self.calibrated.fill_(True)
        self.calibration_record = {
            "definition": self.definition,
            "provenance": deepcopy(provenance),
            "raw_mean": averages.tolist(),
            "scales": scales.tolist(),
            "essential_present": (counts > 0).tolist(),
            "essential_counts": counts.tolist(),
            "point_normalization": norm,
            "units": [unit[1] for unit in self.units],
            "dimensions": list(self.dimensions),
        }
        self._calibration_accumulator = None
        return deepcopy(self.calibration_record)

    def forward(self, generated, reference, *, second_generated, **kwargs):
        if second_generated is None or second_generated.shape != generated.shape:
            raise ValueError("fair Shape requires two aligned prediction members")
        self._second_generated = second_generated
        try:
            result = super().forward(generated, reference, **kwargs)
            attraction, separation = self._last_fair_leaves
            selection = kwargs.get("query_selection")
            if selection is None:
                rows = [(i, None) for i in range(len(self.units))]
            else:
                lookup = {
                    (
                        "self" if component == "self.persistence" else "mutual",
                        tuple(self.fields[j] for j in ids),
                        signs,
                        line,
                    ): i
                    for i, (component, _, ids, signs, line) in enumerate(self.units)
                }
                rows = [(lookup[q.query_id], q) for q in selection.queries]
            counts = {
                "self.persistence": 2 * len(self.self_fields),
                "mutual.persistence": 4 * len(self.groups),
            }
            aggregates = {}
            for row, (i, query) in enumerate(rows):
                component, _, _, _, line = self.units[i]
                weight = (
                    float(1.0 if line is None else self.line_weights[line]) / counts[component]
                    if query is None
                    else query.estimator_weight
                )
                for label, leaves in (("attraction", attraction), ("separation", separation)):
                    part = 0.9 * leaves[row, :, :, 0] / self.leaf_scales[i, :, 0].to(leaves)
                    part = part + 0.1 * leaves[row, :, :, 1] / self.leaf_scales[i, :, 1].to(leaves)
                    value = weight * part.mean(-1)
                    aggregates[(component, label)] = aggregates.get((component, label), 0) + value
            batch = generated.shape[0]
            totals = {}
            for label in ("attraction", "separation"):
                total = generated.new_zeros(batch)
                for component in self.weights:
                    total = total + self.weights[component] * aggregates.get((component, label), 0)
                totals[label] = total.detach().cpu().tolist()
            result.diagnostics.update(
                {
                    "attraction_leaves": attraction.detach().cpu().tolist(),
                    "separation_leaves": separation.detach().cpu().tolist(),
                    "attraction_per_sample": totals["attraction"],
                    "separation_per_sample": totals["separation"],
                    "component_attraction_separation": {
                        component: {
                            label: aggregates[(component, label)].detach().cpu().tolist()
                            for label in ("attraction", "separation")
                        }
                        for component in self.weights
                        if (component, "attraction") in aggregates
                    },
                    "predicted_endpoint_filtrations_per_condition": 2 * len(rows),
                    "reference_filtrations_per_condition": len(rows),
                    "predicted_endpoint_filtrations_total": 2 * len(rows) * batch,
                    "reference_filtrations_total": len(rows) * batch,
                    "observer_qualification": "pending_external_domain_and_source_calibration",
                    "gradient_cost_readiness": "pending_model_qualification",
                }
            )
            return result
        finally:
            self._second_generated = None
            self._last_fair_leaves = None

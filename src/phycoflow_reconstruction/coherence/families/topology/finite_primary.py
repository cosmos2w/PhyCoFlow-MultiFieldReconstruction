"""Versioned finite-primary persistence with an explicit four-sign pair bank.

The fixed lower-star diagrams summarize excursion geometry on a declared
raster. They are not a positional field metric or a complete multiparameter
invariant. The target supplies per-snapshot descriptor mean and scale; the
separate loss scales must be frozen from paired TRAIN/SOURCE samples first.
"""

from __future__ import annotations

import math
from copy import deepcopy

import torch

from ....contracts import CoherenceComponentSpec, CoherenceFamilySpec, FamilyResult, TermResult
from .persistence import cubical_diagrams, sliced_diagram_distances
from .persistence_objective import PersistenceTopologyObjective


class FinitePrimaryTopologyObjective(PersistenceTopologyObjective):
    """H0/H1 finite and essential leaves, with the declared pair-line quadrature."""

    definition = "finite_primary_v1"
    signs = (
        (1, 1, "positive_positive"),
        (-1, -1, "negative_negative"),
        (1, -1, "positive_negative"),
        (-1, 1, "negative_positive"),
    )

    def __init__(self, config, field_names):
        super().__init__(config, field_names)
        settings = config.get("persistence", {})
        calibration = settings.get("calibration")
        self.finite_floor = float(calibration["finite_floor"])
        self.essential_floor = float(calibration["essential_floor"])
        directions = torch.tensor(
            [
                [1 / math.sqrt(2), 1 / math.sqrt(2)],
                [2 / math.sqrt(5), 1 / math.sqrt(5)],
                [1 / math.sqrt(5), 2 / math.sqrt(5)],
            ],
            dtype=torch.float64,
        )
        offsets = torch.tensor([[0.0, 0.0], [0.5, -0.5], [-0.5, 0.5]], dtype=torch.float64)
        for i in range(len(self.groups)):
            setattr(self, f"line_directions_{i}", directions.clone())
            setattr(self, f"line_offsets_{i}", offsets.clone())
        self.register_buffer(
            "line_weights", directions.min(dim=1).values / directions.min(dim=1).values.sum()
        )
        # Each unit is (component, readable label, field indices, signs, line).
        units = []
        if self.weights.get("self.persistence", 0) > 0:
            for index, name in zip(self.self_ids, self.self_fields):
                for sign, label in ((1, "positive"), (-1, "negative")):
                    units.append(
                        ("self.persistence", f"self.{name}.{label}", (index,), (sign,), None)
                    )
        if self.weights.get("mutual.persistence", 0) > 0:
            for group in self.groups:
                name = "+".join(self.fields[i] for i in group)
                for left, right, label in self.signs:
                    for line in range(3):
                        units.append(
                            (
                                "mutual.persistence",
                                f"mutual.{name}.{label}.line{line}",
                                group,
                                (left, right),
                                line,
                            )
                        )
        self.units = tuple(units)
        shape = (len(units), len(self.dimensions))
        self.register_buffer("leaf_scales", torch.ones(*shape, 2))
        self.register_buffer("essential_present", torch.zeros(*shape, dtype=torch.bool))
        self.register_buffer("essential_counts", torch.zeros(*shape, dtype=torch.long))
        self.register_buffer("calibrated", torch.tensor(False))
        self.calibration_record = None
        names = list(self.weights)
        names.extend(
            f"{unit[1]}.h{dim}.{kind}"
            for unit in units
            for dim in self.dimensions
            for kind in ("finite", "essential")
        )
        self.spec = CoherenceFamilySpec(
            "topology",
            "4",
            tuple(
                CoherenceComponentSpec(
                    name,
                    "paired_supervised",
                    config.get("units", "model_units"),
                    name in self.weights,
                    required_geometry="fixed_2d_raster",
                    aggregation="per_sample",
                    metadata={}
                    if name in self.weights
                    else {"evaluation_only": True, "weight": 0.0},
                )
                for name in names
            ),
            metadata={"strategy": "cubical_persistence", "definition": self.definition},
        )
        self.spec.validate()

    def _filtrations(self, x):
        banks = []
        for _, _, ids, signs, line in self.units:
            if line is None:
                banks.append(signs[0] * x[:, ids[0]])
            else:
                v = self.line_directions_0[line].to(x)
                offset = self.line_offsets_0[line].to(x)
                s = x.new_tensor(signs)
                banks.append(
                    (
                        (x[:, ids] * s[None, :, None, None] - offset[None, :, None, None])
                        / v[None, :, None, None]
                    ).amax(1)
                )
        return torch.stack(banks)

    def raw_leaves(
        self, generated, reference, *, valid_mask=None, reference_cache=None, return_counts=False
    ):
        """Return raw F/E and E presence, or exact domain counts if requested."""
        if reference_cache is not None:
            # The family-level signature already guards the original target;
            # its rasterized tensor is reconstructed on each family call.
            target = None if "signature" in reference_cache else reference
            identity = (
                target,
                None if target is None else target._version,
                valid_mask,
                None if valid_mask is None else valid_mask._version,
            )
            old = reference_cache.setdefault("finite_primary_identity", identity)
            if (
                old[0] is not target
                or old[1] != identity[1]
                or old[2] is not valid_mask
                or old[3] != identity[3]
            ):
                raise ValueError("persistence reference cache target or mask changed")
        prepared = (
            None if reference_cache is None else reference_cache.get("finite_primary_prepared")
        )
        if prepared is None:
            y = self._descriptor_fields(reference.detach())
            support = y if valid_mask is None else y[..., valid_mask].unsqueeze(-2)
            mean = support.mean((-2, -1), keepdim=True)
            scale = support.std((-2, -1), keepdim=True, unbiased=False).clamp_min(self.scale_floor)
            bank = self._filtrations((y - mean) / scale)
            target_diagrams = cubical_diagrams(
                bank.flatten(0, 1),
                periodic=self.periodic,
                dimensions=self.dimensions,
                valid_mask=valid_mask,
                cacheable=True,
            )
            prepared = (mean, scale, target_diagrams, tuple(reference.shape), id(self))
            if reference_cache is not None:
                reference_cache["finite_primary_prepared"] = prepared
        mean, scale, target_diagrams, shape, owner = prepared
        if tuple(reference.shape) != shape or owner != id(self):
            raise ValueError("persistence reference cache belongs to another objective or shape")
        x = (self._descriptor_fields(generated) - mean) / scale
        bank = self._filtrations(x)
        n, batch, h, w = bank.shape
        diagrams = cubical_diagrams(
            bank.flatten(0, 1),
            periodic=self.periodic,
            dimensions=self.dimensions,
            valid_mask=valid_mask,
        )
        left = [item[d] for item in diagrams for d in self.dimensions]
        right = [item[d] for item in target_diagrams for d in self.dimensions]
        normalization = h * w if valid_mask is None else int(valid_mask.sum())
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
        counts = torch.tensor([b.essential.numel() for b in right], device=x.device).reshape(
            n, batch, len(self.dimensions)
        )
        if not torch.equal(counts, counts[:, :1, :].expand_as(counts)):
            raise ValueError("essential class counts differ on the fixed domain")
        domain_counts = counts[:, 0, :]
        return (
            torch.stack((finite, essential), dim=-1).reshape(n, batch, len(self.dimensions), 2),
            domain_counts if return_counts else domain_counts > 0,
            normalization,
        )

    @torch.no_grad()
    def fit_calibration(self, generated, reference, *, valid_mask=None, provenance):
        if self.calibrated:
            raise ValueError("finite_primary_v1 calibration is already frozen")
        if not isinstance(provenance, dict) or provenance.get("split") != "train":
            raise ValueError("finite_primary_v1 calibration requires TRAIN provenance")
        ids = provenance.get("sample_ids")
        source = provenance.get("source_checkpoint_sha256")
        if (
            not isinstance(ids, (list, tuple))
            or len(ids) != generated.shape[0]
            or any(i is None or not str(i).strip() for i in ids)
        ):
            raise ValueError("calibration sample_ids must identify every TRAIN/SOURCE pair")
        if (
            not isinstance(source, str)
            or len(source) != 64
            or any(c not in "0123456789abcdef" for c in source)
        ):
            raise ValueError("calibration requires source_checkpoint_sha256")
        raw, counts, normalization = self.raw_leaves(
            generated, reference, valid_mask=valid_mask, return_counts=True
        )
        present = counts > 0
        averages = raw.mean(dim=1)
        floors = averages.new_tensor([self.finite_floor, self.essential_floor])
        scales = torch.maximum(averages, floors)
        if not torch.isfinite(scales).all():
            raise FloatingPointError("finite_primary_v1 calibration produced nonfinite scales")
        self.leaf_scales.copy_(scales)
        self.essential_present.copy_(present)
        self.essential_counts.copy_(counts)
        self.calibrated.fill_(True)
        self.calibration_record = {
            "definition": self.definition,
            "provenance": deepcopy(provenance),
            "raw_mean": averages.cpu().tolist(),
            "scales": scales.cpu().tolist(),
            "essential_present": present.cpu().tolist(),
            "essential_counts": counts.cpu().tolist(),
            "point_normalization": normalization,
            "units": [unit[1] for unit in self.units],
            "dimensions": list(self.dimensions),
        }
        return deepcopy(self.calibration_record)

    def forward(self, generated, reference, *, reference_cache=None, valid_mask=None):
        if not self.calibrated:
            raise ValueError(
                "finite_primary_v1 must be calibrated on TRAIN/SOURCE pairs before scoring"
            )
        raw, counts, normalization = self.raw_leaves(
            generated,
            reference,
            valid_mask=valid_mask,
            reference_cache=reference_cache,
            return_counts=True,
        )
        if not torch.equal(counts, self.essential_counts):
            raise ValueError("fixed-domain essential class counts differ from calibration")
        results, sums = {}, {}
        recorded_scales = self.calibration_record["scales"]
        recorded_presence = self.calibration_record["essential_present"]
        unit_counts = {
            "self.persistence": 2 * len(self.self_fields),
            "mutual.persistence": 4 * len(self.groups),
        }
        for i, (component, label, _, _, line) in enumerate(self.units):
            for j, dim in enumerate(self.dimensions):
                finite, essential = raw[i, :, j, 0], raw[i, :, j, 1]
                for k, (kind, value) in enumerate((("finite", finite), ("essential", essential))):
                    key = f"topology.{label}.h{dim}.{kind}"
                    results[key] = TermResult(
                        value,
                        value.mean(),
                        diagnostics={
                            "evaluation_only": True,
                            "scale": recorded_scales[i][j][k],
                            "structurally_present": k == 0 or recorded_presence[i][j],
                        },
                    )
                cost = 0.9 * finite / self.leaf_scales[i, j, 0]
                if recorded_presence[i][j]:
                    cost = cost + 0.1 * essential / self.leaf_scales[i, j, 1]
                weight = 1.0 if line is None else self.line_weights[line].to(cost)
                sums[component] = sums.get(component, 0) + weight * cost / (
                    unit_counts[component] * len(self.dimensions)
                )
        total = generated.sum(dim=(1, 2, 3)) * 0
        for name, value in sums.items():
            results[f"topology.{name}"] = TermResult(value, value.mean())
            total = total + self.weights[name] * value
        return FamilyResult(
            results,
            total,
            total.mean(),
            diagnostics={
                "definition": self.definition,
                "scalar_filtrations": len(self.units),
                "point_normalization": normalization,
                "finite_fraction": 0.9,
                "essential_fraction": 0.1,
                "line_weights": self.line_weights.detach().cpu().tolist(),
                "diagram_distance": "sliced_wasserstein",
                "essential_classes_included": True,
                "finite_slice_approximation": True,
            },
        )

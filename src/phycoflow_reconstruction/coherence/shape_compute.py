"""Lazy pooled fair persistence: new observer, distinct from per-pair v5."""

from __future__ import annotations

import hashlib
import math

import torch

from .families.topology.persistence import cubical_diagrams, sliced_diagram_distances
from .query_sampling import LINE_WEIGHTS, PAIR_SIGNS, index_pair, pair_index

VERSION = "round8_lazy_pooled_fair_shape_v1"
PAIRED_VERSION = "scalable_lazy_pooled_paired_shape_v1"
_SIGNS = ((1, 1), (-1, -1), (1, -1), (-1, 1))
_DIRECTIONS = (
    (1 / math.sqrt(2), 1 / math.sqrt(2)),
    (2 / math.sqrt(5), 1 / math.sqrt(5)),
    (1 / math.sqrt(5), 2 / math.sqrt(5)),
)
_OFFSETS = ((0.0, 0.0), (0.5, -0.5), (-0.5, 0.5))


class LazyPooledShape:
    """Only selected field IDs are turned into filtrations; no pair bank.

    Fixed transform is provided from TRAIN. Calibrate with bounded TRAIN query
    batches using SOURCE/reference, then freeze positive pooled leaf scales.
    Counts record the number of calibration condition-query observations.
    """

    def __init__(
        self,
        field_mean,
        field_scale,
        *,
        dimensions=(0, 1),
        projections=32,
        periodic=False,
        floors=(1e-6, 1e-6),
    ):
        self.mean = torch.as_tensor(field_mean).detach().cpu().double().flatten()
        self.scale = torch.as_tensor(field_scale).detach().cpu().double().flatten()
        if (
            not self.mean.numel()
            or self.mean.shape != self.scale.shape
            or not torch.isfinite(self.mean).all()
            or not torch.isfinite(self.scale).all()
            or not (self.scale > 0).all()
        ):
            raise ValueError("positive finite TRAIN field transforms required")
        self.dimensions = tuple(dimensions)
        if (
            not self.dimensions
            or len(set(self.dimensions)) != len(self.dimensions)
            or any(type(d) is not int or d not in (0, 1) for d in self.dimensions)
            or isinstance(projections, bool)
            or not isinstance(projections, int)
            or projections < 2
        ):
            raise ValueError("invalid homology dimensions or projections")
        self.projections, self.periodic = int(projections), bool(periodic)
        self.floors = torch.as_tensor(floors, dtype=torch.float64)
        if (
            self.floors.shape != (2,)
            or not torch.isfinite(self.floors).all()
            or not (self.floors > 0).all()
        ):
            raise ValueError("positive finite category floors required")
        self.sums, self.counts, self.scales = {}, {}, None
        self.domain = None
        self.line_weights = list(LINE_WEIGHTS)
        self.provenance = None

    def _domain(self, x, mask):
        mask_digest = (
            None
            if mask is None
            else hashlib.sha256(mask.detach().cpu().numpy().tobytes()).hexdigest()
        )
        domain = (tuple(x.shape[-2:]), mask_digest, self.periodic)
        if self.domain is None:
            self.domain = domain
        elif self.domain != domain:
            raise ValueError("fixed Shape domain changed")
        return x.shape[-2] * x.shape[-1] if mask is None else int(mask.sum())

    def _queries(self, batch):
        n = self.mean.numel()
        for d in batch.self_queries:
            if d.index >= n:
                raise ValueError("invalid self query")
            for sign in (1, -1):
                yield ("self", sign, None), (d.index,), (sign,), None, d.weight / 2
        for d in batch.pair_queries:
            a, b = index_pair(n, d.index)
            for signs in _SIGNS:
                for line, weight in enumerate(self.line_weights):
                    yield ("mutual", signs, line), (a, b), signs, line, d.weight * weight / 4

    def _sampled_queries(self, batch):
        """One explicitly sampled sign/line per coalesced outer field ID."""
        n = self.mean.numel()
        for draw in batch.self_queries:
            if (
                not 0 <= draw.index < n
                or draw.inner_sign not in ((1,), (-1,))
                or draw.inner_line is not None
                or draw.inner_probability != 0.5
            ):
                raise ValueError("invalid sampled self Shape query")
            yield (
                ("self", draw.inner_sign[0], None),
                (draw.index,),
                draw.inner_sign,
                None,
                draw.weight,
            )
        for draw in batch.pair_queries:
            pair = index_pair(n, draw.index)
            line = draw.inner_line
            if (
                draw.inner_sign not in PAIR_SIGNS
                or isinstance(line, bool)
                or not isinstance(line, int)
                or not 0 <= line < len(LINE_WEIGHTS)
                or draw.inner_probability is None
                or not math.isclose(draw.inner_probability, LINE_WEIGHTS[line] / 4, rel_tol=1e-12)
            ):
                raise ValueError("invalid sampled mutual Shape query")
            yield ("mutual", draw.inner_sign, line), pair, draw.inner_sign, line, draw.weight

    def _filtration(self, x, ids, signs, line):
        indices = list(ids)
        y = (x[:, indices] - self.mean[indices].to(x)[None, :, None, None]) / self.scale[
            indices
        ].to(x)[None, :, None, None]
        if line is None:
            return signs[0] * y[:, 0]
        s = x.new_tensor(signs)[None, :, None, None]
        v = x.new_tensor(_DIRECTIONS[line])[None, :, None, None]
        off = x.new_tensor(_OFFSETS[line])[None, :, None, None]
        return ((y * s - off) / v).amax(1)

    def _diagram_bank(self, x, queries, mask):
        """Construct only selected filtrations in query-major, then batch order."""
        if not queries:
            raise ValueError("at least one selected Shape query is required")
        left = torch.tensor([q[1][0] for q in queries], device=x.device, dtype=torch.long)
        right = torch.tensor([q[1][-1] for q in queries], device=x.device, dtype=torch.long)
        mean, scale = self.mean.to(x), self.scale.to(x)

        def transformed(index):
            return (x.index_select(1, index) - mean[index][None, :, None, None]) / scale[index][
                None, :, None, None
            ]

        a, b = transformed(left), transformed(right)
        signs_left = x.new_tensor([q[2][0] for q in queries])[None, :, None, None]
        signs_right = x.new_tensor([q[2][-1] for q in queries])[None, :, None, None]
        lines = [0 if q[3] is None else q[3] for q in queries]
        directions = x.new_tensor([_DIRECTIONS[i] for i in lines])
        offsets = x.new_tensor([_OFFSETS[i] for i in lines])
        lhs = (a * signs_left - offsets[:, 0][None, :, None, None]) / directions[:, 0][
            None, :, None, None
        ]
        rhs = (b * signs_right - offsets[:, 1][None, :, None, None]) / directions[:, 1][
            None, :, None, None
        ]
        mutual = torch.tensor(
            [q[3] is not None for q in queries], device=x.device, dtype=torch.bool
        )[None, :, None, None]
        filtrations = torch.where(mutual, torch.maximum(lhs, rhs), a * signs_left)
        flat = filtrations.permute(1, 0, 2, 3).reshape(-1, *x.shape[-2:])
        return cubical_diagrams(
            flat, periodic=self.periodic, dimensions=self.dimensions, valid_mask=mask
        )

    def _distance(self, a, b, norm):
        left = [item[d] for item in a for d in self.dimensions]
        right = [item[d] for item in b for d in self.dimensions]
        finite = sliced_diagram_distances(
            left, right, projections=self.projections, essential_weight=0, normalization=norm
        )
        essential = torch.stack(
            [
                (x.essential.sort().values - y.essential.sort().values).abs().sum()
                for x, y in zip(left, right)
            ]
        )
        return torch.stack((finite, essential), -1).reshape(len(a), len(self.dimensions), 2)

    def _diagram(self, x, ids, signs, line, mask):
        return cubical_diagrams(
            self._filtration(x, ids, signs, line),
            periodic=self.periodic,
            dimensions=self.dimensions,
            valid_mask=mask,
        )

    @torch.no_grad()
    def accumulate(self, source, reference, batch, *, valid_mask=None):
        if self.scales is not None:
            raise ValueError("calibration already frozen")
        self._validate(source, reference)
        norm = self._domain(source, valid_mask)
        for d in (*batch.self_queries, *batch.pair_queries):
            if not math.isclose(d.probability, 1 / d.population, rel_tol=1e-12):
                raise ValueError("Shape calibration requires a fixed uniform query sample")
        queries = tuple(self._queries(batch))
        a = self._diagram_bank(source, queries, valid_mask)
        b = self._diagram_bank(reference, queries, valid_mask)
        raw = self._distance(a, b, norm).reshape(len(queries), len(source), len(self.dimensions), 2)
        if not torch.isfinite(raw).all():
            raise FloatingPointError("nonfinite calibration")
        sums = raw.double().sum(1).cpu()
        self_draws = {d.index: d for d in batch.self_queries}
        pair_draws = {d.index: d for d in batch.pair_queries}
        for row, (key, ids, _, _, _) in enumerate(queries):
            index = ids[0] if key[0] == "self" else pair_index(self.mean.numel(), *ids)
            multiplicity = (self_draws if key[0] == "self" else pair_draws)[index].count
            self.sums[key] = (
                self.sums.get(key, torch.zeros_like(sums[row])) + multiplicity * sums[row]
            )
            self.counts[key] = self.counts.get(key, 0) + multiplicity * len(source)

    def freeze(self, *, provenance):
        if self.scales is not None or not self.sums:
            raise ValueError("calibration missing or frozen")
        if not isinstance(provenance, dict):
            raise TypeError("TRAIN provenance required")
        ids, digest = (
            provenance.get("sample_ids", ()),
            provenance.get("source_checkpoint_sha256", ""),
        )
        if (
            provenance.get("split") != "train"
            or not ids
            or len(set(ids)) != len(ids)
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)
        ):
            raise ValueError("unique TRAIN IDs and SOURCE sha256 required")
        required = {("self", s, None) for s in (1, -1)}
        if self.mean.numel() > 1:
            required |= {("mutual", s, line) for s in _SIGNS for line in range(3)}
        if set(self.sums) != required:
            raise ValueError("all pooled sign/line categories require calibration")
        self.scales = {
            k: torch.maximum(v / self.counts[k], self.floors) for k, v in self.sums.items()
        }
        self.sums = {}
        self.provenance = dict(provenance)
        return self.state_dict()

    def _validate(self, x, y):
        if x.ndim != 4 or x.shape != y.shape or x.shape[1] != self.mean.numel():
            raise ValueError("Shape fields must align as [B,C,H,W]")

    def score(
        self,
        first,
        second,
        reference,
        batch,
        *,
        valid_mask=None,
        return_leaves=False,
        exhaustive=False,
        mode="conditional",
    ):
        if mode not in {"paired", "conditional"} or (mode == "paired") != (second is None):
            raise ValueError(
                "paired Shape requires one member; conditional Shape requires independent draws"
            )
        if self.scales is None:
            raise ValueError("pooled TRAIN calibration must be frozen")
        self._validate(first, reference)
        if second is not None:
            self._validate(second, reference)
        norm = self._domain(first, valid_mask)
        queries = tuple(self._queries(batch) if exhaustive else self._sampled_queries(batch))
        a = self._diagram_bank(first, queries, valid_mask)
        b = None if second is None else self._diagram_bank(second, queries, valid_mask)
        with torch.no_grad():
            t = self._diagram_bank(reference.detach(), queries, valid_mask)
        shape = (len(queries), len(first), len(self.dimensions), 2)
        raw = self._distance(a, t, norm)
        if mode == "conditional":
            raw = (raw + self._distance(b, t, norm) - self._distance(a, b, norm)) / 2
        raw = raw.reshape(shape)
        scales = torch.stack([self.scales[q[0]] for q in queries]).to(raw)
        leaf = raw / scales[:, None]
        values = (0.9 * leaf[..., 0] + 0.1 * leaf[..., 1]).mean(-1)
        weights = values.new_tensor([q[4] for q in queries])
        totals = {}
        for kind in ("self", "mutual"):
            indices = [i for i, q in enumerate(queries) if q[0][0] == kind]
            if indices:
                selection = torch.tensor(indices, device=values.device, dtype=torch.long)
                totals[kind] = (
                    values.index_select(0, selection) * weights.index_select(0, selection)[:, None]
                ).sum(0)
            else:
                totals[kind] = first.new_zeros(len(first))
        if not return_leaves:
            return totals
        leaves = {"self": {}, "pair": {}}
        for kind, draws in (("self", batch.self_queries), ("pair", batch.pair_queries)):
            for draw in draws:
                indices = [
                    i
                    for i, q in enumerate(queries)
                    if (q[0][0] == "self") == (kind == "self")
                    and (q[1][0] if kind == "self" else pair_index(self.mean.numel(), *q[1]))
                    == draw.index
                ]
                selection = torch.tensor(indices, device=values.device, dtype=torch.long)
                leaves[kind][draw.index] = (
                    values.index_select(0, selection)
                    * (weights.index_select(0, selection) / draw.weight)[:, None]
                ).sum(0)
        return totals, leaves

    def state_dict(self):
        if self.scales is None:
            raise ValueError("calibration must be frozen")
        return {
            "version": VERSION,
            "field_mean": self.mean.tolist(),
            "field_scale": self.scale.tolist(),
            "dimensions": self.dimensions,
            "projections": self.projections,
            "periodic": self.periodic,
            "floors": self.floors.tolist(),
            "domain": self.domain,
            "provenance": self.provenance,
            "scales": [(k, v.tolist(), self.counts[k]) for k, v in self.scales.items()],
        }

    @classmethod
    def from_state_dict(cls, state):
        required = {
            "version",
            "field_mean",
            "field_scale",
            "dimensions",
            "projections",
            "periodic",
            "floors",
            "domain",
            "provenance",
            "scales",
        }
        if not isinstance(state, dict) or set(state) != required or state["version"] != VERSION:
            raise ValueError("incompatible Shape artifact")
        obj = cls(
            state["field_mean"],
            state["field_scale"],
            dimensions=state["dimensions"],
            projections=state["projections"],
            periodic=state["periodic"],
            floors=state["floors"],
        )

        def tuples(x):
            return tuple(tuples(y) for y in x) if isinstance(x, (list, tuple)) else x

        domain = tuples(state["domain"])
        if (
            not isinstance(domain, tuple)
            or len(domain) != 3
            or not isinstance(domain[0], tuple)
            or len(domain[0]) != 2
            or any(isinstance(n, bool) or not isinstance(n, int) or n < 2 for n in domain[0])
            or domain[2] != obj.periodic
            or (
                domain[1] is not None
                and (
                    not isinstance(domain[1], str)
                    or len(domain[1]) != 64
                    or any(c not in "0123456789abcdef" for c in domain[1])
                )
            )
        ):
            raise ValueError("invalid Shape domain")
        provenance = state["provenance"]
        if not isinstance(provenance, dict):
            raise TypeError("invalid Shape TRAIN provenance")
        ids, digest = (
            provenance.get("sample_ids", ()),
            provenance.get("source_checkpoint_sha256", ""),
        )
        if (
            provenance.get("split") != "train"
            or not isinstance(ids, (list, tuple))
            or not ids
            or any(not isinstance(x, str) or not x for x in ids)
            or len(set(ids)) != len(ids)
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)
        ):
            raise ValueError("invalid Shape TRAIN provenance")
        keys = {("self", sign, None) for sign in (1, -1)}
        if obj.mean.numel() > 1:
            keys |= {("mutual", signs, line) for signs in _SIGNS for line in range(3)}
        scales, counts = {}, {}
        try:
            for key, value, count in state["scales"]:
                key = tuples(key)
                scale = torch.as_tensor(value, dtype=torch.float64)
                if (
                    key in scales
                    or key not in keys
                    or scale.shape != (len(obj.dimensions), 2)
                    or not torch.isfinite(scale).all()
                    or (scale < obj.floors).any()
                    or isinstance(count, bool)
                    or not isinstance(count, int)
                    or count <= 0
                ):
                    raise ValueError("invalid Shape calibration scales")
                scales[key], counts[key] = scale, count
        except (TypeError, KeyError) as error:
            raise ValueError("invalid Shape calibration scales") from error
        if set(scales) != keys:
            raise ValueError("incomplete Shape calibration scales")
        obj.domain, obj.provenance, obj.scales, obj.counts = (
            domain,
            dict(provenance),
            scales,
            counts,
        )
        return obj

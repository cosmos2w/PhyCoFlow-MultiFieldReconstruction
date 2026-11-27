"""Fixed, TRAIN-fitted regional observations of selected Shape filtrations.

The raster measure and axis extents are supplied by the case; no native-area
claim follows from a sparse raster. Fitting is two-pass and exact: it retains
the selected TRAIN filtration values on CPU until weighted quantiles are fit.
"""

from __future__ import annotations

import math

import torch


VERSION = "fixed_soft_excursion_windows_v1"


def _key(key):
    """Encode a filtration category without depending on dictionary key order."""
    kind, sign, line = key
    signs = (sign,) if isinstance(sign, int) else tuple(sign)
    return f"{kind}:{','.join(str(int(value)) for value in signs)}:{line}"


def _weighted_quantiles(values, weights, probabilities):
    order = torch.argsort(values)
    values, weights = values[order], weights[order]
    cumulative = weights.cumsum(0)
    targets = cumulative[-1] * torch.as_tensor(probabilities, dtype=weights.dtype)
    return values[torch.searchsorted(cumulative, targets).clamp_max(len(values) - 1)]


class PositionalShape:
    """Evaluate a frozen fixed feature map without any persistence calls."""

    def __init__(self, state):
        required = {"version", "grid_shape", "coordinates", "raster_mass", "valid_mask",
                    "axis_extents", "periods", "centers", "widths", "window_weights",
                    "removed_windows", "thresholds", "temperatures", "quantiles",
                    "location_unit", "kind_weights", "threshold_chunk_size", "fit_counts"}
        if not isinstance(state, dict) or set(state) != required or state["version"] != VERSION:
            raise ValueError("incompatible positional Shape artifact")
        self.state = state
        self.grid_shape = tuple(state["grid_shape"])
        p = math.prod(self.grid_shape)
        self.windows = torch.as_tensor(state["window_weights"]).detach().cpu().double()
        self.mass = torch.as_tensor(state["raster_mass"]).detach().cpu().double()
        self.mask = torch.as_tensor(state["valid_mask"]).detach().cpu().bool()
        coords = torch.as_tensor(state["coordinates"]).detach().cpu().double()
        if (len(self.grid_shape) != 2 or min(self.grid_shape) < 2
                or coords.shape != (*self.grid_shape, 2)
                or self.mass.shape != self.grid_shape or self.mask.shape != self.grid_shape
                or self.windows.ndim != 2 or self.windows.shape[1] != p or not len(self.windows)
                or not torch.isfinite(coords).all() or not torch.isfinite(self.mass).all()
                or (self.mass < 0).any() or (self.mass[~self.mask] != 0).any()
                or self.mass[self.mask].sum() <= 0 or not torch.isfinite(self.windows).all()
                or (self.windows < 0).any()
                or not torch.allclose(self.windows.sum(-1), torch.ones(len(self.windows), dtype=torch.float64))
                or (self.windows[:, ~self.mask.flatten()] != 0).any()):
            raise ValueError("invalid positional Shape geometry")
        self.thresholds = state["thresholds"]
        self.temperatures = state["temperatures"]
        if (not self.thresholds or set(self.thresholds) != set(self.temperatures)
                or any(len(row) != 3 or not all(math.isfinite(float(x)) for x in row)
                       or list(row) != sorted(row) for row in self.thresholds.values())
                or any(not math.isfinite(float(v)) or float(v) < 0.1
                       for v in self.temperatures.values())):
            raise ValueError("invalid frozen positional thresholds")
        self.unit = float(state["location_unit"])
        self.chunk = state["threshold_chunk_size"]
        if (not math.isfinite(self.unit) or self.unit <= 0 or type(self.chunk) is not int
                or self.chunk < 1):
            raise ValueError("positive fitted positional unit and chunk size required")

    def features(self, filtrations, queries):
        """Return [query,batch,window,threshold], contracting one threshold chunk."""
        if (filtrations.ndim != 4 or tuple(filtrations.shape[1:2]) != (len(queries),)
                or tuple(filtrations.shape[-2:]) != self.grid_shape):
            raise ValueError("positional filtrations must be [B,Q,H,W] on the fitted grid")
        keys = [_key(q[0]) for q in queries]
        if any(key not in self.thresholds for key in keys):
            raise ValueError("unfitted positional filtration category")
        active = self.mask.to(device=filtrations.device).flatten()
        flat = filtrations.permute(1, 0, 2, 3).flatten(-2)
        if not torch.isfinite(flat[..., active]).all():
            raise ValueError("nonfinite valid positional filtration")
        flat = torch.where(active, flat, torch.zeros_like(flat))
        thresholds = flat.new_tensor([self.thresholds[key] for key in keys])
        temperatures = flat.new_tensor([self.temperatures[key] for key in keys])
        windows = self.windows.to(flat)
        output = []
        for start in range(0, thresholds.shape[-1], self.chunk):
            current = thresholds[:, start:start + self.chunk]
            excursion = torch.sigmoid((current[:, None, :, None] - flat[:, :, None, :])
                                      / temperatures[:, None, None, None])
            output.append(torch.einsum("qbtp,kp->qbkt", excursion, windows))
        return torch.cat(output, dim=-1)

    @staticmethod
    def distance(left, right):
        if left.shape != right.shape or left.ndim != 4:
            raise ValueError("positional feature tensors must align")
        return torch.linalg.vector_norm((left - right).flatten(-2), dim=-1) / math.sqrt(
            left.shape[-2] * left.shape[-1]
        )


class PositionalShapeFitter:
    """Fit fixed category thresholds and a SOURCE/reference location unit on TRAIN.

    `observe_reference` collects weighted reference values, then
    `freeze_thresholds` computes exact weighted quantiles. Replay the same panel
    and queries through `observe_pair` once per equally weighted SOURCE member.
    """

    def __init__(self, shape, grid_coordinates, raster_mass, *, valid_mask=None,
                 periods=(None, None), axis_extents=None, kind_weights=None,
                 threshold_chunk_size=1):
        self.shape = shape
        coords = torch.as_tensor(grid_coordinates).detach().cpu().double()
        mass = torch.as_tensor(raster_mass).detach().cpu().double()
        mask = (mass > 0) if valid_mask is None else torch.as_tensor(valid_mask).detach().cpu().bool()
        if (coords.ndim != 3 or coords.shape[-1] != 2 or mass.shape != coords.shape[:2]
                or mask.shape != mass.shape or not torch.isfinite(coords).all()
                or not torch.isfinite(mass).all() or (mass < 0).any()
                or (mass[~mask] != 0).any() or mass[mask].sum() <= 0):
            raise ValueError("physical raster coordinates, valid mask and nonnegative mass required")
        if len(periods) != 2 or any(v is not None and (not math.isfinite(float(v)) or float(v) <= 0)
                                    for v in periods):
            raise ValueError("explicit positive physical axis periods required")
        if ((shape.periodic and any(value is None for value in periods))
                or (not shape.periodic and any(value is not None for value in periods))):
            raise ValueError("axis periods must match the supported Shape complex")
        if axis_extents is None:
            axis_extents = tuple((float(coords[..., i].min()), float(coords[..., i].max()))
                                 for i in range(2))
        if (len(axis_extents) != 2 or any(len(row) != 2 or not all(math.isfinite(float(v)) for v in row)
                or float(row[1]) <= float(row[0]) for row in axis_extents)):
            raise ValueError("positive finite physical axis extents required")
        if type(threshold_chunk_size) is not int or threshold_chunk_size < 1:
            raise ValueError("threshold chunk size must be positive")
        kind_weights = {"self": 1.0, "mutual": 1.0} if kind_weights is None else dict(kind_weights)
        if (set(kind_weights) != {"self", "mutual"} or any(not math.isfinite(float(x)) or float(x) < 0
                for x in kind_weights.values()) or float(kind_weights["self"]) <= 0
                or (shape.mean.numel() > 1 and float(kind_weights["mutual"]) <= 0)):
            raise ValueError("positive fixed weights for active unary/mutual targets required")
        self.kind_weights = {key: float(value) for key, value in kind_weights.items()}
        # Centers wrap at a period; omit the duplicate endpoint on periodic axes.
        centers = [((i / 4 if periods[0] else i / 3), (j / 2 if periods[1] else j))
                   for i in range(4) for j in range(2)]
        normalized = []
        for axis in range(2):
            lo, hi = (float(v) for v in axis_extents[axis])
            normalized.append((coords[..., axis] - lo) / (float(periods[axis]) if periods[axis] else hi - lo))
        windows, removed = [], []
        for index, center in enumerate(centers):
            log_weight = torch.zeros_like(mass)
            for axis, width in enumerate((0.22, 0.39)):
                delta = normalized[axis] - center[axis]
                log_weight += ((torch.cos(2 * math.pi * delta) - 1) / (2 * math.pi * width) ** 2
                               if periods[axis] else -0.5 * (delta / width).square())
            active = mass > 0
            maximum = log_weight[active].max()
            raw = mass * torch.exp(log_weight - maximum)
            total = raw.sum()
            if not torch.isfinite(total) or total <= 0:
                removed.append(index)
            else:
                windows.append((raw / total).flatten())
        if not windows:
            raise ValueError("no nonempty positional Shape windows")
        self.state = {"version": VERSION, "grid_shape": tuple(mass.shape), "coordinates": coords,
                      "raster_mass": mass, "valid_mask": mask,
                      "axis_extents": tuple(tuple(float(v) for v in row) for row in axis_extents),
                      "periods": tuple(None if v is None else float(v) for v in periods),
                      "centers": centers, "widths": (0.22, 0.39),
                      "window_weights": torch.stack(windows), "removed_windows": removed,
                      "thresholds": {}, "temperatures": {}, "quantiles": (0.2, 0.5, 0.8),
                      "location_unit": 1.0, "kind_weights": self.kind_weights,
                      "threshold_chunk_size": threshold_chunk_size, "fit_counts": {}}
        self.samples = {}
        self.reference_mass = mass.flatten()[mass.flatten() > 0]
        self.feature = None
        self.distance_sum = 0.0
        self.distance_weight = 0.0

    @torch.no_grad()
    def observe_reference(self, reference, queries):
        if self.feature is not None:
            raise ValueError("positional thresholds already frozen")
        self._validate_target_queries(queries)
        self.shape._validate(reference, reference)
        if tuple(reference.shape[-2:]) != self.state["grid_shape"]:
            raise ValueError("reference and positional geometry differ")
        values = self.shape._filtration_bank(reference, queries).permute(1, 0, 2, 3).flatten(-2)
        active = self.state["raster_mass"].flatten() > 0
        for index, query in enumerate(queries):
            weight = float(query[4]) * self.kind_weights[query[0][0]]
            if weight < 0 or not math.isfinite(weight):
                raise ValueError("finite nonnegative fixed query weights required")
            if weight:
                sample = values[index, :, active.to(values.device)].detach().cpu()
                if not torch.isfinite(sample).all():
                    raise ValueError("nonfinite TRAIN filtration")
                self.samples.setdefault(_key(query[0]), []).append((sample, weight))

    def freeze_thresholds(self):
        if self.feature is not None or not self.samples:
            raise ValueError("TRAIN reference filtration calibration missing or already frozen")
        for key, rows in self.samples.items():
            values = torch.cat([row.flatten() for row, _ in rows]).double()
            weights = torch.cat([self.reference_mass.repeat(len(row)) * weight for row, weight in rows])
            quantiles = _weighted_quantiles(values, weights, (0.1, 0.2, 0.5, 0.8, 0.9))
            self.state["thresholds"][key] = quantiles[[1, 2, 3]].tolist()
            self.state["temperatures"][key] = max(0.1, float((quantiles[4] - quantiles[0]) / 4))
            self.state["fit_counts"][key] = int(values.numel())
        self.samples = {}
        self.feature = PositionalShape(self.state)

    @torch.no_grad()
    def observe_pair(self, source, reference, queries):
        if self.feature is None:
            raise ValueError("freeze TRAIN thresholds before calibration")
        self._validate_target_queries(queries)
        self.shape._validate(source, reference)
        a = self.feature.features(self.shape._filtration_bank(source, queries), queries)
        b = self.feature.features(self.shape._filtration_bank(reference, queries), queries)
        distance = self.feature.distance(a, b)
        weights = distance.new_tensor([float(q[4]) * self.kind_weights[q[0][0]] for q in queries])
        self.distance_sum += float((distance * weights[:, None]).sum().double().cpu())
        self.distance_weight += float(weights.sum().double().cpu()) * len(source)

    def _validate_target_queries(self, queries):
        from .query_sampling import full_target_query_batch

        expected = tuple(self.shape._queries(full_target_query_batch(self.shape.mean.numel())))
        actual = tuple(queries)
        if len(actual) != len(expected):
            raise ValueError("positional fit requires the complete fixed target query pack")
        keyed = {tuple(row[:4]): float(row[4]) for row in actual}
        if len(keyed) != len(actual) or any(
            tuple(row[:4]) not in keyed or not math.isclose(keyed[tuple(row[:4])], float(row[4]),
                                                           rel_tol=1e-12, abs_tol=1e-15)
            for row in expected
        ):
            raise ValueError("positional fit requires declared target weights, not adaptive frequencies")

    def freeze(self):
        if self.feature is None or self.distance_weight <= 0:
            raise ValueError("SOURCE/reference position calibration is missing")
        unit = self.distance_sum / self.distance_weight
        if not math.isfinite(unit) or unit <= 0:
            raise ValueError("zero positional calibration: inspect TRAIN sensitivity")
        self.state["location_unit"] = unit
        PositionalShape(self.state)
        return self.state

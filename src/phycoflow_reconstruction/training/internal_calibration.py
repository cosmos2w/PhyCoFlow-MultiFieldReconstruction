"""Checkpointable 10-epoch, eight-block internal gradient calibration."""

from __future__ import annotations

import math
import statistics

import torch

VERSION = "internal_gradient_calibration_v1"
BLOCKS = {
    "scale": ("signed", "joint", "log_power_self", "rho"),
    "state": ("marginal", "mixed"),
    "shape": ("self", "mutual"),
}
EXTRA_BLOCKS = {"state": ("group_projection",), "scale": ("third_order",)}
PROBES = (1, 3, 6, 8, 10)


def gradient_norm(loss, parameters):
    """Measure one normalized block on the exact declared trainable parameters."""
    if not loss.requires_grad:
        return 0.0
    gradients = torch.autograd.grad(loss, tuple(parameters), retain_graph=True, allow_unused=True)
    return math.sqrt(
        sum(float(g.detach().double().square().sum()) for g in gradients if g is not None)
    )


def sample_weighted_parameter_gradients(microbatches, parameters):
    """Accumulate complete parameter gradients, before any nonlinear update.

    Each iterable item contains ``(sample_ids, scalar_losses)``. The returned
    named gradients are the gradient of the sample-weighted full-batch mean,
    with all family and field-risk gradients kept separate for the caller to
    apply ConFIG and external fidelity policy exactly once.
    """
    parameters = tuple(parameters)
    if not parameters or any(not p.requires_grad for p in parameters):
        raise ValueError("nonempty trainable parameters are required")
    sums, totals, names, seen = {}, {}, None, []
    for sample_ids, losses in microbatches:
        sample_ids = list(sample_ids)
        if not sample_ids or not losses or (names is not None and tuple(losses) != names):
            raise ValueError("every microbatch requires IDs and the same ordered losses")
        names = tuple(losses)
        n = len(sample_ids)
        seen.extend(sample_ids)
        for name, loss in losses.items():
            if loss.numel() != 1 or not torch.isfinite(loss):
                raise ValueError("microbatch losses must be finite scalars")
            gradients = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True) if loss.requires_grad else (None,) * len(parameters)
            if name not in sums:
                sums[name] = [torch.zeros_like(p) for p in parameters]
                totals[name] = 0.0
            for target, gradient in zip(sums[name], gradients):
                if gradient is not None:
                    target.add_(gradient.detach(), alpha=n)
            totals[name] += float(loss.detach()) * n
    if not seen or len(set(seen)) != len(seen):
        raise ValueError("microbatch IDs must be nonempty and unique")
    size = len(seen)
    return {name: tuple(value / size for value in values) for name, values in sums.items()}, {
        name: value / size for name, value in totals.items()
    }, seen


def fit_common_prefit(calibration, batch_blocks, parameters, *, binding, scale_joint_ratio=1):
    """Fit common coefficients from full semantic-block SOURCE graphs.

    Each item is ``(sample_ids, blocks)`` for one complete panel batch. The
    caller owns constructing the full target and SOURCE rollout; sampled query
    sketches do not meet this interface's scientific contract.
    """
    parameters = tuple(parameters)
    if not parameters or any(not p.requires_grad for p in parameters):
        raise ValueError("SOURCE calibration requires trainable SOURCE parameters")
    expected = {k for k, active in calibration.active.items() if active}
    sums = {k: 0.0 for k in expected}
    ids, batches, count = [], [], 0
    for sample_ids, blocks in batch_blocks:
        sample_ids = list(sample_ids)
        if not sample_ids or set(blocks) != expected:
            raise ValueError("full batch must supply every active semantic block")
        norms = {k: gradient_norm(blocks[k] / calibration.units[k], parameters) for k in sorted(expected)}
        n = len(sample_ids)
        count += n
        ids.extend(sample_ids)
        batches.append({"sample_ids": sample_ids, "sample_count": n, "gradient_norms": norms})
        for k in expected:
            sums[k] += n * norms[k] ** 2
    if not count or len(set(ids)) != count:
        raise ValueError("prefit panel IDs must be nonempty and unique")
    rms = {k: math.sqrt(sums[k] / count) for k in sorted(expected)}
    artifact = {"version": "common_prefit_calibration_v1", "binding": dict(binding), "priorities": calibration.priorities.copy(), "units": calibration.units.copy(), "active": calibration.active.copy(), "rms_gradient_norms": rms, "scale_joint_ratio": scale_joint_ratio, "panel_sample_ids": ids, "batches": batches}
    # The same computation is used by the loader, then independently checked.
    result = {k: 1.0 for k in calibration.active}
    for family, blocks in BLOCKS.items():
        core = [f"{family}.{b}" for b in blocks if calibration.active[f"{family}.{b}"]]
        extras = [f"{family}.{b}" for b in EXTRA_BLOCKS.get(family, ()) if calibration.active.get(f"{family}.{b}", False)]
        positive = [rms[k] for k in core if rms[k] > 0]
        if positive:
            reference = statistics.median(positive)
            for k in core + extras:
                if rms[k] >= max(calibration.floor, 0.1 * reference):
                    result[k] = min(2.0, max(0.5, reference / rms[k]))
    result["scale.joint"] *= scale_joint_ratio
    artifact["coefficients"] = result
    calibration.load_prefit(artifact)
    return artifact


def state_channel_contribution(channel_squared, common_squared, weight):
    """Exact channel chain-rule contribution with zero-distance convention."""
    positive = common_squared > 0
    denom = torch.sqrt(torch.where(positive, common_squared, torch.ones_like(common_squared)))
    factor = torch.where(positive, weight / (2 * denom), torch.zeros_like(denom)).detach()
    return factor * channel_squared


class InternalCalibration:
    def __init__(self, priorities, units, active, numerical_floor=1e-12):
        keys = {f"{f}.{b}" for f, blocks in BLOCKS.items() for b in blocks}
        extra = {f"{f}.{b}" for f, blocks in EXTRA_BLOCKS.items() for b in blocks}
        if (
            set(priorities) not in (keys, keys | extra)
            or set(units) != set(priorities)
            or set(active) != set(priorities)
        ):
            raise ValueError("semantic block declarations must match the C8 or expanded contract")
        keys = set(priorities)
        if (
            not math.isfinite(float(numerical_floor))
            or numerical_floor <= 0
            or any(
                not math.isfinite(float(priorities[k]))
                or priorities[k] <= 0
                or not math.isfinite(float(units[k]))
                or units[k] <= 0
                for k in keys
            )
        ):
            raise ValueError("priorities, units and floor must be positive finite")
        self.priorities, self.units, self.active = dict(priorities), dict(units), dict(active)
        self.floor = float(numerical_floor)
        self.ema, self.records, self.coefficients = {}, [], None

    def observe(self, epoch, norms):
        if (
            self.coefficients is not None
            or epoch not in PROBES
            or (self.records and epoch <= self.records[-1]["epoch"])
        ):
            raise ValueError("probes must occur once in order before freeze")
        expected = {k for k, v in self.active.items() if v}
        if set(norms) != expected or any(
            not math.isfinite(float(v)) or v < 0 for v in norms.values()
        ):
            raise ValueError("norms must contain exactly active blocks and be finite nonnegative")
        for k, v in norms.items():
            self.ema[k] = float(v) if k not in self.ema else 0.8 * self.ema[k] + 0.2 * float(v)
        self.records.append({"epoch": epoch, "norms": dict(norms)})

    def freeze(self, completed_epoch):
        if completed_epoch != 10 or tuple(x["epoch"] for x in self.records) != PROBES:
            raise ValueError("freeze requires all five probes and completed epoch 10")
        result = {k: 1.0 for k in self.active}
        for family, blocks in BLOCKS.items():
            names = [f"{family}.{b}" for b in blocks if self.active[f"{family}.{b}"]]
            extras = [
                f"{family}.{b}"
                for b in EXTRA_BLOCKS.get(family, ())
                if self.active.get(f"{family}.{b}", False)
            ]
            positive = [self.ema[k] for k in names if self.ema[k] > self.floor]
            if not positive:
                continue
            median = statistics.median(positive)
            floor = max(self.floor, 0.1 * median)
            for k in names + extras:
                # Expanded blocks use the core reference, without changing it.
                # Unresolved near-zero blocks remain unamplified.
                if self.ema[k] >= floor:
                    result[k] = min(2.0, max(0.5, median / max(self.ema[k], floor)))
        self.coefficients = result
        return dict(result)

    def load_prefit(self, artifact):
        """Load a TRAIN-fitted common coefficient vector without online probes."""
        if artifact.get("version") != "common_prefit_calibration_v1":
            raise ValueError("invalid common prefit calibration version")
        if any(artifact.get(k) != getattr(self, k) for k in ("priorities", "units", "active")):
            raise ValueError("common prefit block declarations differ from runtime")
        norms = artifact.get("rms_gradient_norms", {})
        expected = {k for k, active in self.active.items() if active}
        if set(norms) != expected or any(not math.isfinite(float(v)) or float(v) < 0 for v in norms.values()):
            raise ValueError("common prefit requires finite RMS norms for active blocks")
        result = {k: 1.0 for k in self.active}
        for family, blocks in BLOCKS.items():
            names = [f"{family}.{b}" for b in blocks if self.active[f"{family}.{b}"]]
            extras = [f"{family}.{b}" for b in EXTRA_BLOCKS.get(family, ()) if self.active.get(f"{family}.{b}", False)]
            positive = [float(norms[k]) for k in names if float(norms[k]) > 0]
            if positive:
                median = statistics.median(positive)
                for k in names + extras:
                    if float(norms[k]) >= max(self.floor, 0.1 * median):
                        result[k] = min(2.0, max(0.5, median / float(norms[k])))
        ratio = artifact.get("scale_joint_ratio", 1)
        if ratio not in (1, 2, 4):
            raise ValueError("scale joint ratio must be 1, 2 or 4")
        result["scale.joint"] *= ratio
        if artifact.get("coefficients") != result:
            raise ValueError("common prefit coefficients do not match RMS fit")
        self.coefficients = result
        self.prefit = dict(artifact)

    def state_dict(self):
        if hasattr(self, "prefit"):
            return {"version": "internal_prefit_state_v1", "artifact": self.prefit}
        return {
            "version": VERSION,
            "priorities": self.priorities.copy(),
            "units": self.units.copy(),
            "active": self.active.copy(),
            "floor": self.floor,
            "ema": self.ema.copy(),
            "records": [{"epoch": r["epoch"], "norms": r["norms"].copy()} for r in self.records],
            "coefficients": None if self.coefficients is None else self.coefficients.copy(),
        }

    @classmethod
    def from_state_dict(cls, state):
        if state.get("version") == "internal_prefit_state_v1":
            artifact = state["artifact"]
            obj = cls(artifact["priorities"], artifact["units"], artifact["active"])
            obj.load_prefit(artifact)
            return obj
        if state["version"] != VERSION:
            raise ValueError("calibration version mismatch")
        obj = cls(state["priorities"], state["units"], state["active"], state["floor"])
        for record in state["records"]:
            obj.observe(record["epoch"], record["norms"])
        if obj.ema != state["ema"]:
            raise ValueError("inconsistent calibration EMA")
        if state["coefficients"] is not None and obj.freeze(10) != state["coefficients"]:
            raise ValueError("inconsistent frozen coefficients")
        return obj

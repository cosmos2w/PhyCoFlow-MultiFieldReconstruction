"""Opt-in source-matched, dimensionless endpoint primal-dual constraints.

Frozen training calibration is distinct from family gradient calibration and
validation checkpoint selection. No controller statistic is fitted on test or
validation data. The controller makes no pointwise or convergence guarantee.
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch

ENDPOINT_VERSION = "endpoint_primal_dual_v1"
NATIVE_VERSION = "endpoint_native_primal_dual_v2"

FIDELITY_DEFAULTS = {
    "enabled": True,
    "risk": "endpoint_model_mse",
    "relative_budget_total": 0.05,
    "relative_budget_per_field": 0.05,
    "absolute_floor": 1e-8,
    "absolute_allowance": 0.0,
    "source_weight_selection": "live",
    "same_noise_source": True,
    "calibration_batches": 8,
    "dual_lr": 0.05,
    "ema_decay": 0.9,
    "augmented_rho": 1.0,
    "multiplier_max": 100.0,
    "native_loss_role": "monitor",
    "anchor": {"enabled": False},
    "diagnostics_every_steps": 20,
}

FIDELITY_V2_DEFAULTS = {
    "version": NATIVE_VERSION,
    "native_loss_role": "constraint",
    "native_budget": 0.0,
    "calibration_seed": 700_103,
}


def fidelity_settings(config: Mapping[str, Any]) -> dict[str, Any]:
    supplied = dict(config.get("fidelity_controller", {}))
    version = supplied.get("version", ENDPOINT_VERSION)
    if version not in {ENDPOINT_VERSION, NATIVE_VERSION}:
        raise ValueError("unknown fidelity controller version")
    if version == ENDPOINT_VERSION:
        # Omitted version and explicit v1 have the historical serialized keys.
        supplied.pop("version", None)
        return {**FIDELITY_DEFAULTS, **supplied}
    return {**FIDELITY_DEFAULTS, **FIDELITY_V2_DEFAULTS, **supplied}


def _native_rng_state(model):
    cuda = any(value.device.type == "cuda" for value in (*model.parameters(), *model.buffers()))
    return (torch.get_rng_state(), torch.cuda.get_rng_state_all() if cuda else None,
            random.getstate(), np.random.get_state())


def _restore_native_rng(state):
    torch.set_rng_state(state[0])
    if state[1] is not None:
        torch.cuda.set_rng_state_all(state[1])
    random.setstate(state[2])
    np.random.set_state(state[3])


def matched_native_losses(model, source_model, batch, *, live_graph=True):
    """Replay the unchanged native objective without shifting the public RNG.

    Both models retain their caller-selected mode. Only LIVE's draws advance
    the visible Torch CPU/CUDA, Python and NumPy streams, including on a failed
    teacher evaluation. The supervised loss receives the usual training batch;
    this helper adds no conditioning inputs to generative reconstruction.
    """
    before = _native_rng_state(model)
    with torch.set_grad_enabled(live_graph):
        live = model.training_loss(batch).total
    after = _native_rng_state(model)
    try:
        _restore_native_rng(before)
        with torch.no_grad():
            source = source_model.training_loss(batch).total.detach()
    finally:
        _restore_native_rng(after)
    if live.ndim != 0 or source.ndim != 0 or not bool(
        torch.isfinite(live) and torch.isfinite(source)
    ):
        raise FloatingPointError("matched native loss must be a finite scalar")
    return live, source


def endpoint_risks(
    prediction: torch.Tensor, reference: torch.Tensor, mask: torch.Tensor | None = None
) -> torch.Tensor:
    """Return aggregate model-unit MSE per field, never per-sample ratios."""
    if prediction.ndim != 3 or prediction.shape != reference.shape:
        raise ValueError("endpoint risks require aligned [B,N,C] tensors")
    if mask is None:
        mask = torch.ones(prediction.shape[:2], device=prediction.device, dtype=torch.bool)
    if mask.shape != prediction.shape[:2] or not bool(mask.any(dim=1).all()):
        raise ValueError("each endpoint risk snapshot needs valid query points")
    errors = (prediction - reference.detach()).square()
    per_sample = (errors * mask[..., None]).sum(dim=1) / mask.sum(dim=1)[:, None]
    return per_sample.mean(dim=0)


def unobserved_endpoint_risks(prediction, reference, batch):
    """Exclude clamped sensor entries separately for each physical channel."""
    query_ids = batch.metadata.get("query_indices")
    if query_ids is None or batch.obs_indices is None:
        return None
    valid = batch.query_valid_mask[..., None].expand_as(prediction).clone()
    for b in range(prediction.shape[0]):
        observed = batch.obs_valid_mask[b]
        for channel in range(prediction.shape[-1]):
            indices = batch.obs_indices[b][observed & (batch.obs_field_ids[b] == channel)]
            if indices.numel():
                valid[b, :, channel] &= ~torch.isin(query_ids[b], indices.to(query_ids.device))
    errors = (prediction.detach() - reference.detach()).square()
    counts = valid.sum(dim=(0, 1))
    risks = (errors * valid).sum(dim=(0, 1)) / counts.clamp_min(1)
    return {"risks": risks, "counts": counts}


class FidelityController:
    """One augmented primal pressure; detached EMA/dual advance after a step.

    EMA starts at zero and is bias-corrected by 1-beta**updates. Multipliers
    start at zero. Only accepted updates advance either accumulator.
    """

    def __init__(self, field_names, settings: Mapping, calibration: Mapping, *, state=None):
        self.field_names = tuple(field_names)
        self.settings = fidelity_settings({"fidelity_controller": settings})
        self.version = self.settings.get("version", ENDPOINT_VERSION)
        self.native_constrained = self.version == NATIVE_VERSION
        expected_role = "constraint" if self.native_constrained else "monitor"
        if self.settings["native_loss_role"] != expected_role:
            raise ValueError("fidelity controller version/native_loss_role mismatch")
        if not self.native_constrained and "native_budget" in self.settings:
            raise ValueError("native_budget requires the native v2 controller")
        self.names = ("total", *self.field_names) + (("native",) if self.native_constrained else ())
        self.calibration = dict(calibration)
        if tuple(calibration["field_names"]) != self.field_names:
            raise ValueError("fidelity calibration field order mismatch")
        scales = list(calibration["source_risks"])
        if self.native_constrained:
            native = calibration.get("native")
            if not isinstance(native, Mapping) or native.get("version") != "native_source_calibration_v2":
                raise ValueError("native v2 controller requires frozen native calibration")
            if calibration.get("split") != "train" or native.get("split") != "train":
                raise ValueError("native calibration must use TRAIN only")
            if calibration.get("version") != "endpoint_native_source_calibration_v2":
                raise ValueError("fidelity calibration/controller version mismatch")
            if not math.isfinite(float(self.settings["native_budget"])) or float(self.settings["native_budget"]) < 0:
                raise ValueError("native budget must be finite and nonnegative")
            scales.append(native["source_scale"])
        elif "native" in calibration or calibration.get("version") == "endpoint_native_source_calibration_v2":
            raise ValueError("native v2 calibration cannot be read by the v1 controller")
        self.normalizers = torch.as_tensor(scales, dtype=torch.float64)
        if self.normalizers.shape != (len(self.names),) or not bool(
            torch.isfinite(self.normalizers).all() and (self.normalizers > 0).all()
        ):
            raise ValueError("fidelity calibration needs finite positive frozen risk scales")
        self.multipliers = torch.zeros_like(self.normalizers)
        self.ema = torch.zeros_like(self.normalizers)
        self.updates = 0
        if state is not None:
            if state.get("version") != self.version:
                raise ValueError("fidelity resume controller version mismatch")
            if state["calibration"] != self.calibration or state["settings"] != self.settings:
                raise ValueError("fidelity resume calibration/settings differ from the frozen state")
            if tuple(state["names"]) != self.names:
                raise ValueError("fidelity resume constraint order mismatch")
            self.multipliers.copy_(state["multipliers"].cpu())
            self.ema.copy_(state["ema"].cpu())
            self.updates = int(state["updates"])
            if self.updates < 0 or not torch.isfinite(self.ema).all() or not bool(
                (self.multipliers >= 0).all()
                and (self.multipliers <= float(self.settings["multiplier_max"])).all()
                and torch.isfinite(self.multipliers).all()
            ):
                raise ValueError("invalid recovered fidelity controller state")

    def violations(self, prediction, source_prediction, reference, mask=None, *,
                   native_loss=None, source_native_loss=None):
        live_fields = endpoint_risks(prediction, reference, mask)
        source_fields = endpoint_risks(source_prediction.detach(), reference, mask)
        live = torch.cat((live_fields.mean()[None], live_fields))
        source = torch.cat((source_fields.mean()[None], source_fields))
        budgets = live.new_tensor([self.settings["relative_budget_total"]] +
                                 [self.settings["relative_budget_per_field"]] * len(self.field_names))
        scales = self.normalizers[:len(live)].to(device=live.device, dtype=live.dtype)
        violations = (live - (1 + budgets) * source -
                      float(self.settings["absolute_allowance"])) / scales
        if self.native_constrained:
            if native_loss is None or source_native_loss is None:
                raise ValueError("native v2 constraint requires matched live/source native losses")
            if native_loss.ndim != 0 or source_native_loss.ndim != 0 or not native_loss.requires_grad:
                raise ValueError("native constraint requires a scalar LIVE autograd graph")
            source_native = source_native_loss.detach()
            native_scale = self.normalizers[-1].to(device=native_loss.device, dtype=native_loss.dtype)
            native_violation = ((native_loss - source_native) / native_scale
                                - float(self.settings["native_budget"]))
            violations = torch.cat((violations, native_violation.reshape(1)))
            live = torch.cat((live, native_loss.reshape(1)))
            source = torch.cat((source, source_native.reshape(1)))
        elif native_loss is not None or source_native_loss is not None:
            raise ValueError("native losses cannot enter the v1 monitor-only controller")
        if not torch.isfinite(violations).all():
            raise FloatingPointError("non-finite endpoint fidelity risks")
        return violations, live, source

    def primal(self, violations: torch.Tensor):
        if violations.shape != self.normalizers.shape:
            raise ValueError("fidelity primal constraint layout mismatch")
        rho = float(self.settings["augmented_rho"])
        multipliers = self.multipliers.to(device=violations.device, dtype=violations.dtype)
        pressure = (multipliers + rho * violations).clamp_min(0)
        terms = (pressure.square() - multipliers.square()) / (2 * rho)
        return terms.sum(), {name: term for name, term in zip(self.names, terms)}, pressure

    def advance(self, violations: torch.Tensor) -> None:
        values = violations.detach().to(device="cpu", dtype=torch.float64)
        if values.shape != self.ema.shape or not torch.isfinite(values).all():
            raise FloatingPointError("invalid detached fidelity controller update")
        beta = float(self.settings["ema_decay"])
        self.ema.mul_(beta).add_(values, alpha=1 - beta)
        self.updates += 1
        corrected = self.ema / (1 - beta**self.updates)
        self.multipliers.add_(corrected, alpha=float(self.settings["dual_lr"]))
        self.multipliers.clamp_(0, float(self.settings["multiplier_max"]))

    def telemetry(self, violations, live, source, pressure):
        corrected = self.ema / (1 - float(self.settings["ema_decay"])**self.updates) if self.updates else self.ema
        row = {}
        # Risks/pressure are pre-step; EMA and multipliers below are post-step.
        for i, name in enumerate(self.names):
            prefix = f"fidelity/{name}"
            row.update({f"{prefix}/source_risk": float(source[i].detach()),
                        f"{prefix}/live_risk": float(live[i].detach()),
                        f"{prefix}/violation": float(violations[i].detach()),
                        f"{prefix}/ema_violation": float(corrected[i]),
                        f"{prefix}/multiplier": float(self.multipliers[i]),
                        f"{prefix}/effective_primal_coefficient": float(pressure[i].detach()),
                        f"{prefix}/cap_saturated": bool(self.multipliers[i] >= float(self.settings["multiplier_max"]))})
        if self.native_constrained:
            row["gradient/native/constraint_active"] = bool(pressure[-1].detach() > 0)
            row["fidelity/native/frozen_scale"] = float(self.normalizers[-1])
            row["fidelity/native/budget"] = float(self.settings["native_budget"])
        return row

    def state_dict(self):
        return {"version": self.version, "names": list(self.names),
                "settings": self.settings, "calibration": self.calibration,
                "multipliers": self.multipliers.clone(), "ema": self.ema.clone(),
                "updates": self.updates}


def coherence_selection_report(source: Mapping, candidate: Mapping, settings: Mapping):
    """Frozen equal-family source ratios within total AND per-field gates."""
    from .topology_selection import fidelity_eligibility

    report = fidelity_eligibility(source, candidate, settings)
    source_families = source["coherence"]["families"]
    candidate_families = candidate["coherence"]["families"]
    # RunStore writes sorted JSON keys. A resumed source report therefore has
    # a different insertion order from an in-memory family result. Membership
    # is a contract; incidental mapping order is not part of this mean score.
    if set(source_families) != set(candidate_families):
        raise ValueError("coherence selection family membership changed")
    ratios = {name: float(candidate_families[name]["total"]) /
              max(float(value["total"]), 1e-12) for name, value in source_families.items()}
    score = sum(ratios.values()) / len(ratios)
    report.update(metric=score, mse=float(candidate["mse_normalized"]),
                  metrics=dict(candidate), family_source_normalized_scores=ratios,
                  definition="equal_active_family_mean_source_ratio_v1",
                  family_scale_floor=1e-12)
    report["eligible"] &= math.isfinite(score) and all(math.isfinite(v) for v in ratios.values())
    return report


def r3_coherence_selection_report(source: Mapping, candidate: Mapping, settings: Mapping):
    """Screen both R3 arms with raw A/B and the same TRAIN finite-primary C.

    Training C remains visible with its original semantics. This explicit
    prospective selector never changes the legacy R2 dispatch or gradients.
    """
    accounting = {}

    def with_selection_c(metrics: Mapping, role: str) -> dict:
        families = dict(metrics["coherence"]["families"])
        topology = families["topology"]
        components = topology.get("component_scalars", {})
        if "topology.finite_primary" not in components:
            raise ValueError("R3 selection requires TRAIN-calibrated topology.finite_primary")
        finite_primary = float(components["topology.finite_primary"])
        if not math.isfinite(finite_primary) or finite_primary < 0:
            raise ValueError("R3 finite-primary selection scalar must be finite and nonnegative")
        accounting[role] = {"training_total": float(topology["total"]),
                            "legacy": float(components.get("topology.legacy", topology["total"])),
                            "finite_primary": finite_primary}
        families["topology"] = {**topology, "total": finite_primary}
        return {**metrics, "coherence": {**metrics["coherence"], "families": families}}

    report = coherence_selection_report(with_selection_c(source, "source"),
                                       with_selection_c(candidate, "candidate"), settings)
    # Preserve the original training-definition metrics and publish the
    # selection accounting separately so no old topology score is relabelled.
    report["metrics"] = dict(candidate)
    report["definition"] = "equal_raw_A_B_finite_primary_C_mean_source_ratio_r3_v1"
    report["topology_selection_accounting"] = accounting
    report["topology_selection_representation"] = "training_raster_full16"
    report["selection_scales"] = "matched_SOURCE_raw_family_values_not_gradient_balance"
    return report

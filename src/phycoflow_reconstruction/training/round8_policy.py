"""Versioned, coordinator-independent Round 8 physical risk and deployment policy.

Arrays use [condition, member, field, site]; native masses are normalized per
condition before sampling. Coupling identities must describe the actual random
field/buffers and preprocessing, not merely an RNG seed.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
import math
import torch

VERSION = 'round8_physical_blend_v1'
FAMILIES = ('scale', 'shape', 'state')
MATERIALITY = {'scale': .01, 'shape': .05, 'state': .02}
PRESERVATION = {'scale': .005, 'shape': .02, 'state': .02}


def blend(source, endpoint, alpha, *, source_identity, endpoint_identity):
    """Blend already clamped, member-paired outputs with verified identities."""
    if source.shape != endpoint.shape or source.ndim != 4:
        raise ValueError('endpoints must have identical [condition,member,field,site] shapes')
    required = ('conditions', 'observations', 'latent', 'prior_buffers', 'solver', 'normalizer', 'support', 'clamp')
    if any(k not in source_identity or k not in endpoint_identity or source_identity[k] != endpoint_identity[k] for k in required):
        raise ValueError('paired endpoint coupling identity mismatch or incomplete')
    if not 0 <= alpha <= 1:
        raise ValueError('alpha must be in [0,1]')
    return source + alpha * (endpoint - source)


def physical_risk(prediction, target, weights, *, design='native', probability=None):
    """Member MSE using native, Horvitz-Thompson or with-replacement design.

    For sampled designs weights are the selected *native normalized* masses;
    probability contains selected inclusion/draw probabilities. Never renormalize
    sampled masses. Native target has shape [B,C,N] or [B,1,C,N].
    """
    if prediction.ndim != 4:
        raise ValueError('prediction must be [B,J,C,N]')
    if target.ndim == 3:
        target = target[:, None]
    if target.shape != (prediction.shape[0], 1, prediction.shape[2], prediction.shape[3]):
        raise ValueError('target shape mismatch')
    w = torch.as_tensor(weights, dtype=prediction.dtype, device=prediction.device)
    if w.ndim == 1:
        w = w.expand(prediction.shape[0], -1)
    if w.shape != (prediction.shape[0], prediction.shape[-1]) or not torch.isfinite(w).all() or (w < 0).any():
        raise ValueError('weights must be finite nonnegative [N] or [B,N]')
    if design == 'native':
        if not torch.allclose(w.sum(-1), torch.ones_like(w.sum(-1)), rtol=1e-5, atol=1e-7):
            raise ValueError('native weights must sum to one per condition')
        mass = w
    elif design in ('without_replacement', 'with_replacement'):
        p = torch.as_tensor(probability, dtype=w.dtype, device=w.device)
        p = torch.broadcast_to(p, w.shape)
        if not torch.isfinite(p).all() or (p <= 0).any() or (p > 1).any():
            raise ValueError('sampling probability must be in (0,1]')
        mass = w / p
        if design == 'with_replacement':
            mass = mass / prediction.shape[-1]
    else:
        raise ValueError('unknown sampling design')
    return (((prediction - target) ** 2) * mass[:, None, None, :]).sum(-1).mean((0, 1))


def quadratic(source, endpoint, target, weights, **risk_options):
    """Return R0, B, A under the same linear spatial estimator as physical_risk."""
    r0 = physical_risk(source, target, weights, **risk_options)
    displacement = endpoint - source
    zeros = torch.zeros_like(target)
    a = physical_risk(displacement, zeros, weights, **risk_options)
    # Polarization with elementwise multiplication avoids cancellation when
    # endpoint and source are close. Reuse the risk integration on positive and
    # negative parts so every sampling design gets the identical measure.
    reference = target[:, None] if target.ndim == 3 else target
    cross = (source - reference) * displacement
    positive = torch.sqrt(torch.clamp(cross, min=0))
    negative = torch.sqrt(torch.clamp(-cross, min=0))
    b = (physical_risk(positive, zeros, weights, **risk_options)
         - physical_risk(negative, zeros, weights, **risk_options))
    return r0, b, a


def alpha_cap(r0, b, a, allowance=.08):
    """Stable positive root, including zero risk and zero displacement."""
    r0, b, a = (torch.as_tensor(x, dtype=torch.float64) for x in (r0, b, a))
    if r0.shape != b.shape or b.shape != a.shape or not all(torch.isfinite(x).all() for x in (r0,b,a)) or (r0 < 0).any() or (a < 0).any() or allowance < 0:
        raise ValueError('invalid quadratic')
    caps = []
    for r, linear, curvature in zip(r0.flatten(), b.flatten(), a.flatten()):
        r, linear, curvature = float(r), float(linear), float(curvature)
        if curvature == 0:
            if linear > 0:  # Defensive handling of inconsistent supplied quadratics.
                caps.append(allowance * r / (2 * linear))
            continue
        root = math.hypot(linear, math.sqrt(curvature * allowance * r))
        if linear >= 0:
            cap = (allowance * r / (root + linear)) if root + linear else 0.0
        else:
            cap = (root - linear) / curvature
        caps.append(cap)
    cap = min(1.0, *caps) if caps else 1.0
    # Round the numerical boundary toward its feasible side. A root rounded
    # upward by one ulp otherwise fails the identical strict TRAIN guard.
    for _ in range(64):
        if bool(torch.all(r0 + 2 * cap * b + cap * cap * a <= (1 + allowance) * r0)):
            return cap
        cap = math.nextafter(cap, 0.0)
    raise FloatingPointError('risk boundary did not round to a feasible coefficient')


def controller_violation(risk, source_risk, field_scale, floor):
    if floor <= 0 or (field_scale <= 0).any():
        raise ValueError('positive scales and floor required')
    return (risk - 1.10 * source_risk) / torch.maximum(source_risk, field_scale.square() * floor)


@dataclass(frozen=True)
class DeploymentPolicy:
    alpha: float | None
    families: tuple[str, ...]
    version: str = VERSION


def select_train_policy(cap, source_risk, risks, source_scores, raw_scores, blend_scores, units):
    """TRAIN-only selection; score dictionaries are keyed by exact candidate alpha."""
    if not 0 <= cap <= 1:
        raise ValueError('cap must lie in [0,1]')
    if set(source_scores) != set(FAMILIES) or set(raw_scores) != set(FAMILIES) or set(units) != set(FAMILIES):
        raise ValueError('all three common family scores and units are required')
    if any(not math.isfinite(float(units[k])) or units[k] <= 0 for k in FAMILIES):
        raise ValueError('positive finite TRAIN-frozen score units required')
    for alpha in sorted({min(.25, cap), min(.5, cap)} - {0.0}, reverse=True):
        risk, scores = risks[alpha], blend_scores[alpha]
        if not bool(torch.all(torch.isfinite(risk) & (risk <= 1.08 * source_risk))):
            continue
        gains = {k: (source_scores[k] - scores[k]) / units[k] for k in FAMILIES}
        raw = {k: (source_scores[k] - raw_scores[k]) / units[k] for k in FAMILIES}
        if any(not math.isfinite(float(gains[k])) or not math.isfinite(float(raw[k])) for k in FAMILIES):
            raise ValueError('scores must be finite')
        if any(gains[k] < -PRESERVATION[k] for k in FAMILIES):
            continue
        qualified = tuple(k for k in FAMILIES if raw[k] > 0 and gains[k] >= MATERIALITY[k] and gains[k] >= .3 * raw[k])
        if qualified:
            return DeploymentPolicy(alpha, qualified)
    return DeploymentPolicy(None, ())

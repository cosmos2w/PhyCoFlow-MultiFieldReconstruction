"""Default-off execution settings for the bounded Run-11 acceleration study.

Independent counter-seeded streams make sampler decisions a pure function of
the accepted training cursor. They never consume rollout or data RNG state.
"""

import hashlib
from collections.abc import Mapping
from pathlib import Path

import torch


def validate_fast_lagrangian(config):
    opt = config["optimization"]
    if "fast_lagrangian" not in opt:
        return
    settings = opt["fast_lagrangian"]
    allowed = {"topology_estimator", "family_probability", "coefficient_interval", "seed"}
    if not isinstance(settings, Mapping) or set(settings) - allowed:
        raise ValueError("invalid optimization.fast_lagrangian settings")
    if settings.get("topology_estimator", "full") not in {"full", "p3", "s2"}:
        raise ValueError("topology_estimator must be full, p3, or s2")
    probability = settings.get("family_probability", 1.0)
    interval = settings.get("coefficient_interval", 1)
    seed = settings.get("seed", 700211)
    if isinstance(probability, bool) or probability not in (0.5, 1.0):
        raise ValueError("family_probability must be 0.5 or 1.0")
    if isinstance(interval, bool) or interval not in (1, 2, 4):
        raise ValueError("coefficient_interval must be 1, 2, or 4")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("fast_lagrangian seed must be a nonnegative integer")
    mt = opt.get("multitask", {})
    external = mt.get("external_policy", {})
    if (
        external.get("method") != "lagrangian"
        or external.get("lagrangian", {}).get("penalty_interval_updates", 1) != 1
    ):
        raise ValueError("fast_lagrangian requires dense Lagrangian fidelity")
    internal = "coherence_aggregation" in mt
    if probability != 1.0 and (internal or interval != 1):
        raise ValueError("family sampling requires scalar sum, without periodic ConFIG")
    if interval != 1 and not internal:
        raise ValueError("periodic coefficients require internal ConFIG")
    if mt.get("family_scalarization", {}).get("method", "none") not in {"none", "fixed_sum"}:
        raise ValueError("fast_lagrangian requires fixed family weights")
    if not internal and mt.get("family_scalarization", {}).get("method") != "fixed_sum":
        raise ValueError("scalar acceleration requires explicit fixed_sum")


def training_execution(config, step):
    """Return a training-only decision; calibration/evaluation never call this."""
    settings = config["optimization"].get("fast_lagrangian")
    if settings is None:
        return None
    seed = settings.get("seed", 700211)
    probability = settings.get("family_probability", 1.0)
    generator = torch.Generator(device="cpu").manual_seed(seed + 2_000_033 + step)
    included = probability == 1.0 or bool(torch.rand((), generator=generator) < probability)
    return {
        "estimator": settings.get("topology_estimator", "full"),
        "topology_included": included,
        "family_probability": probability,
        "topology_seed": seed + 3_000_017 + step,
        "accepted_cursor": step,
    }


def implementation_binding():
    """Bind recovery to the default-off estimator and coefficient implementation."""
    package = Path(__file__).resolve().parents[1]
    paths = [
        "training/fast_lagrangian.py",
        "training/periodic_config.py",
        "coherence/families/topology/filtration_sampling.py",
    ]
    return {path: hashlib.sha256((package / path).read_bytes()).hexdigest() for path in paths}

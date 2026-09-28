"""Fail early if a persistence run optimizes a different reconstruction operator."""

import torch

from .model_lifecycle import evaluation_weight_context
from .rollout import differentiable_reconstruction


def _endpoint_agreement(training, inference):
    """Compare each physical field without singular tolerances at zero crossings.

    Float32 grad/no-grad kernels need not round identically. Bound both the
    worst pixel error (0.01%) and RMS error (0.001%) relative to that field's
    reference RMS, rather than letting a large field mask a smaller one. An
    identically zero reference must match exactly. No physical-unit absolute
    tolerance is shared between fields.
    """
    if training.shape != inference.shape or training.ndim != 3:
        raise ValueError("rollout endpoints must have matching [batch, points, fields] shapes")
    actual = training.detach().double()
    reference = inference.detach().double()
    difference = actual - reference
    scale = reference.square().mean((0, 1)).sqrt()
    maximum = difference.abs().amax((0, 1))
    rms = difference.square().mean((0, 1)).sqrt()
    max_rtol, rms_rtol = 1e-4, 1e-5
    finite = torch.isfinite(actual).all() & torch.isfinite(reference).all()
    matches = bool(finite & (maximum <= max_rtol * scale).all() & (rms <= rms_rtol * scale).all())
    denominator = scale.clamp_min(torch.finfo(torch.float64).tiny)
    return {
        "passed": matches,
        "comparison": "per_field_reference_rms",
        "max_error_rtol": max_rtol,
        "rms_error_rtol": rms_rtol,
        "reference_rms_per_field": scale.cpu().tolist(),
        "max_abs_error_per_field": maximum.cpu().tolist(),
        "rms_error_per_field": rms.cpu().tolist(),
        "relative_max_error_per_field": (maximum / denominator).cpu().tolist(),
        "relative_rms_error_per_field": (rms / denominator).cpu().tolist(),
        "pointwise_allclose_legacy": torch.allclose(actual, reference, rtol=1e-4, atol=2e-6),
    }


def verify_rollout_contract(model, batch, config):
    """Compare live differentiable and production endpoints with identical noise.

    This is deliberately performed with autograd enabled on the training path,
    including activation checkpointing, and without dense targets in either path.
    The caller supplies a small fixed validation batch. No optimizer update occurs.
    """
    if batch.target_fields is not None:
        raise ValueError("rollout contract checks must not expose targets to the model")
    if any(
        getattr(module, "_ema", None) is not None and getattr(module, "_ema_eval", False)
        for module in model.modules()
    ):
        raise ValueError(
            "persistence training requires live evaluation weights; set model_ema_eval=false"
        )
    model.eval()
    steps = int(config["rollout"]["steps"])
    if steps != int(config["evaluation"]["generation_steps"]):
        raise ValueError("persistence training/evaluation generation steps must match")
    device = batch.query_coords.device
    seed = int(config["evaluation"].get("seed", 2027))
    devices = [device.index or 0] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        training = differentiable_reconstruction(
            model,
            batch,
            steps=steps,
            solver=config["rollout"]["solver"],
            generator=torch.Generator(device=device).manual_seed(seed),
            observation_config=dict(config["observation_consistency"]),
        )
        with evaluation_weight_context(model), torch.no_grad():
            inference = model.reconstruct(
                batch, steps=steps, generator=torch.Generator(device=device).manual_seed(seed)
            ).prediction
        report = {
            **_endpoint_agreement(training, inference),
            "seed": seed,
            "steps": steps,
            "model_mode": "eval",
            "evaluation_weights": "live",
            "training_path_has_autograd": training.requires_grad,
        }
    if not report["passed"]:
        raise ValueError(
            f"training/inference rollout mismatch: {report}; check source solver, observation settings, and per-field numerical errors"
        )
    return report

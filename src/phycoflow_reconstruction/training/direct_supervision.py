"""Fixed native/SSS objective and transactional finite field action."""
from copy import deepcopy
import math
import torch

FAMILY_ORDER = ("global_distribution", "cross_spectrum", "topology")
LOSS_ORDER = ("native", "state", "scale", "shape")


def direct_scalar(native, families, units, scales, modules):
    result = native / float(units['native'])
    from ..coherence.compose import three_family_order
    for name, unit in zip(three_family_order(families), LOSS_ORDER[1:]):
        result = result + (families[name] * float(scales[name]) *
                           float(getattr(modules[name], 'family_weight', 1.0)) /
                           float(units[unit]) / 3.0)
    return result


def finite_field_action(model, optimizer, loss, pre_excess, source_mse, tolerance,
                        evaluate_excess, grad_clip=None, dot_tolerance=0.0):
    """One AdamW proposal, optionally shrink its full displacement, or roll back.

    ``evaluate_excess`` must replay the identical paired samples under no_grad.
    The caller owns RNG/sampler consumption and performs EMA only on acceptance.
    """
    if not torch.isfinite(pre_excess).all():
        raise FloatingPointError("nonfinite pre-action field risk")
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    gradients = [None if p.grad is None else p.grad.detach().clone() for p in parameters]
    if any(g is not None and not torch.isfinite(g).all() for g in gradients):
        raise FloatingPointError('nonfinite direct objective gradient')
    gradient_norm = math.sqrt(sum(float(g.double().square().sum()) for g in gradients if g is not None))
    clip_coefficient = 1.0
    if grad_clip is not None:
        torch.nn.utils.clip_grad_norm_(parameters, float(grad_clip))
        clip_coefficient = min(1.0, float(grad_clip) / (gradient_norm + 1e-6))
    old_parameters = [p.detach().clone() for p in parameters]
    old_optimizer = deepcopy(optimizer.state_dict())
    accepted = False
    post_checks = 0
    try:
        optimizer.step()
        displacement = [p.detach().clone() - old for p, old in zip(parameters, old_parameters)]
        dot = sum(float((g.double() * d.double()).sum()) for g, d in zip(gradients, displacement) if g is not None)
        norm2 = sum(float(d.double().square().sum()) for d in displacement)
        if not math.isfinite(dot) or not math.isfinite(norm2):
            optimizer.load_state_dict(old_optimizer)
            with torch.no_grad():
                for p, old in zip(parameters, old_parameters): p.copy_(old)
            raise FloatingPointError('nonfinite AdamW proposal')
        scales = torch.as_tensor(source_mse, device=pre_excess.device, dtype=pre_excess.dtype).clamp_min(1e-8)
        tolerance = torch.as_tensor(tolerance, device=pre_excess.device, dtype=pre_excess.dtype)
        allowance = 0.1 * torch.as_tensor(source_mse, device=pre_excess.device, dtype=pre_excess.dtype)
        before = pre_excess.detach()
        violation = ((before - allowance).clamp_min(0) / scales).sum()
        violation_tolerance = (tolerance / scales).sum()
        alpha = 0.0
        after = before
        reason = 'zero_action' if norm2 == 0 else 'ascending_action'
        if norm2 and dot < -float(dot_tolerance):
            reason = 'field_filter'
            for trial in (1.0, 0.5):
                with torch.no_grad():
                    for p, old, delta in zip(parameters, old_parameters, displacement):
                        p.copy_(old + trial * delta)
                    post_checks += 1
                    after = evaluate_excess().detach()
                if not torch.isfinite(after).all():
                    with torch.no_grad():
                        for p, old in zip(parameters, old_parameters): p.copy_(old)
                    optimizer.load_state_dict(old_optimizer)
                    raise FloatingPointError('nonfinite post-action field risk')
                feasible = bool(torch.all(after <= torch.maximum(allowance, before) + tolerance))
                restored = bool(violation <= violation_tolerance or
                                ((after - allowance).clamp_min(0) / scales).sum() < violation - violation_tolerance)
                if feasible and restored:
                    accepted, alpha, reason = True, trial, 'accepted'
                    break
    finally:
        if not accepted:
            with torch.no_grad():
                for p, old in zip(parameters, old_parameters): p.copy_(old)
            optimizer.load_state_dict(old_optimizer)
    return {'update_accepted': accepted, 'action_alpha': alpha, 'action_dot': dot,
            'action_norm': math.sqrt(norm2), 'action_reason': reason,
            'model_grad_norm': gradient_norm, 'gradient_clip_coefficient': clip_coefficient,
            'field_post_ensemble_checks': post_checks,
            'field_excess_pre': before.cpu().tolist(), 'field_excess_post': after.cpu().tolist()}

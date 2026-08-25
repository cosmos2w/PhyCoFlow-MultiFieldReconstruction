import warnings
from typing import Optional, Sequence

import torch


OBS_CONSISTENCY_MODES = ("none", "default_hard", "endpoint", "endpoint_smooth")


def normalize_obs_consistency_mode(mode: str) -> str:
    mode = str(mode or "default_hard").strip().lower()
    if mode == "hard":
        warnings.warn(
            "obs_consistency_mode='hard' is deprecated; use 'default_hard'.",
            DeprecationWarning,
            stacklevel=2,
        )
        return "default_hard"
    if mode not in OBS_CONSISTENCY_MODES:
        raise ValueError(
            f"Unknown obs_consistency_mode={mode!r}. "
            f"Expected one of {OBS_CONSISTENCY_MODES}."
        )
    return mode


def _obs_values_2d(obs_values: torch.Tensor) -> torch.Tensor:
    return obs_values[..., 0] if obs_values.ndim == 3 else obs_values


def scatter_observed_values(
    x: torch.Tensor,
    obs_values: torch.Tensor,
    obs_mask: torch.Tensor,
    obs_indices: torch.Tensor,
    obs_field_ids: torch.Tensor,
    strength: float = 1.0,
) -> torch.Tensor:
    """
    Return a cloned tensor with observed point/channel entries blended in.

    SenConsis means sensor consistency between generated values and sparse
    observed values. This helper performs the pointwise sensor replacement used
    by the default hard mode and by the final trusted-sensor clamp.
    """
    out = x.clone()
    values = _obs_values_2d(obs_values).to(dtype=out.dtype)
    strength = float(strength)
    bsz = out.shape[0]
    for b in range(bsz):
        valid = obs_mask[b].bool()
        if not torch.any(valid):
            continue
        idx = obs_indices[b, valid].long()
        fld = obs_field_ids[b, valid].long()
        val = values[b, valid]
        out[b, idx, fld] = (1.0 - strength) * out[b, idx, fld] + strength * val
    return out


def build_pointwise_observation_maps(
    coords: torch.Tensor,
    obs_values: torch.Tensor,
    obs_mask: torch.Tensor,
    obs_indices: torch.Tensor,
    obs_field_ids: torch.Tensor,
    n_fields: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    value_map = torch.zeros(
        coords.shape[0], coords.shape[1], int(n_fields),
        device=coords.device, dtype=obs_values.dtype,
    )
    mask_map = torch.zeros_like(value_map)
    values = _obs_values_2d(obs_values)
    for b in range(coords.shape[0]):
        valid = obs_mask[b].bool()
        if not torch.any(valid):
            continue
        idx = obs_indices[b, valid].long()
        fld = obs_field_ids[b, valid].long()
        value_map[b, idx, fld] = values[b, valid]
        mask_map[b, idx, fld] = 1.0
    return value_map, mask_map


def build_smooth_observation_maps(
    coords: torch.Tensor,
    obs_coords: torch.Tensor,
    obs_values: torch.Tensor,
    obs_mask: torch.Tensor,
    obs_field_ids: torch.Tensor,
    n_fields: int,
    sigma: float,
    chunk_size: int = 8192,
    mask_mode: str = "max",
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build Gaussian/RBF sensor value and mask maps on query coordinates.

    The computation is chunked over query points and stays on the input device.
    Fields without sensors keep zero masks and therefore do not guide sampling.
    """
    if sigma <= 0:
        raise ValueError(f"obs_consistency_sigma must be positive, got {sigma}.")
    if mask_mode not in ("max", "sum"):
        raise ValueError("mask_mode must be 'max' or 'sum'.")

    bsz, n_pts, _ = coords.shape
    n_fields = int(n_fields)
    value_map = torch.zeros(bsz, n_pts, n_fields, device=coords.device, dtype=obs_values.dtype)
    mask_map = torch.zeros_like(value_map)
    values = _obs_values_2d(obs_values).to(dtype=coords.dtype)
    sigma2 = float(sigma) ** 2
    chunk_size = max(1, int(chunk_size))

    with torch.no_grad():
        for b in range(bsz):
            for c in range(n_fields):
                valid = obs_mask[b].bool() & (obs_field_ids[b].long() == c)
                if not torch.any(valid):
                    continue
                ocoords = obs_coords[b, valid].to(dtype=coords.dtype)
                ovalues = values[b, valid].to(dtype=coords.dtype)
                for start in range(0, n_pts, chunk_size):
                    end = min(start + chunk_size, n_pts)
                    q = coords[b, start:end].to(dtype=coords.dtype)
                    d2 = torch.cdist(q, ocoords, p=2.0).pow(2)
                    weights = torch.exp(-d2 / (2.0 * sigma2))
                    wsum = weights.sum(dim=1)
                    value_map[b, start:end, c] = (weights @ ovalues) / wsum.clamp_min(eps)
                    if mask_mode == "sum":
                        mask_map[b, start:end, c] = wsum.clamp(max=1.0)
                    else:
                        mask_map[b, start:end, c] = weights.max(dim=1).values
    return value_map, mask_map


def apply_endpoint_observation_consistency(
    x_t: torch.Tensor,
    v: torch.Tensor,
    t: torch.Tensor,
    value_map: torch.Tensor,
    mask_map: torch.Tensor,
    strength: float = 1.0,
    schedule_power: float = 2.0,
    eps: float = 1e-5,
) -> torch.Tensor:
    """
    Rectified-flow clean-endpoint observation masking.

    RF predicts v_theta(x_t, t, obs), so the clean endpoint estimate is
    x1_hat = x_t + (1 - t) * v. We mask that endpoint toward sensor values and
    convert the consistent endpoint back into a guided velocity.
    """
    if t.ndim == 0:
        t = t.expand(x_t.shape[0])
    tau = (1.0 - t).to(device=x_t.device, dtype=x_t.dtype).clamp_min(eps).view(-1, 1, 1)
    gamma = float(strength) * tau.pow(float(schedule_power))
    guide = gamma * mask_map.to(dtype=x_t.dtype)
    x1_hat = x_t + tau * v
    x1_hat_consistent = x1_hat * (1.0 - guide) + value_map.to(dtype=x_t.dtype) * guide
    return (x1_hat_consistent - x_t) / tau


def observation_consistency_metrics(
    recon: torch.Tensor,
    obs_values: torch.Tensor,
    obs_mask: torch.Tensor,
    obs_indices: torch.Tensor,
    obs_field_ids: torch.Tensor,
    field_names: Optional[Sequence[str]] = None,
    eps: float = 1e-12,
) -> dict:
    """Compute relative L2 SenConsis metrics only."""
    values = _obs_values_2d(obs_values).to(device=recon.device, dtype=recon.dtype)
    obs_mask = obs_mask.to(device=recon.device)
    obs_indices = obs_indices.to(device=recon.device)
    obs_field_ids = obs_field_ids.to(device=recon.device)

    bsz, n_obs = obs_mask.shape
    batch_idx = torch.arange(bsz, device=recon.device).view(-1, 1).expand(bsz, n_obs)
    gathered = recon[batch_idx, obs_indices.long(), obs_field_ids.long().clamp_min(0)]
    valid = obs_mask.bool() & (obs_field_ids >= 0)

    metrics = {"obs_count_SenConsis_total": int(valid.sum().item())}
    if torch.any(valid):
        diff = gathered[valid] - values[valid]
        ref = values[valid]
        metrics["obs_rel_l2_SenConsis"] = float(
            torch.linalg.vector_norm(diff) / (torch.linalg.vector_norm(ref) + eps)
        )
    else:
        metrics["obs_rel_l2_SenConsis"] = float("nan")

    n_fields = recon.shape[-1]
    if field_names is None:
        field_names = [f"field_{i}" for i in range(n_fields)]
    for c in range(n_fields):
        name = str(field_names[c]) if c < len(field_names) else f"field_{c}"
        fvalid = valid & (obs_field_ids.long() == c)
        metrics[f"obs_count_SenConsis_{name}"] = int(fvalid.sum().item())
        if torch.any(fvalid):
            diff = gathered[fvalid] - values[fvalid]
            ref = values[fvalid]
            metrics[f"obs_rel_l2_SenConsis_{name}"] = float(
                torch.linalg.vector_norm(diff) / (torch.linalg.vector_norm(ref) + eps)
            )
        else:
            metrics[f"obs_rel_l2_SenConsis_{name}"] = float("nan")
    return metrics

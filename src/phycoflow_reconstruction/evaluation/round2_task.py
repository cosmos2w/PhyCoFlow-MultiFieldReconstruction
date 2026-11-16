"""Frozen regional signed covariance used for the Round 2 final assessment."""

import hashlib

import torch

from ..coherence.measures import measure_fingerprint


def regional_covariance(fields: torch.Tensor, physical_masses: torch.Tensor, manifest: dict,
                        field_names: tuple[str, ...]) -> torch.Tensor:
    """Return one dimensionless covariance per sample in physical field units.

    Inputs must use the manifest's native support and physical mass definition;
    the manifest is fitted once on TRAIN truth and never on model predictions.
    """
    if manifest.get("schema") != "car_round2_reserved_task_v1":
        raise ValueError("unsupported frozen regional task manifest")
    if fields.ndim != 3 or fields.shape[-1] != len(field_names):
        raise ValueError("fields must be [batch, native points, named fields]")
    receipt = manifest["native_measure"]
    fingerprint = measure_fingerprint(physical_masses, metadata=receipt["metadata"])
    if fingerprint["sha256"] != receipt["sha256"]:
        raise ValueError("native physical masses do not match the frozen task measure")
    masses = torch.as_tensor(physical_masses, device=fields.device, dtype=fields.dtype)
    if masses.ndim != 1 or masses.shape[0] != fields.shape[1]:
        raise ValueError("native physical masses must align with field points")
    ids = torch.as_tensor(manifest["region_point_indices"], device=fields.device, dtype=torch.long)
    index_hash = hashlib.sha256(ids.cpu().numpy().astype("<i8", copy=False).tobytes()).hexdigest()
    if index_hash != manifest["region_indices_sha256"]:
        raise ValueError("frozen task region index hash mismatch")
    if ids.numel() == 0 or torch.any(ids < 0) or torch.any(ids >= fields.shape[1]):
        raise ValueError("frozen task region is empty or outside native support")
    weights = masses[ids]
    if not torch.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("task region must contain positive finite physical masses")
    weights = weights / weights.sum()
    selected = fields[:, ids][:, :, [field_names.index(name) for name in manifest["field_names"]]]
    centered = selected - (selected * weights[None, :, None]).sum(dim=1, keepdim=True)
    scales = torch.as_tensor(manifest["truth_scales"], device=fields.device, dtype=fields.dtype)
    if scales.shape != (2,) or not torch.isfinite(scales).all() or (scales <= 0).any():
        raise ValueError("frozen task scales must be positive and finite")
    return (centered[..., 0] * centered[..., 1] * weights).sum(dim=1) / scales.prod()

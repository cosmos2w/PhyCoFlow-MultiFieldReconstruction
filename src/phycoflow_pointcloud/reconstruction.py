"""Tensor-level reconstruction helpers independent of any dataset or loader."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Protocol

import torch


class ReconstructionModel(Protocol):
    """Minimal sampling protocol implemented by :class:`PointCloudFFM`."""

    def sample(self, **kwargs: Any) -> torch.Tensor: ...


@dataclass(frozen=True)
class ReconstructionConfig:
    """Model-generic solver and cache choices for one reconstruction call."""

    n_steps: int = 4
    ode_solver: str = "euler"
    obs_consistency_mode: str = "endpoint_smooth"
    obs_consistency_strength: float = 1.0
    obs_consistency_sigma: float = 0.05
    obs_consistency_schedule_power: float = 2.0
    obs_consistency_final_clamp: bool = True
    obs_consistency_chunk_size: int = 8192
    execution_mode: str = "cached_streamed"
    query_chunk_size: int = 8192
    cache_level: str = "static_features"


@torch.no_grad()
def reconstruct_from_tensors(
    model: ReconstructionModel,
    *,
    coords: torch.Tensor,
    obs_coords: torch.Tensor,
    obs_values: torch.Tensor,
    obs_mask: torch.Tensor,
    obs_field_ids: torch.Tensor,
    obs_indices: Optional[torch.Tensor] = None,
    config: ReconstructionConfig | None = None,
    geometry_cache: Any = None,
) -> torch.Tensor:
    """Reconstruct from the portable tensor contract.

    Dataset loading, normalization, field naming, sensor selection, and output
    serialization are intentionally owned by the downstream application.
    """
    cfg = config or ReconstructionConfig()
    return model.sample(
        coords=coords,
        obs_coords=obs_coords,
        obs_values=obs_values,
        obs_mask=obs_mask,
        obs_field_ids=obs_field_ids,
        clamp_indices=obs_indices,
        n_steps=cfg.n_steps,
        ode_solver=cfg.ode_solver,
        obs_consistency_mode=cfg.obs_consistency_mode,
        obs_consistency_strength=cfg.obs_consistency_strength,
        obs_consistency_sigma=cfg.obs_consistency_sigma,
        obs_consistency_schedule_power=cfg.obs_consistency_schedule_power,
        obs_consistency_final_clamp=cfg.obs_consistency_final_clamp,
        obs_consistency_chunk_size=cfg.obs_consistency_chunk_size,
        reconstruction_execution_mode=cfg.execution_mode,
        reconstruction_query_chunk_size=cfg.query_chunk_size,
        reconstruction_cache_level=cfg.cache_level,
        reconstruction_geometry_cache=geometry_cache,
    )


__all__ = [
    "ReconstructionConfig",
    "ReconstructionModel",
    "reconstruct_from_tensors",
]

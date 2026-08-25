"""Geometry-only persistent Top-K cache utilities for GL-RBF / CQ backbones.

This helper deliberately delegates neighbor search to the backbone's existing
`_get_topk_neighbors` implementation so Torch/KeOps semantics remain unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch


@dataclass(frozen=True)
class PersistentTopKGeometryCache:
    topk_d2: torch.Tensor
    topk_idx: torch.Tensor
    topk_valid: torch.Tensor

    n_query: int
    n_obs_slots: int
    k: int
    gather_mode: str

    coords_shape: tuple[int, ...]
    obs_coords_shape: tuple[int, ...]
    obs_mask_shape: tuple[int, ...]

    coords_data_ptr: int
    obs_coords_data_ptr: int
    obs_mask_data_ptr: int

    coords_version: int
    obs_coords_version: int
    obs_mask_version: int
    coords_stride: tuple[int, ...]
    obs_coords_stride: tuple[int, ...]
    obs_mask_stride: tuple[int, ...]
    coords_storage_offset: int
    obs_coords_storage_offset: int
    obs_mask_storage_offset: int

    coords_device: str
    obs_coords_device: str
    obs_mask_device: str
    coords_dtype: str
    obs_coords_dtype: str
    obs_mask_dtype: str

    def nbytes(self) -> int:
        total = 0
        seen: set[tuple[int, int]] = set()
        for tensor in (self.topk_d2, self.topk_idx, self.topk_valid):
            storage = tensor.untyped_storage()
            key = (storage.data_ptr(), storage.nbytes())
            if key not in seen:
                seen.add(key)
                total += storage.nbytes()
        return total

    def as_mapping(self) -> dict[str, Any]:
        return {
            "topk_d2": self.topk_d2,
            "topk_idx": self.topk_idx,
            "topk_valid": self.topk_valid,
            "n_query": self.n_query,
            "n_obs_slots": self.n_obs_slots,
            "k": self.k,
            "gather_mode": self.gather_mode,
            "coords_shape": self.coords_shape,
            "obs_coords_shape": self.obs_coords_shape,
            "obs_mask_shape": self.obs_mask_shape,
            "coords_data_ptr": self.coords_data_ptr,
            "obs_coords_data_ptr": self.obs_coords_data_ptr,
            "obs_mask_data_ptr": self.obs_mask_data_ptr,
            "coords_version": self.coords_version,
            "obs_coords_version": self.obs_coords_version,
            "obs_mask_version": self.obs_mask_version,
            "coords_stride": self.coords_stride,
            "obs_coords_stride": self.obs_coords_stride,
            "obs_mask_stride": self.obs_mask_stride,
            "coords_storage_offset": self.coords_storage_offset,
            "obs_coords_storage_offset": self.obs_coords_storage_offset,
            "obs_mask_storage_offset": self.obs_mask_storage_offset,
            "coords_device": self.coords_device,
            "obs_coords_device": self.obs_coords_device,
            "obs_mask_device": self.obs_mask_device,
            "coords_dtype": self.coords_dtype,
            "obs_coords_dtype": self.obs_coords_dtype,
            "obs_mask_dtype": self.obs_mask_dtype,
        }


def _ptr(tensor: torch.Tensor) -> int:
    return int(tensor.untyped_storage().data_ptr())


def _version(tensor: torch.Tensor) -> int:
    return int(tensor._version)


@torch.no_grad()
def build_persistent_topk_geometry_cache(
    backbone: Any,
    *,
    coords: torch.Tensor,
    obs_coords: torch.Tensor,
    obs_mask: torch.Tensor,
    chunk_size: int = 8192,
) -> PersistentTopKGeometryCache:
    """Build Top-K geometry using the backbone's existing neighbor backend."""

    if coords.device != obs_coords.device or coords.device != obs_mask.device:
        raise ValueError("Query coordinates, sensor coordinates, and mask must share a device.")
    if coords.dtype != obs_coords.dtype:
        raise ValueError("Query and sensor coordinates must share a dtype.")
    if coords.ndim != 3 or obs_coords.ndim != 3 or obs_mask.ndim != 2:
        raise ValueError("Expected coords [B,N,D], obs_coords [B,M,D], obs_mask [B,M].")
    if coords.shape[0] != obs_coords.shape[0] or obs_coords.shape[:2] != obs_mask.shape:
        raise ValueError("Geometry batch/sensor dimensions do not match the mask.")

    if not hasattr(backbone, "_get_topk_neighbors"):
        raise TypeError(
            "Backbone must expose _get_topk_neighbors; adapt this helper only if "
            "the current CQ implementation renamed the existing neighbor primitive."
        )

    gather_mode = str(getattr(backbone, "gather_mode", ""))
    if gather_mode not in {"topk_rbf", "topk_rbf_glres"}:
        raise ValueError(
            "Persistent geometry caching currently supports topk_rbf and "
            f"topk_rbf_glres, got {gather_mode!r}."
        )

    gather_topk = int(getattr(backbone, "gather_topk"))
    k = min(gather_topk, int(obs_coords.shape[1]))
    chunk_size = max(1, int(chunk_size))

    topk_d2 = coords.new_empty(coords.shape[0], coords.shape[1], k)
    topk_idx = torch.empty(
        coords.shape[0],
        coords.shape[1],
        k,
        dtype=torch.long,
        device=coords.device,
    )
    topk_valid = torch.empty(
        coords.shape[0],
        coords.shape[1],
        k,
        dtype=torch.bool,
        device=coords.device,
    )

    # The historical unified KNN helper also gathers a feature tensor. Geometry
    # itself is independent of refined features, so a one-channel dummy tensor
    # avoids coupling this cache to observation values or latent state.
    dummy_feat = obs_coords[..., :1].contiguous()

    for start in range(0, int(coords.shape[1]), chunk_size):
        end = min(start + chunk_size, int(coords.shape[1]))
        d2, idx, _, _, valid = backbone._get_topk_neighbors(
            query_coords=coords[:, start:end],
            obs_coords=obs_coords,
            refined_sensor_feat=dummy_feat,
            obs_mask=obs_mask,
            k=k,
        )
        topk_d2[:, start:end] = d2
        topk_idx[:, start:end] = idx
        topk_valid[:, start:end] = valid

    return PersistentTopKGeometryCache(
        topk_d2=topk_d2,
        topk_idx=topk_idx,
        topk_valid=topk_valid,
        n_query=int(coords.shape[1]),
        n_obs_slots=int(obs_coords.shape[1]),
        k=k,
        gather_mode=gather_mode,
        coords_shape=tuple(int(v) for v in coords.shape),
        obs_coords_shape=tuple(int(v) for v in obs_coords.shape),
        obs_mask_shape=tuple(int(v) for v in obs_mask.shape),
        coords_data_ptr=_ptr(coords),
        obs_coords_data_ptr=_ptr(obs_coords),
        obs_mask_data_ptr=_ptr(obs_mask),
        coords_version=_version(coords),
        obs_coords_version=_version(obs_coords),
        obs_mask_version=_version(obs_mask),
        coords_stride=tuple(int(v) for v in coords.stride()),
        obs_coords_stride=tuple(int(v) for v in obs_coords.stride()),
        obs_mask_stride=tuple(int(v) for v in obs_mask.stride()),
        coords_storage_offset=int(coords.storage_offset()),
        obs_coords_storage_offset=int(obs_coords.storage_offset()),
        obs_mask_storage_offset=int(obs_mask.storage_offset()),
        coords_device=str(coords.device),
        obs_coords_device=str(obs_coords.device),
        obs_mask_device=str(obs_mask.device),
        coords_dtype=str(coords.dtype),
        obs_coords_dtype=str(obs_coords.dtype),
        obs_mask_dtype=str(obs_mask.dtype),
    )


def validate_persistent_topk_geometry_cache(
    cache: PersistentTopKGeometryCache | Mapping[str, Any],
    backbone: Any,
    *,
    coords: torch.Tensor,
    obs_coords: torch.Tensor,
    obs_mask: torch.Tensor,
) -> None:
    """Fail fast if a cache no longer matches the live query/sensor geometry."""

    c = cache.as_mapping() if isinstance(cache, PersistentTopKGeometryCache) else cache

    gather_mode = str(getattr(backbone, "gather_mode", ""))
    gather_topk = int(getattr(backbone, "gather_topk"))
    k = min(gather_topk, int(obs_coords.shape[1]))

    expected = {
        "n_query": int(coords.shape[1]),
        "n_obs_slots": int(obs_coords.shape[1]),
        "k": k,
        "gather_mode": gather_mode,
        "coords_shape": tuple(int(v) for v in coords.shape),
        "obs_coords_shape": tuple(int(v) for v in obs_coords.shape),
        "obs_mask_shape": tuple(int(v) for v in obs_mask.shape),
        "coords_data_ptr": _ptr(coords),
        "obs_coords_data_ptr": _ptr(obs_coords),
        "obs_mask_data_ptr": _ptr(obs_mask),
        "coords_version": _version(coords),
        "obs_coords_version": _version(obs_coords),
        "obs_mask_version": _version(obs_mask),
        "coords_stride": tuple(int(v) for v in coords.stride()),
        "obs_coords_stride": tuple(int(v) for v in obs_coords.stride()),
        "obs_mask_stride": tuple(int(v) for v in obs_mask.stride()),
        "coords_storage_offset": int(coords.storage_offset()),
        "obs_coords_storage_offset": int(obs_coords.storage_offset()),
        "obs_mask_storage_offset": int(obs_mask.storage_offset()),
        "coords_device": str(coords.device),
        "obs_coords_device": str(obs_coords.device),
        "obs_mask_device": str(obs_mask.device),
        "coords_dtype": str(coords.dtype),
        "obs_coords_dtype": str(obs_coords.dtype),
        "obs_mask_dtype": str(obs_mask.dtype),
    }

    for key, value in expected.items():
        actual = c.get(key)
        if actual != value:
            raise ValueError(
                "Persistent Top-K geometry cache is stale/incompatible: "
                f"{key}: cache={actual!r}, current={value!r}"
            )

    for tensor_key in ("topk_d2", "topk_idx", "topk_valid"):
        if tensor_key not in c:
            raise ValueError(f"Persistent geometry cache missing {tensor_key!r}.")

    expected_tensor_shape = (int(coords.shape[0]), int(coords.shape[1]), k)
    topk_d2, topk_idx, topk_valid = cache_tensors(cache)
    tensor_checks = (
        ("topk_d2", topk_d2, expected_tensor_shape, coords.dtype),
        ("topk_idx", topk_idx, expected_tensor_shape, torch.long),
        ("topk_valid", topk_valid, expected_tensor_shape, torch.bool),
    )
    for name, tensor, shape, dtype in tensor_checks:
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"Persistent geometry cache {name} is not a tensor.")
        if tuple(tensor.shape) != shape:
            raise ValueError(
                f"Persistent geometry cache {name} has shape {tuple(tensor.shape)}, "
                f"expected {shape}."
            )
        if tensor.device != coords.device or tensor.dtype != dtype:
            raise ValueError(
                f"Persistent geometry cache {name} has device/dtype "
                f"{tensor.device}/{tensor.dtype}, expected {coords.device}/{dtype}."
            )


def cache_tensors(
    cache: PersistentTopKGeometryCache | Mapping[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    c = cache.as_mapping() if isinstance(cache, PersistentTopKGeometryCache) else cache
    return c["topk_d2"], c["topk_idx"], c["topk_valid"]

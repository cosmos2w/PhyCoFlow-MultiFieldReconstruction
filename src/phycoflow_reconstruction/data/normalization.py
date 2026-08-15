"""Serializable per-field normalization without writes beside shared datasets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import h5py
import torch


@dataclass
class FieldNormalizer:
    offset: torch.Tensor
    scale: torch.Tensor
    method: str = "mean_std"

    def __post_init__(self) -> None:
        self.offset = torch.as_tensor(self.offset, dtype=torch.float32).flatten()
        self.scale = torch.as_tensor(self.scale, dtype=torch.float32).flatten().clamp_min(1e-8)
        if self.offset.shape != self.scale.shape:
            raise ValueError("normalization offset and scale must align")

    def encode(self, values: torch.Tensor) -> torch.Tensor:
        return (values - self.offset.to(values.device)) / self.scale.to(values.device)

    def decode(self, values: torch.Tensor) -> torch.Tensor:
        return values * self.scale.to(values.device) + self.offset.to(values.device)

    def state_dict(self) -> dict[str, Any]:
        return {"offset": self.offset, "scale": self.scale, "method": self.method}

    @classmethod
    def identity(cls, channels: int) -> FieldNormalizer:
        return cls(torch.zeros(channels), torch.ones(channels), "none")

    @classmethod
    def from_h5(cls, handle: h5py.File, method: str = "auto") -> FieldNormalizer | None:
        stats = handle.get("statistics")
        if stats is None:
            return None
        if (
            method in {"auto", "robust_99"}
            and "channel_offset" in stats
            and "channel_scale_99" in stats
        ):
            return cls(stats["channel_offset"][:], stats["channel_scale_99"][:], "robust_99")
        if method in {"auto", "mean_std"} and "train_mean" in stats and "train_std" in stats:
            return cls(stats["train_mean"][:], stats["train_std"][:], "mean_std")
        return None

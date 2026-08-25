"""Serializable per-field normalization without writes beside shared datasets."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
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

    def digest(self) -> str:
        payload = {
            "method": self.method,
            "offset": [float(value) for value in self.offset],
            "scale": [float(value) for value in self.scale],
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

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

    @classmethod
    def from_artifact(
        cls,
        path: str | Path,
        *,
        field_names: tuple[str, ...],
        dataset_fingerprint: str,
    ) -> FieldNormalizer:
        """Load verified training-only statistics without modifying shared data."""

        payload = json.loads(Path(path).resolve().read_text(encoding="utf-8"))
        expected_digest = payload.pop("artifact_sha256", None)
        actual_digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        if expected_digest != actual_digest:
            raise ValueError("normalization statistics artifact checksum mismatch")
        if str(payload.get("version")) != "1":
            raise ValueError("unsupported normalization statistics artifact version")
        if tuple(payload.get("field_names", ())) != tuple(field_names):
            raise ValueError("normalization statistics field order mismatch")
        if payload.get("dataset_fingerprint") != dataset_fingerprint:
            raise ValueError("normalization statistics dataset fingerprint mismatch")
        if payload.get("statistics_split") != "train":
            raise ValueError("normalization statistics must come from the training split")
        method = str(payload.get("method", ""))
        if method not in {"mean_std", "robust_99"}:
            raise ValueError("normalization statistics method must be mean_std or robust_99")
        offset = payload.get("offset", ())
        scale = payload.get("scale", ())
        if len(offset) != len(field_names) or len(scale) != len(field_names):
            raise ValueError("normalization statistics channel count mismatch")
        offset_tensor = torch.as_tensor(offset, dtype=torch.float64)
        scale_tensor = torch.as_tensor(scale, dtype=torch.float64)
        if not bool(torch.isfinite(offset_tensor).all()):
            raise ValueError("normalization statistics offsets must be finite")
        if not bool(torch.isfinite(scale_tensor).all()) or bool((scale_tensor <= 0).any()):
            raise ValueError("normalization statistics scales must be finite and positive")
        return cls(offset, scale, method)

"""Training-only empirical reference banks for target-free coherence.

The fitter accepts only a dataset opened on the training split. It stores exact
sample and point IDs, normalization, field order, dataset fingerprint, and RNG
seed so a bank can be audited or reused without consulting validation/test
targets.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from ..data.manifest import dataset_fingerprint
from ..data.normalization import FieldNormalizer


@dataclass
class ReferenceBank:
    values: torch.Tensor
    sample_ids: tuple[str, ...]
    point_indices: torch.Tensor
    metadata: dict[str, Any]

    def validate(self) -> None:
        if self.values.ndim != 3:
            raise ValueError("reference values must have shape [R,N,C]")
        if self.point_indices.shape != self.values.shape[:2]:
            raise ValueError("reference point indices must align with [R,N]")
        if len(self.sample_ids) != self.values.shape[0]:
            raise ValueError("reference sample IDs must align with bank entries")
        if self.metadata.get("split") != "train":
            raise ValueError("reference banks must be fitted from split='train'")
        if not torch.isfinite(self.values).all():
            raise FloatingPointError("reference bank contains non-finite values")

    def digest(self) -> str:
        self.validate()
        digest = hashlib.sha256(json.dumps(self.metadata, sort_keys=True).encode())
        digest.update("\n".join(self.sample_ids).encode())
        digest.update(self.point_indices.contiguous().numpy().tobytes())
        digest.update(self.values.contiguous().numpy().tobytes())
        return digest.hexdigest()

    def payload(self) -> dict[str, Any]:
        return {
            "version": 1,
            "values": self.values.cpu(),
            "sample_ids": self.sample_ids,
            "point_indices": self.point_indices.cpu(),
            "metadata": self.metadata,
            "sha256": self.digest(),
        }

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        torch.save(self.payload(), temporary)
        os.replace(temporary, destination)
        return destination

    @classmethod
    def load(cls, path: str | Path) -> ReferenceBank:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        bank = cls(
            values=payload["values"],
            sample_ids=tuple(payload["sample_ids"]),
            point_indices=payload["point_indices"],
            metadata=dict(payload["metadata"]),
        )
        if bank.digest() != payload["sha256"]:
            raise ValueError("reference-bank digest mismatch")
        return bank

    def select(
        self,
        batch_size: int,
        *,
        step: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, tuple[str, ...]]:
        self.validate()
        indices = [
            (int(step) * int(batch_size) + offset) % self.values.shape[0]
            for offset in range(batch_size)
        ]
        values = self.values[indices].to(device=device, dtype=dtype)
        return values, tuple(self.sample_ids[index] for index in indices)


def fit_reference_bank(
    dataset,
    *,
    max_samples: int,
    points_per_sample: int,
    seed: int,
) -> ReferenceBank:
    """Fit an empirical bank using deterministic samples from a training dataset."""
    if (
        getattr(dataset, "split_name", None) != "train"
        or getattr(dataset.selection, "split", None) != "train"
    ):
        raise ValueError("reference-bank fitting is permitted only on the training split")
    if max_samples < 1 or points_per_sample < 2:
        raise ValueError("reference bank requires max_samples>=1 and points_per_sample>=2")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    sample_indices = torch.randperm(len(dataset), generator=generator)[
        : min(max_samples, len(dataset))
    ]
    values = []
    point_indices = []
    sample_ids = []
    for dataset_index in sample_indices.tolist():
        sample = dataset[dataset_index]
        count = min(int(points_per_sample), sample.values.shape[0])
        if count < 2:
            raise ValueError("reference sample contains fewer than two valid points")
        indices = torch.randperm(sample.values.shape[0], generator=generator)[:count].sort().values
        if count != points_per_sample:
            raise ValueError(
                f"requested {points_per_sample} bank points from a sample with only {sample.values.shape[0]}"
            )
        values.append(sample.values[indices].cpu())
        point_indices.append(indices.cpu())
        suffix = sample.time_index if sample.time_index is not None else "all"
        sample_ids.append(f"{sample.trajectory_id}:{suffix}")
    normalizer: FieldNormalizer = dataset.normalizer
    coordinates = dataset.raw_spatial_coords.contiguous().cpu()
    coordinate_digest = hashlib.sha256(coordinates.numpy().tobytes()).hexdigest()
    trajectory_indices = torch.as_tensor(dataset.selection.trajectory_indices, dtype=torch.long)
    training_conditions = dataset.conditions[trajectory_indices]
    condition_min = training_conditions.amin(dim=0).tolist() if training_conditions.shape[1] else []
    condition_max = training_conditions.amax(dim=0).tolist() if training_conditions.shape[1] else []
    bank = ReferenceBank(
        values=torch.stack(values),
        sample_ids=tuple(sample_ids),
        point_indices=torch.stack(point_indices),
        metadata={
            "split": "train",
            "selection_strategy": dataset.selection.strategy,
            "dataset_path": str(dataset.path),
            "dataset_fingerprint": dataset_fingerprint(dataset.path),
            "field_names": tuple(dataset.field_names),
            "field_units": tuple(dataset.field_units),
            "data_spec": asdict(dataset.data_spec),
            "coordinate_sha256": coordinate_digest,
            "coordinate_bounds": {
                "minimum": coordinates.amin(dim=0).tolist(),
                "maximum": coordinates.amax(dim=0).tolist(),
            },
            "conditions": {
                "dimension": int(training_conditions.shape[1]),
                "training_minimum": condition_min,
                "training_maximum": condition_max,
            },
            "normalization": {
                "method": normalizer.method,
                "offset": normalizer.offset.tolist(),
                "scale": normalizer.scale.tolist(),
            },
            "seed": int(seed),
            "max_samples": int(max_samples),
            "points_per_sample": int(points_per_sample),
        },
    )
    bank.validate()
    return bank

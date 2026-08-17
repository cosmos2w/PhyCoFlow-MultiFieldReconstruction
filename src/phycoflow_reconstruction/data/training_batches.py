"""Dual-path compact batch construction for sparse reconstruction training."""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from ..contracts import ObservationBatch
from .h5_dataset import H5FieldDataset
from .sensor_protocols import SensorProtocol, build_observation_batch


def dataset_field_bytes(dataset) -> int:
    """Return logical uncompressed field bytes used for residency decisions."""
    if isinstance(dataset, H5FieldDataset):
        with h5py.File(dataset.path, "r") as handle:
            fields = handle["fields"]
            return int(fields.size * fields.dtype.itemsize)
    fields = getattr(dataset, "fields", None)
    if isinstance(fields, torch.Tensor):
        return int(fields.numel() * fields.element_size())
    return int(Path(dataset.path).stat().st_size)


def _sample_ids(dataset, item_indices: Sequence[int]) -> tuple[str, ...]:
    values = []
    for item_index in item_indices:
        trajectory, frame = dataset._items[int(item_index)]
        suffix = "all" if frame is None else str(frame)
        values.append(f"{dataset.trajectory_ids[trajectory]}:{suffix}")
    return tuple(values)


def _field_settings(protocol: SensorProtocol, field_names: Sequence[str]):
    lookup = {name: index for index, name in enumerate(field_names)}
    fixed = dict(protocol.field_counts or {})
    ranges = dict(protocol.field_count_ranges or {})
    names = list(fixed) + [name for name in ranges if name not in fixed]
    if not names:
        names = [field_names[0]]
        fixed[names[0]] = 64
    missing = sorted(set(names) - set(lookup))
    if missing:
        raise KeyError(f"observation fields are absent from the dataset: {missing}")
    return names, lookup, fixed, ranges


def _batched_random_indices(
    batch_size: int,
    point_count: int,
    count: int,
    *,
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor:
    """Vectorized sampling without replacement using random ranks."""
    if count > point_count:
        raise ValueError(f"requested {count} points from only {point_count}")
    scores = torch.rand(batch_size, point_count, device=device, generator=generator)
    return scores.topk(count, dim=1, largest=False, sorted=False).indices


def fixed_query_indices(
    point_count: int,
    query_points: int | None,
    *,
    seed: int,
) -> torch.Tensor | None:
    """Build one CPU-selected point set shared by every sample and step."""
    query_count = point_count if query_points is None else min(int(query_points), point_count)
    if query_count == point_count:
        return None
    if query_count < 2:
        raise ValueError("fixed shared coherence queries require at least two points")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    return torch.randperm(point_count, generator=generator)[:query_count].sort().values


def _counts_for_field(
    name: str,
    batch_size: int,
    fixed: Mapping[str, int],
    ranges: Mapping[str, tuple[int, int]],
    *,
    device: torch.device,
    generator: torch.Generator,
) -> tuple[torch.Tensor, int]:
    if name in fixed:
        maximum = int(fixed[name])
        return torch.full((batch_size,), maximum, device=device, dtype=torch.long), maximum
    low, high = (int(value) for value in ranges[name])
    return (
        torch.randint(low, high + 1, (batch_size,), device=device, generator=generator),
        high,
    )


def _random_observation_plan(
    protocol: SensorProtocol,
    field_names: Sequence[str],
    batch_size: int,
    point_count: int,
    *,
    device: torch.device,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    names, lookup, fixed, ranges = _field_settings(protocol, field_names)
    counts = [
        _counts_for_field(
            name, batch_size, fixed, ranges, device=device, generator=generator
        )
        for name in names
    ]
    shared = None
    if protocol.shared_locations:
        shared = _batched_random_indices(
            batch_size,
            point_count,
            max(maximum for _, maximum in counts),
            device=device,
            generator=generator,
        )
    point_parts = []
    field_parts = []
    mask_parts = []
    for name, (count, maximum) in zip(names, counts):
        points = (
            shared[:, :maximum]
            if shared is not None
            else _batched_random_indices(
                batch_size,
                point_count,
                maximum,
                device=device,
                generator=generator,
            )
        )
        point_parts.append(points)
        field_parts.append(
            torch.full_like(points, int(lookup[name]), dtype=torch.long, device=device)
        )
        mask_parts.append(torch.arange(maximum, device=device)[None, :] < count[:, None])
    return (
        torch.cat(point_parts, dim=1),
        torch.cat(field_parts, dim=1),
        torch.cat(mask_parts, dim=1),
    )


def _gather_coordinates(coordinates: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    batch_size, count = indices.shape
    expanded = coordinates.unsqueeze(0).expand(batch_size, -1, -1)
    return torch.gather(
        expanded,
        1,
        indices.unsqueeze(-1).expand(batch_size, count, coordinates.shape[-1]),
    )


def _assemble_random_batch(
    values: torch.Tensor,
    coordinates: torch.Tensor,
    protocol: SensorProtocol,
    field_names: Sequence[str],
    *,
    query_points: int | None,
    generator: torch.Generator,
    query_generator: torch.Generator,
    sample_ids: tuple[str, ...],
    logical_shape: tuple[int, ...],
    conditions: torch.Tensor,
    times: torch.Tensor,
    fixed_query_indices: torch.Tensor | None = None,
    validate: bool = True,
) -> ObservationBatch:
    batch_size, point_count, channels = values.shape
    obs_indices, obs_fields, obs_mask = _random_observation_plan(
        protocol,
        field_names,
        batch_size,
        point_count,
        device=values.device,
        generator=generator,
    )
    query_count = point_count if query_points is None else min(int(query_points), point_count)
    flat_values = values.reshape(batch_size * point_count, channels)
    offsets = torch.arange(batch_size, device=values.device)[:, None] * point_count
    complete_grid = query_count == point_count
    if complete_grid:
        # Expanded views preserve the [B,N,*] observation contract while the
        # underlying index/grid storage exists only once for the whole batch.
        query_indices = torch.arange(point_count, device=values.device)[None, :].expand(
            batch_size, -1
        )
        query_coordinates = coordinates[None, :, :].expand(batch_size, -1, -1)
        query_values = values
        query_mask = torch.ones(
            1, point_count, device=values.device, dtype=torch.bool
        ).expand(batch_size, -1)
    elif fixed_query_indices is not None:
        shared = fixed_query_indices.to(device=values.device, dtype=torch.long)
        if shared.numel() != query_count:
            raise ValueError("fixed query index count disagrees with query_points")
        query_indices = shared[None, :].expand(batch_size, -1)
        query_coordinates = _gather_coordinates(coordinates, query_indices)
        query_values = flat_values[query_indices + offsets]
        query_mask = torch.ones_like(query_indices, dtype=torch.bool)
    else:
        query_indices = _batched_random_indices(
            batch_size,
            point_count,
            query_count,
            device=values.device,
            generator=query_generator,
        )
        query_coordinates = _gather_coordinates(coordinates, query_indices)
        query_values = flat_values[query_indices + offsets]
        query_mask = torch.ones_like(query_indices, dtype=torch.bool)
    obs_all_fields = flat_values[obs_indices + offsets]
    obs_values = torch.gather(obs_all_fields, 2, obs_fields.unsqueeze(-1))
    batch = ObservationBatch(
        obs_coords=_gather_coordinates(coordinates, obs_indices),
        obs_values=obs_values,
        obs_field_ids=obs_fields,
        obs_valid_mask=obs_mask,
        query_coords=query_coordinates,
        query_valid_mask=query_mask,
        target_fields=query_values,
        sample_ids=sample_ids,
        obs_indices=obs_indices,
        logical_shapes=(logical_shape,) * batch_size,
        metadata={
            "protocol": protocol.to_dict(),
            "query_indices": query_indices,
            "sample_context": {
                "conditions": conditions,
                # Keep one compact tensor rather than B scalar tensor views;
                # every tensor crossing worker IPC consumes a shared-memory
                # descriptor under PyTorch multiprocessing.
                "time": times,
            },
        },
    )
    if validate:
        batch.validate()
    return batch


class ResidentH5BatchSource(nn.Module):
    """Keep the complete selected split normalized in non-persistent CUDA buffers."""

    strategy = "vram_resident"

    def __init__(
        self,
        dataset: H5FieldDataset,
        batches: Iterable[Sequence[int]],
        protocol: SensorProtocol,
        *,
        query_points: int | None,
        device: torch.device,
        start_step: int,
        fixed_query_indices: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if device.type != "cuda":
            raise ValueError("VRAM residency requires a CUDA device")
        self.dataset = dataset
        self.batches = batches
        self.protocol = protocol
        self.query_points = query_points
        self.device = device
        self.start_step = int(start_step)
        self.fixed_query_indices = fixed_query_indices
        self.sample_ids = _sample_ids(dataset, range(len(dataset)))
        self.logical_shape = dataset.data_spec.logical_shape

        point_count = int(math.prod(dataset.grid_shape))
        channels = dataset.data_spec.num_fields
        fields = torch.empty(
            (len(dataset), point_count, channels), dtype=torch.float32, device=device
        )
        with h5py.File(dataset.path, "r") as handle:
            cursor = 0
            frames = np.asarray(dataset.selection.frame_indices, dtype=np.int64)
            permutation = dataset.point_permutation.numpy()
            identity = np.array_equal(permutation, np.arange(point_count))
            bytes_per_frame = point_count * channels * handle["fields"].dtype.itemsize
            frames_per_chunk = max(1, (256 * 1024**2) // bytes_per_frame)
            for trajectory in dataset.selection.trajectory_indices:
                for chunk_start in range(0, len(frames), frames_per_chunk):
                    chunk_frames = frames[chunk_start : chunk_start + frames_per_chunk]
                    raw = handle["fields"][trajectory, chunk_frames].reshape(
                        -1, point_count, channels
                    )
                    if not identity:
                        raw = raw[:, permutation]
                    count = raw.shape[0]
                    fields[cursor : cursor + count].copy_(
                        torch.from_numpy(raw), non_blocking=False
                    )
                    cursor += count
        offset = dataset.normalizer.offset.to(device)
        scale = dataset.normalizer.scale.to(device)
        fields.sub_(offset).div_(scale)
        active = dataset.data_spec.coordinate_dim
        self.register_buffer("fields", fields.contiguous(), persistent=False)
        self.register_buffer(
            "coordinates", dataset.spatial_coords[:, :active].to(device).contiguous(), persistent=False
        )
        item_conditions = torch.stack(
            [dataset.conditions[trajectory] for trajectory, _ in dataset._items]
        ).to(device)
        item_times = torch.stack(
            [dataset.times[int(frame)] for _, frame in dataset._items]
        ).to(device)
        self.register_buffer("conditions", item_conditions, persistent=False)
        self.register_buffer("times", item_times, persistent=False)

    def __iter__(self) -> Iterator[ObservationBatch]:
        for offset, batch_indices in enumerate(self.batches):
            global_step = self.start_step + offset
            index = torch.as_tensor(batch_indices, device=self.device, dtype=torch.long)
            values = self.fields[index]
            generator = torch.Generator(device=self.device).manual_seed(
                self.protocol.seed + global_step
            )
            query_generator = torch.Generator(device=self.device).manual_seed(
                self.protocol.seed + 100_003 + global_step
            )
            yield _assemble_random_batch(
                values,
                self.coordinates,
                SensorProtocol(**{**self.protocol.to_dict(), "seed": self.protocol.seed + global_step}),
                self.dataset.field_names,
                query_points=self.query_points,
                generator=generator,
                query_generator=query_generator,
                sample_ids=tuple(self.sample_ids[int(item)] for item in batch_indices),
                logical_shape=self.logical_shape,
                conditions=self.conditions[index],
                times=self.times[index],
                fixed_query_indices=self.fixed_query_indices,
                # All later batches are built from the same shape-safe tensor
                # operations. Avoid repeated CUDA-to-host synchronization in
                # contract checks after validating the first batch.
                validate=offset == 0,
            )

    def close(self) -> None:
        self.fields = torch.empty(0, device=self.device)
        torch.cuda.empty_cache()


class _CompactH5Dataset(Dataset):
    """Worker-side bulk fetcher that returns only compact observation batches."""

    def __init__(
        self,
        dataset: H5FieldDataset,
        protocol: SensorProtocol,
        query_points: int | None,
        fixed_query_indices: torch.Tensor | None = None,
    ) -> None:
        self.dataset = dataset
        self.protocol = protocol
        self.query_points = query_points
        self.fixed_query_indices = fixed_query_indices

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, request):  # pragma: no cover - DataLoader uses bulk path
        return self.__getitems__([request])[0]

    def __getitems__(self, requests):
        indices = [int(request[0]) for request in requests]
        global_step = int(requests[0][1])
        batch_size = len(indices)
        point_count = int(math.prod(self.dataset.grid_shape))
        device = torch.device("cpu")
        generator = torch.Generator().manual_seed(self.protocol.seed + global_step)
        query_generator = torch.Generator().manual_seed(
            self.protocol.seed + 100_003 + global_step
        )
        obs_indices, obs_fields, obs_mask = _random_observation_plan(
            self.protocol,
            self.dataset.field_names,
            batch_size,
            point_count,
            device=device,
            generator=generator,
        )
        query_count = point_count if self.query_points is None else min(self.query_points, point_count)
        query_indices = (
            torch.arange(point_count).expand(batch_size, -1)
            if query_count == point_count
            else self.fixed_query_indices.expand(batch_size, -1)
            if self.fixed_query_indices is not None
            else _batched_random_indices(
                batch_size,
                point_count,
                query_count,
                device=device,
                generator=query_generator,
            )
        )
        channels = self.dataset.data_spec.num_fields
        obs_values = torch.empty(batch_size, obs_indices.shape[1], 1)
        query_values = torch.empty(batch_size, query_count, channels)
        handle = self.dataset._handle()
        permutation = self.dataset.point_permutation
        item_pairs = [self.dataset._items[item_index] for item_index in indices]
        groups: dict[int, list[tuple[int, int]]] = {}
        for row, (trajectory, frame) in enumerate(item_pairs):
            groups.setdefault(int(trajectory), []).append((row, int(frame)))
        for trajectory, entries in groups.items():
            ordered = sorted(entries, key=lambda item: item[1])
            rows = torch.tensor([row for row, _ in ordered], dtype=torch.long)
            frames = np.asarray([frame for _, frame in ordered], dtype=np.int64)
            raw = torch.from_numpy(
                handle["fields"][trajectory, frames]
                .reshape(len(ordered), point_count, channels)
                .astype(np.float32)
            )[:, permutation]
            group_obs = obs_indices[rows]
            group_queries = query_indices[rows]
            group_size = len(ordered)
            offsets = torch.arange(group_size)[:, None] * point_count
            flat = raw.reshape(group_size * point_count, channels)
            selected_obs = self.dataset.normalizer.encode(flat[group_obs + offsets])
            obs_values[rows, :, 0] = torch.gather(
                selected_obs, 2, obs_fields[rows, :, None]
            )[:, :, 0]
            query_values[rows] = self.dataset.normalizer.encode(
                flat[group_queries + offsets]
            )
        trajectory_indices = torch.tensor(
            [int(trajectory) for trajectory, _ in item_pairs], dtype=torch.long
        )
        frame_indices = torch.tensor([int(frame) for _, frame in item_pairs], dtype=torch.long)
        conditions = self.dataset.conditions[trajectory_indices]
        times = self.dataset.times[frame_indices]
        coordinates = self.dataset.spatial_coords[:, : self.dataset.data_spec.coordinate_dim]
        step_protocol = SensorProtocol(
            **{**self.protocol.to_dict(), "seed": self.protocol.seed + global_step}
        )
        batch = ObservationBatch(
            obs_coords=_gather_coordinates(coordinates, obs_indices),
            obs_values=obs_values,
            obs_field_ids=obs_fields,
            obs_valid_mask=obs_mask,
            query_coords=_gather_coordinates(coordinates, query_indices),
            query_valid_mask=torch.ones_like(query_indices, dtype=torch.bool),
            target_fields=query_values,
            sample_ids=_sample_ids(self.dataset, indices),
            obs_indices=obs_indices,
            logical_shapes=(self.dataset.data_spec.logical_shape,) * batch_size,
            metadata={
                "protocol": step_protocol.to_dict(),
                "query_indices": query_indices,
                "sample_context": {
                    "conditions": conditions,
                    "time": times,
                },
            },
        )
        batch.validate()
        return [batch]


def _unwrap_compact_batch(items):
    if len(items) != 1 or not isinstance(items[0], ObservationBatch):
        raise TypeError("compact HDF5 bulk fetch must return one ObservationBatch")
    return items[0]


class _StepTaggedBatchSampler:
    """Attach the optimizer step without retaining all future requests."""

    def __init__(self, batches: Iterable[Sequence[int]], start_step: int) -> None:
        self.batches = batches
        self.start_step = int(start_step)

    def __iter__(self):
        for offset, batch in enumerate(self.batches):
            yield [(int(item), self.start_step + offset) for item in batch]


def _identity_samples(samples):
    """Pickle-safe legacy collator for worker processes."""
    return samples


class AsyncCompactH5BatchSource:
    strategy = "async_compact_cpu"

    def __init__(
        self,
        dataset: H5FieldDataset,
        batches: Iterable[Sequence[int]],
        protocol: SensorProtocol,
        *,
        query_points: int | None,
        device: torch.device,
        start_step: int,
        num_workers: int,
        fixed_query_indices: torch.Tensor | None = None,
    ) -> None:
        self.device = device
        workers = max(1, int(num_workers))
        self.loader = DataLoader(
            _CompactH5Dataset(dataset, protocol, query_points, fixed_query_indices),
            batch_sampler=_StepTaggedBatchSampler(batches, start_step),
            num_workers=workers,
            collate_fn=_unwrap_compact_batch,
            persistent_workers=True,
            prefetch_factor=2,
            pin_memory=device.type == "cuda",
            pin_memory_device=str(device) if device.type == "cuda" else "",
        )

    def __iter__(self) -> Iterator[ObservationBatch]:
        for batch in self.loader:
            yield batch.to(self.device, non_blocking=self.device.type == "cuda")

    def close(self) -> None:
        iterator = getattr(self.loader, "_iterator", None)
        if iterator is not None:
            iterator._shutdown_workers()


class LegacyBatchSource:
    strategy = "legacy_full_sample"

    def __init__(
        self,
        dataset,
        batches,
        protocol,
        *,
        query_points,
        device,
        start_step,
        num_workers,
        fixed_query_indices=None,
    ) -> None:
        self.dataset = dataset
        self.protocol = protocol
        self.query_points = query_points
        self.device = device
        self.start_step = start_step
        self.fixed_query_indices = fixed_query_indices
        self.loader = DataLoader(
            dataset,
            batch_sampler=batches,
            num_workers=num_workers,
            collate_fn=_identity_samples,
            persistent_workers=bool(num_workers),
        )

    def __iter__(self):
        for offset, samples in enumerate(self.loader):
            protocol = SensorProtocol(
                **{
                    **self.protocol.to_dict(),
                    "seed": self.protocol.seed + self.start_step + offset,
                }
            )
            yield build_observation_batch(
                samples,
                protocol,
                query_points=self.query_points,
                query_indices=self.fixed_query_indices,
            ).to(self.device)

    def close(self) -> None:
        return None


def build_training_batch_source(
    dataset,
    batches: Iterable[Sequence[int]],
    config: Mapping[str, Any],
    *,
    query_points: int | None,
    device: torch.device,
    start_step: int,
):
    """Choose VRAM residency below the threshold, otherwise compact async HDF5."""
    observations = config["observations"]
    protocol = SensorProtocol(
        name=observations.get("protocol", "random_uniform"),
        field_counts={
            name: int(settings["count"])
            for name, settings in observations["fields"].items()
            if "count" in settings
        }
        or None,
        field_count_ranges={
            name: (int(settings["count_min"]), int(settings["count_max"]))
            for name, settings in observations["fields"].items()
            if "count" not in settings
        }
        or None,
        spatial_downsample_ratio=int(observations.get("spatial_downsample_ratio", 1)),
        temporal_downsample_ratio=int(observations.get("temporal_downsample_ratio", 1)),
        phase=int(observations.get("phase", 0)),
        seed=int(observations.get("seed", config["runtime"].get("seed", 42))),
        shared_locations=bool(observations.get("shared_locations", False)),
    )
    runtime = config["runtime"]
    compute = config.get("coherence", {}).get("compute_budget", {})
    query_policy = str(compute.get("query_policy", "random_per_sample"))
    if query_policy not in {"random_per_sample", "fixed_shared"}:
        raise ValueError("coherence.compute_budget.query_policy is invalid")
    shared_queries = None
    if query_policy == "fixed_shared":
        point_count = int(math.prod(dataset.data_spec.logical_shape))
        shared_queries = fixed_query_indices(
            point_count,
            query_points,
            seed=int(compute.get("query_seed", protocol.seed + 100_003)),
        )
    threshold = float(runtime.get("vram_dataset_threshold_gb", 20.0)) * 1_000_000_000
    size = dataset_field_bytes(dataset)
    forced = runtime.get("data_strategy", "auto")
    resident_supported = (
        isinstance(dataset, H5FieldDataset)
        and dataset.reconstruction_unit == "snapshot"
        and protocol.name == "random_uniform"
        and not dataset.include_temporal_derivative
    )
    if forced == "vram" or (
        forced == "auto" and device.type == "cuda" and size < threshold and resident_supported
    ):
        if not resident_supported:
            raise ValueError("VRAM strategy currently requires random snapshot HDF5 training")
        print(
            f"data pipeline: VRAM residency on {device} "
            f"(complete HDF5 fields={size / 1e9:.2f} GB); loading training split once"
        )
        source = ResidentH5BatchSource(
            dataset,
            batches,
            protocol,
            query_points=query_points,
            device=device,
            start_step=start_step,
            fixed_query_indices=shared_queries,
        )
        print(
            f"data pipeline: resident training tensor ready "
            f"({source.fields.numel() * source.fields.element_size() / 1e9:.2f} GB)"
        )
        return source
    compact_supported = (
        isinstance(dataset, H5FieldDataset)
        and dataset.reconstruction_unit == "snapshot"
        and protocol.name == "random_uniform"
        and not dataset.include_temporal_derivative
    )
    if forced in {"auto", "async_cpu"} and compact_supported:
        workers = int(runtime.get("num_workers", 4))
        print(
            f"data pipeline: compact asynchronous HDF5 loading with "
            f"{max(1, workers)} workers (complete fields={size / 1e9:.2f} GB)"
        )
        return AsyncCompactH5BatchSource(
            dataset,
            batches,
            protocol,
            query_points=query_points,
            device=device,
            start_step=start_step,
            num_workers=workers,
            fixed_query_indices=shared_queries,
        )
    print("data pipeline: using compatibility loader for this protocol or dataset format")
    return LegacyBatchSource(
        dataset,
        batches,
        protocol,
        query_points=query_points,
        device=device,
        start_step=start_step,
        num_workers=int(runtime.get("num_workers", 0)),
        fixed_query_indices=shared_queries,
    )

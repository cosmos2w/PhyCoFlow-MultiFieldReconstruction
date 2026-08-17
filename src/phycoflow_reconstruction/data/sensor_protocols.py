"""Deterministic sparse-observation protocols shared by every model adapter.

Training may vary the supplied seed. Evaluation persists the selected indices
through `SensorManifest`, so point and grid models receive identical evidence.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass

import torch

from ..contracts import FieldSample, ObservationBatch


@dataclass(frozen=True)
class SensorProtocol:
    name: str = "random_uniform"
    field_counts: Mapping[str, int] | None = None
    field_count_ranges: Mapping[str, tuple[int, int]] | None = None
    spatial_downsample_ratio: int = 1
    temporal_downsample_ratio: int = 1
    phase: int = 0
    seed: int = 42
    shared_locations: bool = False

    def validate(self) -> None:
        if self.name not in {"random_uniform", "structured_stride", "uniform_spacetime_stride"}:
            raise ValueError(f"unsupported sensor protocol {self.name!r}")
        if self.spatial_downsample_ratio < 1 or self.temporal_downsample_ratio < 1:
            raise ValueError("spatial and temporal downsample ratios must be positive")
        if self.field_counts and any(int(value) < 1 for value in self.field_counts.values()):
            raise ValueError("all field sensor counts must be positive")
        if self.field_count_ranges:
            for name, bounds in self.field_count_ranges.items():
                if len(bounds) != 2 or int(bounds[0]) < 1 or int(bounds[1]) < int(bounds[0]):
                    raise ValueError(f"invalid sensor count range for {name!r}: {bounds}")
        overlap = set(self.field_counts or {}).intersection(self.field_count_ranges or {})
        if overlap:
            raise ValueError(
                f"fields cannot define fixed counts and count ranges together: {sorted(overlap)}"
            )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _sample_observation_indices(
    sample: FieldSample,
    protocol: SensorProtocol,
    generator: torch.Generator,
) -> list[tuple[int, int]]:
    field_lookup = {name: index for index, name in enumerate(sample.field_names)}
    pairs: list[tuple[int, int]] = []

    if protocol.name == "uniform_spacetime_stride":
        if sample.reconstruction_unit != "space_time_trajectory" or len(sample.logical_shape) != 2:
            raise ValueError("uniform_spacetime_stride requires a [T,X] space-time sample")
        time_count, space_count = sample.logical_shape
        time_ids = torch.arange(protocol.phase, time_count, protocol.temporal_downsample_ratio)
        space_ids = torch.arange(protocol.phase, space_count, protocol.spatial_downsample_ratio)
        if time_ids.numel() < 2 or space_ids.numel() < 2:
            raise ValueError("space-time downsampling must retain at least two points per axis")
        point_ids = (time_ids[:, None] * space_count + space_ids[None, :]).reshape(-1)
        fields = _resolved_field_counts(protocol, sample, generator, int(point_ids.numel()))
        for name in fields:
            field_id = field_lookup[name]
            pairs.extend((int(point_id), field_id) for point_id in point_ids)
        return pairs

    fields = _resolved_field_counts(protocol, sample, generator, min(64, sample.values.shape[0]))
    shared_order = None
    if protocol.name == "random_uniform" and protocol.shared_locations:
        shared_order = torch.randperm(sample.values.shape[0], generator=generator)
    for name, count in fields.items():
        if name not in field_lookup:
            raise KeyError(f"observation field {name!r} not in {sample.field_names}")
        if int(count) > sample.values.shape[0]:
            raise ValueError(f"requested {count} sensors from only {sample.values.shape[0]} points")
        if protocol.name == "structured_stride":
            ratio = protocol.spatial_downsample_ratio
            if ratio > 1 and len(sample.logical_shape) == 2:
                rows, columns = sample.logical_shape
                row_ids = torch.arange(protocol.phase, rows, ratio)
                column_ids = torch.arange(protocol.phase, columns, ratio)
                point_ids = (row_ids[:, None] * columns + column_ids[None, :]).reshape(-1)
                point_ids = point_ids[: int(count)] if int(count) < point_ids.numel() else point_ids
            elif ratio > 1:
                point_ids = torch.arange(protocol.phase, sample.values.shape[0], ratio)[
                    : int(count)
                ]
            else:
                step = max(sample.values.shape[0] // int(count), 1)
                point_ids = torch.arange(protocol.phase, sample.values.shape[0], step)[: int(count)]
        elif shared_order is not None:
            point_ids = shared_order[: int(count)]
        else:
            point_ids = torch.randperm(sample.values.shape[0], generator=generator)[: int(count)]
        pairs.extend((int(point_id), field_lookup[name]) for point_id in point_ids)
    return pairs


def _resolved_field_counts(
    protocol: SensorProtocol,
    sample: FieldSample,
    generator: torch.Generator,
    default_count: int,
) -> dict[str, int]:
    counts = {name: int(value) for name, value in (protocol.field_counts or {}).items()}
    for name, bounds in (protocol.field_count_ranges or {}).items():
        low, high = int(bounds[0]), int(bounds[1])
        counts[name] = int(torch.randint(low, high + 1, (1,), generator=generator).item())
    return counts or {sample.field_names[0]: int(default_count)}


def build_observation_batch(
    samples: Sequence[FieldSample],
    protocol: SensorProtocol,
    *,
    query_points: int | None = None,
    manifest_indices: Mapping[str, Sequence[Sequence[int]]] | None = None,
    query_indices: Sequence[int] | torch.Tensor | None = None,
) -> ObservationBatch:
    """Build one reproducible sparse-input/query batch.

    The protocol seed drives both random sensor selection and efficient query
    subsampling. Training loops must offset that seed by the global optimizer
    step; evaluation intentionally keeps it fixed for comparable metrics.
    """
    protocol.validate()
    if not samples:
        raise ValueError("cannot build an empty observation batch")
    generator = torch.Generator(device="cpu").manual_seed(protocol.seed)
    obs_payloads = []
    query_payloads = []
    sample_ids = []
    logical_shapes = []
    context_payloads: list[dict[str, object]] = []

    for sample_index, sample in enumerate(samples):
        sample.validate()
        sample_id = f"{sample.trajectory_id}:{sample.time_index if sample.time_index is not None else 'all'}"
        if manifest_indices and sample_id in manifest_indices:
            pairs = [(int(item[0]), int(item[1])) for item in manifest_indices[sample_id]]
        else:
            pairs = _sample_observation_indices(sample, protocol, generator)
        if not pairs:
            raise ValueError(f"sample {sample_id!r} has no observations")
        if len(pairs) != len(set(pairs)):
            raise ValueError(f"sample {sample_id!r} contains duplicate point/field observations")
        for point_id, field_id in pairs:
            if not 0 <= point_id < sample.values.shape[0]:
                raise ValueError(
                    f"sample {sample_id!r} observation point {point_id} is out of range"
                )
            if not 0 <= field_id < sample.values.shape[1]:
                raise ValueError(
                    f"sample {sample_id!r} observation field {field_id} is out of range"
                )
        point_ids = torch.tensor([item[0] for item in pairs], dtype=torch.long)
        field_ids = torch.tensor([item[1] for item in pairs], dtype=torch.long)
        obs_payloads.append(
            (
                sample.coordinates[point_ids],
                sample.values[point_ids, field_ids].unsqueeze(-1),
                field_ids,
                point_ids,
            )
        )

        if query_indices is not None:
            query_ids = torch.as_tensor(query_indices, dtype=torch.long)
            if query_ids.ndim != 1 or query_ids.numel() < 1:
                raise ValueError("query_indices must be a non-empty one-dimensional sequence")
            if torch.any(query_ids < 0) or torch.any(query_ids >= sample.values.shape[0]):
                raise ValueError("query_indices contain an out-of-range point")
            if torch.unique(query_ids).numel() != query_ids.numel():
                raise ValueError("query_indices must be unique")
        elif query_points is None or query_points >= sample.values.shape[0]:
            query_ids = torch.arange(sample.values.shape[0])
        else:
            # Offset seeds by sample while retaining deterministic adapter-independent queries.
            query_generator = torch.Generator(device="cpu").manual_seed(
                protocol.seed + 100_003 + sample_index
            )
            query_ids = (
                torch.randperm(sample.values.shape[0], generator=query_generator)[:query_points]
                .sort()
                .values
            )
        query_payloads.append((sample.coordinates[query_ids], sample.values[query_ids], query_ids))
        sample_context: dict[str, object] = {
            "conditions": sample.conditions,
            "time": sample.time,
        }
        for group_name in ("physics", "auxiliary", "diagnostics"):
            group = sample.metadata.get(group_name, {})
            if not isinstance(group, Mapping):
                continue
            selected_group: dict[str, object] = {}
            for name, value in group.items():
                if (
                    isinstance(value, torch.Tensor)
                    and value.ndim
                    and value.shape[0] == sample.values.shape[0]
                ):
                    selected_group[str(name)] = value[query_ids]
                else:
                    selected_group[str(name)] = value
            sample_context[group_name] = selected_group
        context_payloads.append(sample_context)
        sample_ids.append(sample_id)
        logical_shapes.append(sample.logical_shape)

    coordinate_dims = {sample.coordinates.shape[1] for sample in samples}
    channel_counts = {sample.values.shape[1] for sample in samples}
    field_orders = {sample.field_names for sample in samples}
    if len(coordinate_dims) != 1 or len(channel_counts) != 1 or len(field_orders) != 1:
        raise ValueError(
            "all samples in an observation batch must share coordinate dimension and field order"
        )

    max_obs = max(payload[0].shape[0] for payload in obs_payloads)
    max_query = max(payload[0].shape[0] for payload in query_payloads)
    coord_dim = samples[0].coordinates.shape[1]
    channels = samples[0].values.shape[1]
    bsz = len(samples)

    obs_coords = torch.zeros(bsz, max_obs, coord_dim)
    obs_values = torch.zeros(bsz, max_obs, 1)
    obs_fields = torch.zeros(bsz, max_obs, dtype=torch.long)
    obs_mask = torch.zeros(bsz, max_obs, dtype=torch.bool)
    obs_indices = torch.full((bsz, max_obs), -1, dtype=torch.long)
    query_coords = torch.zeros(bsz, max_query, coord_dim)
    query_values = torch.zeros(bsz, max_query, channels)
    query_mask = torch.zeros(bsz, max_query, dtype=torch.bool)
    query_indices = torch.full((bsz, max_query), -1, dtype=torch.long)

    for batch_index, (obs_payload, query_payload) in enumerate(zip(obs_payloads, query_payloads)):
        coords, values, fields, point_ids = obs_payload
        count = coords.shape[0]
        obs_coords[batch_index, :count] = coords
        obs_values[batch_index, :count] = values
        obs_fields[batch_index, :count] = fields
        obs_mask[batch_index, :count] = True
        obs_indices[batch_index, :count] = point_ids
        coords_q, values_q, ids_q = query_payload
        count_q = coords_q.shape[0]
        query_coords[batch_index, :count_q] = coords_q
        query_values[batch_index, :count_q] = values_q
        query_mask[batch_index, :count_q] = True
        query_indices[batch_index, :count_q] = ids_q

    context: dict[str, object] = {}
    context["conditions"] = torch.stack([payload["conditions"] for payload in context_payloads])
    context["time"] = tuple(payload["time"] for payload in context_payloads)
    for group_name in ("physics", "auxiliary", "diagnostics"):
        groups = [payload.get(group_name, {}) for payload in context_payloads]
        keys = set.intersection(*(set(group) for group in groups)) if groups else set()
        collated: dict[str, object] = {}
        for name in sorted(keys):
            values = [group[name] for group in groups]
            collated[name] = (
                torch.stack(values)
                if all(isinstance(value, torch.Tensor) for value in values)
                else tuple(values)
            )
        if collated:
            context[group_name] = collated

    batch = ObservationBatch(
        obs_coords=obs_coords,
        obs_values=obs_values,
        obs_field_ids=obs_fields,
        obs_valid_mask=obs_mask,
        query_coords=query_coords,
        query_valid_mask=query_mask,
        target_fields=query_values,
        sample_ids=tuple(sample_ids),
        obs_indices=obs_indices,
        logical_shapes=tuple(logical_shapes),
        metadata={
            "protocol": protocol.to_dict(),
            "query_indices": query_indices,
            "sample_context": context,
        },
    )
    batch.validate()
    return batch

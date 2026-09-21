"""Training-only stratified trajectory sampling and resumable full passes."""

import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import replace
from hashlib import sha256
from typing import Any

import numpy as np
import torch

from .manifest import dataset_fingerprint


def _allocate(capacity, total, minimum):
    capacity = np.asarray(capacity, dtype=np.int64)
    counts = np.asarray(minimum, dtype=np.int64).copy()
    if total < counts.sum() or total > capacity.sum():
        raise ValueError("subset budget cannot satisfy the requested trajectory/time coverage")
    target = capacity / capacity.sum() * total
    while counts.sum() < total:
        deficit = np.where(counts < capacity, target - counts, -np.inf)
        counts[int(np.argmax(deficit))] += 1
    return counts.tolist()


def select_topology_subset(
    rows: Sequence[Mapping[str, Any]],
    *,
    fraction: float = 0.3,
    seed: int = 42,
    min_frames_per_trajectory: int = 4,
    strata_keys: Sequence[str] = (),
    trajectory_key: str = "trajectory_id",
    time_key: str = "time",
    frame_key: str = "frame",
    split_key: str = "split",
) -> tuple[list[int], list[dict[str, Any]]]:
    """Select temporal bins within each trajectory using metadata only.

    ``strata_keys`` identify categorical metadata columns for proportional
    allocation. Every trajectory receives its declared minimum. Rows must belong
    to the training split; field values and holdout targets are never inspected.
    """
    if not rows or not 0 < fraction <= 1 or min_frames_per_trajectory < 1:
        raise ValueError("invalid topology subset settings")
    groups = defaultdict(lambda: defaultdict(list))
    required = set(strata_keys) | {trajectory_key, time_key, frame_key, split_key}
    for index, row in enumerate(rows):
        missing = required - row.keys()
        if missing:
            raise ValueError(f"subset metadata is missing columns: {sorted(missing)}")
        if row[split_key] != "train":
            raise ValueError("topology subset must contain training rows only")
        key = tuple(row[column] for column in strata_keys)
        groups[key][str(row[trajectory_key])].append(index)
    keys = sorted(groups)
    capacities = [sum(map(len, groups[key].values())) for key in keys]
    minima = [
        sum(min(len(trajectory), min_frames_per_trajectory) for trajectory in groups[key].values())
        for key in keys
    ]
    quotas = _allocate(capacities, math.ceil(len(rows) * fraction), minima)
    rng = np.random.default_rng(seed)
    selected, strata = [], []
    for key, quota in zip(keys, quotas):
        trajectories = sorted(groups[key])
        capacities = [len(groups[key][name]) for name in trajectories]
        counts = _allocate(
            capacities, quota, [min(n, min_frames_per_trajectory) for n in capacities]
        )
        for name, count in zip(trajectories, counts):
            ordered = sorted(
                groups[key][name], key=lambda i: (rows[i][time_key], rows[i][frame_key])
            )
            for temporal_bin in np.array_split(ordered, count):
                selected.append(int(rng.choice(temporal_bin)))
        strata.append(
            {
                "labels": dict(zip(strata_keys, key)),
                "available": sum(capacities),
                "selected": quota,
                "trajectories": len(trajectories),
            }
        )
    selected.sort()
    return selected, strata


def apply_training_subset(dataset, settings):
    """Apply a metadata subset consistently to snapshot and resident loaders."""
    settings = {
        "fraction": 0.3,
        "seed": 42,
        "min_frames_per_trajectory": 4,
        "strata_keys": [],
        "trajectory_key": "trajectory_id",
        "time_key": "time",
        "frame_key": "frame",
        "split_key": "split",
        **settings,
    }
    if dataset.split_name != "train" or dataset.reconstruction_unit != "snapshot":
        raise ValueError("topology subset requires training snapshots")
    metadata = dataset.dataset_metadata.get("samples")
    if not metadata or any(t != 0 for _, t in dataset._items):
        raise ValueError("topology subset requires canonical per-snapshot metadata")
    items = list(dataset._items)
    rows = [metadata[b] for b, _ in items]
    selected, strata = select_topology_subset(rows, **settings)
    payload = {
        "version": 2,
        "dataset_fingerprint": dataset_fingerprint(dataset.path),
        "settings": dict(settings),
        "original_count": len(items),
        "selected_count": len(selected),
        "strata": strata,
        "original_indices": selected,
        "samples": [{"dataset_index": items[i][0], **rows[i]} for i in selected],
        "sampling": "full_pass_without_replacement; shuffled per epoch",
    }
    payload["sha256"] = sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    dataset._items = [items[i] for i in selected]
    # Resident loading uses the Cartesian split selection; compact/legacy paths
    # use _items. Canonical metadata above guarantees one stored frame per row.
    dataset.selection = replace(
        dataset.selection,
        trajectory_indices=tuple(b for b, _ in dataset._items),
        frame_indices=(0,),
        strategy=dataset.selection.strategy + "_topology_subset",
    )
    dataset.training_subset_manifest = payload
    return payload


def iter_full_pass_indices(dataset_size, num_batches, batch_size, *, seed, start_step=0):
    """Stateless epoch permutations: exact mid-epoch resume despite prefetch.

    The final batch may be smaller; every selected sample occurs exactly once
    per full epoch. Epochs and update counts remain distinct from sample draws.
    """
    if not 1 <= batch_size <= dataset_size or num_batches < 0 or start_step < 0:
        raise ValueError("invalid full-pass sampler dimensions")
    steps = math.ceil(dataset_size / batch_size)
    current, order = None, None
    for step in range(start_step, start_step + num_batches):
        epoch, offset = divmod(step, steps)
        if epoch != current:
            order = torch.randperm(
                dataset_size, generator=torch.Generator().manual_seed(seed + epoch)
            )
            current = epoch
        yield order[offset * batch_size : (offset + 1) * batch_size].tolist()

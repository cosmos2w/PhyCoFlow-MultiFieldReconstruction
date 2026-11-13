"""Physical mass stays aligned from the case provider into the State loss."""

from __future__ import annotations

import h5py
import numpy as np
import pytest
import torch

from phycoflow_reconstruction.coherence import build_coherence_family
from phycoflow_reconstruction.data.factory import open_field_dataset
from phycoflow_reconstruction.data.physical_measure import (
    attach_batch_measure,
    batch_measure_context,
)
from phycoflow_reconstruction.data.sensor_protocols import SensorProtocol, build_observation_batch
from phycoflow_reconstruction.data.training_batches import _assemble_random_batch, _CompactH5Dataset
from phycoflow_reconstruction.training.rollout import subset_query_batch


def _fixture(path):
    y = np.array([0.0, 1.0, 4.0], dtype=np.float32)
    x = np.array([0.0, 0.5, 3.0, 7.0], dtype=np.float32)
    yy, xx = np.meshgrid(y, x, indexing="ij")
    coords = np.stack((xx, yy, np.zeros_like(xx)), axis=-1)[..., None, :]
    node = np.arange(12, dtype=np.float32).reshape(3, 4)
    fields = np.empty((1, 10, 3, 4, 1, 3), dtype=np.float32)
    for frame in range(10):
        fields[0, frame, ..., 0, 0] = node + frame
        fields[0, frame, ..., 0, 1] = (node % 4) ** 2 - yy + frame / 10
        fields[0, frame, ..., 0, 2] = np.sin(node) + frame / 5
    with h5py.File(path, "w") as handle:
        handle.create_dataset("fields", data=fields)
        handle.create_dataset("coordinates", data=coords)
        handle.create_dataset("time", data=np.arange(10, dtype=np.float32))
        handle.create_dataset("conditions", data=np.zeros((1, 0), dtype=np.float32))
        handle.attrs["field_names"] = '["a", "b", "c"]'
        handle.attrs["grid_shape"] = "[3, 4]"


def _brute_w2(left, right, weights):
    """Independent two-pointer integration of the weighted quantile coupling."""
    left_order = np.argsort(left, kind="stable")
    right_order = np.argsort(right, kind="stable")
    left_values, right_values = left[left_order], right[right_order]
    left_mass = weights[left_order].astype(np.float64)
    right_mass = weights[right_order].astype(np.float64)
    left_mass, right_mass = left_mass / left_mass.sum(), right_mass / right_mass.sum()
    i = j = 0
    cost = 0.0
    while i < len(left_values) and j < len(right_values):
        shared = min(left_mass[i], right_mass[j])
        cost += shared * (left_values[i] - right_values[j]) ** 2
        left_mass[i] -= shared
        right_mass[j] -= shared
        if left_mass[i] < 1e-14:
            i += 1
        if right_mass[j] < 1e-14:
            j += 1
    return cost


def test_provider_generic_and_compact_batches_subset_and_resume(tmp_path):
    path = tmp_path / "nonuniform.h5"
    _fixture(path)
    dataset = open_field_dataset(
        {
            "path": str(path),
            "split": "train",
            "normalization": "none",
            "coordinate_dim": 2,
            "physical_measure": "native_rectilinear_area",
        }
    )
    try:
        weights = dataset.physical_weights
        assert weights.shape == (12,)
        torch.testing.assert_close(weights.sum(), torch.tensor(28.0, dtype=weights.dtype))
        ids = torch.tensor([0, 2, 3, 5, 7, 8, 10, 11])
        protocol = SensorProtocol(field_counts={"a": 2}, seed=17)
        generic = build_observation_batch([dataset[0], dataset[1]], protocol, query_indices=ids)
        compact = _CompactH5Dataset(dataset, protocol, 8, fixed_query_indices=ids).__getitems__(
            [(0, 0), (1, 0)]
        )[0]
        # Exercise the vectorized resident assembler on CPU without opening a
        # CUDA context; the resident source adds the same dataset measure.
        resident_assembled = _assemble_random_batch(
            torch.stack([dataset[0].values, dataset[1].values]),
            dataset.spatial_coords[:, :2],
            protocol,
            dataset.field_names,
            query_points=8,
            generator=torch.Generator().manual_seed(17),
            query_generator=torch.Generator().manual_seed(100020),
            sample_ids=generic.sample_ids,
            logical_shape=dataset.grid_shape,
            conditions=torch.zeros(2, 0),
            times=torch.tensor([0.0, 1.0]),
            fixed_query_indices=ids,
        )
        attach_batch_measure(dataset, resident_assembled)
        batches = (generic, compact, resident_assembled)
        for batch in batches:
            torch.testing.assert_close(batch.target_fields[0], dataset[0].values[ids])
            torch.testing.assert_close(batch.target_fields[1], dataset[1].values[ids])
            expected_mass = weights[ids].expand(2, -1)
            torch.testing.assert_close(
                batch_measure_context(batch)["generated_weights"], expected_mass
            )
            assert (
                batch_measure_context(batch)["measure_metadata"]["native_measure_sha256"]
                == dataset.physical_measure_receipt["sha256"]
            )

        chosen = torch.tensor([0, 3, 4, 7])
        subsets = [
            subset_query_batch(batch, 4, indices=chosen, generator=torch.Generator().manual_seed(4))
            for batch in batches
        ]
        selected_ids = ids[chosen]
        for selected in subsets:
            torch.testing.assert_close(
                selected.metadata["query_indices"], selected_ids.expand(2, -1)
            )
            torch.testing.assert_close(
                batch_measure_context(selected)["generated_weights"],
                weights[selected_ids].expand(2, -1),
            )
            torch.testing.assert_close(selected.target_fields, generic.target_fields[:, chosen])

        cfg = {
            "definition": "marginal_mixed_weighted_v3",
            "enabled": True,
            "weight": 1.0,
            "target_use": "paired_supervised",
            "units": "model_units",
            "reference_bank": {"enabled": False},
            "components": {
                "self": {"enabled": True, "weight": 1.0},
                "mutual": {"enabled": False},
                "cross": {
                    "enabled": True,
                    "weight": 1.0,
                    "directions": 32,
                    "top_fraction": 1.0,
                    "seed": 1234,
                    "include_axes": False,
                    "qmc": True,
                },
            },
        }
        f = build_coherence_family(
            "global_distribution", cfg, dataset.data_spec, dataset.normalizer
        )
        f.bind_measure_artifact(dataset.physical_measure_receipt, weights=weights)
        prediction = (
            subsets[0].target_fields
            + torch.tensor([0.3, -0.2, 0.1]) * torch.tensor([1.0, -1.0, 0.5, 2.0])[None, :, None]
        )
        outcomes = [
            f(prediction, selected.target_fields, context=batch_measure_context(selected))
            for selected in subsets
        ]
        for outcome in outcomes[1:]:
            torch.testing.assert_close(
                outcomes[0].per_sample_cost, outcome.per_sample_cost, rtol=0, atol=0
            )
        left = prediction[0].numpy().astype(np.float64)
        right = subsets[0].target_fields[0].numpy().astype(np.float64)
        mass = weights[selected_ids].numpy()
        marginal = np.mean([_brute_w2(left[:, i], right[:, i], mass) for i in range(3)])
        bank = (
            f.components_by_key["cross_joint_topk_swd"]
            .directions.detach()
            .numpy()
            .astype(np.float64)
        )
        mixed = np.mean(
            [_brute_w2(left @ direction, right @ direction, mass) for direction in bank]
        )
        assert float(outcomes[0].per_sample_cost[0]) == pytest.approx(
            marginal + mixed, rel=2e-6, abs=2e-7
        )

        checkpoint = tmp_path / "family.pt"
        torch.save(f.state_artifact(), checkpoint)
        restored = build_coherence_family(
            "global_distribution", cfg, dataset.data_spec, dataset.normalizer
        )
        restored.load_state_artifact(torch.load(checkpoint, map_location="cpu", weights_only=True))
        torch.testing.assert_close(restored.state_artifact()["measure_support"]["weights"], weights)
        assert restored.state_artifact()["measure"] == dataset.physical_measure_receipt
        restored_result = restored(
            prediction, subsets[0].target_fields, context=batch_measure_context(subsets[0])
        )
        torch.testing.assert_close(
            restored_result.per_sample_cost, outcomes[0].per_sample_cost, rtol=0, atol=0
        )
    finally:
        dataset.close()


def test_explicit_provider_measure_preserves_receipt_and_validates_weights(tmp_path):
    path = tmp_path / "explicit.h5"
    _fixture(path)
    base = {"path": str(path), "split": "train", "normalization": "none", "coordinate_dim": 2}
    policy = {
        "provider": "explicit_weights",
        "weights": [1.0, 2.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        "receipt": {"mesh": "example", "units": "m^2"},
    }
    dataset = open_field_dataset({**base, "physical_measure": policy})
    try:
        assert dataset.physical_measure_receipt["metadata"]["provider_receipt"] == policy["receipt"]
        torch.testing.assert_close(
            dataset[0].metadata["auxiliary"]["physical_weights"],
            torch.tensor(policy["weights"], dtype=torch.float64),
        )
        assert dataset[0].metadata["physical_measure"] == "explicit_weights"
    finally:
        dataset.close()
    for bad in ([-1.0] * 12, [0.0] * 12, [1.0] * 11):
        with pytest.raises(ValueError):
            open_field_dataset({**base, "physical_measure": {**policy, "weights": bad}})


def test_explicit_provider_rejects_mass_outside_mask():
    from types import SimpleNamespace

    from phycoflow_reconstruction.data.physical_measure import configure_physical_measure

    sample = SimpleNamespace(
        reconstruction_unit="snapshot",
        coordinates_raw=torch.tensor([[0.0], [1.0], [2.0]]),
        logical_shape=(3,),
        valid_points=torch.tensor([True, False, True]),
    )

    class Dataset:
        def __getitem__(self, index):
            return sample

    policy = {"provider": "explicit_weights", "weights": [1.0, 0.0, 2.0], "receipt": {"units": "m"}}
    result = configure_physical_measure(Dataset(), policy)
    assert result.physical_weights.tolist() == [1.0, 0.0, 2.0]
    assert result.physical_measure_receipt["valid_point_count"] == 2
    with pytest.raises(ValueError):
        configure_physical_measure(Dataset(), {**policy, "weights": [1.0, 1.0, 2.0]})

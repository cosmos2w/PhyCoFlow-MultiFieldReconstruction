"""Acceptance coverage for the bounded topology upgrades in work plan 1002."""

from __future__ import annotations

import itertools
import json
from types import SimpleNamespace

import pytest
import torch

from phycoflow_reconstruction.coherence.families.topology.family import TopologyFamily
from phycoflow_reconstruction.coherence.families.topology.geometry import (
    build_raster_map,
    rasterize_fields,
)
from phycoflow_reconstruction.coherence.families.topology.persistence import (
    Diagram,
    sliced_diagram_distance,
    sliced_diagram_distances,
)
from phycoflow_reconstruction.coherence.families.topology.persistence_objective import (
    PersistenceTopologyObjective,
)
from phycoflow_reconstruction.contracts import DataSpec
from phycoflow_reconstruction.data.normalization import FieldNormalizer
from phycoflow_reconstruction.evaluation.topology_set import TopologySetAccumulator
from phycoflow_reconstruction.training.run_store import file_sha256


def _mutual_config(*, bank_size: int | None = 8, subset_size: int = 4) -> dict:
    mutual = {
        "enabled": True,
        "groups": [["u", "v"]],
        "seed": 117,
        "direction_floor": 0.25,
        "offset_range": 0.7,
    }
    if bank_size is None:
        mutual["lines"] = 4
    else:
        mutual["line_bank_size"] = bank_size
        mutual["training_subset_size"] = subset_size
    return {
        "strategy": "cubical_persistence",
        "target_use": "paired_supervised",
        "units": "model_units",
        "fields": ["u", "v"],
        "geometry": {"grid_shape": [5, 6], "periodic": True},
        "filtration": {
            "dimensions": [0, 1],
            "directions": ["sublevel", "superlevel"],
            "smoothing_sigma": 0.0,
        },
        "persistence": {"projections": 8, "essential_weight": 0.1},
        "components": {
            "self": {"enabled": False, "weight": 0.0},
            "mutual": mutual,
        },
    }


def _coordinates(height: int, width: int) -> torch.Tensor:
    y = torch.linspace(0.0, 1.0, height)
    x = torch.linspace(0.0, 1.0, width)
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    return torch.stack((grid_x, grid_y), dim=-1).reshape(-1, 2)


def test_master_bank_subset_resume_and_full_bank_phases() -> None:
    objective = PersistenceTopologyObjective(_mutual_config(bank_size=16, subset_size=4), ("u", "v"))
    selected = objective.selected_line_indices(37, "train")
    assert len(selected) == 1
    assert len(selected[0]) == 4
    assert len(set(selected[0])) == 4
    assert selected == objective.selected_line_indices(37, "train")
    assert selected != objective.selected_line_indices(38, "train")
    assert objective.selected_line_indices(37, "evaluation") == (tuple(range(16)),)
    assert objective.selected_line_indices(37, "calibration") == (tuple(range(16)),)

    restored = PersistenceTopologyObjective(_mutual_config(bank_size=16, subset_size=4), ("u", "v"))
    restored.load_state_dict(objective.state_dict(), strict=True)
    assert restored.selected_line_indices(37, "train") == selected
    torch.testing.assert_close(
        restored.line_directions_0, objective.line_directions_0, rtol=0, atol=0
    )
    torch.testing.assert_close(restored.line_offsets_0, objective.line_offsets_0, rtol=0, atol=0)

    legacy = PersistenceTopologyObjective(_mutual_config(bank_size=None), ("u", "v"))
    assert legacy.version == "3"
    assert legacy.selected_line_indices(None, "train") == (tuple(range(4)),)


def test_master_bank_set_evaluation_restores_frozen_artifact(tmp_path) -> None:
    config = _mutual_config(bank_size=8, subset_size=4)
    spec = DataSpec(field_names=("u", "v"), field_units=("a.u.", "a.u."),
                    coordinate_dim=2, logical_shape=(5, 6),
                    mesh_type="structured")
    normalizer = FieldNormalizer.identity(2)
    family = TopologyFamily(config, spec, normalizer)
    objective = family.spatial_objective
    # A valid serialized permutation must be replayed, not silently replaced
    # by regenerating the default seed bank at evaluation time.
    objective.line_directions_0.copy_(objective.line_directions_0.roll(1, 0))
    objective.line_offsets_0.copy_(objective.line_offsets_0.roll(1, 0))
    run_dir = tmp_path / "child"
    (run_dir / "artifacts").mkdir(parents=True)
    artifact = run_dir / "artifacts/topology_family.pt"
    torch.save(family.state_artifact(), artifact)
    source = tmp_path / "source.pt"
    torch.save({"source": 1}, source)
    (run_dir / "run_manifest.json").write_text(json.dumps({
        "stage": "post_training", "parent_run": str(tmp_path),
        "source_checkpoint": str(source),
        "source_hashes": {"checkpoint": file_sha256(source)},
        "coherence_family_state_sha256s": {"topology": file_sha256(artifact)},
    }))
    runtime = SimpleNamespace(
        config={"coherence": {"families": {"topology": config},
                              "compute_budget": {"point_count": 30, "query_seed": 17}}},
        dataset=SimpleNamespace(data_spec=spec, normalizer=normalizer, field_names=("u", "v")),
        device=torch.device("cpu"), run_dir=run_dir, coherence_artifact_run_dir=None,
    )
    accumulator = TopologySetAccumulator.build(runtime)
    restored = accumulator.family.spatial_objective
    torch.testing.assert_close(restored.line_directions_0, objective.line_directions_0,
                               rtol=0, atol=0)
    torch.testing.assert_close(restored.line_offsets_0, objective.line_offsets_0, rtol=0, atol=0)
    assert restored.selected_line_indices(3, "evaluation") == (tuple(range(8)),)
    assert accumulator.family_artifact_provenance["artifact_sha256"] == file_sha256(artifact)
    source.write_bytes(b"changed source")
    with pytest.raises(ValueError, match="source checkpoint hash"):
        TopologySetAccumulator.build(runtime)


def test_positive_line_uses_the_max_reduction_and_subset_mean_weights() -> None:
    config = _mutual_config(bank_size=5, subset_size=3)
    config["filtration"]["directions"] = ["sublevel"]
    objective = PersistenceTopologyObjective(config, ("u", "v"))
    x = torch.randn(2, 2, 5, 6, generator=torch.Generator().manual_seed(28))
    selected = ((0, 2, 4),)
    bank, labels, weights, _ = objective._filtrations(x, selected)
    a = objective.line_directions_0[0].to(x)
    b = objective.line_offsets_0[0].to(x)
    expected = ((x[:, :2] - b[None, :, None, None]) / a[None, :, None, None]).amax(1)
    torch.testing.assert_close(bank[0], expected)
    assert labels == ["mutual.persistence"] * 3
    assert len(weights) == 3


def test_scalar_and_batched_distance_match_for_empty_and_essential_diagrams() -> None:
    pytest.importorskip("gudhi")
    generator = torch.Generator().manual_seed(51)
    left, right, inputs = [], [], []
    for left_count, right_count, essential_count in ((0, 0, 1), (0, 3, 2), (4, 1, 0)):
        left_finite = torch.randn(
            left_count, 2, generator=generator, dtype=torch.float64, requires_grad=True
        )
        right_finite = torch.randn(
            right_count, 2, generator=generator, dtype=torch.float64, requires_grad=True
        )
        left_essential = torch.randn(
            essential_count, generator=generator, dtype=torch.float64, requires_grad=True
        )
        right_essential = torch.randn(
            essential_count, generator=generator, dtype=torch.float64, requires_grad=True
        )
        left.append(Diagram(left_finite, left_essential))
        right.append(Diagram(right_finite, right_essential))
        inputs.extend((left_finite, right_finite, left_essential, right_essential))

    batched = sliced_diagram_distances(
        left, right, projections=8, essential_weight=0.2, normalization=30
    )
    scalar = torch.stack(
        [
            sliced_diagram_distance(
                a, b, projections=8, essential_weight=0.2, normalization=30
            )
            for a, b in zip(left, right)
        ]
    )
    torch.testing.assert_close(batched, scalar, rtol=1e-12, atol=1e-12)
    batch_gradient = torch.autograd.grad(batched.sum(), inputs, retain_graph=True)
    scalar_gradient = torch.autograd.grad(scalar.sum(), inputs)
    for actual, expected in zip(batch_gradient, scalar_gradient):
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


def test_family_sampling_state_artifact_roundtrip_and_cache_identity() -> None:
    pytest.importorskip("gudhi")
    config = _mutual_config(bank_size=8, subset_size=3)
    spec = DataSpec(("u", "v"), ("1", "1"), 2, (5, 6), mesh_type="structured")
    family = TopologyFamily(config, spec, FieldNormalizer.identity(2))
    assert family.version == "4"
    artifact = family.state_artifact()
    restored = TopologyFamily(config, spec, FieldNormalizer.identity(2))
    restored.load_state_artifact(artifact)
    assert (
        restored.spatial_objective.selected_line_indices(12, "train")
        == family.spatial_objective.selected_line_indices(12, "train")
    )
    assert restored.state_artifact()["sampling_state"] == artifact["sampling_state"]

    generator = torch.Generator().manual_seed(95)
    reference = torch.randn(1, 30, 2, generator=generator)
    prediction = reference + 0.1 * torch.randn_like(reference)
    coordinates = _coordinates(5, 6).unsqueeze(0)
    cache = {}
    family(
        prediction,
        reference,
        coordinates=coordinates,
        context={"persistence_cache": cache, "phase": "train", "global_step": 12},
    )
    family(
        prediction + 0.02,
        reference,
        coordinates=coordinates,
        context={"persistence_cache": cache, "phase": "train", "global_step": 12},
    )
    with pytest.raises(ValueError, match="reference cache target/coordinates changed"):
        family(
            prediction,
            reference.clone(),
            coordinates=coordinates,
            context={"persistence_cache": cache, "phase": "train", "global_step": 12},
        )
    with pytest.raises(ValueError, match="reference cache line selection changed"):
        family(
            prediction,
            reference,
            coordinates=coordinates,
            context={"persistence_cache": cache, "phase": "train", "global_step": 13},
        )


def test_uniform_subset_gradients_approximate_the_full_bank_gradient() -> None:
    pytest.importorskip("gudhi")
    config = _mutual_config(bank_size=8, subset_size=4)
    objective = PersistenceTopologyObjective(config, ("u", "v"))
    generator = torch.Generator().manual_seed(863)
    reference = torch.randn(1, 2, 5, 6, generator=generator)
    base = reference + 0.23 * torch.randn(reference.shape, generator=generator)

    full_prediction = base.clone().requires_grad_()
    full = objective(full_prediction, reference, phase="evaluation")
    full_loss = full.component_results["topology.mutual.persistence"].scalar_loss
    full_gradient = torch.autograd.grad(full_loss, full_prediction)[0]
    assert full.diagnostics["line_sampling"]["mode"] == "full_master_bank"
    assert full.diagnostics["scalar_filtrations"] == 16

    subset_gradients = []
    for step in range(64):
        prediction = base.clone().requires_grad_()
        result = objective(
            prediction,
            reference,
            global_step=step,
            phase="train",
            reference_cache={},
        )
        assert result.diagnostics["line_sampling"]["selected_indices"] == [
            list(objective.selected_line_indices(step, "train")[0])
        ]
        assert result.diagnostics["scalar_filtrations"] == 8
        loss = result.component_results["topology.mutual.persistence"].scalar_loss
        subset_gradients.append(torch.autograd.grad(loss, prediction)[0])

    mean_gradient = torch.stack(subset_gradients).mean(0)
    relative_error = torch.linalg.vector_norm(mean_gradient - full_gradient) / torch.linalg.vector_norm(
        full_gradient
    ).clamp_min(1e-12)
    assert relative_error.item() < 0.25


def test_complete_native_gather_order_sparse_discrepancy_and_average_guard() -> None:
    height, width = 8, 8
    coordinates = _coordinates(height, width)
    y, x = coordinates[:, 1], coordinates[:, 0]
    values = torch.sin(torch.pi * x) * torch.cos(torch.pi * y)
    values += 0.35 * torch.sin(3 * torch.pi * x + 2 * torch.pi * y)
    native_fields = values.reshape(1, height * width, 1)

    order = torch.randperm(height * width, generator=torch.Generator().manual_seed(74))
    shuffled_map = build_raster_map(
        coordinates[order],
        grid_shape=(height, width),
        native_shape=(height, width),
        neighbors=4,
    )
    gathered = rasterize_fields(
        native_fields[:, order],
        shuffled_map.neighbor_indices,
        shuffled_map.neighbor_weights,
        (height, width),
    )
    torch.testing.assert_close(gathered[0, 0], values.reshape(height, width), rtol=0, atol=0)
    assert shuffled_map.diagnostics["native_gather_applied"] == 1
    assert shuffled_map.neighbor_indices.shape == (height * width, 1)

    sparse_coords = coordinates[torch.tensor(
        [row * width + column for row, column in itertools.product((0, 2, 5, 7), repeat=2)]
    )]
    sparse_values = (
        torch.sin(torch.pi * sparse_coords[:, 0]) * torch.cos(torch.pi * sparse_coords[:, 1])
        + 0.35 * torch.sin(3 * torch.pi * sparse_coords[:, 0] + 2 * torch.pi * sparse_coords[:, 1])
    )
    sparse_map = build_raster_map(
        sparse_coords,
        grid_shape=(height, width),
        native_shape=(height, width),
        neighbors=4,
        antialias_downsample=True,
    )
    sparse_raster = rasterize_fields(
        sparse_values[None, :, None],
        sparse_map.neighbor_indices,
        sparse_map.neighbor_weights,
        (height, width),
    )[0, 0]
    discrepancy = (sparse_raster - values.reshape(height, width)).abs().mean().item()
    assert discrepancy > 0.01
    assert sparse_map.diagnostics["complete_native_cartesian_grid"] == 0
    assert sparse_map.diagnostics["native_area_average_applied"] == 0
    assert sparse_map.diagnostics["antialias_applied"] == 0

    downsampled_map = build_raster_map(
        coordinates,
        grid_shape=(4, 4),
        native_shape=(8, 8),
        antialias_downsample=True,
    )
    assert downsampled_map.diagnostics["native_area_average_applied"] == 1
    assert downsampled_map.diagnostics["antialias_applied"] == 1
    reduced = rasterize_fields(
        native_fields,
        downsampled_map.neighbor_indices,
        downsampled_map.neighbor_weights,
        (4, 4),
    )
    expected = torch.nn.functional.avg_pool2d(
        native_fields.reshape(1, 8, 8, 1).permute(0, 3, 1, 2), 2
    )
    torch.testing.assert_close(reduced, expected, rtol=0, atol=1e-7)

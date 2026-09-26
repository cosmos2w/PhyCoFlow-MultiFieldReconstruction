"""Focused correctness checks for the unified three-family coherence runtime."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from helpers.multifamily import _cross_config, _global_config

from phycoflow_reconstruction.coherence import build_coherence_family
from phycoflow_reconstruction.coherence.families.cross_spectrum.statistics import (
    symmetric_relative_coherence_score,
)
from phycoflow_reconstruction.coherence.families.topology.betti_curves import betti_curves
from phycoflow_reconstruction.config import load_config
from phycoflow_reconstruction.config.validate import validate_config
from phycoflow_reconstruction.contracts import DataSpec, ModelCapabilities, ObservationBatch
from phycoflow_reconstruction.data.normalization import FieldNormalizer
from phycoflow_reconstruction.training.post_training import _coherence_objective


def _coordinates(size: int = 6) -> torch.Tensor:
    return torch.stack(
        torch.meshgrid(torch.linspace(0, 1, size), torch.linspace(0, 1, size), indexing="ij"),
        dim=-1,
    ).reshape(-1, 2)


def test_symmetric_spectral_coherence_score_is_bounded_and_calibrated():
    reference = torch.tensor([1.0, 2.0, 3.0])

    assert symmetric_relative_coherence_score(reference, reference, 1.0e-8) == 1.0
    assert symmetric_relative_coherence_score(torch.zeros_like(reference), reference, 1.0e-8) == 0.0
    partial = symmetric_relative_coherence_score(reference * 0.8, reference, 1.0e-8)
    assert 0.0 < partial < 1.0


def _topology_config() -> dict:
    return {
        "target_use": "paired_supervised",
        "units": "model_units",
        "fields": ["u", "v"],
        "geometry": {
            "grid_shape": [6, 6],
            "axes": [0, 1],
            "neighbors": 1,
            "periodic": False,
        },
        "filtration": {
            "quantiles": [0.25, 0.5, 0.75],
            "dimensions": [0, 1],
            "directions": ["superlevel"],
            "sharpness": 12.0,
            "smoothing_sigma": 0.0,
        },
        "components": {
            "self": {"enabled": True, "weight": 1.0},
            "mutual": {
                "enabled": True,
                "weight": 1.0,
                "pairs": [["u", "v"]],
                "lines": 2,
            },
        },
    }


def _persistence_topology_config() -> dict:
    return {
        "strategy": "cubical_persistence",
        "target_use": "paired_supervised",
        "units": "model_units",
        "fields": ["u", "v"],
        "geometry": {
            "grid_shape": [6, 6],
            "axes": [0, 1],
            "neighbors": 1,
            "periodic": False,
        },
        "filtration": {
            "dimensions": [0, 1],
            "directions": ["sublevel", "superlevel"],
            "smoothing_sigma": 0.0,
        },
        "persistence": {"distance": "sliced_wasserstein", "projections": 4},
        "components": {
            "self": {"enabled": True, "weight": 1.0},
            "mutual": {
                "enabled": True,
                "weight": 0.5,
                "groups": [["u", "v"]],
                "lines": 2,
                "seed": 11,
            },
        },
    }


def test_cross_spectrum_is_ensemble_only_geometry_fixed_and_differentiable():
    spec = DataSpec(("u", "v"), ("1", "1"), 2, (6, 6))
    family = build_coherence_family(
        "cross_spectrum", _cross_config(), spec, FieldNormalizer.identity(2)
    )
    coordinates = _coordinates().unsqueeze(0).expand(4, -1, -1)
    generated = torch.randn(4, 36, 2, requires_grad=True)
    reference = torch.randn_like(generated)
    result = family(generated, reference, coordinates=coordinates)
    assert result.per_sample_cost is None
    assert len(result.component_results) == 3
    assert (
        len(
            result.component_results["cross_spectrum.same_frequency.magnitude_squared"].diagnostics[
                "per_pair_mean_square"
            ]
        )
        == 1
    )
    same_scores = result.component_results[
        "cross_spectrum.same_frequency.magnitude_squared"
    ].diagnostics["per_pair_coherence_score"]
    assert len(same_scores) == 1
    assert 0.0 <= same_scores[0] <= 1.0
    assert (
        len(
            result.component_results[
                "cross_spectrum.cross_frequency.band_energy_coupling"
            ].diagnostics["per_pair_mean_square"]
        )
        == 1
    )
    cross_scores = result.component_results[
        "cross_spectrum.cross_frequency.band_energy_coupling"
    ].diagnostics["per_pair_coherence_score"]
    assert len(cross_scores) == 1
    assert 0.0 <= cross_scores[0] <= 1.0
    assert (
        len(
            result.component_results["cross_spectrum.band_energy.log_power"].diagnostics[
                "per_band_field_mean_square"
            ]
        )
        == 3
    )
    result.scalar_loss.backward()
    assert generated.grad is not None and torch.isfinite(generated.grad).all()
    assert torch.linalg.vector_norm(generated.grad) > 0
    restored = build_coherence_family(
        "cross_spectrum", _cross_config(), spec, FieldNormalizer.identity(2)
    )
    restored.load_state_artifact(family.state_artifact())
    assert restored.geometry_sha256 == family.geometry_sha256
    assert torch.equal(restored.eigenvectors, family.eigenvectors)


def test_exact_forward_betti_curves_distinguish_component_and_hole():
    levels = torch.tensor([0.5])
    full = torch.ones(5, 5)
    full_curves = betti_curves(full, levels, (0, 1), sharpness=12.0, periodic=False)
    assert full_curves[0].item() == 1
    assert full_curves[1].item() == 0

    ring = torch.ones(5, 5)
    ring[1:4, 1:4] = 0
    ring_curves = betti_curves(ring, levels, (0, 1), sharpness=12.0, periodic=False)
    assert ring_curves[0].item() == 1
    assert ring_curves[1].item() == 1


@pytest.mark.parametrize(
    "topology_strategy", ["betti_curves", "spatial_self_mutual", "cubical_persistence"]
)
def test_one_rollout_composes_all_three_families_and_backpropagates(topology_strategy):
    class ToyFlow(torch.nn.Module):
        capabilities = ModelCapabilities("point", True, True, False, True)

        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(0.2))
            self.rollout_calls = 0

        def sample_source(self, batch, *, generator=None):
            assert batch.target_fields is None
            self.rollout_calls += 1
            return torch.zeros(batch.query_coords.shape[0], batch.query_coords.shape[1], 2)

        def velocity(self, batch, state, time):
            assert batch.target_fields is None
            x, y = batch.query_coords.unbind(dim=-1)
            pattern = torch.stack(
                (
                    torch.sin(torch.pi * x) * torch.sin(torch.pi * y),
                    torch.cos(2.0 * torch.pi * x) * torch.sin(torch.pi * y),
                ),
                dim=-1,
            )
            return self.scale * pattern

    spec = DataSpec(("u", "v"), ("1", "1"), 2, (6, 6))
    normalizer = FieldNormalizer.identity(2)
    if topology_strategy == "cubical_persistence":
        pytest.importorskip("gudhi", reason="cubical persistence requires the topology extra")
        topology_config = _persistence_topology_config()
    else:
        topology_config = _topology_config()
    if topology_strategy == "spatial_self_mutual":
        topology_config.update(
            {
                "strategy": topology_strategy,
                "anchor": {"provider": "raw", "fields": ["v"]},
                "components": {
                    "self": {"enabled": True, "weight": 1.0},
                    "anchor_self": {"enabled": True, "weight": 0.2},
                    "mutual": {
                        "enabled": True,
                        "weight": 1.0,
                        "carrier_field": "u",
                        "detach_carrier": True,
                        "gradient_scale": 0.2,
                        "lines": 4,
                    },
                },
            }
        )
    families = {
        "global_distribution": build_coherence_family(
            "global_distribution", _global_config(), spec, normalizer
        ),
        "cross_spectrum": build_coherence_family(
            "cross_spectrum", _cross_config(), spec, normalizer
        ),
        "topology": build_coherence_family("topology", topology_config, spec, normalizer),
    }
    coordinates = _coordinates().unsqueeze(0).expand(4, -1, -1)
    batch = ObservationBatch(
        obs_coords=coordinates[:, :2],
        obs_values=torch.zeros(4, 2, 1),
        obs_field_ids=torch.tensor([[0, 1]]).expand(4, -1),
        obs_valid_mask=torch.ones(4, 2, dtype=torch.bool),
        query_coords=coordinates,
        query_valid_mask=torch.ones(4, 36, dtype=torch.bool),
        target_fields=torch.randn(4, 36, 2, generator=torch.Generator().manual_seed(23)),
        sample_ids=("a", "b", "c", "d"),
        obs_indices=torch.tensor([[0, 1]]).expand(4, -1),
        logical_shapes=((6, 6),) * 4,
        metadata={"query_indices": torch.arange(36).expand(4, -1)},
    )
    model = ToyFlow()
    result, reference_ids = _coherence_objective(
        model,
        batch,
        families,
        {name: None for name in families},
        {
            "coherence": {"compute_budget": {"batch_size": 4, "point_count": 36}},
            "rollout": {"steps": 1, "solver": "euler"},
            "observation_consistency": {"mode": "none", "final_clamp": False},
        },
        step=0,
        generator=torch.Generator().manual_seed(2),
    )
    assert reference_ids == ("a", "b", "c", "d")
    assert model.rollout_calls == 1
    assert result.per_sample_cost is None
    assert {path.split(".")[0] for path in result.component_results} == {
        "global_distribution",
        "cross_spectrum",
        "topology",
    }
    family_losses = {
        name: details["scalar_loss"] for name, details in result.diagnostics["families"].items()
    }
    assert set(family_losses) == {"global_distribution", "cross_spectrum", "topology"}
    assert all(torch.isfinite(loss) and loss.item() > 0 for loss in family_losses.values())
    torch.testing.assert_close(result.scalar_loss.detach(), sum(family_losses.values()))
    if topology_strategy == "spatial_self_mutual":
        assert {path for path in result.component_results if path.startswith("topology.")} == {
            "topology.self.region",
            "topology.self.connectivity",
            "topology.anchor_self.spatial",
            "topology.mutual.spatial",
        }
        assert (
            len(result.component_results) == 10
        )  # Three distribution, three spectral, four topology.
    if topology_strategy == "cubical_persistence":
        assert result.diagnostics["families"]["topology"]["diagram_distance"] == (
            "sliced_wasserstein"
        )
        for path in ("topology.self.persistence", "topology.mutual.persistence"):
            term = result.component_results[path]
            assert torch.isfinite(term.scalar_loss) and term.scalar_loss.item() > 0
            gradient, = torch.autograd.grad(term.scalar_loss, model.scale, retain_graph=True)
            assert torch.isfinite(gradient) and gradient.abs() > 0
    # The existing trainer receives one coherence scalar alongside its native data loss.
    data_loss = (model.scale - 0.5).square()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    before = model.scale.detach().clone()
    (data_loss + result.scalar_loss).backward()
    assert model.scale.grad is not None and torch.isfinite(model.scale.grad)
    optimizer.step()
    assert not torch.equal(model.scale.detach(), before)
    restored_topology = build_coherence_family("topology", topology_config, spec, normalizer)
    restored_topology.load_state_artifact(families["topology"].state_artifact())
    assert restored_topology.geometry_sha256 == families["topology"].geometry_sha256
    assert torch.equal(restored_topology.neighbor_indices, families["topology"].neighbor_indices)


def test_combined_demo50_template_satisfies_strict_config_contract():
    project = Path(__file__).resolve().parents[1]
    config = load_config(
        project / "cases/turbulent_combustion/configs/posttrain/demo50_all_coherence.yaml"
    )
    validate_config(config)

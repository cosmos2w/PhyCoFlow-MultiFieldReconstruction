"""Real-pairing checks for unbiased complete-filtration topology sampling."""

import pytest
import torch

pytest.importorskip("gudhi")

from helpers.persistence import config as persistence_config

from phycoflow_reconstruction.coherence.families.topology.family import TopologyFamily
from phycoflow_reconstruction.coherence.families.topology.filtration_sampling import (
    TopologyFiltrationSampler,
    sampled_family_forward,
)
from phycoflow_reconstruction.coherence.families.topology.persistence_objective import (
    PersistenceTopologyObjective,
)
from phycoflow_reconstruction.contracts import DataSpec
from phycoflow_reconstruction.data.normalization import FieldNormalizer


def run11_objective():
    config = persistence_config()
    config["fields"] = ["a", "b"]
    config["filtration"] = {
        "directions": ["superlevel", "sublevel"],
        "dimensions": [0, 1],
    }
    config["components"] = {
        "self": {"enabled": True, "weight": 1.7, "fields": ["a", "b"]},
        "mutual": {
            "enabled": True,
            "weight": 0.63,
            "groups": [["a", "b"]],
            "lines": 1,
            "seed": 197,
        },
    }
    config["persistence"] = {"projections": 32, "essential_weight": 0.1}
    return config


def tensors():
    generator = torch.Generator().manual_seed(8241)
    reference = torch.randn((2, 2, 7, 9), generator=generator, dtype=torch.float64)
    prediction = reference + 0.17 * torch.randn(
        reference.shape, generator=generator, dtype=torch.float64
    )
    return prediction, reference


def value_and_input_gradient(objective, prediction, reference):
    generated = prediction.detach().clone().requires_grad_()
    result = objective(generated, reference)
    gradient = torch.autograd.grad(result.scalar_loss, generated)[0]
    return result.scalar_loss.detach(), gradient.detach()


@pytest.mark.parametrize("strategy, expected_choices, expected_units", [("p3", 2, 3), ("s2", 8, 2)])
def test_exact_probability_weighted_expectation_matches_gudhi_oracle(
    strategy, expected_choices, expected_units
):
    objective = PersistenceTopologyObjective(run11_objective(), ("a", "b"))
    prediction, reference = tensors()
    oracle_value, oracle_gradient = value_and_input_gradient(objective, prediction, reference)
    sampler = TopologyFiltrationSampler(objective, strategy=strategy, seed=92)
    choices = sampler.enumerate_choices()
    assert len(choices) == expected_choices
    assert {len(choice.indices) for choice in choices} == {expected_units}
    assert len(sampler.units) == 6

    values = []
    gradients = []
    reference_cache = {}
    for choice in choices:
        generated = prediction.detach().clone().requires_grad_()
        estimate = sampler.evaluate(
            generated, reference, choice=choice, reference_cache=reference_cache
        )
        gradients.append(torch.autograd.grad(estimate.scalar_loss, generated)[0].detach())
        values.append(estimate.scalar_loss.detach())
        assert estimate.diagnostics["estimator_value"] == "T_estimate"
        assert len(estimate.sampled_units) == expected_units

    torch.testing.assert_close(torch.stack(values).mean(), oracle_value, rtol=1e-10, atol=1e-11)
    torch.testing.assert_close(
        torch.stack(gradients).mean(0), oracle_gradient, rtol=1e-10, atol=1e-11
    )


def test_full_helper_matches_legacy_value_and_input_gradient():
    objective = PersistenceTopologyObjective(run11_objective(), ("a", "b"))
    prediction, reference = tensors()
    oracle_value, oracle_gradient = value_and_input_gradient(objective, prediction, reference)
    sampler = TopologyFiltrationSampler(objective, strategy="full", seed=1)
    generated = prediction.detach().clone().requires_grad_()
    estimate = sampler.evaluate(generated, reference)
    gradient = torch.autograd.grad(estimate.scalar_loss, generated)[0]

    torch.testing.assert_close(estimate.scalar_loss, oracle_value, rtol=1e-10, atol=1e-11)
    torch.testing.assert_close(gradient, oracle_gradient, rtol=1e-10, atol=1e-11)


def test_sampler_rng_is_private_reproducible_and_resumable():
    objective = PersistenceTopologyObjective(run11_objective(), ("a", "b"))
    sampler = TopologyFiltrationSampler(objective, strategy="s2", seed=510)
    initial_global_state = torch.get_rng_state()
    sampler.draw()
    saved = sampler.state_dict()
    expected_next = sampler.draw()

    resumed = TopologyFiltrationSampler(objective, strategy="s2", seed=999)
    resumed.load_state_dict(saved)
    actual_next = resumed.draw()
    assert actual_next == expected_next
    assert torch.equal(torch.get_rng_state(), initial_global_state)

    incompatible = PersistenceTopologyObjective(
        {
            **run11_objective(),
            "components": {
                **run11_objective()["components"],
                "mutual": {
                    "enabled": True,
                    "weight": 0.63,
                    "groups": [["a", "b"]],
                    "lines": 1,
                    "seed": 198,
                },
            },
        },
        ("a", "b"),
    )
    with pytest.raises(ValueError, match="objective changed"):
        TopologyFiltrationSampler(incompatible, strategy="s2").load_state_dict(saved)


def test_sampled_family_adapter_preserves_full_family_result():
    height, width = 5, 6
    config = run11_objective()
    config["geometry"] = {
        "grid_shape": [height, width],
        "neighbors": 1,
        "periodic": True,
        "periods": [1.0, 1.0],
    }
    config["filtration"]["smoothing_sigma"] = 0.0
    spec = DataSpec(("a", "b"), ("1", "1"), 2, (height, width), mesh_type="structured")
    family = TopologyFamily(config, spec, FieldNormalizer.identity(2))
    y, x = torch.meshgrid(
        torch.arange(height, dtype=torch.float64) / height,
        torch.arange(width, dtype=torch.float64) / width,
        indexing="ij",
    )
    coordinates = torch.stack((x, y), dim=-1).reshape(1, height * width, 2).expand(2, -1, -1)
    generator = torch.Generator().manual_seed(19)
    reference = torch.randn((2, height * width, 2), generator=generator, dtype=torch.float64)
    prediction = (
        reference + 0.1 * torch.randn(reference.shape, generator=generator, dtype=torch.float64)
    ).requires_grad_()
    context = {"persistence_cache": {}}
    expected = family(prediction, reference, coordinates=coordinates, context=context)
    sampler = TopologyFiltrationSampler(family.spatial_objective, strategy="full", seed=3)
    actual = sampled_family_forward(
        family,
        prediction,
        reference,
        coordinates,
        sampler,
        context={"persistence_cache": {}},
    )

    torch.testing.assert_close(actual.scalar_loss, expected.scalar_loss, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(
        actual.per_sample_cost, expected.per_sample_cost, rtol=1e-6, atol=1e-7
    )
    assert set(actual.component_results) == {
        "topology.self.persistence",
        "topology.mutual.persistence",
    }
    for name in actual.component_results:
        torch.testing.assert_close(
            actual.component_results[name].per_sample_cost,
            expected.component_results[name].per_sample_cost,
            rtol=1e-6,
            atol=1e-7,
        )
        assert actual.component_results[name].diagnostics["estimator_value"] == "T_full"

    sampled = sampled_family_forward(
        family,
        prediction,
        reference,
        coordinates=coordinates,
        strategy="p3",
        seed=23,
        context={"persistence_cache": {}},
    )
    assert sampled.diagnostics["estimator_value"] == "T_estimate"
    assert sampled.diagnostics["scalar_filtrations"] == 3
    assert sampled.diagnostics["full_scalar_filtrations"] == 6
    assert all(
        term.diagnostics["estimator_value"] == "T_estimate"
        for term in sampled.component_results.values()
    )

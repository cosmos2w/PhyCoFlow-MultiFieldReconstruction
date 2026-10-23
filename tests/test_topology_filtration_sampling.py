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


def generalized_objective():
    config = persistence_config()
    config["fields"] = ["b", "a", "c"]
    config["filtration"] = {
        "directions": ["superlevel", "sublevel"],
        "dimensions": [0, 1],
    }
    config["components"] = {
        "self": {"enabled": True, "weight": 2.3, "fields": ["speed", "c"]},
        "mutual": {
            "enabled": True,
            "weight": 0.47,
            "groups": [["a", "b"], ["b", "c"]],
            "lines": 2,
            "seed": 197,
        },
    }
    config["persistence"] = {
        "projections": 7,
        "essential_weight": 0.1,
        "descriptors": {"speed": {"provider": "vector_magnitude", "fields": ["a", "b"]}},
    }
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


def test_polarity_sampling_generalizes_and_matches_reordered_subset_oracle():
    objective = PersistenceTopologyObjective(generalized_objective(), ("a", "b", "c"))
    assert objective.fields == ("b", "a", "c", "speed")
    assert objective.self_fields == ("speed", "c")
    assert objective.self_ids == (3, 2)
    generator = torch.Generator().manual_seed(42017)
    reference = torch.randn((2, 3, 7, 8), generator=generator, dtype=torch.float64)
    prediction = reference + 0.13 * torch.randn(
        reference.shape, generator=generator, dtype=torch.float64
    )
    oracle_value, oracle_gradient = value_and_input_gradient(objective, prediction, reference)

    sampler = TopologyFiltrationSampler(objective, strategy="polarity", seed=91)
    assert sampler.PROVIDER_ID == "topology.complete_filtration.v1"
    assert len(sampler.units) == 12
    assert sum(unit.component == "self" for unit in sampler.units) == 4
    assert sum(unit.component == "mutual" for unit in sampler.units) == 8
    for unit in sampler.units:
        if unit.component == "self":
            assert (
                unit.descriptor_index == objective.self_ids[objective.self_fields.index(unit.field)]
            )
            assert unit.coefficient == pytest.approx(objective.weights["self.persistence"] / 4)
        else:
            assert unit.descriptor_index is None
            assert unit.coefficient == pytest.approx(
                objective.weights["mutual.persistence"] * unit.line_weight / 8
            )

    support = sampler.support_summary()
    assert support.choice_probabilities == (0.5, 0.5)
    assert support.inclusion_probabilities == (0.5,) * 12
    assert len(support.choices) == 2
    for choice in support.choices:
        selected = [sampler.units[index] for index in choice.indices]
        assert len(selected) == 6
        assert len({unit.direction for unit in selected}) == 1
        assert choice.inclusion_probabilities == (0.5,) * 6
    for left, left_unit in enumerate(sampler.units):
        for right, right_unit in enumerate(sampler.units):
            expected_joint = 0.5 if left_unit.direction == right_unit.direction else 0.0
            assert support.joint_inclusion_probabilities[left][right] == expected_joint

    values = []
    gradients = []
    reference_cache = {}
    for choice in support.choices:
        generated = prediction.detach().clone().requires_grad_()
        estimate = sampler.evaluate(
            generated, reference, choice=choice, reference_cache=reference_cache
        )
        values.append(estimate.scalar_loss.detach())
        gradients.append(torch.autograd.grad(estimate.scalar_loss, generated)[0].detach())
    probabilities = torch.tensor(support.choice_probabilities, dtype=prediction.dtype)
    values = torch.stack(values)
    gradients = torch.stack(gradients)
    torch.testing.assert_close((values * probabilities).sum(), oracle_value, rtol=1e-10, atol=1e-11)
    torch.testing.assert_close(
        (gradients * probabilities[:, None, None, None, None]).sum(0),
        oracle_gradient,
        rtol=1e-10,
        atol=1e-11,
    )
    support_variance = (probabilities * (values - oracle_value).square()).sum()
    assert support_variance > 0


def test_support_summary_preserves_s2_correlations_and_nested_probability():
    objective = PersistenceTopologyObjective(run11_objective(), ("a", "b"))
    sampler = TopologyFiltrationSampler(objective, strategy="s2", seed=17)
    support = sampler.support_summary()
    assert support.choice_probabilities == (0.125,) * 8
    self_indices = [unit.index for unit in sampler.units if unit.component == "self"]
    mutual_indices = [unit.index for unit in sampler.units if unit.component == "mutual"]
    assert all(support.inclusion_probabilities[index] == 0.25 for index in self_indices)
    assert all(support.inclusion_probabilities[index] == 0.5 for index in mutual_indices)
    assert support.joint_inclusion_probabilities[self_indices[0]][self_indices[1]] == 0.0
    assert support.joint_inclusion_probabilities[mutual_indices[0]][mutual_indices[1]] == 0.0
    assert support.joint_inclusion_probabilities[self_indices[0]][mutual_indices[0]] == 0.125

    # Nest the polarity draw under a whole-family Bernoulli action with p=1/2.
    # The exact support includes the no-family point and both conditional strata.
    polarity = TopologyFiltrationSampler(objective, strategy="polarity", seed=19)
    nested = [((), 0.5)]
    polarity_support = polarity.support_summary()
    nested.extend(
        (choice.indices, 0.5 * mass)
        for choice, mass in zip(polarity_support.choices, polarity_support.choice_probabilities)
    )
    for unit in polarity.units:
        marginal = sum(mass for indices, mass in nested if unit.index in indices)
        assert marginal == 0.25
        assert 1.0 / marginal == 4.0
    first_direction = polarity.units[0].direction
    same_direction = [unit.index for unit in polarity.units if unit.direction == first_direction]
    other_direction = [unit.index for unit in polarity.units if unit.direction != first_direction]
    same_pair_joint = sum(
        mass
        for indices, mass in nested
        if same_direction[0] in indices and same_direction[1] in indices
    )
    cross_pair_joint = sum(
        mass
        for indices, mass in nested
        if same_direction[0] in indices and other_direction[0] in indices
    )
    assert same_pair_joint == 0.25
    assert cross_pair_joint == 0.0


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


def test_legacy_v1_state_replays_for_identity_mapping_and_legacy_strategy():
    objective = PersistenceTopologyObjective(run11_objective(), ("a", "b"))
    sampler = TopologyFiltrationSampler(objective, strategy="s2", seed=510)
    assert objective.self_ids == (0, 1)
    sampler.draw()
    legacy_state = sampler.state_dict()
    assert legacy_state["version"] == TopologyFiltrationSampler.STATE_VERSION == 2
    legacy_state["version"] = 1
    expected_next = sampler.draw()

    resumed = TopologyFiltrationSampler(objective, strategy="s2", seed=999)
    resumed.load_state_dict(legacy_state)
    assert resumed.draw() == expected_next


@pytest.mark.parametrize("self_fields", [("b", "a"), ("b",)])
def test_legacy_v1_state_rejects_reordered_or_subset_descriptor_mapping(self_fields):
    config = run11_objective()
    config["components"]["self"]["fields"] = list(self_fields)
    objective = PersistenceTopologyObjective(config, ("a", "b"))
    assert tuple(objective.self_ids) != tuple(range(len(objective.self_fields)))
    sampler = TopologyFiltrationSampler(objective, strategy="full", seed=31)
    legacy_state = sampler.state_dict()
    legacy_state["version"] = 1

    resumed = TopologyFiltrationSampler(objective, strategy="full", seed=999)
    with pytest.raises(ValueError, match="legacy v1 sampler state requires identity descriptor"):
        resumed.load_state_dict(legacy_state)


def test_legacy_v1_state_rejects_new_polarity_strategy_even_for_identity_mapping():
    objective = PersistenceTopologyObjective(run11_objective(), ("a", "b"))
    assert objective.self_ids == (0, 1)
    sampler = TopologyFiltrationSampler(objective, strategy="polarity", seed=11)
    legacy_state = sampler.state_dict()
    legacy_state["version"] = 1

    resumed = TopologyFiltrationSampler(objective, strategy="polarity", seed=999)
    with pytest.raises(ValueError, match="legacy v1 sampler state requires identity descriptor"):
        resumed.load_state_dict(legacy_state)


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

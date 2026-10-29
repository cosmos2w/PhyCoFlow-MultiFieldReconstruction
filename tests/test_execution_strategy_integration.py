"""Focused checks for generic family dispatch and dense endpoint execution."""

import pytest
import torch
from torch import nn

from phycoflow_reconstruction.contracts import FamilyResult, ModelCapabilities, ObservationBatch
from phycoflow_reconstruction.training.fidelity_budget import field_mse
from phycoflow_reconstruction.training.post_training import _coherence_objective

FAMILY_WEIGHTS = {
    "global_distribution": 0.4,
    "cross_spectrum": 0.7,
    "topology": 1.2,
}


class _EndpointModel(nn.Module):
    capabilities = ModelCapabilities("point", False, True, False, False)

    def __init__(self):
        super().__init__()
        self.value = nn.Parameter(torch.tensor(0.25, dtype=torch.float64))
        self.calls = 0
        self.input_batches = []

    def differentiable_reconstruct(self, batch, *, steps, generator):
        assert batch.target_fields is None
        self.calls += 1
        self.input_batches.append(batch)
        return self.value.expand(*batch.query_coords.shape[:2], 1)


class _RecordingFamily:
    target_use = "paired_supervised"

    def __init__(self, family_name):
        self.family_name = family_name
        self.family_weight = FAMILY_WEIGHTS[family_name]
        self.calls = 0

    def __call__(self, prediction, reference, *, coordinates, context):
        self.calls += 1
        assert coordinates.shape[:2] == prediction.shape[:2]
        assert tuple(context["sample_ids"]) == ("case-0", "case-1")
        per_sample = (prediction - reference).square().mean((1, 2))
        return FamilyResult({}, per_sample, per_sample.mean())


def _batch():
    batch_size, point_count = 2, 5
    generator = torch.Generator().manual_seed(35)
    target = torch.randn((batch_size, point_count, 1), generator=generator, dtype=torch.float64)
    return ObservationBatch(
        obs_coords=torch.zeros((batch_size, 2, 1), dtype=torch.float64),
        obs_values=torch.zeros((batch_size, 2, 1), dtype=torch.float64),
        obs_field_ids=torch.zeros((batch_size, 2), dtype=torch.long),
        obs_valid_mask=torch.ones((batch_size, 2), dtype=torch.bool),
        query_coords=torch.linspace(0, 1, point_count, dtype=torch.float64)
        .view(1, point_count, 1)
        .expand(batch_size, -1, -1),
        query_valid_mask=torch.ones((batch_size, point_count), dtype=torch.bool),
        target_fields=target,
        sample_ids=("case-0", "case-1"),
        logical_shapes=((point_count,), (point_count,)),
    )


def _call_objective(model, batch, families, *, execution, endpoint_context):
    family_losses = {}
    result, reference_ids = _coherence_objective(
        model,
        batch,
        families,
        None,
        {
            "coherence": {"compute_budget": {"batch_size": 2, "point_count": 3}},
            "rollout": {"steps": 1, "solver": "euler"},
            "observation_consistency": {"mode": "none", "final_clamp": False},
        },
        step=4,
        generator=torch.Generator().manual_seed(751),
        endpoint_context=endpoint_context,
        family_losses=family_losses,
        execution=execution,
    )
    return result, reference_ids, family_losses


@pytest.mark.parametrize("excluded_family", tuple(FAMILY_WEIGHTS))
def test_generic_dispatch_bypasses_each_a_b_t_family_and_preserves_dense_endpoint(
    excluded_family,
):
    batch = _batch()
    model = _EndpointModel()
    families = {name: _RecordingFamily(name) for name in FAMILY_WEIGHTS}
    endpoint_context = {}
    probability = 0.5
    execution = {
        "family_probabilities": {name: probability for name in FAMILY_WEIGHTS},
        "family_inclusions": {name: name != excluded_family for name in FAMILY_WEIGHTS},
        "within_family": {},
    }

    sampled, reference_ids, family_losses = _call_objective(
        model,
        batch,
        families,
        execution=execution,
        endpoint_context=endpoint_context,
    )

    included_families = set(FAMILY_WEIGHTS) - {excluded_family}
    assert model.calls == 1
    assert families[excluded_family].calls == 0
    assert all(families[name].calls == 1 for name in included_families)
    assert set(sampled.diagnostics["families"]) == included_families
    assert set(family_losses) == included_families
    assert reference_ids == ("case-0", "case-1")
    assert sampled.per_sample_cost is None
    assert "prediction" in endpoint_context and "target" in endpoint_context
    assert "mask" in endpoint_context
    assert endpoint_context["target"].shape == (2, 3, 1)
    assert endpoint_context["mask"].shape == (2, 3)

    expected_scalar = sampled.scalar_loss.new_zeros(())
    for name in included_families:
        family_value = (endpoint_context["prediction"] - endpoint_context["target"]).square().mean()
        expected_scalar = expected_scalar + FAMILY_WEIGHTS[name] * family_value / probability
    torch.testing.assert_close(sampled.scalar_loss, expected_scalar)

    # Family selection happens after the one shared reconstruction and cannot
    # change the dense endpoint payload or its fieldwise risk.
    dense_model = _EndpointModel()
    dense_endpoint = {}
    dense_families = {name: _RecordingFamily(name) for name in FAMILY_WEIGHTS}
    dense, _, _ = _call_objective(
        dense_model,
        batch,
        dense_families,
        execution=None,
        endpoint_context=dense_endpoint,
    )
    assert dense_model.calls == 1
    assert all(family.calls == 1 for family in dense_families.values())
    torch.testing.assert_close(endpoint_context["prediction"], dense_endpoint["prediction"])
    torch.testing.assert_close(endpoint_context["target"], dense_endpoint["target"])
    torch.testing.assert_close(endpoint_context["mask"], dense_endpoint["mask"])
    torch.testing.assert_close(
        field_mse(
            endpoint_context["prediction"],
            endpoint_context["target"],
            endpoint_context["mask"],
        ),
        field_mse(dense_endpoint["prediction"], dense_endpoint["target"], dense_endpoint["mask"]),
    )
    assert dense.per_sample_cost is not None

from copy import deepcopy

import pytest
import torch

pytest.importorskip("gudhi")
from scalable_test_helpers import config as scale_config
from test_topology_finite_primary import config as shape_config
from test_topology_finite_primary import coords

from phycoflow_reconstruction.coherence import build_coherence_family
from phycoflow_reconstruction.coherence.families.multiscale_relation import MultiscaleRelationFamily
from phycoflow_reconstruction.coherence.families.topology.family import TopologyFamily
from phycoflow_reconstruction.contracts import DataSpec
from phycoflow_reconstruction.data.normalization import FieldNormalizer
from phycoflow_reconstruction.evaluation.frozen_coherence import FrozenCoherenceObserver
from phycoflow_reconstruction.training.scalable_fitting import fit_scalable_artifacts


def make_observer(mode):
    fields = ("CO", "T")
    normalizer = FieldNormalizer.identity(2)
    spec = DataSpec(fields, ("1", "1"), 2, (16,))
    g = torch.Generator().manual_seed(56)
    states = torch.randn(8, 16, 2, generator=g, dtype=torch.double)
    points = coords(8).double()
    sc = scale_config(fields)
    sc["geometry"] = {"operator": "fem_p1", "coordinate_scaling": "none"}
    sc["bands"] = {"count": 1}
    scale = MultiscaleRelationFamily(sc, spec, normalizer)
    scale.fit_reference_tensors(points[0], states, sample_ids=[str(i) for i in range(8)])
    shape_settings = shape_config()
    shape_settings["persistence"] = {"definition": "scalable_pooled_v1", "projections": 4}
    shape_settings["components"]["mutual"] = {"enabled": True}
    shape = TopologyFamily(shape_settings, spec, normalizer)
    state_cfg = {
        "definition": "conditional_quantile_fair_v1",
        "base_definition": "marginal_mixed_weighted_v3",
        "conditional_draws": 2,
        "fields": fields,
        "target_use": "paired_supervised",
        "units": "model_units",
        "components": {
            "self": {"enabled": True},
            "mutual": {"enabled": False},
            "cross": {
                "enabled": True,
                "directions": 8,
                "top_fraction": 1.0,
                "include_axes": False,
                "qmc": True,
            },
        },
    }
    state = build_coherence_family("global_distribution", state_cfg, spec, normalizer)
    families = {"global_distribution": state, "multiscale_spectrum": scale, "topology": shape}
    provenance = {
        "split": "train",
        "sample_ids": ["a", "b"],
        "field_order": list(fields),
        "source_checkpoint_sha256": "a" * 64,
        "normalizer_sha256": "b" * 64,
        "support_sha256": "c" * 64,
        "config_sha256": "d" * 64,
        "definition_version": "v1",
    }
    target = states[:2]
    source = states[2:6].reshape(2, 2, 16, 2)
    mass = torch.ones(2, 16, dtype=torch.double)
    artifact = fit_scalable_artifacts(
        families=families,
        source=source,
        reference=target,
        coordinates=points[:2],
        provenance=provenance,
        source_weights=mass,
        reference_weights=mass,
        shape_batches=1,
        candidate_count=8,
    )
    return (
        FrozenCoherenceObserver(
            artifact,
            families,
            expected_provenance=provenance,
            reference=target,
            coordinates=points[:2],
            sample_ids=["v0", "v1"],
            member_weights=mass,
            reference_weights=mass,
            mode=mode,
            query_seed=3,
            query_batches=2,
        ),
        source.transpose(0, 1),
        families,
        artifact,
    )


@pytest.mark.parametrize("mode", ["deterministic", "conditional"])
def test_frozen_observer_repeat_swapping_draws_and_no_training_mutation(mode):
    observer, draws, families, artifact = make_observer(mode)
    draws = draws[:1] if mode == "deterministic" else draws
    states = {k: deepcopy(v.state_dict()) for k, v in families.items()}
    before = deepcopy(artifact)
    rng = torch.random.get_rng_state()
    first = observer.score(draws)
    second = observer.score(draws)
    assert first["observer_sha256"] == second["observer_sha256"]
    for name in ("state", "scale", "shape"):
        torch.testing.assert_close(
            first["per_sample"][name], second["per_sample"][name], rtol=0, atol=0
        )
    if mode == "conditional":
        reverse = observer.score(draws.flip(0))
        for name in ("state", "scale", "shape"):
            torch.testing.assert_close(
                first["per_sample"][name], reverse["per_sample"][name], rtol=1e-6, atol=1e-6
            )
    assert torch.equal(rng, torch.random.get_rng_state())
    for k, v in families.items():
        for name, tensor in v.state_dict().items():
            torch.testing.assert_close(states[k][name], tensor, rtol=0, atol=0)
    assert before["sha256"] == artifact["sha256"]
    with pytest.raises(ValueError, match="draw count"):
        observer.score(draws[:0])


def test_frozen_observer_rejects_mutated_comparison_definition():
    observer, draws, _, _ = make_observer("deterministic")
    observer.coefficients["state.marginal"] = 2.0
    with pytest.raises(ValueError, match="observer content changed"):
        observer.score(draws[:1])

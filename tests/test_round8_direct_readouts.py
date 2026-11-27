from copy import deepcopy
import itertools
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts.analysis import car_round8_support_readout as support
from scripts.analysis.car_round8_child_relations import (
    FROZEN_KEYS, summarize_direct, validate_direct_companion,
)
from scripts.analysis.car_round8_cached_summary import ref
from phycoflow_reconstruction.coherence.families.multiscale_spectrum.statistics import piecewise_means


def test_state_groups_are_direct_weighted_root_norms_on_one_shared_embedding(monkeypatch):
    family = SimpleNamespace(
        component_weights={"self_marginal_w2": 2., "cross_joint_topk_swd": 3.},
        components_by_key={"self_marginal_w2": SimpleNamespace(channel_weights=torch.ones(2, dtype=torch.float64))},
        _in_declared_units=lambda member, truth: (member, truth),
        _endpoint_measures=lambda member, truth, context, required: (context["generated_weights"], context["reference_weights"]),
    )
    truth = torch.zeros((1, 2, 2), dtype=torch.float64)
    points = [(1., 1.), (2., 0.), (0., 2.), (3., -1.)]
    members = [torch.tensor([[point, point]], dtype=torch.float64) for point in points]
    bank = {"directions": torch.tensor([[1 / math.sqrt(2), 1 / math.sqrt(2)]], dtype=torch.float64)}
    masses = torch.tensor([.25, .75], dtype=torch.float64)
    historical = support.state_triplet(family, bank, members, truth, masses)
    original, calls = support.exact_common_embeddings, []

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(support, "exact_common_embeddings", counted)
    total, groups = support.state_triplet(family, bank, members, truth, masses, return_groups=True)
    assert len(calls) == 1
    assert total == historical

    def distance(a, b):
        d0, d1 = a[0]-b[0], a[1]-b[1]
        marginal = math.sqrt(2 * (d0*d0 + d1*d1) / 2)
        mixed = math.sqrt(3 * ((d0+d1) / math.sqrt(2)) ** 2)
        return marginal, mixed, math.hypot(marginal, mixed)

    for index, name in enumerate(("marginal", "mixed", "total")):
        attraction = sum(distance(point, (0., 0.))[index] for point in points) / 4
        separation = sum(distance(a, b)[index] for a, b in itertools.combinations(points, 2)) / 12
        actual = total if name == "total" else groups[name]
        assert np.allclose(actual, [attraction, separation, attraction-separation], rtol=1e-12, atol=1e-12)
    assert not np.isclose(groups["marginal"][0] + groups["mixed"][0], total[0])


def test_piece_mean_errors_use_frozen_piece_mass_and_memberwise_absolute(monkeypatch):
    measures = torch.tensor([[1., 3., 0.], [0., 0., 2.]], dtype=torch.float64)
    truth = torch.tensor([[[1., 10.], [3., 20.], [10., 30.]]], dtype=torch.float64)
    errors = torch.tensor([
        [[1., -2.], [3., 2.], [-1., 4.]],
        [[-1., 4.], [-3., -2.], [3., -4.]],
        [[2., 1.], [2., 1.], [-2., 2.]],
        [[-2., -3.], [-2., 3.], [4., -2.]],
    ], dtype=torch.float64)
    members = (truth + errors[:, None]).contiguous()
    calls = []

    def statistics(_family, fields):
        calls.append(1)
        return torch.ones((len(fields), 1, 2), dtype=torch.float64), piecewise_means(fields, measures), None

    monkeypatch.setattr(support, "_scale_statistics", statistics)
    family = SimpleNamespace(filters=torch.ones((1, 1)), power_floor=torch.ones((1, 2)),
                             piece_measures=measures, units="model_units", field_names=("first", "second"))
    _, direct = support.scale_context(family, members, truth, return_piece_means=True)
    assert len(calls) == 1
    individual = np.stack(((errors[:, 0].numpy() + 3 * errors[:, 1].numpy()) / 4,
                           errors[:, 2].numpy()), axis=1)
    assert np.allclose(direct["signed_error_mean_member"], individual.mean(axis=0))
    assert np.allclose(direct["absolute_error_mean_member"], np.abs(individual).mean(axis=0))
    assert not np.allclose(direct["absolute_error_mean_member"], np.abs(individual.mean(axis=0)))
    state = SimpleNamespace(component_weights={"self_marginal_w2": 2., "cross_joint_topk_swd": 3.})
    weights, geometry = support.direct_readout_metadata(state, family, ["first", "second"])
    assert weights == {"marginal": 2., "mixed": 3.}
    assert geometry == {"piece_indices": [0, 1], "mass_fractions": [2/3, 1/3],
                        "unit_space": "model_units", "field_order": ["first", "second"]}


def test_direct_summary_and_companion_bindings_reject_changed_source(tmp_path):
    source_file = tmp_path / "source.json"
    source_file.write_text("{}")
    source_ref = ref(source_file)
    def fake_ref(name): return {"path": "/unused/" + name, "sha256": name}
    spec = {name: fake_ref(name) for name in FROZEN_KEYS}
    assessment = {"sample_ids": ["frame-1"], "field_order": ["first"], "field_units": ["unknown"],
                  "physical_mask": "selected support", "source_selected": fake_ref("selected")}
    historical = {"spec": fake_ref("spec")}
    cached = {"inputs": {"joint500_assessment": fake_ref("joint500")}}
    companion = {"schema": "car_round8_state_scale_direct_companion_v1", "status": "completed",
                 "split": "validation", "members": 4, "test_locked": True, **{k: assessment[k] for k in
                    ("sample_ids", "field_order", "field_units", "physical_mask")},
                 "records": [{"sample_id": "frame-1"}],
                 "inputs": {"source_readout": source_ref, "historical_assessment": fake_ref("joint500"),
                            "source_selected": fake_ref("selected"), "spec": fake_ref("spec"), **spec}}
    validate_direct_companion(companion, assessment, cached, historical, spec, spec, source_file)
    bad = deepcopy(companion)
    bad["inputs"]["source_selected"] = fake_ref("different")
    with pytest.raises(ValueError, match="source_selected"):
        validate_direct_companion(bad, assessment, cached, historical, spec, spec, source_file)
    def side(value):
        return {"state_groups": {"marginal": [value, 1., value-1.], "mixed": [value+1, 2., value-1.]},
                "piece_mean": {"signed_error_mean_member": [[value]], "absolute_error_mean_member": [[value+1]]}}
    records = [{"models": {"SOURCE": side(3.), "candidate": side(5.)}} for _ in range(3)]
    indices = np.zeros((8, 3), dtype=int)
    result = summarize_direct(records, 2., indices)
    assert np.allclose(result["state_groups"]["marginal"]["paired_delta_frozen_train_units"]["mean"], [1., 0., 1.])
    assert np.allclose(result["piece_mean"]["signed_error_mean_member"]["paired_delta"]["mean"], [[2.]])

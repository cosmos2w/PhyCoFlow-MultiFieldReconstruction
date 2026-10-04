"""R3 acceptance: genuine GUDHI structure, legacy arithmetic, and frozen scales."""
import math
from copy import deepcopy

import pytest
import torch

from phycoflow_reconstruction.coherence.families.topology.persistence import (
    Diagram, cubical_diagrams, sliced_diagram_distance, sliced_diagram_distances,
)
from phycoflow_reconstruction.coherence.families.topology.persistence_objective import PersistenceTopologyObjective


def configuration(mode="finite_primary_v1"):
    return {"strategy": "cubical_persistence", "target_use": "paired_supervised",
            "fields": ["a", "b"], "geometry": {"periodic": False, "grid_shape": [8, 8]},
            "filtration": {"smoothing_sigma": 0},
            "persistence": {"aggregation": mode, "projections": 8},
            "components": {"self": {"enabled": True, "weight": 2},
                           "mutual": {"enabled": True, "weight": 1, "groups": [["a", "b"]],
                                      "line_bank_size": 16, "training_subset_size": 4, "seed": 1729}}}


def legacy_oracle(left, right, projections, normalization):
    x, y = left.finite, right.finite
    theta = (torch.arange(projections, dtype=x.dtype) + 0.5) * math.pi / projections
    directions = torch.stack((theta.cos(), theta.sin()))
    a = (torch.cat((x, y.mean(-1, keepdim=True).expand(-1, 2))) @ directions).sort(0).values
    b = (torch.cat((y, x.mean(-1, keepdim=True).expand(-1, 2))) @ directions).sort(0).values
    return ((a-b).abs().sum(0).mean() / normalization
            + 0.1 * (left.essential.sort().values-right.essential.sort().values).abs().sum())


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("batch_mode", ["0", "1"])
def test_legacy_exact_values_gradients_and_decomposition(packed, batch_mode, monkeypatch):
    monkeypatch.setenv("PHYCOFLOW_TOPOLOGY_BATCHED", batch_mode)
    x = torch.tensor([[0.1, 1.2], [0.2, 0.8]], dtype=torch.float64, requires_grad=True)
    e = torch.tensor([0.3], dtype=torch.float64, requires_grad=True)
    left = [Diagram(x, e), Diagram(x[:0], e)]
    right = [Diagram(x.detach()+0.2, e.detach()+0.1), Diagram(x.detach()[:0], e.detach()+0.1)]
    oracle = sum(legacy_oracle(a,b,8,64) for a,b in zip(left,right))
    if packed:
        actual = sliced_diagram_distances(left,right,projections=8,normalization=64).sum()
        f,e_part = sliced_diagram_distances(left,right,projections=8,normalization=64,return_components=True)
        rebuilt = (f+0.1*e_part).sum()
    else:
        actual = sum(sliced_diagram_distance(a,b,projections=8,normalization=64) for a,b in zip(left,right))
        parts = [sliced_diagram_distance(a,b,projections=8,normalization=64,return_components=True) for a,b in zip(left,right)]
        rebuilt = sum(f+0.1*e_part for f,e_part in parts)
    torch.testing.assert_close(actual,oracle,atol=1e-15,rtol=1e-14)
    torch.testing.assert_close(actual,rebuilt,atol=0,rtol=0)
    oracle_grad = torch.autograd.grad(oracle,(x,e),retain_graph=True)
    actual_grad = torch.autograd.grad(actual,(x,e),retain_graph=True)
    rebuilt_grad = torch.autograd.grad(rebuilt,(x,e))
    for a,b,c in zip(oracle_grad,actual_grad,rebuilt_grad):
        torch.testing.assert_close(a,b,atol=1e-15,rtol=1e-14)
        torch.testing.assert_close(b,c,atol=0,rtol=0)


def test_constant_rectangle_h0_minimum_and_extrema_only_change():
    target = torch.full((1,8,8),2.0,dtype=torch.float64)
    shifted = (target+0.5).requires_grad_()
    left,right = cubical_diagrams(shifted,periodic=False),cubical_diagrams(target,periodic=False)
    assert right[0][0].essential.tolist() == [2.0]
    assert right[0][0].finite.shape == (0,2)
    assert right[0][1].essential.numel() == 0
    f,e = sliced_diagram_distance(left[0][0],right[0][0],return_components=True)
    assert f == 0 and e == 0.5
    assert torch.autograd.grad(e,shifted)[0].abs().sum() == 1


def test_hole_change_has_finite_signal_at_matching_extrema():
    # A raised ring vertex delays the H1 birth without changing either extremum.
    target = torch.ones(1,8,8,dtype=torch.float64)
    target[:,2:6,2] = 0; target[:,2:6,5] = 0
    target[:,2,2:6] = 0; target[:,5,2:6] = 0
    prediction = target.clone(); prediction[:,2,3] = 0.4
    prediction.requires_grad_()
    left,right = cubical_diagrams(prediction,periodic=False),cubical_diagrams(target,periodic=False)
    f,e = sliced_diagram_distance(left[0][1],right[0][1],normalization=64,return_components=True)
    assert f > 0 and e == 0
    assert target.min() == prediction.min() and target.max() == prediction.max()
    assert torch.autograd.grad(f,prediction)[0].abs().sum() > 0


def test_ties_identity_nonfinite_and_essential_counts_match_historical_behavior():
    field = torch.zeros(1,8,8,requires_grad=True)
    d = cubical_diagrams(field,periodic=False)[0][0]
    f,e = sliced_diagram_distance(d,d,return_components=True)
    assert f == e == 0
    with pytest.raises(ValueError,match="essential class counts differ"):
        sliced_diagram_distance(d,Diagram(d.finite,d.essential[:0]),return_components=True)
    field = field.detach(); field[0,0,0] = float("nan")
    with pytest.raises(FloatingPointError,match="finite"):
        cubical_diagrams(field,periodic=False)


def test_train_scales_weights_sampling_freeze_and_resume():
    cfg = configuration(); obj = PersistenceTopologyObjective(cfg,("a","b"))
    target = torch.randn(2,2,8,8,generator=torch.Generator().manual_seed(6))
    prediction = (target+0.2*target.roll(1,-1)).requires_grad_()
    with pytest.raises(ValueError,match="frozen TRAIN"):
        obj(prediction,target,global_step=0)
    source = obj(prediction,target,phase="calibration")
    records = [obj.collect_source_components(source)]
    calibration = obj.freeze_source_calibration(records,{"split":"train","source_checkpoint_sha256":"source"})
    result = obj(prediction,target,phase="evaluation")
    weights = obj.reporting_group_weights()
    assert sum(weights.values()) == pytest.approx(1)
    assert sum(w for g,w in weights.items() if g.startswith("self.")) == pytest.approx(2/3)
    expected = sum(w*(0.9*records[0][g]["finite"]/calibration["scales"]["finite"][g]
                      +0.1*records[0][g]["essential"]/calibration["scales"]["essential"][g])
                   for g,w in weights.items())
    assert result.scalar_loss.item() == pytest.approx(expected,rel=2e-6)
    legacy = sum(obj.weights[category]*source.component_results[f"topology.{category}"].scalar_loss
                 for category in obj.weights)
    torch.testing.assert_close(result.component_results["topology.legacy"].scalar_loss,legacy)
    restored = PersistenceTopologyObjective(cfg,("a","b")); restored.load_source_calibration(calibration)
    torch.testing.assert_close(restored(prediction,target,phase="evaluation").scalar_loss,result.scalar_loss,atol=0,rtol=0)
    assert obj.selected_line_indices(19,"train") == restored.selected_line_indices(19,"train")
    assert len(obj.selected_line_indices(19,"train")[0]) == 4
    assert len(obj.selected_line_indices(19,"evaluation")[0]) == 16
    with pytest.raises(ValueError,match="already frozen"):
        obj.freeze_source_calibration(records,{"split":"train","source_checkpoint_sha256":"source"})
    corrupt = deepcopy(calibration); corrupt["scientific_source"]["gudhi_version"] = "wrong"
    with pytest.raises(ValueError,match="identity"):
        restored.load_source_calibration(corrupt)
    with pytest.raises(ValueError,match="identity"):
        restored.load_source_calibration(calibration,representation="native_grid")


def test_zero_source_terms_positive_bounded_scales_and_explicit_undefined_ratios():
    obj = PersistenceTopologyObjective(configuration(),("a","b"))
    target = torch.ones(2,2,8,8)
    result = obj(target,target,phase="calibration")
    artifact = obj.freeze_source_calibration([obj.collect_source_components(result)],
                                            {"split":"train","source_checkpoint_sha256":"source"})
    for part in ("finite","essential"):
        assert all(s == 1e-4 for s in artifact["scales"][part].values())
    result = obj(target,target,phase="evaluation")
    assert result.scalar_loss == 0
    assert all(value is None for row in result.diagnostics["component_source_relative_ratios"].values() for value in row.values())
    with pytest.raises(ValueError,match="TRAIN"):
        PersistenceTopologyObjective(configuration(),("a","b")).freeze_source_calibration(
            [obj.collect_source_components(result)],{"split":"test","source_checkpoint_sha256":"source"})


def test_one_ph_pass_per_generated_and_reference_bank(monkeypatch):
    import phycoflow_reconstruction.coherence.families.topology.persistence_objective as module
    obj = PersistenceTopologyObjective(configuration(),("a","b"))
    calls = []
    engine = module.cubical_diagrams
    def counted(*args,**kwargs):
        calls.append(kwargs.get("cacheable",False))
        return engine(*args,**kwargs)
    monkeypatch.setattr(module,"cubical_diagrams",counted)
    target = torch.randn(2,2,8,8,generator=torch.Generator().manual_seed(3))
    source = obj(target+0.1,target,phase="calibration")
    assert calls == [True,False]
    obj.freeze_source_calibration([obj.collect_source_components(source)],
                                  {"split":"train","source_checkpoint_sha256":"source"})
    cache = {}; calls.clear()
    obj(target+0.2,target,global_step=3,reference_cache=cache)
    obj(target+0.3,target,global_step=3,reference_cache=cache)
    assert calls == [True,False,False]


@pytest.mark.parametrize("mode", ["legacy", "finite_primary_v1"])
def test_family_artifact_binds_scales_and_resume_line_bank(mode):
    from phycoflow_reconstruction.coherence.families.topology.family import TopologyFamily
    from phycoflow_reconstruction.contracts import DataSpec
    from phycoflow_reconstruction.data.normalization import FieldNormalizer
    from phycoflow_reconstruction.training.post_training import _component_reports
    cfg = configuration(mode); spec = DataSpec(("a","b"),("1","1"),2,(8,8),mesh_type="structured")
    family = TopologyFamily(cfg,spec,FieldNormalizer.identity(2))
    y,x = torch.meshgrid(torch.arange(8)/7,torch.arange(8)/7,indexing="ij")
    coords = torch.stack((x,y),-1).reshape(1,64,2)
    target = torch.randn(1,64,2,generator=torch.Generator().manual_seed(8))
    prediction = (target+0.1*target.roll(1,1)).requires_grad_()
    source = family(prediction,target,coordinates=coords,context={"phase":"calibration"})
    family.spatial_objective.freeze_source_calibration(
        [family.spatial_objective.collect_source_components(source)],
        {"split":"train","source_checkpoint_sha256":"source"})
    result = family(prediction,target,coordinates=coords,context={"global_step":17})
    artifact = family.state_artifact()
    restored = TopologyFamily(cfg,spec,FieldNormalizer.identity(2)); restored.load_state_artifact(artifact)
    actual = restored(prediction,target,coordinates=coords,context={"global_step":17})
    torch.testing.assert_close(actual.scalar_loss,result.scalar_loss,atol=0,rtol=0)
    assert actual.diagnostics["line_sampling"] == result.diagnostics["line_sampling"]
    report = _component_reports(family,result)
    assert sum(v["weighted_contribution"] for v in report.values()) == pytest.approx(result.scalar_loss.item())
    corrupt = deepcopy(artifact); corrupt["scientific_source"]["implementation"] = "old"
    with pytest.raises(ValueError,match="source/backend mismatch"):
        restored.load_state_artifact(corrupt)
    corrupt = deepcopy(artifact)
    corrupt["scientific_source"]["source_sha256"]["persistence.py"] = "0" * 64
    with pytest.raises(ValueError,match="source/backend mismatch"):
        restored.load_state_artifact(corrupt)
    corrupt = deepcopy(artifact["finite_primary_calibration"])
    corrupt["scientific_source"]["source_sha256"]["persistence_objective.py"] = "0" * 64
    with pytest.raises(ValueError,match="identity/provenance mismatch"):
        restored.spatial_objective.load_source_calibration(corrupt)

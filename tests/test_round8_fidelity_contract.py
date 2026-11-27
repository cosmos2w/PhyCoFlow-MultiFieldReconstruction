"""Changed field-risk semantics, calibration reuse, and controller contracts."""
from itertools import combinations
import json

import pytest
import torch

from phycoflow_reconstruction.coherence.physical_risk import physical_ensemble_risk, physical_ensemble_risks
from phycoflow_reconstruction.training.fidelity_budget import ExternalPolicy, FidelityBudget, MeanRiskDifferenceBudget
from phycoflow_reconstruction.training.fidelity_quadrature import stratified_field_risks
from phycoflow_reconstruction.training.post_training import _load_shared_fidelity_reference
from phycoflow_reconstruction.training.run_store import file_sha256


@pytest.mark.parametrize("member_count", [1, 2, 4])
def test_exact_complement_member_mse_value_and_gradient(member_count):
    torch.manual_seed(4)
    base = torch.randn(2, member_count, 6, 3, dtype=torch.float64)
    target = torch.randn(2, 6, 3, dtype=torch.float64)
    masses = torch.tensor([0., 1., 2., 4., 8., 12.], dtype=torch.float64)
    factor = torch.tensor(0.7, dtype=torch.float64, requires_grad=True)
    members = factor * base
    full = physical_ensemble_risk(members, target, masses, metric="member_mse")
    selected = torch.tensor([[0, 3], [0, 3]])
    values = []
    for combo in combinations([1, 2, 4, 5], 2):
        complement = torch.tensor([combo, combo])
        values.append(stratified_field_risks(members[:, :, selected[0]], target[:, selected[0]],
            members[:, :, complement[0]], target[:, complement[0]], selected, complement,
            masses, metric="member_mse"))
    estimated = torch.stack(values).mean(0)
    torch.testing.assert_close(estimated, full, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(torch.autograd.grad(estimated.sum(), factor, retain_graph=True)[0],
                               torch.autograd.grad(full.sum(), factor)[0], atol=1e-12, rtol=1e-12)
    if member_count == 1:
        with pytest.raises(ValueError, match="at least two"):
            physical_ensemble_risk(members, target, masses, metric="mean_risk")
    else:
        torch.testing.assert_close(physical_ensemble_risk(members, target, masses, metric="mean_risk"),
                                   physical_ensemble_risks(members, target, masses)[1])


def test_ordered_field_tolerances_units_and_pi_reference_roundtrip():
    pytest.importorskip("cooper")
    ref = FidelityBudget.from_reference(["tracer", "velocity", "pressure"], [0., 2., 3e-12], 2.,
        relative_increase=[0.1, 0.2, 0.3], absolute_allowance=[0., 0.5, 2e-12],
        denominator_floor=[1e-5, 1e-4, 1e-8], field_units_squared=["tracer²", "velocity²", "pressure²"])
    assert ref.upper_bounds == pytest.approx([0., 2.9, 5.9e-12])
    assert ref.denominators == pytest.approx([1e-5, 2., 1e-8])
    assert FidelityBudget.from_state_dict(json.loads(json.dumps(ref.state_dict()))) == ref
    policy = ExternalPolicy({"method": "lagrangian", "lagrangian": {"epoch_calibrated_pi":
        {"gamma": 3., "tau_epochs": 1., "horizon_epochs": 10., "steps_per_epoch": 2}}}, ref, "cpu")
    assert policy.state_dict()["epoch_calibrated_pi"]["z"] == pytest.approx(
        ref.violations(torch.tensor(ref.reference_risks)).tolist())
    legacy = FidelityBudget.from_reference(["field"], [1.], 1.).state_dict()
    assert isinstance(legacy["absolute_allowance"], float) and "field_units_squared" not in legacy
    assert FidelityBudget.from_state_dict(legacy).state_dict() == legacy


def test_existing_signed_panel_is_reused_only_under_matching_law(tmp_path):
    old_budget = MeanRiskDifferenceBudget.from_reference(["a", "b"], [-.2, .3], [.01, .02], [1., 2.], 1.4)
    shared = {"metric": "physical_mean_risk_model_units", "mean_risk_scales": [1., 2.],
        "member_allowance_fraction": .1, "reference_law": "training_selected_complement_v1",
        "rotating_complement": {"points": 3, "seed": 19}}
    binding = {"sha256": "old", "source_checkpoint_sha256": "source", "observations": {"a": 1},
               "compute_budget": {"batch_size": 64, "point_count": 4096},
               "core_family_config": {"definition": "fair", "conditional_draws": 2}, "fidelity_settings": shared}
    report = {"binding": binding, "reference": old_budget.state_dict(), "settings": shared,
        "risk_reference_panel": {"field_order": ["a", "b"], "source_mean_risk": [-.2, .3],
            "source_member_mse": [1., 2.], "members": 2,
            "reference_law": "training_selected_complement_v1"}}
    path = tmp_path / "source.json"
    path.write_text(json.dumps(report))
    declaration = {"path": str(path), "sha256": file_sha256(path)}
    settings = {"metric": "physical_member_mse_model_units", "relative_increase": .1,
        "reference_member_mse_from_panel": True, "reference_law": shared["reference_law"],
        "rotating_complement": shared["rotating_complement"]}
    expected = {**binding, "sha256": "new", "fidelity_settings": settings}
    reference, converted = _load_shared_fidelity_reference(declaration, expected)
    assert reference.reference_risks == (1., 2.) and reference.upper_bounds == pytest.approx((1.1, 2.2))
    assert converted["binding"] == expected and converted["member_mse_calibration_reuse"]["sha256"] == declaration["sha256"]
    with pytest.raises(ValueError, match="law differs"):
        _load_shared_fidelity_reference(declaration, {**expected, "observations": {"a": 2}})
    changed_count = {**expected, "core_family_config": {"definition": "fair", "conditional_draws": 3}}
    with pytest.raises(ValueError, match="law differs"):
        _load_shared_fidelity_reference(declaration, changed_count)
    permitted = {**changed_count, "compute_budget": {"batch_size": 32, "point_count": 4096},
                 "fidelity_settings": {**settings, "reference_member_count_reuse": True}}
    _, converted = _load_shared_fidelity_reference(declaration, permitted)
    assert converted["member_mse_calibration_reuse"]["source_panel_member_count"] == 2
    assert converted["member_mse_calibration_reuse"]["training_member_count"] == 3
    assert converted["member_mse_calibration_reuse"]["source_batch_size"] == 64
    assert converted["member_mse_calibration_reuse"]["training_batch_size"] == 32
    with pytest.raises(ValueError, match="query and support law"):
        _load_shared_fidelity_reference(declaration,
            {**permitted, "compute_budget": {"batch_size": 32, "point_count": 2048}})
    with pytest.raises(ValueError, match="independent-member family law"):
        _load_shared_fidelity_reference(declaration,
            {**permitted, "core_family_config": {"definition": "different", "conditional_draws": 3}})


@pytest.mark.parametrize("member_count,structured", [(1, False), (1, True), (3, False)])
def test_rotating_complement_replays_genuine_members_and_grid(monkeypatch, member_count, structured):
    from types import SimpleNamespace
    from phycoflow_reconstruction.contracts import FieldSample
    from phycoflow_reconstruction.data.sensor_protocols import SensorProtocol, build_observation_batch
    from phycoflow_reconstruction.training.coherence_sampling import _conditional_draw_seed
    from phycoflow_reconstruction.training.fidelity_quadrature import rotating_field_risks
    import phycoflow_reconstruction.training.rollout as rollout

    coords = torch.arange(9, dtype=torch.float32).reshape(-1, 1) / 9
    sample = FieldSample(values=coords.clone(), coordinates=coords, coordinates_raw=coords,
        time=torch.tensor(0.), trajectory_id="frame", time_index=0, conditions=torch.empty(0),
        field_names=("observed",), logical_shape=(9,))
    protocol = SensorProtocol(field_counts={"observed": 2}, seed=2)
    selected = build_observation_batch([sample], protocol, query_indices=[0, 2, 4],
        manifest_indices={"frame:0": [(1, 0), (7, 0)]})
    class Dataset:
        physical_weights = torch.arange(1, 10, dtype=torch.float32)
        trajectory_ids = ["frame"]
        _items = [(0, 0)]
        def __getitem__(self, index):
            assert index == 0
            return sample
    model = SimpleNamespace(capabilities=SimpleNamespace(structured_grid_required=structured))
    parameter = torch.tensor(0.8, requires_grad=True)
    calls = []
    def fake(_model, batch, *, generator, **_kwargs):
        assert batch.target_fields is None
        calls.append(batch.query_coords.shape[1])
        return parameter * batch.query_coords + torch.rand((), generator=generator)
    monkeypatch.setattr(rollout, "differentiable_reconstruction", fake)
    seed = 111
    first = torch.Generator().manual_seed(seed)
    state = first.get_state().clone()
    selected_members = [parameter * selected.query_coords + torch.rand((), generator=first)]
    for index in range(1, member_count):
        generator = torch.Generator().manual_seed(_conditional_draw_seed(seed, index))
        selected_members.append(parameter * selected.query_coords + torch.rand((), generator=generator))
    endpoint = {"coupling_selected_batch": selected, "members": selected_members,
        "target": selected.target_fields, "rollout_generator_state": state}
    config = {"observations": {"fields": {"observed": {"count": 2}}, "seed": 2},
        "rollout": {"steps": 1, "solver": "euler"}, "observation_consistency": {},
        "optimization": {"multitask": {"fidelity_budget": {"metric": "physical_member_mse_model_units",
            "rotating_complement": {"points": 2, "seed": 25}}}}}
    risk, complement = rotating_field_risks(model, endpoint, Dataset(), config, absolute_step=4,
        rollout_seed=seed, device=torch.device("cpu"), capture_endpoints=True)
    assert calls == [9 if structured else 2] * member_count
    assert len(endpoint["complement_members"]) == member_count
    for selected_member, complement_member in zip(selected_members, endpoint["complement_members"]):
        torch.testing.assert_close(selected_member[:, :1] - parameter * selected.query_coords[:, :1],
            complement_member[:, :1] - parameter * coords[complement[:, 0]])
    assert risk.shape == (1, 1) and torch.isfinite(torch.autograd.grad(risk.sum(), parameter)[0])


def test_actual_senseiver_calibrates_its_own_j1_complement_reference(tmp_path):
    from helpers.coherence import _post_config, _write_fixture
    from phycoflow_reconstruction.data.factory import open_field_dataset
    from phycoflow_reconstruction.models import build_model
    from phycoflow_reconstruction.training.post_training import _fit_source_physical_risk

    path = tmp_path / "fields.h5"
    _write_fixture(path)
    config = _post_config(path, tmp_path / "unused-source")
    config["dataset"].update(coordinate_dim=2, physical_measure="native_rectilinear_area")
    config["model"] = {"name": "senseiver", "width": 8, "num_latents": 4, "heads": 2, "depth": 1}
    config["coherence"]["compute_budget"].update(batch_size=1, point_count=8,
        query_policy="fixed_shared", query_seed=11)
    config["optimization"]["multitask"] = {"fidelity_budget": {
        "metric": "physical_member_mse_model_units", "reference_law": "training_selected_complement_v1",
        "rotating_complement": {"points": 4, "seed": 17}}}
    dataset = open_field_dataset(config["dataset"], split="train")
    try:
        model = build_model(config["model"], dataset.data_spec)
        risk, receipt = _fit_source_physical_risk(model, dataset, config, torch.device("cpu"),
            seed=71, families={})
        assert receipt["sample_count"] == 1 and receipt["members"] == 1
        assert receipt["reference_law"] == "training_selected_complement_v1"
        assert receipt["source_mean_risk"] is None and risk.tolist() == receipt["source_member_mse"]
        assert torch.isfinite(risk).all() and (risk >= 0).all()
    finally:
        dataset.close()

"""Native risk protection, common randomness and versioned recovery contracts."""

import json
import random
from copy import deepcopy
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch
from helpers.coherence import _base_config, _write_fixture
from test_upgrade_1002_lifecycle import upgraded_config

from phycoflow_reconstruction.config.validate import validate_config
from phycoflow_reconstruction.training.base_training import run_base_training
from phycoflow_reconstruction.training.fidelity_controller import (
    FIDELITY_DEFAULTS,
    NATIVE_VERSION,
    FidelityController,
    matched_native_losses,
)
from phycoflow_reconstruction.training.gradient_balance import (
    calibrate_coherence_direction,
    coherence_primal_dual_update,
    combine_coherence_gradients,
)
from phycoflow_reconstruction.training.post_training import run_post_training
from phycoflow_reconstruction.training.run_store import file_sha256, load_project_checkpoint


def native_controller(*, state=None, settings=None, calibration=None):
    fields = ["CH4", "CO", "T", "U_1", "p"]
    artifact = {"version": "endpoint_native_source_calibration_v2", "split": "train",
                "field_names": fields, "source_risks": [1.] * 6,
                "native": {"version": "native_source_calibration_v2", "split": "train",
                           "source_scale": 2.}}
    options = {"version": NATIVE_VERSION, "native_loss_role": "constraint", "native_budget": 0.}
    return FidelityController(fields, {**options, **(settings or {})}, calibration or artifact, state=state)


class RandomNative(torch.nn.Module):
    def __init__(self, value=1.):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(value, dtype=torch.float64))

    def training_loss(self, batch):
        draw = torch.rand(3, dtype=self.weight.dtype).sum() + random.random() + np.random.rand()
        return SimpleNamespace(total=(self.weight * draw - batch).square())


def test_common_native_draws_live_graph_teacher_detachment_and_public_rng():
    live = RandomNative()
    source = deepcopy(live).requires_grad_(False)
    torch.manual_seed(321)
    random.seed(321)
    np.random.seed(321)
    before = (torch.get_rng_state(), random.getstate(), np.random.get_state())
    ordinary = live.training_loss(0.).total
    after = (torch.get_rng_state(), random.getstate(), np.random.get_state())
    torch.set_rng_state(before[0]); random.setstate(before[1]); np.random.set_state(before[2])
    candidate, reference = matched_native_losses(live, source, 0.)
    assert torch.equal(candidate, ordinary) and torch.equal(candidate, reference)
    assert candidate.requires_grad and not reference.requires_grad
    assert torch.equal(torch.get_rng_state(), after[0])
    assert random.getstate() == after[1]
    assert np.array_equal(np.random.get_state()[1], after[2][1])
    assert np.random.get_state()[2:] == after[2][2:]
    candidate.backward()
    assert torch.isfinite(live.weight.grad) and live.weight.grad != 0
    assert source.weight.grad is None


def test_cuda_common_native_draws_restore_cpu_cuda_python_numpy_public_streams():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for native CPU/CUDA common-randomness replay")

    class MixedDeviceNative(RandomNative):
        def training_loss(self, batch):
            cpu_draw = torch.rand(3, dtype=self.weight.dtype, device="cpu").sum()
            cuda_draw = torch.rand(3, dtype=self.weight.dtype, device=self.weight.device).sum()
            draw = cpu_draw.to(self.weight.device) + cuda_draw + random.random() + np.random.rand()
            return SimpleNamespace(total=(self.weight * draw - batch).square())

    live = MixedDeviceNative().to("cuda:0").eval()
    source = deepcopy(live).requires_grad_(False).eval()
    source_before = {name: value.detach().clone() for name, value in source.state_dict().items()}
    original_python, original_numpy = random.getstate(), np.random.get_state()
    try:
        with torch.random.fork_rng(devices=[0]):
            torch.manual_seed(321)
            torch.cuda.manual_seed(321)
            random.seed(321)
            np.random.seed(321)
            before = (torch.get_rng_state(), torch.cuda.get_rng_state(0),
                      random.getstate(), np.random.get_state())
            ordinary = live.training_loss(0.).total
            after = (torch.get_rng_state(), torch.cuda.get_rng_state(0),
                     random.getstate(), np.random.get_state())
            expected_next = (torch.rand(4, device="cpu"), torch.rand(4, device="cuda:0"),
                             random.random(), np.random.rand(4))
            torch.set_rng_state(before[0])
            torch.cuda.set_rng_state(before[1], 0)
            random.setstate(before[2])
            np.random.set_state(before[3])

            candidate, reference = matched_native_losses(live, source, 0.)
            assert candidate.device == reference.device == torch.device("cuda:0")
            assert torch.equal(candidate, ordinary) and torch.equal(candidate, reference)
            assert candidate.requires_grad and candidate.grad_fn is not None
            assert not reference.requires_grad and reference.grad_fn is None
            assert torch.equal(torch.get_rng_state(), after[0])
            assert torch.equal(torch.cuda.get_rng_state(0), after[1])
            assert random.getstate() == after[2]
            assert np.array_equal(np.random.get_state()[1], after[3][1])
            assert np.random.get_state()[2:] == after[3][2:]
            assert torch.equal(torch.rand(4, device="cpu"), expected_next[0])
            assert torch.equal(torch.rand(4, device="cuda:0"), expected_next[1])
            assert random.random() == expected_next[2]
            assert np.array_equal(np.random.rand(4), expected_next[3])

            candidate.backward()
            assert torch.isfinite(live.weight.grad) and live.weight.grad != 0
            assert source.weight.grad is None
            assert all(torch.equal(value, source_before[name]) for name, value in source.state_dict().items())
    finally:
        random.setstate(original_python)
        np.random.set_state(original_numpy)


def test_native_teacher_failure_restores_post_live_stream():
    live = RandomNative()
    source = deepcopy(live)
    def fail(batch):
        torch.rand(8); random.random(); np.random.rand()
        raise RuntimeError("teacher failed")
    source.training_loss = fail
    torch.manual_seed(11); random.seed(11); np.random.seed(11)
    state = (torch.get_rng_state(), random.getstate(), np.random.get_state())
    live.training_loss(0.)
    expected = (torch.rand(1), random.random(), np.random.rand())
    torch.set_rng_state(state[0]); random.setstate(state[1]); np.random.set_state(state[2])
    with pytest.raises(RuntimeError, match="teacher failed"):
        matched_native_losses(live, source, 0.)
    assert torch.equal(torch.rand(1), expected[0])
    assert random.random() == expected[1] and np.random.rand() == expected[2]


def test_native_monitor_can_evaluate_without_graph():
    live = RandomNative()
    a, b = matched_native_losses(live, deepcopy(live), 0., live_graph=False)
    assert torch.equal(a, b) and not a.requires_grad and not b.requires_grad


def test_seven_risk_source_identity_and_native_primal_finite_difference():
    controller = native_controller()
    assert controller.names == ("total", "CH4", "CO", "T", "U_1", "p", "native")
    prediction = torch.ones(2, 3, 5, dtype=torch.float64, requires_grad=True)
    target = torch.zeros_like(prediction)
    native = torch.tensor(3., dtype=torch.float64, requires_grad=True)
    teacher_native = torch.tensor(2., dtype=torch.float64, requires_grad=True)
    violations, live, source = controller.violations(
        prediction, prediction.detach(), target, native_loss=native, source_native_loss=teacher_native)
    assert violations[:6].tolist() == pytest.approx([-.05] * 6)
    assert violations[-1] == .5 and live[-1] == 3 and source[-1] == 2
    loss, terms, pressure = controller.primal(violations)
    assert set(terms) == set(controller.names)
    gradient = torch.autograd.grad(loss, native, retain_graph=True)[0]
    epsilon = 1e-6
    def value(d):
        g, _, _ = controller.violations(prediction, prediction.detach(), target,
            native_loss=torch.tensor(d, dtype=torch.float64, requires_grad=True),
            source_native_loss=teacher_native)
        return float(controller.primal(g)[0])
    assert gradient == pytest.approx((value(3 + epsilon) - value(3 - epsilon)) / (2 * epsilon))
    assert gradient == pressure[-1] / 2
    loss.backward()
    assert teacher_native.grad is None
    assert torch.count_nonzero(prediction.grad) == 0
    identity, _, _ = controller.violations(prediction, prediction.detach(), target,
        native_loss=native, source_native_loss=native.detach())
    assert identity[-1] == 0


def test_native_constraint_rejects_detached_live_and_missing_native_inputs():
    controller = native_controller()
    prediction = torch.ones(1, 1, 5, requires_grad=True)
    with pytest.raises(ValueError, match="matched"):
        controller.violations(prediction, prediction.detach(), prediction.detach())
    with pytest.raises(ValueError, match="autograd"):
        controller.violations(prediction, prediction.detach(), prediction.detach(),
                              native_loss=torch.tensor(1.), source_native_loss=torch.tensor(1.))


def test_v1_native_monitor_supplies_zero_native_gradient_by_construction():
    model = torch.nn.Linear(2, 1, bias=False, dtype=torch.float64)
    with torch.no_grad():
        native_monitor = model.weight[0, 0].square().detach()
    before = model.weight.detach().clone()
    coherence = model.weight[0, 1].square()
    coherence_primal_dual_update(model, torch.optim.SGD(model.parameters(), lr=.1),
        {"A": coherence}, coherence * 0, method="weighted_sum", grad_clip=None)
    assert not native_monitor.requires_grad
    assert model.weight[0, 0] == before[0, 0]
    assert model.weight[0, 1] != before[0, 1]


def test_native_and_endpoint_constraints_update_separate_parameters_with_displacement():
    model = torch.nn.Linear(2, 1, bias=False, dtype=torch.float64)
    with torch.no_grad(): model.weight.fill_(2.)
    native = model.weight[0, 0].square()
    prediction = model.weight[0, 1].expand(1, 2, 5)
    controller = native_controller()
    violations, _, _ = controller.violations(prediction, torch.ones_like(prediction),
        torch.zeros_like(prediction), native_loss=native, source_native_loss=native.new_tensor(1.))
    pressure, terms, _ = controller.primal(violations)
    row = coherence_primal_dual_update(model, torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=0),
        {"A": model.weight.sum() * 0}, pressure, method="weighted_sum", grad_clip=1,
        diagnostics=True, native_loss=native, constraint_losses=terms)
    assert (model.weight < 2).all()
    assert row["gradient/native/norm"] > 0 and row["update/actual_dot/native"] < 0
    assert "update/actual_dot/fidelity/native" in row and "update/actual_dot/fidelity/p" in row


def test_native_multiplier_sustained_violation_slack_and_exact_state_resume():
    controller = native_controller()
    values = torch.tensor([0.] * 6 + [.2])
    for _ in range(20): controller.advance(values)
    assert controller.multipliers[-1] > 0
    recovered = native_controller(state=controller.state_dict())
    for _ in range(100):
        slack = torch.tensor([0.] * 6 + [-.5])
        controller.advance(slack); recovered.advance(slack)
    assert controller.multipliers[-1] == 0
    assert torch.equal(controller.multipliers, recovered.multipliers)
    assert torch.equal(controller.ema, recovered.ema)
    assert controller.updates == recovered.updates


def test_v1_state_compatibility_and_v1_v2_artifact_rejection():
    old_calibration = {"field_names": ["u"], "source_risks": [1., 1.]}
    old = FidelityController(["u"], FIDELITY_DEFAULTS, old_calibration)
    old.advance(torch.tensor([.1, -.1]))
    state = old.state_dict()
    assert state["version"] == "endpoint_primal_dual_v1" and "version" not in state["settings"]
    restored = FidelityController(["u"], FIDELITY_DEFAULTS, old_calibration, state=state)
    assert torch.equal(restored.multipliers, old.multipliers)
    with pytest.raises(ValueError, match="version"):
        native_controller(state=state)
    with pytest.raises(ValueError, match="native v2"):
        FidelityController(native_controller().field_names, FIDELITY_DEFAULTS, native_controller().calibration)
    bad = deepcopy(native_controller().calibration); bad["native"]["split"] = "validation"
    with pytest.raises(ValueError, match="TRAIN"): native_controller(calibration=bad)
    bad = deepcopy(native_controller().calibration); bad["native"]["source_scale"] = 0.
    with pytest.raises(ValueError, match="positive"): native_controller(calibration=bad)
    state = native_controller().state_dict(); state["names"] = list(reversed(state["names"]))
    with pytest.raises(ValueError, match="order"): native_controller(state=state)


@pytest.mark.parametrize("method", ["config", "weighted_sum", "cagrad"])
def test_external_fixed_scalar_preserves_raw_direction_and_fidelity_magnitude(method):
    first = torch.nn.Linear(2, 1, bias=False, dtype=torch.float64)
    with torch.no_grad(): first.weight.copy_(torch.tensor([[2., 3.]], dtype=torch.float64))
    second = deepcopy(first)
    reports = []
    for model, scale in ((first, 1.), (second, 2.)):
        reports.append(coherence_primal_dual_update(model, torch.optim.SGD(model.parameters(), lr=.1),
            {"A": model.weight[0, 0], "B": model.weight[0, 1]}, model.weight.sum() * 0,
            method=method, grad_clip=None, coherence_direction_scale=scale))
    original = torch.tensor([[2., 3.]], dtype=torch.float64)
    assert torch.allclose(original - second.weight, 2 * (original - first.weight))
    assert reports[1]["gradient/coherence_raw_norm"] == reports[0]["gradient/coherence_raw_norm"]
    assert reports[1]["gradient/coherence_calibrated_norm"] == 2 * reports[0]["gradient/coherence_calibrated_norm"]


@pytest.mark.parametrize("matrix", [
    [[1., 0., 1., 3.], [0., 1., 3., 1.], [1., 1., 2., 2.]],
    [[1., 0., 0., 0.], [2., 0., 0., 0.], [3., 0., 0., 0.]],
    [[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 0., 0.]],
])
def test_train_gram_calibration_preserves_full_vector_raw_norms(matrix):
    vectors = torch.tensor(matrix, dtype=torch.float64)
    scales = dict(zip("ABC", [1., 2., 3.]))
    calibration = calibrate_coherence_direction([(vectors @ vectors.T).tolist()], list("ABC"), scales)
    bank = {name: vector * scales[name] for name, vector in zip("ABC", vectors)}
    for method in ("config", "weighted_sum", "cagrad"):
        direction, _ = combine_coherence_gradients(bank, method=method)
        assert calibration["raw_median_norms"][method] == pytest.approx(float(direction.norm()), rel=1e-7)
        assert calibration["resolved_scales"][method] > 0


def test_zero_nonfinite_non_psd_direction_calibration_is_rejected():
    for gram in ([[0., 0.], [0., 0.]], [[float("nan"), 0.], [0., 1.]], [[1., 2.], [2., 1.]]):
        with pytest.raises((ValueError, FloatingPointError)):
            calibrate_coherence_direction([gram], ["A", "B"], {"A": 1., "B": 1.})


def test_native_version_strict_config_roles_and_frozen_calibration_policy():
    config = upgraded_config("unused.h5", "unused_source")
    config["fidelity_controller"].update(version=NATIVE_VERSION, native_loss_role="constraint")
    config["optimization"]["coherence_direction_calibration"] = "train_median_to_weighted_sum"
    config["runtime"]["plot_format"] = "pdf"
    validate_config(config)
    for changes, pattern in [
        ({"native_loss_role": "monitor"}, "role"),
        ({"version": "unknown"}, "version"),
        ({"native_budget": float("nan")}, "finite"),
        ({"typo_budget": 0.}, "unknown"),
    ]:
        invalid = deepcopy(config); invalid["fidelity_controller"].update(changes)
        with pytest.raises((ValueError, TypeError), match=pattern): validate_config(invalid)
    invalid = deepcopy(config); invalid["optimization"]["model_mode"] = "train"
    with pytest.raises(ValueError, match="inference"): validate_config(invalid)


def test_native_v2_interrupted_resume_calibration_rng_optimizer_teacher_exact(tmp_path):
    path = tmp_path / "fixture.h5"
    _write_fixture(path)
    source = run_base_training(_base_config(path), case_dir=tmp_path / "source_case")
    source_hash = file_sha256(source / "checkpoints/last.pt")
    config = upgraded_config(path, source)
    config["fidelity_controller"].update(version=NATIVE_VERSION, native_loss_role="constraint")
    config["optimization"]["coherence_direction_calibration"] = "train_median_to_weighted_sum"
    child = run_post_training(config, case_dir=tmp_path / "child_case", max_steps=2)
    calibration_hash = file_sha256(child / "artifacts/coherence_calibration.json")
    run_post_training(config, case_dir=tmp_path / "child_case", max_steps=2, resume=child)
    whole = run_post_training(config, case_dir=tmp_path / "whole_case", max_steps=4)
    recovered = load_project_checkpoint(child / "checkpoints/last.pt")
    uninterrupted = load_project_checkpoint(whole / "checkpoints/last.pt")
    assert recovered["fidelity_controller"]["version"] == NATIVE_VERSION
    assert recovered["fidelity_controller"]["updates"] == uninterrupted["fidelity_controller"]["updates"] == 4
    assert recovered["epoch"] == uninterrupted["epoch"]
    for key in ("multipliers", "ema"):
        assert torch.equal(recovered["fidelity_controller"][key], uninterrupted["fidelity_controller"][key])
    for key in recovered["model"]:
        assert torch.equal(recovered["model"][key], uninterrupted["model"][key]), key
    for key, state in recovered["optimizer"]["state"].items():
        for name, value in state.items():
            assert torch.equal(value, uninterrupted["optimizer"]["state"][key][name])
    assert torch.equal(recovered["rng_state"]["torch_cpu"], uninterrupted["rng_state"]["torch_cpu"])
    assert recovered["rng_state"]["python"] == uninterrupted["rng_state"]["python"]
    assert np.array_equal(recovered["rng_state"]["numpy"][1], uninterrupted["rng_state"]["numpy"][1])
    assert calibration_hash == file_sha256(child / "artifacts/coherence_calibration.json")
    assert file_sha256(source / "checkpoints/last.pt") == source_hash


def test_r2_legacy_panel_native_monitor_stream_pdf_and_reserved_audit(tmp_path):
    path = tmp_path / "fixture.h5"
    _write_fixture(path)
    with h5py.File(path, "a") as handle:
        del handle["fields"]; del handle["time"]
        handle.create_dataset("fields", data=np.random.default_rng(5).normal(
            size=(3, 512, 16, 1, 1, 2)).astype("float32"))
        handle.create_dataset("time", data=np.arange(512, dtype="float32"))
    base = _base_config(path)
    base["optimization"]["batch_size"] = 512
    source = run_base_training(base, case_dir=tmp_path / "source_case")
    config = upgraded_config(path, source)
    config.pop("fidelity_controller")
    config["optimization"].pop("coherence_gradient_method")
    config["optimization"].update(update_policy="legacy", epochs=1, steps_per_epoch=1, sampling="full_pass")
    config["objectives"]["data_retention"].update(enabled=True, weight=.1)
    config["runtime"].update(plot_format="pdf", progress=False)
    config["evaluation"].update(max_samples=64, r2_protocol={
        "enabled": True, "native_monitor_every_epochs": 1,
        "previously_used_validation_indices": list(range(96)),
    }, native_topology_audit={"enabled": True, "every_epochs": 25,
                              "max_samples": 2, "seed": 2027})
    # Native audit configuration uses the unchanged supported native gather.
    config["coherence"]["families"]["topology"]["geometry"].update(antialias_downsample=True)
    run = run_post_training(config, case_dir=tmp_path / "child_case", max_steps=1)
    before = json.loads((run / "evaluation/before.json").read_text())
    assert before["inference"]["samples"] == 64
    assert [len(group) for group in before["panel_groups"]] == [32, 32]
    baseline_native = json.loads((run / "evaluation/native_before.json").read_text())
    assert baseline_native["train"]["ratio"] == pytest.approx(1.)
    assert baseline_native["validation"]["ratio"] == pytest.approx(1.)
    history = [json.loads(line) for line in (run / "metrics/history.jsonl").read_text().splitlines()]
    stream = [json.loads(line) for line in (run / "metrics/coherence_updates.jsonl").read_text().splitlines()]
    assert len(stream) == 1 and history[0]["native_monitor/train/candidate"] > 0
    assert history[0]["native_monitor/validation/source"] > 0
    declaration = json.loads((run / "artifacts/r2_panel_declaration.json").read_text())
    selection = set(declaration["selection"]["dataset_indices"])
    native_validation = set(declaration["native_monitor"]["dataset_indices"]["validation"])
    assert native_validation <= selection and len(native_validation) == 32
    contract = json.loads((run / "evaluation/native_topology_audit/contract.json").read_text())
    metadata = contract
    assert not set(metadata["sample_ids"]) & set(metadata["reserved_final_audit_sample_ids"])
    assert metadata["panel_role"] == "periodic_development_native_audit"
    assert not list(run.rglob("*.png")) and (run / "loss_history.pdf").is_file()
    assert load_project_checkpoint(run / "checkpoints/last.pt")["epoch"] == 1


@pytest.mark.parametrize("change,pattern", [
    (lambda c: c["evaluation"].update(split="test"), "test data"),
    (lambda c: c["evaluation"].update(max_samples=32), "64 snapshots"),
    (lambda c: c["evaluation"]["r2_protocol"].update(native_seeds=[]), "seeds"),
    (lambda c: c["evaluation"]["r2_protocol"].update(native_validation_count=31), "16 strata"),
    (lambda c: c["model"].update(model_ema_eval=True), "LIVE"),
])
def test_r2_protocol_strict_contracts(change, pattern):
    config = upgraded_config("unused.h5", "unused_source")
    config["evaluation"].update(max_samples=64, r2_protocol={"enabled": True})
    change(config)
    with pytest.raises(ValueError, match=pattern): validate_config(config)

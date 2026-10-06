"""R5's declared scalar equations, source identity and actual child recovery."""

from copy import deepcopy
import json
import random
from unittest.mock import patch

import numpy as np
import pytest
import torch

from helpers.coherence import _base_config, _post_config, _write_fixture
from phycoflow_reconstruction.config.validate import validate_config
from phycoflow_reconstruction.training.base_training import run_base_training
from phycoflow_reconstruction.training.parameter_retention import (
    CALIBRATION, DEFINITION, SourceParameterRetention, private_call,
    scalar_objective, scalar_update, trainable_parameter_identity,
)
from phycoflow_reconstruction.training.post_training import run_post_training
from phycoflow_reconstruction.training.run_store import file_sha256, load_project_checkpoint
from phycoflow_reconstruction.training.source import load_source_model, set_trainable_scope


class MixedParameters(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.real = torch.nn.Parameter(torch.tensor([1., -2.], dtype=torch.float64))
        self.complex = torch.nn.Parameter(torch.tensor([1. + 2j, -3j], dtype=torch.complex128))
        self.unused = torch.nn.Parameter(torch.tensor([2., 3.], dtype=torch.float64))
        self.frozen = torch.nn.Parameter(torch.tensor([4.]), requires_grad=False)
        self.register_buffer("buffer", torch.tensor([7.]))

    def losses(self):
        native = self.real.square().sum() + self.complex.abs().square().sum()
        coherence = .2 * self.real.sin().sum() + .3 * self.complex.real.square().sum()
        return native, coherence


def artifact(model, source_hash="source"):
    return {"definition": DEFINITION, "calibration": CALIBRATION, "split": "train",
            "lambda_sp": .25, "native_gradient_median": .5, "reference_displacement": 2.,
            "epsilon_displacement": 1e-12, "source_checkpoint_sha256": source_hash,
            "source_parameter_identity": trainable_parameter_identity(model),
            "reference_parameter_sha256": "reference",
            "train_probes": [{"sample_ids": [f"train:{i}"], "native_seed": 19+i,
                              "query_sha256": f"query{i}"} for i in range(8)]}


def test_source_penalty_zero_complex_gradient_frozen_and_buffers():
    model = MixedParameters()
    retention = SourceParameterRetention(model, artifact(model), source_checkpoint_sha256="source")
    snapshots = [value.clone() for value in retention.source_parameters]
    omega, penalty = retention.penalty()
    assert omega.item() == penalty.item() == 0
    penalty.backward()
    assert all(torch.count_nonzero(parameter.grad) == 0 for parameter in retention.parameters)
    with torch.no_grad():
        model.real.add_(.4)
        model.complex.add_(.2 + .3j)
        model.unused.add_(.6)
        model.frozen.add_(100)
        model.buffer.add_(100)
    model.zero_grad()
    omega, penalty = retention.penalty()
    expected = .5 * (2*.4**2 + 2*(.2**2+.3**2) + 2*.6**2)
    assert omega.item() == pytest.approx(expected)
    penalty.backward()
    for parameter, source in zip(retention.parameters, snapshots):
        torch.testing.assert_close(parameter.grad, .25*(parameter-source))
    assert model.frozen.grad is None
    for source, saved in zip(retention.source_parameters, snapshots):
        assert not source.requires_grad
        torch.testing.assert_close(source, saved, rtol=0, atol=0)


@pytest.mark.parametrize("arm", ["N", "F", "S"])
@pytest.mark.parametrize("grad_clip", [None, .5])
def test_production_scalar_value_gradient_adamw_matches_separate_oracle(arm, grad_clip):
    model = MixedParameters()
    retention = SourceParameterRetention(model, artifact(model), source_checkpoint_sha256="source")
    with torch.no_grad():
        model.real.add_(.4)
        model.complex.add_(.2 + .3j)
    oracle = deepcopy(model)
    parameters = tuple(parameter for parameter in model.parameters() if parameter.requires_grad)
    oracle_parameters = tuple(parameter for parameter in oracle.parameters() if parameter.requires_grad)
    native, coherence = model.losses()
    use_retention = retention if arm == "S" else None
    use_coherence = coherence if arm != "N" else None
    declared = .1*native + (coherence if arm != "N" else 0)
    if arm == "S":
        declared = declared + .25*.5*sum((p-s).abs().square().sum()
                                       for p,s in zip(parameters, retention.source_parameters))
    torch.testing.assert_close(scalar_objective(native, use_coherence, retention=use_retention), declared)
    d, c = oracle.losses()
    losses = [.1*d] + ([c] if arm != "N" else [])
    gradients = [torch.autograd.grad(loss, oracle_parameters, retain_graph=True, allow_unused=True)
                 for loss in losses]
    for index, parameter in enumerate(oracle_parameters):
        value = sum((bank[index] if bank[index] is not None else torch.zeros_like(parameter))
                    for bank in gradients)
        if arm == "S":
            value = value + .25*(parameter-retention.source_parameters[index])
        parameter.grad = value.detach().clone()
    norm = torch.stack([p.grad.abs().double().square().sum() for p in oracle_parameters]).sum().sqrt()
    if grad_clip:
        coefficient = (grad_clip/(norm+1e-6)).clamp(max=1.)
        for parameter in oracle_parameters:
            parameter.grad.mul_(coefficient)
    optimizers = [torch.optim.AdamW(m.parameters(), lr=.002, weight_decay=.07) for m in (model,oracle)]
    optimizers[1].step()
    with patch("torch.autograd.grad", side_effect=AssertionError("ordinary scalar extra gradient")):
        report = scalar_update(model, optimizers[0], native, use_coherence,
                               retention=use_retention, grad_clip=grad_clip)
    assert report["loss/scalar_objective"] == pytest.approx(float(declared.detach()))
    for left,right in zip(parameters,oracle_parameters):
        torch.testing.assert_close(left.grad,right.grad,rtol=2e-14,atol=2e-14)
        torch.testing.assert_close(left,right,rtol=2e-14,atol=2e-14)
        for key,value in optimizers[0].state[left].items():
            torch.testing.assert_close(value,optimizers[1].state[right][key],rtol=2e-14,atol=2e-14)
    assert ("parameter_retention_loss" in report) == (arm == "S")


@pytest.mark.parametrize("key,value", [("split", "validation"), ("split", "test"),
    ("lambda_sp", float("nan")), ("lambda_sp", .5), ("reference_displacement", 0),
    ("source_checkpoint_sha256", "different"), ("train_probes", []),
    ("source_parameter_identity", {})])
def test_bad_calibration_and_resume_rejected(key,value):
    model = MixedParameters()
    bad = artifact(model)
    bad[key] = value
    with pytest.raises(ValueError):
        SourceParameterRetention(model,bad,source_checkpoint_sha256="source")
    good = SourceParameterRetention(model,artifact(model),source_checkpoint_sha256="source")
    state = good.state_dict()
    good.verify_resume(state)
    state["lambda_sp"] = .6
    with pytest.raises(ValueError,match="resume"):
        good.verify_resume(state)


def test_private_native_draws_unaffected_by_omitted_coherence_and_evaluation():
    def draw():
        return torch.rand(3), random.random(), np.random.rand()
    state = (torch.get_rng_state(),random.getstate(),np.random.get_state())
    expected = private_call(draw,seed=3042,device=torch.device("cpu"))
    private_call(draw,seed=9001,device=torch.device("cpu"))
    torch.rand(17)
    actual = private_call(draw,seed=3042,device=torch.device("cpu"))
    torch.testing.assert_close(actual[0],expected[0],rtol=0,atol=0)
    assert actual[1:] == expected[1:]
    assert random.getstate() == state[1]
    assert np.array_equal(np.random.get_state()[1],state[2][1])


def test_cpu_private_stream_never_seeds_or_reads_cuda_generators():
    before = torch.get_rng_state()
    expected = torch.rand(4,generator=torch.Generator().manual_seed(3123))
    with patch("torch.cuda.manual_seed_all",side_effect=AssertionError("CPU call changed CUDA RNG")), \
         patch("torch.cuda.get_rng_state_all",side_effect=AssertionError("CPU call initialized CUDA")):
        actual = private_call(lambda: torch.rand(4),seed=3123,device=torch.device("cpu"))
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    assert torch.equal(torch.get_rng_state(),before)


def matched_config(path,source,arm):
    post = _post_config(path,source)
    post["runtime"].update(execution_mode="r4_exact",random_stream_policy="matched_native_v1",
                           diagnostics_every_epochs=25,plot_format="pdf")
    post["optimization"].update(epochs=4)
    post["objectives"]["coherence"]["enabled"] = arm != "N"
    post["coherence"]["compute_budget"].update(query_policy="fixed_shared")
    post["evaluation"]["preview"] = {"enabled": False}
    post["checkpointing"] = {"every_epochs": 1}
    return post


@pytest.mark.parametrize("change", [
    lambda config: config["optimization"].update(gradient_balance="config"),
    lambda config: config["runtime"].update(execution_mode="legacy"),
    lambda config: config["runtime"].update(random_stream_policy="unknown"),
    lambda config: config["runtime"].update(diagnostics_every_epochs=0),
    lambda config: config["objectives"]["data_retention"].update(weight=.2),
    lambda config: config["objectives"].update(source_anchor={"enabled":True,"weight":1.}),
    lambda config: config["coherence"]["schedule"].update(every_n_steps=2),
    lambda config: config["objectives"].update(parameter_retention={"enabled":True}),
])
def test_matched_contract_rejects_changed_equations_and_teacher(change):
    config = matched_config("not_opened.h5","not_opened_source","F")
    change(config)
    with pytest.raises(ValueError):
        validate_config(config)


def test_native_and_coherence_arms_match_actual_source_native_draw_sample_sensor_query(tmp_path):
    path = tmp_path/"fields.h5"
    _write_fixture(path)
    source = run_base_training(_base_config(path),case_dir=tmp_path/"source_case")
    rows = []
    for arm in ("N","F"):
        config = matched_config(path,source,arm)
        child = run_post_training(config,case_dir=tmp_path/arm,max_steps=1)
        rows.append(json.loads((child/"metrics/history.jsonl").read_text().splitlines()[0]))
    assert rows[0]["stream_identity"] == rows[1]["stream_identity"]
    assert rows[0]["native_data_loss"] == rows[1]["native_data_loss"]


def test_scheduled_scalar_diagnostics_reuse_one_live_coherence_forward(tmp_path):
    import phycoflow_reconstruction.training.post_training as training
    path = tmp_path/"fields.h5"
    _write_fixture(path)
    source = run_base_training(_base_config(path),case_dir=tmp_path/"source_case")
    config = matched_config(path,source,"F")
    config["runtime"]["diagnostics_every_epochs"] = 2
    with patch.object(training,"_coherence_objective",wraps=training._coherence_objective) as coherence:
        child = run_post_training(config,case_dir=tmp_path/"child",max_steps=2)
    assert coherence.call_count == 2
    rows = [json.loads(line) for line in (child/"metrics/history.jsonl").read_text().splitlines()]
    assert "family_gradient_diagnostics" not in rows[0]
    assert rows[1]["family_gradient_diagnostics"]["reused_training_rollout"]
    metadata = json.loads((child/"run_manifest.json").read_text())["source_metadata"]
    assert metadata["live_state_exact_match"]
    assert metadata["initial_trainable_parameter_identity"]["tensor_sha256"]


def test_deploy_only_checkpoint_rejected_for_both_initialization_and_resume(tmp_path):
    path = tmp_path/"fields.h5"
    _write_fixture(path)
    source = run_base_training(_base_config(path),case_dir=tmp_path/"source_case")
    config = matched_config(path,source,"N")
    child = run_post_training(config,case_dir=tmp_path/"child",max_steps=1)
    checkpoint_path = child/"checkpoints/last.pt"
    payload = load_project_checkpoint(checkpoint_path)
    payload["inference_only"] = True
    torch.save(payload,checkpoint_path)
    with pytest.raises(ValueError,match="inference-only"):
        run_post_training(config,case_dir=tmp_path/"child",max_steps=1,resume=child)
    checkpoint_path = source/"checkpoints/last.pt"
    payload = load_project_checkpoint(checkpoint_path)
    payload["inference_only"] = True
    torch.save(payload,checkpoint_path)
    with pytest.raises(ValueError,match="inference-only"):
        run_post_training(config,case_dir=tmp_path/"new_child",max_steps=1)


@pytest.mark.parametrize("arm", ["N", "F", "S"])
def test_real_post_training_resume_identity_frozen_coefficient_and_native_only(tmp_path,arm):
    path = tmp_path/"fields.h5"
    _write_fixture(path)
    source = run_base_training(_base_config(path),case_dir=tmp_path/"source_case")
    source_hash = file_sha256(source/"checkpoints/last.pt")
    post = matched_config(path,source,arm)
    if arm == "S":
        model,dataset,_ = load_source_model(post,torch.device("cpu"))
        set_trainable_scope(model,post["trainable"])
        calibration_path = tmp_path/"frozen_train.json"
        calibration_path.write_text(json.dumps(artifact(model,source_hash)))
        dataset.close()
        post["objectives"]["parameter_retention"] = {
            "enabled": True,"definition": DEFINITION,"calibration": CALIBRATION,
            "calibration_path": str(calibration_path)}
    validate_config(post)
    context = patch("phycoflow_reconstruction.training.post_training._coherence_objective",
                    side_effect=AssertionError("native-only invoked coherence/teacher")) if arm == "N" else patch(
                        "phycoflow_reconstruction.training.post_training.matched_native_losses",
                        side_effect=AssertionError("scalar hot path invoked source teacher"))
    with context:
        child = run_post_training(post,case_dir=tmp_path/"split",max_steps=2)
        run_post_training(post,case_dir=tmp_path/"split",max_steps=2,resume=child)
        whole = run_post_training(post,case_dir=tmp_path/"whole",max_steps=4)
    resumed = load_project_checkpoint(child/"checkpoints/last.pt")
    uninterrupted = load_project_checkpoint(whole/"checkpoints/last.pt")
    assert resumed["global_step"] == uninterrupted["global_step"] == 4
    for name,value in resumed["model"].items():
        torch.testing.assert_close(value,uninterrupted["model"][name],rtol=0,atol=0)
    for index,state in resumed["optimizer"]["state"].items():
        for key,value in state.items():
            torch.testing.assert_close(value,uninterrupted["optimizer"]["state"][index][key],rtol=0,atol=0)
    for family,state in resumed["family_states"].items():
        for key,value in state.items():
            torch.testing.assert_close(value,uninterrupted["family_states"][family][key],rtol=0,atol=0)
    assert resumed["random_stream_identity"] == uninterrupted["random_stream_identity"]
    assert {"python","numpy","torch_cpu","index_generator"} <= resumed["rng_state"].keys()
    assert file_sha256(source/"checkpoints/last.pt") == source_hash
    if arm == "S":
        assert resumed["parameter_retention"] == uninterrupted["parameter_retention"]
        assert resumed["parameter_retention"]["lambda_sp"] == .25
    histories = [[json.loads(row) for row in (run/"metrics/history.jsonl").read_text().splitlines()]
                 for run in (child,whole)]
    assert [row["stream_identity"] for row in histories[0]] == [row["stream_identity"] for row in histories[1]]
    for row in histories[0]:
        assert row["data_loss"] == row["native_data_loss"]
        assert row["total"] == pytest.approx(.1*row["data_loss"]+row.get("coherence_loss",0)+row.get("parameter_retention_loss",0))
        assert ("parameter_retention_loss" in row) == (arm == "S")
        assert row["coherence_applied"] == (arm != "N")

"""Fixed scalar-strength value, dispatch, receipt and recovery contracts."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from helpers.coherence import _base_config, _post_config, _write_fixture

from phycoflow_reconstruction.config.validate import validate_config
from phycoflow_reconstruction.training import post_training
from phycoflow_reconstruction.training.base_training import run_base_training
from phycoflow_reconstruction.training.parameter_retention import scalar_objective
from phycoflow_reconstruction.training.post_training import run_post_training
from phycoflow_reconstruction.training.run_store import load_project_checkpoint
from phycoflow_reconstruction.training.training_policy import (
    persist_training_policy,
    resolve_training_policy,
)


def _matched_config(dataset, source, *, beta=0.25):
    config = _post_config(dataset, source)
    config["runtime"].update(
        execution_mode="r4_exact",
        random_stream_policy="matched_native_v1",
        diagnostics_every_epochs=25,
        plot_format="pdf",
    )
    config["optimization"].update(epochs=4, gradient_balance="weighted_sum", update_policy="legacy")
    config["objectives"]["coherence"].update(enabled=True, weight=beta)
    config["objectives"]["parameter_retention"] = {"enabled": False}
    config["coherence"]["compute_budget"].update(query_policy="fixed_shared")
    config["evaluation"]["preview"] = {"enabled": False}
    config["checkpointing"] = {"every_epochs": 1}
    return config


@pytest.mark.parametrize("beta", [0.0, 0.25, 1.0])
def test_scalar_strength_is_applied_once_outside_unchanged_calibrated_families(beta):
    parameter = torch.tensor([1.2, -0.7, 0.8], dtype=torch.float64, requires_grad=True)
    native = parameter.square().sum()
    families = {
        "A": parameter[0].pow(3),
        "B": parameter[1].square() + 0.4,
        "C": parameter[2].sin(),
    }
    scales = {"A": 2.5, "B": 0.4, "C": 1.2}
    scales_before = dict(scales)
    family_values_before = {name: value.detach().clone() for name, value in families.items()}
    calibrated_abc = sum(scales[name] * families[name] for name in ("A", "B", "C"))

    actual = scalar_objective(native, calibrated_abc, data_weight=0.1, coherence_weight=beta)
    expected = 0.1 * native + beta * sum(
        scales[name] * families[name] for name in ("A", "B", "C")
    )
    actual_gradient, = torch.autograd.grad(actual, (parameter,), retain_graph=True)
    expected_native_gradient, = torch.autograd.grad(native, (parameter,), retain_graph=True)
    expected_coherence_gradient = torch.zeros_like(parameter)
    for name in ("A", "B", "C"):
        family_gradient, = torch.autograd.grad(families[name], (parameter,), retain_graph=True)
        if beta > 0:
            assert torch.count_nonzero(family_gradient) > 0
        expected_coherence_gradient = expected_coherence_gradient + scales[name] * family_gradient
    expected_gradient = 0.1 * expected_native_gradient + beta * expected_coherence_gradient
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual_gradient, expected_gradient, rtol=0, atol=0)
    assert scales == scales_before
    for name, value in families.items():
        torch.testing.assert_close(value.detach(), family_values_before[name], rtol=0, atol=0)

    if beta == 0:
        native_only = 0.1 * native
        native_gradient, = torch.autograd.grad(native_only, (parameter,))
        torch.testing.assert_close(actual_gradient, native_gradient, rtol=0, atol=0)
    if beta == 1:
        # With retention off, beta=1 is the existing R5 F scalar equation.
        torch.testing.assert_close(actual, 0.1 * native + calibrated_abc, rtol=0, atol=0)


@pytest.mark.parametrize("beta", [0.0, 0.25, 1.0])
def test_matched_scalar_validator_accepts_fixed_nonnegative_beta(beta):
    config = _matched_config(Path("unused.h5"), Path("unused_source"), beta=beta)
    validate_config(config)


@pytest.mark.parametrize("beta", [-0.01, float("nan"), float("inf")])
def test_matched_scalar_validator_rejects_invalid_beta(beta):
    config = _matched_config(Path("unused.h5"), Path("unused_source"), beta=beta)
    with pytest.raises(ValueError):
        validate_config(config)


@pytest.mark.parametrize(
    "change",
    [
        lambda config: config["optimization"].update(gradient_balance="config"),
        lambda config: config["optimization"].update(update_policy="coherence_primal_dual"),
        lambda config: config["runtime"].update(execution_mode="legacy"),
    ],
)
def test_matched_scalar_validator_rejects_dispatch_that_would_misreport_policy(change):
    config = _matched_config(Path("unused.h5"), Path("unused_source"))
    change(config)
    with pytest.raises(ValueError):
        validate_config(config)


def test_resolved_abc_policy_names_fixed_beta_and_live_source_weights():
    class RectifiedFlow:
        sample_source = None
        velocity = None

    class Family:
        version = "1"
        family_weight = 1.0

    config = _matched_config(Path("unused.h5"), Path("unused_source"), beta=0.25)
    families = {name: Family() for name in ("global_distribution", "cross_spectrum", "topology")}
    for family, definition in zip(families.values(), ("marginal_copula_v2", "second_order_blocks_v4", "cubical_persistence")):
        family.definition = definition
    source_identity = {
        "names": ["weight"],
        "layout": [{"name": "weight", "shape": [1], "dtype": "torch.float32"}],
        "tensor_sha256": "a" * 64,
    }
    policy = resolve_training_policy(
        config,
        model=RectifiedFlow(),
        families=families,
        family_scales={name: 1.0 for name in families},
        calibration={"mode": "initial_grad_norm"},
        calibration_sha256="b" * 64,
        source_hashes={"checkpoint": "c" * 64},
        source_metadata={"kind": "native_run", "initial_trainable_parameter_identity": source_identity},
        normalizer_digest="d" * 64,
    )
    assert policy["summary"] == (
        "Fixed weighted sum: 0.1 * native_RF_loss + beta * calibrated_ABC; "
        "no ConFIG; no fidelity controller; no parameter-retention penalty."
    )
    assert policy["resolved_summary"].startswith("Resolved coefficients: native=0.1, beta=0.25;")
    assert policy["objective"]["equation"] == "0.1 * native_RF_loss + 0.25 * calibrated_ABC"
    assert policy["weights"]["selection"] == "live"
    assert policy["weights"]["evaluation_selection"] == "live"
    assert policy["descriptor_versions"]["cross_spectrum"]["definition"] == "second_order_blocks_v4"


def test_legacy_resume_receipt_can_be_reconstructed_on_multiple_own_resumes(tmp_path):
    class Store:
        def __init__(self, run_dir):
            self.run_dir = run_dir

        def write_json(self, relative_path, payload):
            path = self.run_dir / relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(dict(payload), sort_keys=True) + "\n")
            return path

        def update_manifest(self, **details):
            self.details = details

    store = Store(tmp_path)
    policy = {
        "schema_version": "matched_scalar_training_policy_v1",
        "summary": "Fixed weighted sum: beta stays fixed.",
        "resolved_summary": "Resolved beta=0.25.",
        "initialization": {
            "kind": "independent_initialization_from_declared_SOURCE",
            "source_checkpoint_sha256": "e" * 64,
            "source_trainable_tensor_sha256": "f" * 64,
        },
    }
    first = persist_training_policy(store, policy, is_resume=True, start_step=5)
    second = persist_training_policy(store, policy, is_resume=True, start_step=8)
    assert first["initialization"]["receipt_origin"] == "reconstructed_on_own_resume"
    assert second["initialization"] == first["initialization"]
    assert second["execution"]["start_global_step"] == 8
    assert second["policy_sha256"] == first["policy_sha256"]


def _assert_rng_equal(left, right):
    assert left["python"] == right["python"]
    assert left["numpy"] == right["numpy"]
    torch.testing.assert_close(left["torch_cpu"], right["torch_cpu"], rtol=0, atol=0)
    torch.testing.assert_close(left["index_generator"], right["index_generator"], rtol=0, atol=0)


def test_r6_receipt_and_own_resume_preserve_beta_scales_source_optimizer_and_rng(tmp_path):
    dataset = tmp_path / "fields.h5"
    _write_fixture(dataset)
    source = run_base_training(_base_config(dataset), case_dir=tmp_path / "source_case")
    config = _matched_config(dataset, source, beta=0.25)
    validate_config(config)

    with patch.object(post_training, "_coherence_objective", wraps=post_training._coherence_objective) as coherence_objective:
        child = run_post_training(config, case_dir=tmp_path / "resumed", max_steps=1)
    assert coherence_objective.call_count == 1
    assert coherence_objective.call_args.kwargs["require_source_prediction"] is False
    assert coherence_objective.call_args.kwargs["source_anchor_model"] is None
    receipt_path = child / "training_policy.json"
    initial_receipt = json.loads(receipt_path.read_text())
    first = load_project_checkpoint(child / "checkpoints/last.pt")
    assert initial_receipt["execution"]["kind"] == "independent_source_initialization"
    assert initial_receipt["execution"]["start_global_step"] == 0
    assert initial_receipt["execution"]["resume_checkpoint_sha256"] is None
    assert initial_receipt["objective"]["effective_coherence_coefficient"] == 0.25
    assert initial_receipt["objective"]["native_data_coefficient"] == 0.1
    assert initial_receipt["objective"]["parameter_retention"] == {
        "active": False,
        "coefficient": None,
        "definition": None,
    }
    assert initial_receipt["update"]["implementation"] == "training.parameter_retention.scalar_update"
    assert initial_receipt["update"]["conFIG_active"] is False
    assert initial_receipt["update"]["fidelity_controller_active"] is False
    assert initial_receipt["source_identity"]["source_hashes"]["checkpoint"]
    assert initial_receipt["source_identity"]["initial_trainable_parameters"]["tensor_sha256"]
    assert initial_receipt["weights"]["selection"] == "live"
    assert initial_receipt["weights"]["evaluation_selection"] == "live"
    assert initial_receipt["sampler"]["endpoint_rollout"]["steps"] == 1
    assert initial_receipt["objective"]["family_scales"] == first["family_scales"]
    assert initial_receipt["initialization"]["kind"] == "independent_initialization_from_declared_SOURCE"
    assert "no parameter-retention penalty" in initial_receipt["summary"]

    run_post_training(config, case_dir=tmp_path / "resumed", max_steps=1, resume=child)
    resumed_receipt = json.loads(receipt_path.read_text())
    resumed = load_project_checkpoint(child / "checkpoints/last.pt")
    assert resumed_receipt["execution"]["kind"] == "own_run_resume"
    assert resumed_receipt["execution"]["start_global_step"] == 1
    assert len(resumed_receipt["execution"]["resume_checkpoint_sha256"]) == 64
    assert resumed_receipt["policy_sha256"] == initial_receipt["policy_sha256"]
    assert resumed_receipt["objective"]["effective_coherence_coefficient"] == 0.25
    assert resumed_receipt["objective"]["family_scales"] == resumed["family_scales"]
    assert resumed_receipt["source_identity"] == initial_receipt["source_identity"]
    assert resumed_receipt["initialization"] == initial_receipt["initialization"]

    whole = run_post_training(config, case_dir=tmp_path / "whole", max_steps=2)
    uninterrupted = load_project_checkpoint(whole / "checkpoints/last.pt")
    assert resumed["global_step"] == uninterrupted["global_step"] == 2
    for name, value in resumed["model"].items():
        torch.testing.assert_close(value, uninterrupted["model"][name], rtol=0, atol=0)
    for index, state in resumed["optimizer"]["state"].items():
        for key, value in state.items():
            torch.testing.assert_close(value, uninterrupted["optimizer"]["state"][index][key], rtol=0, atol=0)
    assert resumed["random_stream_identity"] == uninterrupted["random_stream_identity"]
    _assert_rng_equal(resumed["rng_state"], uninterrupted["rng_state"])

    rows = [json.loads(line) for line in (child / "metrics/history.jsonl").read_text().splitlines()]
    assert [row["step"] for row in rows] == [1, 2]
    assert [row["epoch"] for row in rows] == [1, 2]
    for row in rows:
        assert row["coherence_applied"] is True
        assert row["total"] == pytest.approx(0.1 * row["native_data_loss"] + 0.25 * row["coherence_loss"])
        assert "parameter_retention_loss" not in row

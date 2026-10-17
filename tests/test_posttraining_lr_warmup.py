"""Contracts for the opt-in post-training model AdamW LR warmup."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import torch
from torch import nn

from phycoflow_reconstruction.config import load_config
from phycoflow_reconstruction.config.validate import validate_config
from phycoflow_reconstruction.training.post_training import _scheduled_model_lr
from phycoflow_reconstruction.training.run_store import RunStore

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_CONFIG_DIR = (
    PROJECT_ROOT / "cases/turbulent_combustion/configs/posttrain/fidelity_budget"
)
_RUN11_CONFIG = _CONFIG_DIR / "run11_config_lagrangian_1000ep.yaml"
_CANDIDATE_CONFIG = _CONFIG_DIR / "run11w_config_lagrangian_lr_warmup_20ep.yaml"
_WARMUP = {"enabled": True, "start_lr": 5.0e-6, "updates": 76}


def _candidate() -> dict:
    return load_config(_CANDIDATE_CONFIG)


def test_candidate_is_run11_with_only_bounded_lr_warmup_changes():
    run11 = load_config(_RUN11_CONFIG)
    candidate = _candidate()
    validate_config(candidate)

    expected = deepcopy(run11)
    expected["optimization"]["epochs"] = 20
    expected["optimization"]["lr_warmup"] = _WARMUP
    expected["output"]["experiment_name"] = "_MultiTask/11w_lagrangian_lr_warmup_20ep"
    assert candidate == expected
    assert "lr_warmup" not in run11["optimization"]


@pytest.mark.parametrize(
    ("warmup", "error", "message"),
    [
        (None, TypeError, "must be a mapping"),
        ({"enabled": 1, "start_lr": 5.0e-6, "updates": 76}, TypeError, "enabled"),
        (
            {"enabled": True, "start_lr": 5.0e-6, "updates": 76, "scheduler": "linear"},
            ValueError,
            "unknown optimization.lr_warmup keys",
        ),
        ({"enabled": True, "start_lr": 0.0, "updates": 76}, ValueError, "start_lr"),
        (
            {"enabled": True, "start_lr": float("nan"), "updates": 76},
            ValueError,
            "start_lr",
        ),
        ({"enabled": True, "start_lr": 5.1e-5, "updates": 76}, ValueError, "<= optimization.lr"),
        ({"enabled": True, "start_lr": 5.0e-6, "updates": 1}, ValueError, "updates must be >= 2"),
        ({"enabled": True, "start_lr": 5.0e-6, "updates": 76.0}, ValueError, "updates must be an integer"),
        ({"enabled": True, "start_lr": 5.0e-6, "updates": True}, ValueError, "updates must be an integer"),
        ({"enabled": True, "updates": 76}, ValueError, "start_lr is required"),
        ({"enabled": True, "start_lr": 5.0e-6}, ValueError, "updates is required"),
    ],
)
def test_lr_warmup_validation_rejects_invalid_enabled_profiles(warmup, error, message):
    config = _candidate()
    config["optimization"]["lr_warmup"] = warmup
    with pytest.raises(error, match=message):
        validate_config(config)


def test_lr_warmup_can_be_disabled_without_schedule_parameters():
    config = _candidate()
    config["optimization"]["lr_warmup"] = {"enabled": False}
    validate_config(config)
    assert _scheduled_model_lr(config["optimization"], 38) is None


def test_scheduled_model_lr_uses_global_completed_updates_and_plateaus():
    optimization = {
        "lr": 5.0e-5,
        "lr_warmup": _WARMUP,
    }

    # The helper receives the zero-based completed-update count before the step.
    assert _scheduled_model_lr(optimization, 0) == pytest.approx(5.0e-6)
    assert _scheduled_model_lr(optimization, 75) == pytest.approx(5.0e-5)
    assert _scheduled_model_lr(optimization, 76) == pytest.approx(5.0e-5)
    assert _scheduled_model_lr(optimization, 1000) == pytest.approx(5.0e-5)

    # After 38 completed updates, the next step is step 39 and uses 38/75 progress.
    assert _scheduled_model_lr(optimization, 38) == pytest.approx(2.78e-5)


def test_missing_or_disabled_lr_warmup_is_a_no_op():
    assert _scheduled_model_lr({"lr": 5.0e-5}, 0) is None
    assert _scheduled_model_lr(
        {"lr": 5.0e-5, "lr_warmup": {"enabled": False, "start_lr": 1.0e-6, "updates": 4}},
        3,
    ) is None


def test_lr_warmup_telemetry_recovery_trims_uncommitted_and_torn_rows(tmp_path):
    metrics = tmp_path / "metrics"
    metrics.mkdir()
    committed = '{"step": 9, "optimizer_lr": 2.78e-5}'
    (metrics / "lr_warmup_updates.jsonl").write_text(
        '{"step": 9, "optimizer_lr": 1.0e-5}\n'
        + committed
        + '\n{"step": 18, "optimizer_lr": 3.0e-5}\n{"step": 27,'
    )

    RunStore(tmp_path, {}).recover_metric_histories(9)

    assert (metrics / "lr_warmup_updates.jsonl").read_text() == committed + "\n"


class _ThreeScalarModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(2.0))
        self.b = nn.Parameter(torch.tensor(2.0))
        self.c = nn.Parameter(torch.tensor(2.0))


def _family_losses(model: _ThreeScalarModel) -> dict[str, torch.Tensor]:
    return {
        "global_distribution": model.a,
        "cross_spectrum": model.b,
        "topology": model.c,
    }


def _endpoint_risk(model: _ThreeScalarModel) -> torch.Tensor:
    return model.a.square().reshape(1).to(torch.float64)


def test_lr_warmup_resume_keeps_adamw_lagrangian_and_abt_step_contract():
    pytest.importorskip("torchjd")
    pytest.importorskip("cooper")

    from phycoflow_reconstruction.training.fidelity_budget import ExternalPolicy, FidelityBudget
    from phycoflow_reconstruction.training.multitask import OuterAggregator

    optimization = {"lr": 5.0e-5, "weight_decay": 1.0e-6, "lr_warmup": _WARMUP}
    policy_settings = {
        "method": "lagrangian",
        "vector_reduction": "sum",
        "objective_scale": "source_mean",
        "lagrangian": {"dual_lr": 0.01},
    }
    reference = FidelityBudget.from_reference(("CH4",), (1.0,), 3.0)
    model = _ThreeScalarModel()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=optimization["lr"], weight_decay=optimization["weight_decay"]
    )
    policy = ExternalPolicy(policy_settings, reference, "cpu")
    aggregator = OuterAggregator({"method": "torchjd_config"})

    for global_step in range(38):
        learning_rate = _scheduled_model_lr(optimization, global_step)
        assert learning_rate is not None
        optimizer.param_groups[0]["lr"] = learning_rate
        result = policy.update_with_coherence_aggregation(
            model,
            optimizer,
            _family_losses(model),
            _endpoint_risk(model),
            None,
            aggregator,
        )
        assert result["family_order"] == ["global_distribution", "cross_spectrum", "topology"]

    checkpoint = {
        "global_step": 38,
        "model": deepcopy(model.state_dict()),
        "optimizer": deepcopy(optimizer.state_dict()),
        "external_policy": deepcopy(policy.state_dict()),
    }
    assert checkpoint["external_policy"]["completed_updates"] == checkpoint["global_step"]
    assert checkpoint["external_policy"]["settings"]["lagrangian"]["dual_lr"] == 0.01
    assert checkpoint["external_policy"]["dual_optimizer"]["param_groups"][0]["lr"] == 0.01

    resumed_model = _ThreeScalarModel()
    resumed_model.load_state_dict(checkpoint["model"])
    resumed_optimizer = torch.optim.AdamW(
        resumed_model.parameters(),
        lr=optimization["lr"],
        weight_decay=optimization["weight_decay"],
    )
    resumed_optimizer.load_state_dict(checkpoint["optimizer"])
    resumed_policy = ExternalPolicy(policy_settings, reference, "cpu")
    resumed_policy.load_state_dict(checkpoint["external_policy"])

    # Scheduler progress is derived from the global cursor; it has no state to checkpoint.
    next_lr = _scheduled_model_lr(optimization, checkpoint["global_step"])
    assert next_lr == pytest.approx(2.78e-5)
    resumed_optimizer.param_groups[0]["lr"] = next_lr
    before = torch.stack(tuple(parameter.detach() for parameter in resumed_model.parameters()))
    model_steps = []
    original_model_step = resumed_optimizer.step

    def count_model_step(*args, **kwargs):
        model_steps.append(True)
        return original_model_step(*args, **kwargs)

    resumed_optimizer.step = count_model_step
    original_dual_step = resumed_policy._dual_optimizer.step

    def dual_step_after_model(*args, **kwargs):
        assert model_steps == [True]
        assert not torch.equal(
            torch.stack(tuple(parameter.detach() for parameter in resumed_model.parameters())), before
        )
        return original_dual_step(*args, **kwargs)

    resumed_policy._dual_optimizer.step = dual_step_after_model
    result = resumed_policy.update_with_coherence_aggregation(
        resumed_model,
        resumed_optimizer,
        _family_losses(resumed_model),
        _endpoint_risk(resumed_model),
        None,
        aggregator,
    )

    assert model_steps == [True]
    assert result["family_order"] == ["global_distribution", "cross_spectrum", "topology"]
    assert resumed_policy.completed_updates == 39
    assert resumed_optimizer.param_groups[0]["lr"] == pytest.approx(next_lr)
    assert resumed_policy._dual_optimizer.param_groups[0]["lr"] == pytest.approx(0.01)
    assert all(float(state["step"]) == 39 for state in resumed_optimizer.state.values())
    assert not torch.equal(torch.stack(tuple(resumed_model.parameters())), before)

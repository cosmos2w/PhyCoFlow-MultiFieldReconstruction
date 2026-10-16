"""Formal Run 11–13 config contracts for nested ConFIG and fidelity policies."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
from helpers.coherence import _post_config

from phycoflow_reconstruction.config import load_config
from phycoflow_reconstruction.config.validate import validate_config

_CONFIG_DIR = (
    Path(__file__).resolve().parents[1]
    / "cases/turbulent_combustion/configs/posttrain/fidelity_budget"
)
_FORMAL_CONFIGS = {
    "lagrangian": "run11_config_lagrangian_1000ep.yaml",
    "rectify": "run12_config_rectify_1000ep.yaml",
    "extended_log_barrier": "run13_config_log_barrier_1000ep.yaml",
}


def _formal_config(method: str = "rectify") -> dict:
    return load_config(_CONFIG_DIR / _FORMAL_CONFIGS[method])


@pytest.mark.parametrize("method", tuple(_FORMAL_CONFIGS))
def test_formal_profiles_validate_and_match_scientific_protocol(method):
    config = _formal_config(method)
    validate_config(config)

    assert config["source_run"] == (
        "runs/tc_gl_rbf_cq_cached_kv_5000ep/20260826T212231Z_7b68c461"
    )
    assert config["source_checkpoint"] == "last.pt"
    assert config["model"]["model_ema_eval"] is False
    assert config["trainable"]["scope"] == "full_model"
    assert config["objectives"] == {
        "data_retention": {"enabled": False, "weight": 0.0},
        "coherence": {"enabled": True, "weight": 1.0},
    }

    optimization = config["optimization"]
    assert optimization["epochs"] == 1000
    assert optimization["batch_size"] == 32
    assert optimization["train_fraction"] == 0.15
    assert "steps_per_epoch" not in optimization
    assert optimization["lr"] == 5.0e-5
    assert optimization["weight_decay"] == 1.0e-6
    assert optimization["grad_clip"] == 1.0
    assert optimization["multitask"]["family_scalarization"] == {"method": "none"}
    assert optimization["multitask"]["outer_aggregation"] is None
    assert optimization["multitask"]["coherence_aggregation"] == {
        "method": "torchjd_config"
    }
    assert optimization["multitask"]["fidelity_budget"]["calibration_batches"] == 8
    assert optimization["multitask"]["fidelity_budget"]["calibration_draws"] == 2
    assert optimization["multitask"]["fidelity_budget"]["calibration_seed"] == 700103

    assert config["rollout"] == {"steps": 2, "solver": "euler"}
    assert config["coherence"]["compute_budget"] == {
        "batch_size": 32,
        "point_count": 4096,
        "query_policy": "fixed_shared",
        "query_seed": 100045,
    }
    assert config["coherence"]["family_balance"]["mode"] == "none"
    assert config["coherence"]["family_balance"]["gradient_diagnostics_every_epochs"] == 0
    families = config["coherence"]["families"]
    assert all(families[name]["enabled"] for name in (
        "global_distribution", "cross_spectrum", "topology"
    ))
    topology = families["topology"]
    assert topology["fields"] == ["CO", "T"]
    assert topology["geometry"]["grid_shape"] == [32, 128]
    assert topology["filtration"]["dimensions"] == [0, 1]
    assert topology["filtration"]["directions"] == ["sublevel", "superlevel"]
    assert topology["persistence"]["projections"] == 32
    assert topology["components"]["mutual"]["groups"] == [["CO", "T"]]
    assert topology["components"]["mutual"]["lines"] == 1

    assert config["checkpointing"]["every_epochs"] == 5
    assert config["checkpointing"]["selection_metric"] == "topology_with_fidelity"
    assert config["checkpointing"]["validation_every_epochs"] == 5
    assert config["evaluation"]["max_samples"] == 32
    assert config["posttrain_fidelity"]["max_relative_mse_increase"] == 0.05
    assert config["posttrain_fidelity"]["max_relative_field_mse_increase"] == 0.05
    preview = config["evaluation"]["preview"]
    assert preview["enabled"] is False
    assert preview["loss_every_epochs"] == 10
    assert preview["reconstruct_every_epochs"] == 200

    policy = optimization["multitask"]["external_policy"]
    assert policy["method"] == method
    assert policy["vector_reduction"] == "sum"
    assert policy["objective_scale"] == "source_mean"
    if method == "lagrangian":
        assert policy["lagrangian"] == {"dual_lr": 0.01}
    elif method == "extended_log_barrier":
        assert policy["extended_log_barrier"] == {
            "t_initial": 1.0,
            "growth_factor": 1.1,
            "update_interval": 100,
            "t_max": 100.0,
        }


def test_formal_profiles_share_every_non_policy_setting():
    configs = [_formal_config(method) for method in _FORMAL_CONFIGS]
    baseline = deepcopy(configs[0])
    baseline["optimization"]["multitask"].pop("external_policy")
    for config in configs[1:]:
        candidate = deepcopy(config)
        candidate["optimization"]["multitask"].pop("external_policy")
        assert {key: value for key, value in candidate.items() if key != "output"} == {
            key: value for key, value in baseline.items() if key != "output"
        }

    assert [config["output"]["experiment_name"] for config in configs] == [
        "_MultiTask/11_config_lagrangian_1000ep",
        "_MultiTask/12_config_rectify_1000ep",
        "_MultiTask/13_config_log_barrier_1000ep",
    ]


def test_coherence_aggregation_requires_external_policy_and_explicit_none_scalarizer():
    legacy = _post_config(Path("/tmp/formal-unused.h5"), Path("/tmp/source"))
    legacy["optimization"]["multitask"] = {
        "family_scalarization": {"method": "none"},
        "coherence_aggregation": {"method": "torchjd_config"},
    }
    with pytest.raises(ValueError, match="coherence_aggregation requires external_policy"):
        validate_config(legacy)

    config = _formal_config()
    config["optimization"]["multitask"]["family_scalarization"] = {
        "method": "fixed_sum"
    }
    with pytest.raises(ValueError, match="family_scalarization.method=none"):
        validate_config(config)


@pytest.mark.parametrize(
    ("selector", "message"),
    [
        ({"method": "sum"}, "method must be torchjd_config"),
        ({}, "method must be torchjd_config"),
        ({"method": "torchjd_config", "legacy": True}, "unknown .* keys"),
        (None, "must be a mapping"),
    ],
)
def test_coherence_aggregation_rejects_unknown_or_invalid_selectors(selector, message):
    config = _formal_config()
    config["optimization"]["multitask"]["coherence_aggregation"] = selector
    with pytest.raises((TypeError, ValueError), match=message):
        validate_config(config)


@pytest.mark.parametrize("family", ["global_distribution", "cross_spectrum", "topology"])
def test_coherence_aggregation_requires_all_three_families_enabled(family):
    config = _formal_config()
    config["checkpointing"]["selection_metric"] = "reconstruction_mse"
    config["checkpointing"].pop("validation_every_epochs")
    config["coherence"]["families"][family]["enabled"] = False
    with pytest.raises(ValueError, match="requires enabled coherence families"):
        validate_config(config)


def test_coherence_aggregation_rejects_non_null_outer_aggregation():
    config = _formal_config()
    config["optimization"]["multitask"]["outer_aggregation"] = {
        "method": "torchjd_config"
    }
    with pytest.raises(ValueError, match="outer_aggregation must be null or omitted"):
        validate_config(config)


def test_external_none_family_route_remains_valid_without_coherence_aggregation():
    config = _formal_config()
    config["optimization"]["multitask"].pop("coherence_aggregation")
    config["evaluation"]["preview"].pop("loss_every_epochs")
    config["evaluation"]["preview"].pop("reconstruct_every_epochs")
    validate_config(config)


def test_legacy_none_plus_outer_config_route_keeps_its_existing_meaning():
    config = _post_config(Path("/tmp/legacy-unused.h5"), Path("/tmp/source"))
    config["optimization"]["multitask"] = {
        "family_scalarization": {"method": "none"},
        "outer_aggregation": {"method": "torchjd_config"},
    }
    validate_config(config)

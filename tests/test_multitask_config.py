"""Focused contracts for the explicit post-training multitask configuration."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import yaml
from helpers.coherence import _post_config

from phycoflow_reconstruction.config import load_config
from phycoflow_reconstruction.config.validate import validate_config


def _config() -> dict:
    return _post_config(Path("/tmp/multitask-unused.h5"), Path("/tmp/multitask-source"))


@pytest.mark.parametrize(
    ("family_scalarization", "outer_aggregation"),
    [
        ({}, {}),
        ({"method": "fixed_sum"}, {"method": "sum"}),
        ({"method": "stch", "stch": {"mu": 0.1}}, {"method": "torchjd_config"}),
        ({"method": "dwa", "dwa": {"temperature": 2.0}}, {"method": "upgrad"}),
        ({"method": "fixed_sum"}, {"method": "cagrad", "cagrad": {"c": 0.5}}),
    ],
)
def test_multitask_methods_and_defaults_validate(family_scalarization, outer_aggregation):
    config = _config()
    config["optimization"]["multitask"] = {
        "family_scalarization": family_scalarization,
        "outer_aggregation": outer_aggregation,
    }
    validate_config(config)


def test_multitask_defaults_validate_with_inherited_legacy_config_fields():
    config = _config()
    config["optimization"].update(
        gradient_balance="config",
        config_missing_behavior="weighted_sum",
        config_data_grad_scale=3.0,
        config_coherence_grad_scale=0.5,
        multitask={},
    )
    validate_config(config)


def test_multitask_defaults_compose_and_accept_dotted_cli_overrides(tmp_path):
    base = _config()
    base["optimization"]["multitask"] = {}
    base_path = tmp_path / "base.yaml"
    base_path.write_text(yaml.safe_dump(base), encoding="utf-8")
    child_path = tmp_path / "child.yaml"
    child_path.write_text(
        "defaults: [base.yaml]\n"
        "optimization:\n"
        "  multitask:\n"
        "    family_scalarization: {method: dwa}\n"
        "    outer_aggregation: {method: cagrad}\n",
        encoding="utf-8",
    )

    config = load_config(
        child_path,
        overrides=[
            "optimization.multitask.family_scalarization.dwa.temperature=1.75",
            "optimization.multitask.outer_aggregation.cagrad.c=0.25",
        ],
    )
    validate_config(config)
    assert config["optimization"]["multitask"]["family_scalarization"] == {
        "method": "dwa",
        "dwa": {"temperature": 1.75},
    }
    assert config["optimization"]["multitask"]["outer_aggregation"] == {
        "method": "cagrad",
        "cagrad": {"c": 0.25},
    }


@pytest.mark.parametrize(
    ("interface", "config", "message"),
    [
        ("family_scalarization", {"method": "bad"}, "family_scalarization.method"),
        ("family_scalarization", {"method": "fixed_sum", "stch": {"mu": 0.1}}, "unknown"),
        ("family_scalarization", {"method": "stch", "weights": [1.0]}, "unknown"),
        ("outer_aggregation", {"method": "sum", "cagrad": {"c": 0.5}}, "unknown"),
        ("outer_aggregation", {"method": "cagrad", "cagrad": {"c": True}}, "cagrad.c"),
        ("outer_aggregation", {"method": "cagrad", "cagrad": {"c": float("nan")}}, "cagrad.c"),
        ("outer_aggregation", {"method": "cagrad", "cagrad": {"c": 10**400}}, "cagrad.c"),
        ("family_scalarization", {"method": "stch", "stch": {"mu": 0.0}}, "stch.mu"),
        (
            "family_scalarization",
            {"method": "dwa", "dwa": {"temperature": float("inf")}},
            "dwa.temperature",
        ),
    ],
)
def test_multitask_rejects_inactive_or_invalid_method_settings(interface, config, message):
    valid = _config()
    valid["optimization"]["multitask"] = {interface: config}
    with pytest.raises(ValueError, match=message):
        validate_config(valid)


@pytest.mark.parametrize("mode", ["component_constrained", "topology_regularized"])
def test_multitask_rejects_constrained_topology_modes(mode):
    config = _config()
    config["optimization"].update(
        gradient_balance=mode,
        multitask={"family_scalarization": {}, "outer_aggregation": {}},
    )
    with pytest.raises(ValueError, match="cannot be combined with constrained topology"):
        validate_config(config)


def test_multitask_is_rejected_by_physics_only_posttraining():
    config = _config()
    config.pop("coherence")
    config["physics"] = {}
    config["objectives"] = {
        "data_retention": {"enabled": True, "weight": 0.1},
        "physics": {"enabled": True, "weight": 1.0},
    }
    config["optimization"]["multitask"] = {}
    with pytest.raises(ValueError, match="requires coherence post-training"):
        validate_config(config)


def test_legacy_coherence_config_without_multitask_remains_valid():
    config = deepcopy(_config())
    config["optimization"].update(
        gradient_balance="config",
        config_missing_behavior="error",
        config_data_grad_scale=1.0,
        config_coherence_grad_scale=1.0,
    )
    validate_config(config)

"""Contract for the portable Run 11f-style fidelity-budget recipe."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import yaml

from phycoflow_reconstruction.config import load_config
from phycoflow_reconstruction.config.validate import validate_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_CONFIG_DIR = (
    PROJECT_ROOT / "cases/turbulent_combustion/configs/posttrain/fidelity_budget"
)
_PORTABLE_CONFIG = _CONFIG_DIR / "lagrangian_config_warmup.yaml"
_HISTORICAL_11F_CONFIG = (
    _CONFIG_DIR / "run11f_lagrangian_lr_warmup_dense_fast_100ep.yaml"
)


def _without_deployment_differences(config: dict) -> dict:
    result = deepcopy(config)
    result.pop("source_run")
    result.pop("output")
    result["optimization"].pop("epochs")
    runtime = result["runtime"]
    for key in ("device", "num_workers", "keops_cache_folder"):
        runtime.pop(key, None)
    return result


def test_portable_recipe_matches_historical_11f_science_without_20ep_dependency():
    recipe = yaml.safe_load(_PORTABLE_CONFIG.read_text())
    assert recipe["defaults"] == ["_formal_1000ep_common.yaml"]

    historical = load_config(_HISTORICAL_11F_CONFIG)
    portable = load_config(_PORTABLE_CONFIG)

    assert portable["source_run"] is None
    assert portable["source_checkpoint"] == historical["source_checkpoint"] == "last.pt"
    assert portable["optimization"]["epochs"] == 1000
    assert historical["optimization"]["epochs"] == 100
    assert portable["optimization"]["lr"] == 5.0e-5
    assert portable["optimization"]["lr_warmup"] == {
        "enabled": True,
        "start_lr": 5.0e-6,
        "updates": 76,
    }
    assert portable["optimization"]["multitask"]["family_scalarization"] == {
        "method": "none"
    }
    assert portable["optimization"]["multitask"]["coherence_aggregation"] == {
        "method": "torchjd_config"
    }
    assert portable["optimization"]["multitask"]["outer_aggregation"] is None
    assert portable["optimization"]["multitask"]["external_policy"] == {
        "method": "lagrangian",
        "vector_reduction": "sum",
        "objective_scale": "source_mean",
        "lagrangian": {"dual_lr": 0.01, "penalty_interval_updates": 1},
    }

    assert _without_deployment_differences(portable) == _without_deployment_differences(
        historical
    )
    assert portable["output"]["experiment_name"] == (
        "posttrain_lagrangian_config_warmup"
    )
    assert portable["output"]["experiment_name"] != historical["output"]["experiment_name"]


def test_portable_recipe_validates_when_source_is_supplied(tmp_path):
    config = load_config(_PORTABLE_CONFIG, overrides=[f"source_run={tmp_path / 'source'}"])
    validate_config(config)

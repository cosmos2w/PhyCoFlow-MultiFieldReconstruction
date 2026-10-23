"""Portable profiles preserve the bounded study's resolved scientific contract."""

from pathlib import Path

import pytest

from phycoflow_reconstruction.config import load_config
from phycoflow_reconstruction.config.validate import validate_config

PROFILES = Path(__file__).resolve().parents[1] / "cases/turbulent_combustion/configs/posttrain/fidelity_budget"


@pytest.mark.parametrize("name", ["fast_lagrangian_scalar", "fast_lagrangian_config", "lpc_ordered"])
def test_portable_profile_uses_explicit_source_and_dense_fixed_protocol(tmp_path, name):
    cfg = load_config(PROFILES / f"{name}.yaml", overrides=[f"source_run={tmp_path / 'source'}"])
    validate_config(cfg)
    assert load_config(PROFILES / f"{name}.yaml")["source_run"] is None
    opt = cfg["optimization"]
    assert (opt["epochs"], opt["batch_size"], opt["train_fraction"]) == (100, 32, 0.15)
    assert (opt["lr"], opt["weight_decay"], opt["grad_clip"]) == (5e-5, 1e-6, 1.0)
    assert opt["lr_warmup"] == {"enabled": True, "start_lr": 5e-6, "updates": 76}
    assert opt["fast_lagrangian"] == {
        "topology_estimator": "p3", "family_probability": 1.0,
        "coefficient_interval": 1, "seed": 700211,
    }
    mt = opt["multitask"]
    assert mt["external_policy"]["lagrangian"] == {"dual_lr": 0.01, "penalty_interval_updates": 1}
    assert mt["fidelity_budget"]["relative_increase"] == 0.05
    assert mt["fidelity_budget"]["calibration_batches"] == 8
    assert mt["fidelity_budget"]["calibration_draws"] == 2
    assert cfg["rollout"] == {"steps": 2, "solver": "euler"}
    assert cfg["coherence"]["compute_budget"]["point_count"] == 4096
    topology = cfg["coherence"]["families"]["topology"]
    assert topology["geometry"]["grid_shape"] == [32, 128]
    assert topology["filtration"]["dimensions"] == [0, 1]
    assert topology["filtration"]["directions"] == ["sublevel", "superlevel"]
    assert cfg["model"]["model_ema_eval"] is False
    if name == "fast_lagrangian_config":
        assert mt["coherence_aggregation"] == {"method": "torchjd_config"}
        assert mt["family_scalarization"]["method"] == "none"
    else:
        assert "coherence_aggregation" not in mt
        assert mt["family_scalarization"]["method"] == "fixed_sum"
    assert ("lpc_ordered" in opt) == (name == "lpc_ordered")

"""Static contract checks for the formal GL_rbf_CQ B/C arms."""

from __future__ import annotations

import copy
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_ROOT = PROJECT_ROOT / "benchmarks" / "gl_rbf_cq_migration_200ep"
CONFIG_B = BENCHMARK_ROOT / "configs" / "B_gl_rbf_cq_legacy_mha_200ep.yaml"
CONFIG_C = BENCHMARK_ROOT / "configs" / "C_gl_rbf_cq_cached_kv_200ep.yaml"


def _load(path: Path) -> dict:
    config = yaml.safe_load(path.read_text())
    assert isinstance(config, dict)
    return config


def test_b_c_configs_share_every_scientific_setting_except_execution():
    arm_b = _load(CONFIG_B)
    arm_c = _load(CONFIG_C)
    assert arm_b["model"]["name"] == arm_c["model"]["name"] == "gl_rbf_cq"
    assert arm_b["model"]["condition_attention_execution"] == "legacy_mha"
    assert arm_c["model"]["condition_attention_execution"] == "cached_kv"
    assert arm_b["model"]["sensor_attention_padding_mode"] == arm_c["model"][
        "sensor_attention_padding_mode"
    ] == "full"

    comparable_b = copy.deepcopy(arm_b)
    comparable_c = copy.deepcopy(arm_c)
    comparable_b["model"].pop("condition_attention_execution")
    comparable_c["model"].pop("condition_attention_execution")
    comparable_b["output"]["experiment_name"] = "same"
    comparable_c["output"]["experiment_name"] = "same"
    assert comparable_b == comparable_c

    for config in (arm_b, arm_c):
        assert config["dataset"]["normalization"] == "mean_std"
        assert config["dataset"]["normalization_stats_path"] == (
            "../../benchmarks/gl_rbf_cq_migration_200ep/downstream_train_normalization.json"
        )
        assert config["observations"] == {
            "protocol": "random_uniform",
            "seed": 42,
            "fields": {"T": {"count_min": 192, "count_max": 384}},
        }
        assert config["optimization"] == {
            "epochs": 200,
            "batch_size": 40,
            "lr": 1.0e-4,
            "weight_decay": 1.0e-6,
            "grad_clip": 1.0,
            "backward_loss_scale": 1.0,
        }
        model = config["model"]
        assert model["query_points"] == 4096
        assert model["train_query_microbatch_size"] == 2048
        assert model["reuse_condition_context_across_query_microbatches"] is True
        assert model["model_ema_enabled"] is True
        assert model["model_ema_decay"] == 0.999


def test_b_c_use_the_frozen_checkpoint_and_evaluation_policy():
    arm_b = _load(CONFIG_B)
    arm_c = _load(CONFIG_C)
    for config in (arm_b, arm_c):
        assert config["evaluation"] == {
            "split": "validation",
            "max_samples": 20,
            "query_points": 4096,
            "generation_steps": 32,
            "seed": 2027,
            "preview": {
                "enabled": False,
                "every_epochs": 20,
                "split": "validation",
                "sample_index": 0,
                "query_points": 4096,
                "generation_steps": 32,
                "seed": 2027,
                "keep_history": False,
            },
        }
        assert config["checkpointing"] == {
            "enabled": True,
            "every_epochs": 20,
            "epochs": [1, 20, 40, 60, 100, 150, 200],
            "save_epoch_one": True,
        }
        assert config["benchmark_telemetry"] == {"enabled": True, "sample_steps": 5}

"""Arm-A benchmark configuration and generic telemetry contracts."""

import json
from pathlib import Path

import torch

from phycoflow_reconstruction.config import load_config
from phycoflow_reconstruction.config.validate import validate_config
from phycoflow_reconstruction.contracts import DataSpec
from phycoflow_reconstruction.models import build_model
from phycoflow_reconstruction.training.benchmark_telemetry import BenchmarkTelemetry

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_ROOT = PROJECT_ROOT / "benchmarks" / "gl_rbf_cq_migration_200ep"


def test_arm_a_config_matches_historical_scale_and_common_protocol():
    config = load_config(BENCHMARK_ROOT / "configs" / "A_legacy_gl_rbf_enh_200ep.yaml")
    validate_config(config)
    assert config["model"]["hidden_dim"] == 256
    assert config["model"]["latent_dim"] == 256
    assert config["model"]["num_latents"] == 128
    assert config["model"]["heads"] == 8
    assert config["model"]["latent_blocks"] == 4
    assert config["model"]["field_embedding_dim"] == 128
    assert config["model"]["fourier_bands"] == 32
    assert config["model"]["rff_features"] == 256
    assert config["model"]["rff_lengthscale"] == 0.15
    assert config["model"]["query_points"] == 4096
    assert config["optimization"] == {
        "epochs": 200,
        "batch_size": 40,
        "lr": 1.0e-4,
        "weight_decay": 1.0e-6,
        "grad_clip": 1.0,
        "backward_loss_scale": 7.52316384526264e-37,
    }
    assert config["observations"]["fields"] == {"T": {"count_min": 192, "count_max": 384}}


def test_arm_a_capacity_options_reach_legacy_model():
    spec = DataSpec(("CH4", "CO", "T", "U_1", "p"), ("1",) * 5, 2, (4, 4))
    model = build_model(
        {
            "name": "pointcloud_ffm",
            "backbone": "gl_rbf_enh",
            "hidden_dim": 256,
            "latent_dim": 256,
            "num_latents": 128,
            "heads": 8,
            "latent_blocks": 4,
            "gather_topk": 32,
            "rbf_sigma": 0.05,
            "field_embedding_dim": 128,
            "fourier_bands": 32,
            "fourier_max_frequency": 64.0,
            "rff_features": 256,
            "rff_lengthscale": 0.15,
        },
        spec,
    )
    assert model.velocity_model.field_embedding.embedding_dim == 128
    assert model.velocity_model.position.frequencies.numel() == 32
    assert float(model.velocity_model.position.frequencies[-1]) == 64.0
    assert model.prior.features == 256


def test_benchmark_telemetry_records_cpu_step_and_epoch(tmp_path):
    telemetry = BenchmarkTelemetry(
        tmp_path,
        enabled=True,
        device=torch.device("cpu"),
        steps_per_epoch=2,
        parameter_count=12,
        trainable_parameter_count=10,
    )
    telemetry.start_step(0)
    telemetry.start_phase("forward_native_loss")
    telemetry.end_phase("forward_native_loss")
    telemetry.start_phase("backward")
    telemetry.end_phase("backward")
    telemetry.start_phase("optimizer")
    telemetry.end_phase("optimizer")
    telemetry.finish_step()
    telemetry.finish_epoch(1)
    telemetry.close()
    payload = json.loads(
        (tmp_path / "metrics" / "benchmark_telemetry.json").read_text(encoding="utf-8")
    )
    assert payload["epochs"][0]["epoch"] == 1
    assert payload["epochs"][0]["steps_per_epoch"] == 2
    assert payload["epochs"][0]["parameter_count"] == 12
    assert payload["steps"][0]["forward_native_loss_time_s"] >= 0


def test_benchmark_telemetry_is_noop_when_disabled(tmp_path):
    telemetry = BenchmarkTelemetry(
        tmp_path,
        enabled=False,
        device=torch.device("cpu"),
        steps_per_epoch=1,
        parameter_count=1,
        trainable_parameter_count=1,
    )
    telemetry.start_step(0)
    telemetry.finish_step()
    telemetry.close()
    assert not (tmp_path / "metrics").exists()

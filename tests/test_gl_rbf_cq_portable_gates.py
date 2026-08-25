"""Portable GL_rbf_CQ correctness gates independent of the combustion dataset."""

from __future__ import annotations

import copy
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEMO_ROOT = PROJECT_ROOT.parent / "0_demo_TurbulentCombustion"


def _small_core_config(coord_dim: int, execution: str = "cached_kv") -> dict:
    return {
        "model_name": "GL_rbf_CQ",
        "backbone": "GL_rbf_ENH_CQ",
        "coord_dim": coord_dim,
        "prior": "iid",
        "hidden_dim": 16,
        "cond_dim": 8,
        "field_embed_dim": 4,
        "latent_dim": 16,
        "num_latents": 8,
        "num_heads": 4,
        "num_latent_blocks": 2,
        "ff_mult": 2,
        "attn_dropout": 0.0,
        "mlp_dropout": 0.0,
        "summary_type": "mean",
        "gather_mode": "topk_rbf_glres",
        "gather_topk": 3,
        "gather_query_chunk_size": 5,
        "learnable_rbf_sigma": True,
        "neighbor_backend": "torch",
        "USE_FOURIER_PE": True,
        "fourier_pe_num_bands": 2,
        "fourier_pe_max_freq": 4.0,
        "sensor_coord_encoding": "fourier",
        "latent_sensor_reinject": True,
        "latent_reinject_every": 1,
        "condition_attention_execution": execution,
        "sensor_attention_padding_mode": "full",
        "glres_scale_init": 1.0e-2,
        "cq_query_dim": 8,
        "cq_readout_mode": "lowrank",
        "cq_readout_rank": 4,
        "cq_readout_heads": 2,
        "cq_fusion_mode": "additive",
        "cq_time_conditioning": "sinusoidal_film",
        "cq_time_embed_dim": 8,
        "cq_time_max_period": 10000.0,
        "cq_time_film_zero_init": True,
        "cq_measurement_support_mode": "rbf_value_support",
        "cq_measurement_support_normalize": True,
    }


def _core_inputs(coord_dim: int, n_fields: int, *, query_count: int = 17) -> dict:
    generator = torch.Generator(device="cpu").manual_seed(8201 + coord_dim + n_fields)
    coords = torch.rand(2, query_count, coord_dim, generator=generator)
    first = torch.linspace(0, query_count - 1, 6).round().to(torch.long)
    second = (first + 1).remainder(query_count)
    obs_indices = torch.stack((first, second))
    obs_mask = torch.tensor(
        [[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0]], dtype=torch.bool
    )
    obs_field_ids = torch.arange(6).remainder(n_fields).repeat(2, 1)
    return {
        "x1": torch.randn(2, query_count, n_fields, generator=generator),
        "coords": coords,
        "obs_coords": torch.stack([coords[i, obs_indices[i]] for i in range(2)]),
        "obs_values": torch.randn(2, 6, 1, generator=generator),
        "obs_mask": obs_mask,
        "obs_field_ids": obs_field_ids,
        "obs_indices": obs_indices,
    }


def test_declared_portable_package_imports_in_isolation(tmp_path: Path):
    manifest = yaml.safe_load((DEMO_ROOT / "GL_rbf_CQ_RELEASE_MANIFEST.yaml").read_text())
    for relative in manifest["portable_core_files"]:
        source = DEMO_ROOT / relative
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    script = r'''
import json
import sys
import torch

sys.modules["pykeops"] = None
from phycoflow_pointcloud import build_pointcloud_model

config = {
    "model_name": "GL_rbf_CQ", "backbone": "GL_rbf_ENH_CQ", "coord_dim": 2,
    "prior": "iid", "hidden_dim": 16, "cond_dim": 8, "field_embed_dim": 4,
    "latent_dim": 16, "num_latents": 8, "num_heads": 4, "num_latent_blocks": 1,
    "ff_mult": 2, "summary_type": "mean", "gather_mode": "topk_rbf_glres",
    "gather_topk": 3, "neighbor_backend": "torch", "USE_FOURIER_PE": True,
    "fourier_pe_num_bands": 2, "fourier_pe_max_freq": 4.0,
    "sensor_coord_encoding": "fourier", "latent_sensor_reinject": True,
    "condition_attention_execution": "cached_kv",
    "sensor_attention_padding_mode": "full", "cq_query_dim": 8,
    "cq_readout_mode": "lowrank", "cq_readout_rank": 4,
    "cq_readout_heads": 2, "cq_fusion_mode": "additive",
}
model = build_pointcloud_model(config, n_fields=3, device="cpu").eval()
coords = torch.rand(1, 9, 2)
obs_coords = coords[:, [0, 3, 7]]
out = model.model(
    torch.tensor([0.25]), torch.randn(1, 9, 3), coords, obs_coords,
    torch.randn(1, 3, 1), torch.ones(1, 3), torch.tensor([[0, 1, 2]]),
)
forbidden = ["Model", "helpers", "_legacy_model_full", "train_pointcloud_ffm", "neuralop"]
assert not any(name in sys.modules for name in forbidden)
assert tuple(out.shape) == (1, 9, 3)
print(json.dumps({"shape": list(out.shape), "module": model.model.__class__.__module__}))
'''
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(tmp_path / "src")
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=True,
    )
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert payload["shape"] == [1, 9, 3]


@pytest.mark.parametrize(("coord_dim", "n_fields"), [(2, 2), (3, 5)])
def test_portable_core_forward_backward_and_padding(coord_dim: int, n_fields: int):
    portable = pytest.importorskip("phycoflow_pointcloud")
    torch.manual_seed(301 + coord_dim + n_fields)
    model = portable.build_pointcloud_model(
        _small_core_config(coord_dim), n_fields=n_fields, device="cpu"
    )
    values = _core_inputs(coord_dim, n_fields)
    output = model.model(
        torch.tensor([0.2, 0.8]),
        values["x1"],
        values["coords"],
        values["obs_coords"],
        values["obs_values"],
        values["obs_mask"],
        values["obs_field_ids"],
    )
    assert output.shape == values["x1"].shape
    assert torch.isfinite(output).all()
    output.square().mean().backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_legacy_and_cached_initial_state_identity_and_kv_projection_counts():
    portable = pytest.importorskip("phycoflow_pointcloud")
    values = _core_inputs(2, 5)
    torch.manual_seed(991)
    legacy = portable.build_pointcloud_model(
        _small_core_config(2, "legacy_mha"), n_fields=5, device="cpu"
    )
    torch.manual_seed(991)
    cached = portable.build_pointcloud_model(
        _small_core_config(2, "cached_kv"), n_fields=5, device="cpu"
    )
    assert legacy.state_dict().keys() == cached.state_dict().keys()
    for key in legacy.state_dict():
        torch.testing.assert_close(legacy.state_dict()[key], cached.state_dict()[key], rtol=0, atol=0)

    legacy.model.input_cross_attn.reset_execution_counters()
    cached.model.input_cross_attn.reset_execution_counters()
    with torch.no_grad():
        legacy_out = legacy.model(
            torch.tensor([0.2, 0.8]), values["x1"], values["coords"], values["obs_coords"],
            values["obs_values"], values["obs_mask"], values["obs_field_ids"],
        )
        cached_out = cached.model(
            torch.tensor([0.2, 0.8]), values["x1"], values["coords"], values["obs_coords"],
            values["obs_values"], values["obs_mask"], values["obs_field_ids"],
        )
    assert torch.isfinite(legacy_out).all() and torch.isfinite(cached_out).all()
    assert cached.model.input_cross_attn.kv_projection_calls == 1
    assert legacy.model.input_cross_attn.kv_projection_calls > cached.model.input_cross_attn.kv_projection_calls


def test_query_microbatch_loss_and_gradients_match_monolithic():
    portable = pytest.importorskip("phycoflow_pointcloud")
    values = _core_inputs(2, 5)
    torch.manual_seed(710)
    full = portable.build_pointcloud_model(_small_core_config(2), n_fields=5, device="cpu")
    micro = copy.deepcopy(full)

    torch.manual_seed(991)
    full_loss, _ = full.training_loss(**values)
    full_loss.backward()
    torch.manual_seed(991)
    micro_loss, metrics = micro.training_loss_microbatched(
        **values,
        query_microbatch_size=5,
        backward=True,
        reuse_condition_context=True,
    )
    torch.testing.assert_close(micro_loss, full_loss.detach(), rtol=3e-6, atol=3e-7)
    assert metrics["query_microbatches"] == 4.0
    for full_parameter, micro_parameter in zip(full.parameters(), micro.parameters()):
        if full_parameter.grad is None or micro_parameter.grad is None:
            assert full_parameter.grad is None and micro_parameter.grad is None
        else:
            torch.testing.assert_close(
                micro_parameter.grad, full_parameter.grad, rtol=5e-5, atol=5e-7
            )


def test_persistent_topk_reconstruction_adds_zero_knn_calls():
    portable = pytest.importorskip("phycoflow_pointcloud")
    values = _core_inputs(3, 3)
    model = portable.build_pointcloud_model(_small_core_config(3), n_fields=3, device="cpu").eval()
    original = model.model._get_topk_neighbors
    calls = 0

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    model.model._get_topk_neighbors = counted
    geometry = model.prepare_reconstruction_geometry_cache(
        coords=values["coords"],
        obs_coords=values["obs_coords"],
        obs_mask=values["obs_mask"],
        chunk_size=5,
    )
    assert calls > 0
    calls = 0
    tensor_values = {key: value for key, value in values.items() if key != "x1"}
    reconstruction = portable.ReconstructionConfig(
        n_steps=2,
        obs_consistency_mode="none",
        execution_mode="cached_streamed",
        query_chunk_size=5,
        cache_level="static_features",
    )
    torch.manual_seed(717)
    fresh = portable.reconstruct_from_tensors(model, **tensor_values, config=reconstruction)
    fresh_calls = calls
    calls = 0
    torch.manual_seed(717)
    persistent = portable.reconstruct_from_tensors(
        model, **tensor_values, config=reconstruction, geometry_cache=geometry
    )
    assert calls == 0
    assert fresh_calls > 0
    torch.testing.assert_close(persistent, fresh, rtol=1e-5, atol=2e-6)


def test_ema_save_load_resume_and_evaluation_selection():
    portable = pytest.importorskip("phycoflow_pointcloud")
    values = _core_inputs(2, 3, query_count=13)
    torch.manual_seed(817)
    model = portable.build_pointcloud_model(_small_core_config(2), n_fields=3, device="cpu")
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
    ema = portable.ModelEMA(model, decay=0.9)
    loss, _ = model.training_loss(**values)
    loss.backward()
    optimizer.step()
    ema.update(model)
    assert ema.num_updates == 1

    payload = {
        "model": model.state_dict(),
        "model_ema": ema.state_dict(),
        "model_ema_enabled": True,
        "model_ema_eval": True,
        "optimizer": optimizer.state_dict(),
    }
    torch.manual_seed(817)
    resumed = portable.build_pointcloud_model(_small_core_config(2), n_fields=3, device="cpu")
    selected = portable.resolve_checkpoint_state(payload, model=resumed)
    assert selected.selection == "ema_trainable_plus_live_frozen"
    resumed.load_state_dict(selected.state_dict, strict=True)
    resumed_optimizer = torch.optim.Adam(resumed.parameters(), lr=1.0e-3)
    resumed_optimizer.load_state_dict(payload["optimizer"])
    resumed_ema = portable.ModelEMA(resumed, decay=0.9)
    resumed_ema.load_state_dict(payload["model_ema"])
    assert resumed_ema.num_updates == 1
    assert resumed_optimizer.state_dict()["state"].keys() == payload["optimizer"]["state"].keys()

    with torch.no_grad():
        for parameter in resumed.parameters():
            if parameter.requires_grad:
                parameter.add_(0.123)
                break
    live_state = {key: value.detach().clone() for key, value in resumed.state_dict().items()}
    with resumed_ema.average_parameters(resumed):
        assert any(
            not torch.equal(value, live_state[key])
            for key, value in resumed.state_dict().items()
        )
    for key, value in resumed.state_dict().items():
        torch.testing.assert_close(value, live_state[key], rtol=0, atol=0)

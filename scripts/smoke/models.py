#!/usr/bin/env python
"""Run the short model-registry smoke matrix on a synthetic contract.

This is a local acceptance check, not a benchmark.  It builds each registered
model with deliberately small dimensions, computes one native loss and update,
and performs a one-step reconstruction.  Optional operator dependencies are
reported as skips so the script remains useful on a minimal installation.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import tempfile
from pathlib import Path
from typing import Any

import torch

_KEOPS_CACHE: tempfile.TemporaryDirectory[str] | None = None


def _prepare_keops_cache() -> None:
    """Give optional KeOps a build directory before importing model adapters."""

    global _KEOPS_CACHE
    cache_root = os.environ.get("KEOPS_CACHE_FOLDER")
    if cache_root is None:
        _KEOPS_CACHE = tempfile.TemporaryDirectory(prefix="phycoflow_keops_smoke_")
        cache_root = _KEOPS_CACHE.name
        os.environ["KEOPS_CACHE_FOLDER"] = cache_root
    folder_name = "_".join(platform.uname()[:3]) + f"_p{sys.version.split(' ')[0]}"
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible_devices:
        folder_name += f"_CUDA_VISIBLE_DEVICES_{visible_devices.replace(',', '_')}"
    Path(cache_root, folder_name).mkdir(parents=True, exist_ok=True)


_prepare_keops_cache()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC = PROJECT_ROOT / "src"
CASE_DIR = PROJECT_ROOT / "cases" / "brusselator"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from phycoflow_reconstruction.contracts import DataSpec, ObservationBatch
from phycoflow_reconstruction.data.normalization import FieldNormalizer
from phycoflow_reconstruction.models import build_model

MODEL_CONFIGS: dict[str, dict[str, Any]] = {
    "coordinate_mlp": {"name": "coordinate_mlp", "hidden_dim": 8, "fourier_bands": 2},
    "mlp_rbf": {"name": "mlp_rbf", "hidden_dim": 8, "fourier_bands": 2},
    "deeponet": {"name": "deeponet", "width": 8, "basis_dim": 4},
    "senseiver": {
        "name": "senseiver",
        "width": 8,
        "num_latents": 2,
        "heads": 2,
        "depth": 1,
    },
    "geofno": {"name": "geofno", "hidden_channels": 8, "layers": 1},
    "diffusion_pde": {"name": "diffusion_pde", "hidden_channels": 8},
    "latent_fm_stage1": {"name": "latent_fm", "latent_channels": 4, "stage": 1},
    "pointcloud_ffm_gl_rbf": {
        "name": "pointcloud_ffm",
        "backbone": "gl_rbf_enh",
        "prior": "iid",
        "hidden_dim": 8,
        "latent_dim": 8,
        "num_latents": 2,
        "heads": 2,
        "latent_blocks": 1,
        "gather_topk": 2,
    },
    "pointcloud_ffm_fno": {
        "name": "pointcloud_ffm",
        "backbone": "fno",
        "fno_hidden_channels": 8,
    },
    "gl_rbf_cq": {
        "name": "gl_rbf_cq",
        "backbone": "gl_rbf_enh_cq",
        "prior": "iid",
        "hidden_dim": 8,
        "cond_dim": 8,
        "field_embed_dim": 4,
        "latent_dim": 8,
        "num_latents": 2,
        "num_heads": 2,
        "num_latent_blocks": 1,
        "ff_mult": 2,
        "gather_mode": "topk_rbf_glres",
        "gather_topk": 2,
        "neighbor_backend": "torch",
        "USE_FOURIER_PE": True,
        "fourier_pe_num_bands": 2,
        "fourier_pe_max_freq": 4.0,
        "sensor_coord_encoding": "fourier",
        "condition_attention_execution": "cached_kv",
        "sensor_attention_padding_mode": "full",
        "cq_query_dim": 8,
        "cq_readout_mode": "lowrank",
        "cq_readout_rank": 4,
        "cq_readout_heads": 2,
        "cq_fusion_mode": "additive",
        "cq_time_conditioning": "sinusoidal_film",
        "cq_time_embed_dim": 8,
        "cq_measurement_support_mode": "rbf_value_support",
    },
}

DATA_SPEC = DataSpec(
    ("u", "v"),
    ("dimensionless", "dimensionless"),
    2,
    (4, 4),
    mesh_type="structured",
)


def _tiny_batch(device: torch.device, *, physics: bool = False) -> ObservationBatch:
    """Return a dense 4x4 two-field batch with four valid observations."""

    y, x = torch.meshgrid(
        torch.linspace(0.0, 1.0, 4, device=device),
        torch.linspace(0.0, 1.0, 4, device=device),
        indexing="ij",
    )
    query_coords = torch.stack((x, y), dim=-1).reshape(1, 16, 2)
    target = torch.stack((torch.sin(x * 3.14), torch.cos(y * 3.14)), dim=-1).reshape(
        1, 16, 2
    )
    indices = torch.tensor([0, 5, 10, 15], device=device).reshape(1, -1)
    obs_coords = query_coords[:, indices[0]]
    obs_values = target[:, indices[0], :1]
    batch = ObservationBatch(
        obs_coords=obs_coords,
        obs_values=obs_values,
        obs_field_ids=torch.tensor([[0, 1, 0, 1]], device=device),
        obs_valid_mask=torch.ones(1, 4, dtype=torch.bool, device=device),
        query_coords=query_coords,
        query_valid_mask=torch.ones(1, 16, dtype=torch.bool, device=device),
        target_fields=target,
        sample_ids=("synthetic",),
        obs_indices=indices,
        logical_shapes=((4, 4),),
        metadata={"query_indices": torch.arange(16, device=device).reshape(1, -1)},
    )
    if physics:
        batch.metadata["sample_context"] = {
            "physics": {"temporal_derivative": torch.zeros_like(target)},
            "conditions": torch.ones(1, 4, device=device),
        }
    return batch


def _physics_provider() -> Any:
    """Build the tiny Brusselator provider without importing case code globally."""

    if str(CASE_DIR) not in sys.path:
        sys.path.insert(0, str(CASE_DIR))
    from physics import build_physics_provider

    return build_physics_provider(
        {"temporal_derivative_source": "paired_finite_difference"},
        DATA_SPEC,
        FieldNormalizer.identity(DATA_SPEC.num_fields),
    )


def _stage1_checkpoint(path: Path, device: torch.device) -> None:
    stage1 = build_model(MODEL_CONFIGS["latent_fm_stage1"], DATA_SPEC).to(device)
    torch.save(
        {
            "model_name": "latent_fm",
            "model_config": {"stage": 1},
            "model": stage1.state_dict(),
        },
        path,
    )


def _run_one(name: str, config: dict[str, Any], device: torch.device) -> dict[str, Any]:
    provider = _physics_provider() if name == "pinn" else None
    batch = _tiny_batch(device, physics=provider is not None)
    model = build_model(config, DATA_SPEC, provider).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    optimizer.zero_grad(set_to_none=True)
    loss = model.training_loss(batch)
    if not bool(torch.isfinite(loss.total)):
        raise RuntimeError(f"{name}: native loss is not finite")
    loss.total.backward()
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    if not gradients or not all(bool(torch.isfinite(gradient).all()) for gradient in gradients):
        raise RuntimeError(f"{name}: backward did not produce finite gradients")
    optimizer.step()
    reconstruction = model.reconstruct(
        batch,
        steps=1,
        generator=torch.Generator(device=device).manual_seed(991),
    ).prediction
    if reconstruction.shape != batch.target_fields.shape or not bool(
        torch.isfinite(reconstruction).all()
    ):
        raise RuntimeError(f"{name}: reconstruction shape or finiteness check failed")
    return {
        "status": "pass",
        "loss": float(loss.total.detach().cpu()),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "gradient_tensors": len(gradients),
        "output_shape": list(reconstruction.shape),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--models",
        default=",".join((*MODEL_CONFIGS, "latent_fm_stage2", "pinn")),
        help="comma-separated model labels (default: full matrix)",
    )
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print(json.dumps({"device": str(device), "status": "skip", "reason": "CUDA unavailable"}))
        return 0
    requested = [item.strip() for item in args.models.split(",") if item.strip()]
    unknown = sorted(set(requested) - set(MODEL_CONFIGS) - {"latent_fm_stage2", "pinn"})
    if unknown:
        parser.error(f"unknown model labels: {', '.join(unknown)}")
    results: dict[str, dict[str, Any]] = {}
    with tempfile.TemporaryDirectory(prefix="phycoflow_model_smoke_") as directory:
        stage1_path = Path(directory) / "stage1.pt"
        if "latent_fm_stage2" in requested:
            _stage1_checkpoint(stage1_path, device)
        for name in requested:
            config = dict(MODEL_CONFIGS.get(name, {"name": "pinn"}))
            if name == "latent_fm_stage2":
                config = {
                    "name": "latent_fm",
                    "latent_channels": 4,
                    "stage": 2,
                    "stage1_checkpoint": str(stage1_path),
                }
            try:
                results[name] = _run_one(name, config, device)
            except (ImportError, ModuleNotFoundError) as error:
                results[name] = {"status": "skip", "reason": str(error)}
            except Exception as error:  # noqa: BLE001 - report all matrix failures together
                results[name] = {"status": "fail", "reason": str(error)}
    print(json.dumps({"device": str(device), "results": results}, indent=2, sort_keys=True))
    return int(any(result["status"] == "fail" for result in results.values()))


if __name__ == "__main__":
    raise SystemExit(main())

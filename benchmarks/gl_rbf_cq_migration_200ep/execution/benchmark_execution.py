"""Controlled B-vs-C training-step benchmark on one downstream batch."""

# ruff: noqa: I001 -- project sources are added before local imports.

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import statistics
import subprocess
import sys
from pathlib import Path
from time import perf_counter

import torch

PROJECT = Path(__file__).resolve().parents[3]
REPO = PROJECT.parent
CASE_DIR = PROJECT / "Cases" / "turbulent_combustion"
sys.path.insert(0, str(PROJECT / "src"))

from phycoflow_reconstruction.cli import _load_case_config
from phycoflow_reconstruction.data.factory import open_field_dataset
from phycoflow_reconstruction.data.sensor_protocols import build_observation_batch
from phycoflow_reconstruction.models import build_model
from phycoflow_reconstruction.training.common import (
    iter_unique_batch_indices,
    sensor_protocol_from_config,
)
from phycoflow_reconstruction.training.model_lifecycle import (
    after_optimizer_step,
    backward_and_clip_model_loss,
)
from phycoflow_reconstruction.utils.reproducibility import seed_everything


CONFIGS = {
    "B_legacy_mha_full": PROJECT
    / "benchmarks/gl_rbf_cq_migration_200ep/configs/B_gl_rbf_cq_legacy_mha_200ep.yaml",
    "C_cached_kv_full": PROJECT
    / "benchmarks/gl_rbf_cq_migration_200ep/configs/C_gl_rbf_cq_cached_kv_200ep.yaml",
}


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_hash(named_tensors) -> str:
    digest = hashlib.sha256()
    for name, tensor in named_tensors:
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _summary(values: list[float]) -> dict[str, float]:
    return {
        "median_ms": statistics.median(values),
        "mean_ms": statistics.fmean(values),
        "min_ms": min(values),
        "max_ms": max(values),
    }


class _PhaseTimer:
    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.starts: dict[str, float] = {}
        self.totals: dict[str, float] = {}

    def start(self, name: str) -> None:
        torch.cuda.synchronize(self.device)
        self.starts[name] = perf_counter()

    def end(self, name: str) -> None:
        torch.cuda.synchronize(self.device)
        elapsed = (perf_counter() - self.starts.pop(name)) * 1000.0
        self.totals[name] = self.totals.get(name, 0.0) + elapsed


def _one_step(model, optimizer, batch, *, seed: int, device: torch.device) -> dict:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model.model.input_cross_attn.reset_execution_counters()
    condition_times: list[float] = []
    original_prepare = model.model.prepare_condition_context

    def timed_prepare(*args, **kwargs):
        torch.cuda.synchronize(device)
        started = perf_counter()
        result = original_prepare(*args, **kwargs)
        torch.cuda.synchronize(device)
        condition_times.append((perf_counter() - started) * 1000.0)
        return result

    model.model.prepare_condition_context = timed_prepare
    phases = _PhaseTimer(device)
    torch.cuda.synchronize(device)
    whole_started = perf_counter()
    try:
        losses, norm, _, retries = backward_and_clip_model_loss(
            model,
            batch,
            model.parameters(),
            1.0,
            initial_scale=1.0,
            adaptive=False,
            device=device,
            start_phase=phases.start,
            end_phase=phases.end,
        )
        torch.cuda.synchronize(device)
        optimizer_started = perf_counter()
        optimizer.step()
        after_optimizer_step(model)
        torch.cuda.synchronize(device)
        optimizer_ms = (perf_counter() - optimizer_started) * 1000.0
        whole_ms = (perf_counter() - whole_started) * 1000.0
    finally:
        model.model.prepare_condition_context = original_prepare
    return {
        "loss": float(losses.total),
        "gradient_norm": float(norm),
        "backward_retries": int(retries),
        "condition_context_ms": sum(condition_times),
        "forward_ms": phases.totals.get("forward_native_loss", 0.0),
        "backward_ms": phases.totals.get("backward", 0.0),
        "optimizer_ms": optimizer_ms,
        "whole_step_ms": whole_ms,
        "kv_projection_calls": int(model.model.input_cross_attn.kv_projection_calls),
    }


def _benchmark_arm(
    name: str,
    config: dict,
    initial_state: dict,
    batch,
    *,
    device: torch.device,
    warmup: int,
    repetitions: int,
) -> dict:
    seed_everything(42, True)
    model = build_model(config["model"], batch.metadata["data_spec"]).to(device)
    model.load_state_dict(initial_state, strict=True)

    def optimizer_for_model():
        return torch.optim.AdamW(
            model.parameters(),
            lr=float(config["optimization"]["lr"]),
            weight_decay=float(config["optimization"]["weight_decay"]),
        )

    optimizer = optimizer_for_model()
    for index in range(warmup):
        _one_step(model, optimizer, batch, seed=42 + index, device=device)

    # Both arms enter measurement after the same seeded warmup trajectory.
    # Retaining the warmed optimizer removes one-time state allocation from
    # the measured optimizer distribution.
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    rows = [
        _one_step(model, optimizer, batch, seed=42 + warmup + index, device=device)
        for index in range(repetitions)
    ]
    expected_calls = 4 if name.startswith("B_") else 1
    actual_calls = [row["kv_projection_calls"] for row in rows]
    if actual_calls != [expected_calls] * repetitions:
        raise AssertionError(f"{name} K/V calls {actual_calls} != {expected_calls}")
    return {
        "condition_attention_execution": config["model"]["condition_attention_execution"],
        "warmup_steps": warmup,
        "measured_steps": repetitions,
        "expected_kv_projection_calls_per_step": expected_calls,
        "observed_kv_projection_calls_per_step": actual_calls,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "timing": {
            key: _summary([row[key] for row in rows])
            for key in (
                "condition_context_ms",
                "forward_ms",
                "backward_ms",
                "optimizer_ms",
                "whole_step_ms",
            )
        },
        "losses": [row["loss"] for row in rows],
        "gradient_norms": [row["gradient_norm"] for row in rows],
        "backward_retries": [row["backward_retries"] for row in rows],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=15)
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).with_name("B_vs_C_execution.json")
    )
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("this formal execution benchmark requires CUDA")

    configs = {
        name: _load_case_config(path, CASE_DIR, "turbulent_combustion", [])
        for name, path in CONFIGS.items()
    }
    seed_everything(42, True)
    dataset = open_field_dataset(configs["B_legacy_mha_full"]["dataset"])
    indices = next(
        iter_unique_batch_indices(
            len(dataset),
            1,
            40,
            generator=torch.Generator(device="cpu").manual_seed(42 + 17),
        )
    )
    samples = [dataset[index] for index in indices]
    batch = build_observation_batch(
        samples,
        sensor_protocol_from_config(configs["B_legacy_mha_full"]),
        query_points=4096,
    )
    batch.metadata["data_spec"] = dataset.data_spec
    batch = batch.to(device)

    seed_everything(42, True)
    reference = build_model(
        configs["B_legacy_mha_full"]["model"], dataset.data_spec
    )
    initial_state = {key: value.detach().clone() for key, value in reference.state_dict().items()}
    state_hash = _tensor_hash(initial_state.items())
    del reference
    batch_hash = _tensor_hash(
        (name, getattr(batch, name))
        for name in (
            "obs_coords",
            "obs_values",
            "obs_field_ids",
            "obs_valid_mask",
            "query_coords",
            "query_valid_mask",
            "target_fields",
            "obs_indices",
        )
    )
    arms = {
        name: _benchmark_arm(
            name,
            config,
            initial_state,
            batch,
            device=device,
            warmup=args.warmup,
            repetitions=args.repetitions,
        )
        for name, config in configs.items()
    }
    b = arms["B_legacy_mha_full"]
    c = arms["C_cached_kv_full"]
    output = {
        "protocol": {
            "physical_gpu": 0,
            "visible_device": str(device),
            "seed": 42,
            "batch_size": 40,
            "query_points": 4096,
            "query_microbatch_size": 2048,
            "query_microbatches": 2,
            "neighbor_backend": "keops",
            "sensor_fields": ["T"],
            "sensor_count_range": [192, 384],
            "optimizer": "AdamW",
            "gradient_clip": 1.0,
            "same_initial_state": True,
            "same_batch": True,
            "same_rf_seed_schedule": True,
        },
        "trace": {
            "git_head": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
            ).strip(),
            "config_sha256": {name: _file_hash(path) for name, path in CONFIGS.items()},
            "benchmark_script_sha256": _file_hash(Path(__file__)),
            "initial_state_sha256": state_hash,
            "batch_tensors_sha256": batch_hash,
            "batch_indices": indices,
            "sample_ids": list(batch.sample_ids),
            "dataset_path": str(dataset.path),
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device),
            "gpu_uuid": str(torch.cuda.get_device_properties(device).uuid),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "pykeops": importlib.metadata.version("pykeops"),
        },
        "arms": arms,
        "numerical_equivalence": {
            "max_absolute_loss_difference": max(
                abs(left - right) for left, right in zip(b["losses"], c["losses"])
            ),
            "max_absolute_gradient_norm_difference": max(
                abs(left - right)
                for left, right in zip(b["gradient_norms"], c["gradient_norms"])
            ),
        },
        "execution_effect_C_vs_B": {
            key: {
                "median_ratio": c["timing"][key]["median_ms"]
                / b["timing"][key]["median_ms"],
                "median_percent_change": 100.0
                * (c["timing"][key]["median_ms"] / b["timing"][key]["median_ms"] - 1.0),
            }
            for key in b["timing"]
        },
        "memory_effect_C_vs_B": {
            key: {
                "ratio": c[key] / b[key],
                "percent_change": 100.0 * (c[key] / b[key] - 1.0),
                "bytes_change": c[key] - b[key],
            }
            for key in ("peak_cuda_allocated_bytes", "peak_cuda_reserved_bytes")
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    dataset.close()
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

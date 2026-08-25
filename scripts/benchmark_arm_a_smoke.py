"""Measure one synthetic Arm-A B/Q step without touching the dataset or run store."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from time import perf_counter

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from phycoflow_reconstruction.config import load_config
from phycoflow_reconstruction.contracts import DataSpec, ObservationBatch
from phycoflow_reconstruction.models import build_model


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--query-points", type=int, default=4096)
    parser.add_argument("--sensor-count", type=int, default=384)
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    config = load_config(args.config)
    spec = DataSpec(("CH4", "CO", "T", "U_1", "p"), ("unknown",) * 5, 2, (100, 403))
    model = build_model(config["model"], spec).to(device)
    batch_size = int(args.batch_size)
    query_points = int(args.query_points)
    sensor_count = int(args.sensor_count)
    batch = ObservationBatch(
        obs_coords=torch.rand(batch_size, sensor_count, 2, device=device),
        obs_values=torch.rand(batch_size, sensor_count, 1, device=device),
        obs_field_ids=torch.full(
            (batch_size, sensor_count), 2, dtype=torch.long, device=device
        ),
        obs_valid_mask=torch.ones(batch_size, sensor_count, dtype=torch.bool, device=device),
        query_coords=torch.rand(batch_size, query_points, 2, device=device),
        query_valid_mask=torch.ones(batch_size, query_points, dtype=torch.bool, device=device),
        target_fields=torch.rand(batch_size, query_points, 5, device=device),
        sample_ids=tuple(str(index) for index in range(batch_size)),
        obs_indices=torch.arange(sensor_count, device=device).expand(batch_size, -1),
        metadata={"query_indices": torch.arange(query_points, device=device).expand(batch_size, -1)},
    )
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-4, weight_decay=1.0e-6)
    for _ in range(1):
        optimizer.zero_grad(set_to_none=True)
        losses = model.training_loss(batch)
        losses.total.backward()
        optimizer.step()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = perf_counter()
    optimizer.zero_grad(set_to_none=True)
    forward_started = perf_counter()
    losses = model.training_loss(batch)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    forward_seconds = perf_counter() - forward_started
    backward_started = perf_counter()
    losses.total.backward()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    backward_seconds = perf_counter() - backward_started
    optimizer_started = perf_counter()
    optimizer.step()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    optimizer_seconds = perf_counter() - optimizer_started
    total_seconds = perf_counter() - started
    parameters = sum(parameter.numel() for parameter in model.parameters())
    print(f"parameter_count={parameters}")
    print(f"forward_native_loss_seconds={forward_seconds:.6f}")
    print(f"backward_seconds={backward_seconds:.6f}")
    print(f"optimizer_seconds={optimizer_seconds:.6f}")
    print(f"step_seconds={total_seconds:.6f}")
    if device.type == "cuda":
        print(f"peak_cuda_allocated_bytes={torch.cuda.max_memory_allocated(device)}")
        print(f"peak_cuda_reserved_bytes={torch.cuda.max_memory_reserved(device)}")
    print("steps_per_epoch_dataset_dependent=ceil(dataset_train_samples/128)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

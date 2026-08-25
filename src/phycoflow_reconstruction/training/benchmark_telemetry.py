"""Opt-in, model-agnostic timing and memory telemetry for benchmark runs.

The normal training path does not instantiate this helper.  When enabled, it
records a compact per-step trace and per-epoch summary under ``metrics`` in a
run directory.  CUDA timings synchronize once per completed step and use CUDA
events for the model phases, so asynchronous kernel launches are not reported
as CPU enqueue time.
"""

from __future__ import annotations

import csv
import json
import os
from collections.abc import Mapping
from pathlib import Path
from time import perf_counter
from typing import Any

import torch


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


class BenchmarkTelemetry:
    """Collect optional per-step and per-epoch training performance evidence."""

    def __init__(
        self,
        run_dir: str | Path,
        *,
        enabled: bool,
        device: torch.device,
        steps_per_epoch: int,
        parameter_count: int,
        trainable_parameter_count: int,
        sample_steps: int = 0,
    ) -> None:
        self.enabled = bool(enabled)
        self.run_dir = Path(run_dir)
        self.metrics_dir = self.run_dir / "metrics"
        self.device = device
        self.steps_per_epoch = max(1, int(steps_per_epoch))
        self.parameter_count = int(parameter_count)
        self.trainable_parameter_count = int(trainable_parameter_count)
        self.sample_steps = max(0, int(sample_steps))
        self.steps: list[dict[str, Any]] = []
        self.epochs: list[dict[str, Any]] = []
        self._step: dict[str, Any] | None = None
        self._epoch: dict[str, Any] | None = None
        self._last_step_end: float | None = None
        self._cuda_events: dict[str, tuple[torch.cuda.Event, torch.cuda.Event]] = {}

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        run_dir: str | Path,
        *,
        device: torch.device,
        steps_per_epoch: int,
        parameter_count: int,
        trainable_parameter_count: int,
    ) -> BenchmarkTelemetry:
        settings = config.get("benchmark_telemetry", {})
        if not isinstance(settings, Mapping):
            raise TypeError("benchmark_telemetry must be a mapping")
        return cls(
            run_dir,
            enabled=bool(settings.get("enabled", False)),
            device=device,
            steps_per_epoch=steps_per_epoch,
            parameter_count=parameter_count,
            trainable_parameter_count=trainable_parameter_count,
            sample_steps=int(settings.get("sample_steps", 0)),
        )

    def _reset_peak_memory(self) -> None:
        if self.enabled and self.device.type == "cuda":
            _synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)

    def start_epoch(self, epoch: int) -> None:
        if not self.enabled:
            return
        self._reset_peak_memory()
        self._epoch = {
            "epoch": int(epoch),
            "started": perf_counter(),
            "step_indices": [],
        }

    def start_step(self, global_step: int) -> None:
        if not self.enabled:
            return
        now = perf_counter()
        if self._epoch is None:
            self.start_epoch((int(global_step) // self.steps_per_epoch) + 1)
        data_wait = 0.0 if self._last_step_end is None else max(0.0, now - self._last_step_end)
        self._step = {
            "step": int(global_step) + 1,
            "global_step": int(global_step),
            "started": now,
            "data_wait_time_s": data_wait,
            "phases": {},
            "sampled": self.sample_steps == 0
            or (int(global_step) % self.steps_per_epoch) < self.sample_steps,
        }
        if self.device.type == "cuda" and self._step["sampled"]:
            self._step["total_start"] = torch.cuda.Event(enable_timing=True)
            self._step["total_end"] = torch.cuda.Event(enable_timing=True)
            with torch.cuda.device(self.device):
                self._step["total_start"].record()

    def start_phase(self, name: str) -> None:
        if not self.enabled or self._step is None:
            return
        if not self._step.get("sampled", False):
            return
        if self.device.type == "cuda" and self._step.get("sampled", False):
            start = torch.cuda.Event(enable_timing=True)
            with torch.cuda.device(self.device):
                start.record()
            self._cuda_events[str(name)] = (start, torch.cuda.Event(enable_timing=True))
        else:
            phase = self._step["phases"].setdefault(str(name), {"seconds": 0.0})
            phase["started"] = perf_counter()

    def end_phase(self, name: str) -> None:
        if not self.enabled or self._step is None:
            return
        if not self._step.get("sampled", False):
            return
        key = str(name)
        if self.device.type == "cuda" and self._step.get("sampled", False):
            events = self._cuda_events.pop(key, None)
            if events is None:
                raise RuntimeError(f"benchmark telemetry phase {key!r} was not started")
            with torch.cuda.device(self.device):
                events[1].record()
            phase = self._step["phases"].setdefault(key, {"events": []})
            phase["events"].append(events)
        else:
            phase = self._step["phases"].get(key)
            if phase is None:
                raise RuntimeError(f"benchmark telemetry phase {key!r} was not started")
            phase["seconds"] += max(0.0, perf_counter() - phase.pop("started"))

    def finish_step(self) -> None:
        if not self.enabled or self._step is None:
            return
        if self._cuda_events:
            raise RuntimeError("benchmark telemetry has unfinished phases")
        step = self._step
        if self.device.type == "cuda" and step.get("sampled", False):
            with torch.cuda.device(self.device):
                step["total_end"].record()
            _synchronize(self.device)
            total_ms = float(step["total_start"].elapsed_time(step["total_end"]))
            phase_seconds = {
                name: sum(
                    float(events[0].elapsed_time(events[1])) / 1000.0
                    for events in payload["events"]
                )
                for name, payload in step["phases"].items()
            }
        else:
            total_ms = max(0.0, (perf_counter() - float(step["started"])) * 1000.0)
            phase_seconds = {
                name: float(payload["seconds"])
                for name, payload in step["phases"].items()
            }
        row = {
            "step": step["step"],
            "global_step": step["global_step"],
            "data_wait_time_s": float(step["data_wait_time_s"]),
            "step_time_s": total_ms / 1000.0,
            "forward_native_loss_time_s": phase_seconds.get("forward_native_loss", 0.0),
            "backward_time_s": phase_seconds.get("backward", 0.0),
            "optimizer_time_s": phase_seconds.get("optimizer", 0.0),
            "sampled": bool(step["sampled"]),
        }
        self.steps.append(row)
        if self._epoch is None:
            self.start_epoch((int(step["global_step"]) // self.steps_per_epoch) + 1)
        self._epoch["step_indices"].append(len(self.steps) - 1)
        self._last_step_end = perf_counter()
        self._step = None

    @staticmethod
    def _mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    @staticmethod
    def _median(values: list[float]) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        middle = len(ordered) // 2
        if len(ordered) % 2:
            return ordered[middle]
        return (ordered[middle - 1] + ordered[middle]) / 2.0

    def finish_epoch(self, epoch: int | None = None) -> None:
        if not self.enabled or self._epoch is None:
            return
        indices = list(self._epoch["step_indices"])
        rows = [self.steps[index] for index in indices]
        if self.device.type == "cuda":
            _synchronize(self.device)
        epoch_wall = max(0.0, perf_counter() - float(self._epoch["started"]))
        data_wait = sum(float(row["data_wait_time_s"]) for row in rows)
        training_only = max(0.0, epoch_wall - data_wait)
        sampled_rows = [row for row in rows if row["sampled"]]
        step_times = [float(row["step_time_s"]) for row in sampled_rows]
        if self.device.type == "cuda":
            peak_allocated = int(torch.cuda.max_memory_allocated(self.device))
            peak_reserved = int(torch.cuda.max_memory_reserved(self.device))
        else:
            peak_allocated = 0
            peak_reserved = 0
        summary = {
            "epoch": int(epoch if epoch is not None else self._epoch["epoch"]),
            "steps": len(rows),
            "steps_per_epoch": self.steps_per_epoch,
            "epoch_wall_time_s": epoch_wall,
            "training_only_epoch_time_s": training_only,
            "mean_step_time_s": self._mean(step_times),
            "median_step_time_s": self._median(step_times),
            "sampled_step_count": sum(1 for row in rows if row["sampled"]),
            "mean_forward_native_loss_time_s": self._mean(
                [float(row["forward_native_loss_time_s"]) for row in sampled_rows]
            ),
            "mean_backward_time_s": self._mean(
                [float(row["backward_time_s"]) for row in sampled_rows]
            ),
            "mean_optimizer_time_s": self._mean(
                [float(row["optimizer_time_s"]) for row in sampled_rows]
            ),
            "mean_data_wait_time_s": self._mean(
                [float(row["data_wait_time_s"]) for row in rows]
            ),
            "peak_cuda_allocated_bytes": peak_allocated,
            "peak_cuda_reserved_bytes": peak_reserved,
            "parameter_count": self.parameter_count,
            "trainable_parameter_count": self.trainable_parameter_count,
        }
        self.epochs.append(summary)
        self._epoch = None
        self._last_step_end = None
        self.write()

    def write(self) -> None:
        if not self.enabled:
            return
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": "1",
            "device": str(self.device),
            "steps_per_epoch": self.steps_per_epoch,
            "parameter_count": self.parameter_count,
            "trainable_parameter_count": self.trainable_parameter_count,
            "sample_steps": self.sample_steps,
            "steps": self.steps,
            "epochs": self.epochs,
        }
        json_path = self.metrics_dir / "benchmark_telemetry.json"
        temporary = json_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, json_path)

        csv_path = self.metrics_dir / "benchmark_telemetry_epochs.csv"
        fieldnames = list(self.epochs[0]) if self.epochs else []
        if fieldnames:
            csv_temporary = csv_path.with_suffix(".csv.tmp")
            with csv_temporary.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.epochs)
            os.replace(csv_temporary, csv_path)

    def close(self) -> None:
        if not self.enabled:
            return
        if self._step is not None:
            self.finish_step()
        if self._epoch is not None:
            self.finish_epoch()
        self.write()

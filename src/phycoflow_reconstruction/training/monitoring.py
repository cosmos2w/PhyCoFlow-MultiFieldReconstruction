"""Terminal progress and live loss figures shared by every training stage."""

from __future__ import annotations

import json
import math
import os
import warnings
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from time import perf_counter
from typing import Any

from tqdm.auto import tqdm

_LOSS_KEYS = ("total", "data_loss", "coherence_loss", "physics_loss")


def _format_duration(seconds: float) -> str:
    """Format long epoch estimates without expanding the progress line excessively."""
    total_seconds = max(0, round(seconds))
    days, remainder = divmod(total_seconds, 86_400)
    hours, remainder = divmod(remainder, 3_600)
    minutes, seconds = divmod(remainder, 60)
    if days:
        return f"{days}d{hours:02}h{minutes:02}m"
    if hours:
        return f"{hours}h{minutes:02}m{seconds:02}s"
    if minutes:
        return f"{minutes}m{seconds:02}s"
    return f"{seconds}s"


class TrainingMonitor:
    """Report optimizer progress and refresh a compact diagnostic loss plot."""

    def __init__(
        self,
        run_dir: str | Path,
        *,
        start_step: int,
        final_step: int,
        configured_steps: int,
        steps_per_epoch: int,
        description: str,
        enabled: bool = True,
        plot_every_steps: int = 10,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.history_path = self.run_dir / "metrics" / "history.jsonl"
        self.plot_path = self.run_dir / "loss_history.png"
        self.final_step = int(final_step)
        self.configured_steps = int(configured_steps)
        self.steps_per_epoch = max(1, int(steps_per_epoch))
        self.total_epochs = max(1, math.ceil(self.configured_steps / self.steps_per_epoch))
        self.plot_every_steps = max(1, int(plot_every_steps))
        self.description = str(description)
        self.enabled = bool(enabled)
        self._steps: dict[str, list[int]] = defaultdict(list)
        self._values: dict[str, list[float]] = defaultdict(list)
        self._plot_available = True
        self._load_existing_history()
        self.active_epoch = int(start_step) // self.steps_per_epoch + 1
        self._epoch_started = perf_counter()
        self._epoch_observed_batches = 0
        self.progress = self._new_epoch_bar(
            self.active_epoch,
            initial=int(start_step) % self.steps_per_epoch,
        )
        if enabled:
            tqdm.write(f"Run directory: {self.run_dir}")
            tqdm.write(f"Live loss figure: {self.plot_path}")

    def _epoch_batch_count(self, epoch: int) -> int:
        epoch_start = (epoch - 1) * self.steps_per_epoch
        return min(self.steps_per_epoch, self.configured_steps - epoch_start)

    def _new_epoch_bar(self, epoch: int, *, initial: int):
        self._epoch_started = perf_counter()
        self._epoch_observed_batches = 0
        return tqdm(
            total=self._epoch_batch_count(epoch),
            initial=initial,
            desc=f"{self.description} epoch {epoch}/{self.total_epochs}",
            unit="batch",
            dynamic_ncols=True,
            disable=not self.enabled,
        )

    def _load_existing_history(self) -> None:
        if not self.history_path.exists():
            return
        with self.history_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    self._capture(json.loads(line))

    def _capture(self, row: Mapping[str, Any]) -> None:
        step = int(row.get("step", 0))
        for key in _LOSS_KEYS:
            value = row.get(key)
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                self._steps[key].append(step)
                self._values[key].append(float(value))

    def _epoch_coordinates(self, steps: list[int]) -> list[float]:
        """Express completed optimizer batches as fractional training epochs."""
        return [step / self.steps_per_epoch for step in steps]

    def record(self, row: Mapping[str, Any], *, lr: float | None = None) -> None:
        """Record one already-persisted history row and update live diagnostics."""
        self._capture(row)
        step = int(row["step"])
        epoch = (max(step, 1) - 1) // self.steps_per_epoch + 1
        if epoch != self.active_epoch:
            self.progress.close()
            self.active_epoch = epoch
            self.progress = self._new_epoch_bar(epoch, initial=0)
        batch_in_epoch = (step - 1) % self.steps_per_epoch + 1
        increment = max(0, batch_in_epoch - self.progress.n)
        self._epoch_observed_batches += increment
        elapsed = perf_counter() - self._epoch_started
        epoch_estimate = (
            elapsed * self._epoch_batch_count(epoch) / self._epoch_observed_batches
            if self._epoch_observed_batches
            else 0.0
        )
        primary_key = next((key for key in _LOSS_KEYS if key in row), None)
        postfix: dict[str, str | int] = {
            "epochs_left": max(0, self.total_epochs - epoch),
            "epoch_est": _format_duration(epoch_estimate),
        }
        if primary_key is not None:
            postfix[primary_key] = f"{float(row[primary_key]):.4e}"
        if lr is not None:
            postfix["lr"] = f"{float(lr):.3e}"
        self.progress.set_postfix(postfix, refresh=False)
        self.progress.update(increment)
        if step == 1 or step % self.plot_every_steps == 0 or step == self.final_step:
            self._plot()
        if batch_in_epoch == self._epoch_batch_count(epoch) and step < self.final_step:
            self.progress.close()
            self.active_epoch = epoch + 1
            self.progress = self._new_epoch_bar(self.active_epoch, initial=0)

    def _plot(self) -> None:
        if not self._plot_available or not any(self._values.values()):
            return
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            warnings.warn(
                "matplotlib is unavailable; terminal progress and JSONL history remain enabled, "
                "but loss_history.png cannot be generated. Install the project plot extra.",
                stacklevel=2,
            )
            self._plot_available = False
            return

        plt.rcParams["svg.fonttype"] = "none"
        figure, axis = plt.subplots(figsize=(9, 5.25))
        plotted_values: list[float] = []
        for key in _LOSS_KEYS:
            values = self._values.get(key, [])
            if not values:
                continue
            steps = self._steps[key]
            stride = max(1, math.ceil(len(values) / 4000))
            shown_epochs = self._epoch_coordinates(steps[::stride])
            shown_values = values[::stride]
            plotted_values.extend(shown_values)
            axis.plot(shown_epochs, shown_values, linewidth=1.5, label=key)
        axis.set_xlabel("Training epoch")
        axis.set_xlim(left=0)
        axis.set_ylabel("Loss")
        axis.set_title(f"{self.description} loss history")
        if plotted_values and all(value > 0.0 for value in plotted_values):
            axis.set_yscale("log")
        axis.grid(True, which="both", linestyle="--", alpha=0.35)
        axis.legend(frameon=False)
        figure.tight_layout()
        temporary = self.plot_path.with_name(f".{self.plot_path.name}.tmp")
        figure.savefig(temporary, dpi=150, format="png")
        plt.close(figure)
        os.replace(temporary, self.plot_path)

    def close(self) -> None:
        """Write the latest figure and leave a completed terminal progress line."""
        self._plot()
        self.progress.close()

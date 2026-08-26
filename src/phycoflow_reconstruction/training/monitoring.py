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
_LOSS_LABELS = {
    "total": "total",
    "data_loss": "data",
    "coherence_loss": "coherence",
    "physics_loss": "physics",
}


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
    """Report batch progress and persist compact epoch-level loss diagnostics."""

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
        # Keep the established config key for compatibility, but interpret its
        # value as an epoch interval.  History and plots should scale with the
        # number of epochs, not with potentially millions of optimizer steps.
        self.plot_every_epochs = max(1, int(plot_every_steps))
        self.description = str(description)
        self.enabled = bool(enabled)
        self._steps: dict[str, list[int]] = defaultdict(list)
        self._values: dict[str, list[float]] = defaultdict(list)
        self._epoch_sums: dict[str, float] = defaultdict(float)
        self._epoch_counts: dict[str, int] = defaultdict(int)
        self._epoch_latest: dict[str, Any] = {}
        self._epoch_batch_count_seen = 0
        self._last_step: int | None = None
        self._last_epoch: int | None = None
        self._pending_epoch_report: dict[str, Any] | None = None
        self.last_epoch_report: dict[str, Any] | None = None
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

    def _accumulate_epoch(self, row: Mapping[str, Any]) -> None:
        self._epoch_batch_count_seen += 1
        for key, value in row.items():
            if key == "step":
                continue
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                self._epoch_sums[key] += float(value)
                self._epoch_counts[key] += 1
            else:
                self._epoch_latest[key] = value

    def _flush_epoch(self, *, epoch: int, step: int) -> dict[str, Any]:
        batches = self._epoch_batch_count_seen
        if batches < 1:
            raise RuntimeError("cannot flush an empty training-history epoch")
        row = {
            "step": int(step),
            "epoch": int(epoch),
            "batches": int(batches),
            "epoch_complete": batches == self._epoch_batch_count(epoch),
            **{
                key: value / self._epoch_counts[key]
                for key, value in self._epoch_sums.items()
            },
            **self._epoch_latest,
        }
        with self.history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        self._capture(row)
        self._epoch_sums.clear()
        self._epoch_counts.clear()
        self._epoch_latest.clear()
        self._epoch_batch_count_seen = 0
        return row

    def _live_postfix(
        self,
        row: Mapping[str, Any],
        *,
        epoch: int,
        epoch_estimate: float,
        lr: float | None,
    ) -> dict[str, str | int]:
        postfix: dict[str, str | int] = {
            "epochs_left": max(0, self.total_epochs - epoch),
            "epoch_est": _format_duration(epoch_estimate),
        }
        for key in _LOSS_KEYS:
            value = row.get(key)
            if isinstance(value, (int, float)):
                postfix[_LOSS_LABELS[key]] = f"{float(value):.4e}"
        if lr is not None:
            postfix["lr"] = f"{float(lr):.3e}"
        return postfix

    def _complete_pending_epoch(self, *, start_next: bool = True) -> None:
        pending = self._pending_epoch_report
        if pending is None:
            return
        wall_seconds = perf_counter() - float(pending["started_at"])
        train_seconds = float(pending["train_seconds"])
        batches = int(pending["batches"])
        checkpoint_checked = bool(pending["checkpoint_checked"])
        best_saved = bool(pending["best_checkpoint_saved"])
        best_status = "saved" if best_saved else "unchanged" if checkpoint_checked else "not_checked"
        summary = dict(pending["summary"])
        report = {
            "epoch": int(pending["epoch"]),
            "step": int(pending["step"]),
            "train_seconds": train_seconds,
            "wall_seconds": wall_seconds,
            "batches_per_second": batches / train_seconds if train_seconds > 0.0 else 0.0,
            "best": best_status,
            **{
                key: float(summary[key])
                for key in _LOSS_KEYS
                if isinstance(summary.get(key), (int, float))
            },
        }
        postfix: dict[str, str] = {
            f"{_LOSS_LABELS[key]}_avg": f"{report[key]:.4e}"
            for key in _LOSS_KEYS
            if key in report
        }
        postfix.update(
            {
                "train": _format_duration(train_seconds),
                "wall": _format_duration(wall_seconds),
                "rate": f"{report['batches_per_second']:.2f}batch/s",
                "best": best_status,
            }
        )
        self.progress.set_postfix(postfix, refresh=True)
        self.progress.close()
        self.last_epoch_report = report
        self._pending_epoch_report = None

        if start_next and int(pending["step"]) < self.final_step:
            self.active_epoch = int(pending["epoch"]) + 1
            self.progress = self._new_epoch_bar(self.active_epoch, initial=0)

    def record(self, row: Mapping[str, Any], *, lr: float | None = None) -> None:
        """Update batch progress and persist one mean row at each epoch boundary."""
        # Be defensive for callers outside the built-in trainers: an epoch
        # report that was not explicitly finished is emitted as not checked
        # before the next batch starts.
        self._complete_pending_epoch()
        step = int(row["step"])
        epoch = (max(step, 1) - 1) // self.steps_per_epoch + 1
        self._last_step = step
        self._last_epoch = epoch
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
        self.progress.set_postfix(
            self._live_postfix(row, epoch=epoch, epoch_estimate=epoch_estimate, lr=lr),
            refresh=False,
        )
        self.progress.update(increment)
        self._accumulate_epoch(row)
        epoch_finished = batch_in_epoch == self._epoch_batch_count(epoch)
        if epoch_finished or step == self.final_step:
            train_seconds = perf_counter() - self._epoch_started
            summary = self._flush_epoch(epoch=epoch, step=step)
            if epoch == 1 or epoch % self.plot_every_epochs == 0 or step == self.final_step:
                self._plot()
            self._pending_epoch_report = {
                "epoch": epoch,
                "step": step,
                "batches": self._epoch_observed_batches,
                "started_at": self._epoch_started,
                "train_seconds": train_seconds,
                "summary": summary,
                "checkpoint_checked": False,
                "best_checkpoint_saved": False,
            }

    def finish_step(
        self,
        *,
        checkpoint_checked: bool = False,
        best_checkpoint_saved: bool = False,
    ) -> None:
        """Finalize an epoch line after its optional validation/checkpoint work."""
        if self._pending_epoch_report is None:
            return
        self._pending_epoch_report["checkpoint_checked"] |= bool(checkpoint_checked)
        self._pending_epoch_report["best_checkpoint_saved"] |= bool(best_checkpoint_saved)
        if int(self._pending_epoch_report["step"]) < self.final_step:
            self._complete_pending_epoch()

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

    def close(
        self,
        *,
        checkpoint_checked: bool = False,
        best_checkpoint_saved: bool = False,
    ) -> None:
        """Write the latest figure and leave a completed terminal progress line."""
        if self._epoch_batch_count_seen:
            assert self._last_epoch is not None and self._last_step is not None
            summary = self._flush_epoch(epoch=self._last_epoch, step=self._last_step)
            self._pending_epoch_report = {
                "epoch": self._last_epoch,
                "step": self._last_step,
                "batches": self._epoch_observed_batches,
                "started_at": self._epoch_started,
                "train_seconds": perf_counter() - self._epoch_started,
                "summary": summary,
                "checkpoint_checked": False,
                "best_checkpoint_saved": False,
            }
        if self._pending_epoch_report is not None:
            self._pending_epoch_report["checkpoint_checked"] |= bool(checkpoint_checked)
            self._pending_epoch_report["best_checkpoint_saved"] |= bool(best_checkpoint_saved)
        self._plot()
        if self._pending_epoch_report is not None:
            self._complete_pending_epoch(start_next=False)
        else:
            self.progress.close()

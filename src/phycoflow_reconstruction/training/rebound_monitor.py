"""Fixed-probe rebound monitoring with no optimizer or model intervention."""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from typing import Any


class ReboundMonitor:
    """Request one sparse diagnostic after persistent erasure of useful gain.

    Inputs are the declared fixed-panel equal-family SOURCE-normalized score.
    A trailing 20-epoch window means (epoch - 20, epoch], averaged over its
    scheduled 10-epoch observations. Every unsmoothed observation is retained.
    Missing scheduled events reset the two-event persistence counter.
    """

    version = "fixed_panel_rebound_monitor_v1"

    def __init__(self, *, source_denominators: Mapping[str, float],
                 source_score: float = 1.0) -> None:
        denominators = {str(name): float(value) for name, value in source_denominators.items()}
        if not denominators or any(not math.isfinite(value) or value <= 0
                                   for value in denominators.values()):
            raise ValueError("rebound monitor requires positive finite SOURCE denominators")
        if not math.isfinite(source_score) or source_score <= 0:
            raise ValueError("rebound monitor requires a positive finite SOURCE score")
        self.identity = {
            "version": self.version,
            "metric_name": "validation/coherence_selection_score",
            "scope": "fixed_validation_panel",
            "source_denominators": denominators,
            "source_score": float(source_score),
            "cadence_epochs": 10, "window_epochs": 20,
            "window_convention": "epoch_minus_20_exclusive_to_epoch_inclusive",
            "aggregation": "arithmetic_mean_of_scheduled_unsmoothed_observations",
            "useful_gain_fraction": .05, "source_return_ratio": .98,
            "erased_gain_fraction": .80, "persistent_events": 2,
            "early_gain_reference": "best_trailing_window_mean_observed_so_far",
            "action": "request_sparse_train_diagnostic_only",
        }
        self.observations: list[dict[str, Any]] = []
        self.best_rolling_score = float(source_score)
        self.consecutive_rebound_events = 0
        self.episode_triggered = False
        self.pending_diagnostic = False

    def observe(self, epoch: int, score: float) -> dict[str, Any]:
        """Record a scheduled panel score and return an auditable alarm record."""
        if isinstance(epoch, bool) or int(epoch) != epoch or epoch < 0 or epoch % 10:
            raise ValueError("rebound observations require nonnegative 10-epoch coordinates")
        epoch, score = int(epoch), float(score)
        if not math.isfinite(score) or score < 0:
            raise ValueError("rebound score must be finite and nonnegative")
        previous_epoch = self.observations[-1]["epoch"] if self.observations else None
        if previous_epoch is not None and epoch <= previous_epoch:
            raise ValueError("rebound observations must advance in epochs")
        points = [row["score"] for row in self.observations
                  if epoch - 20 < row["epoch"] <= epoch] + [score]
        rolling = sum(points) / len(points)
        self.best_rolling_score = min(self.best_rolling_score, rolling)
        source = self.identity["source_score"]
        gain = source - self.best_rolling_score
        useful = gain >= source * .05 - 1e-12 * source
        returned = rolling >= .98 * source
        erased = useful and rolling >= source - .2 * gain
        rebound = useful and (returned or erased)
        contiguous = previous_epoch is not None and epoch - previous_epoch == 10
        if rebound:
            self.consecutive_rebound_events = (
                self.consecutive_rebound_events + 1 if contiguous else 1)
        else:
            self.consecutive_rebound_events = 0
            self.episode_triggered = False
        triggered = (self.consecutive_rebound_events >= 2 and not self.episode_triggered)
        if triggered:
            self.pending_diagnostic = True
            self.episode_triggered = True
        record = {"epoch": epoch, "score": score,
            "rolling_score": rolling, "rolling_observation_count": len(points),
            "best_rolling_score": self.best_rolling_score,
            "useful_gain_observed": useful, "returned_to_source_corridor": returned,
            "early_gain_erased": erased, "rebound_event": rebound,
            "consecutive_rebound_events": self.consecutive_rebound_events,
            "diagnostic_triggered": triggered,
            "pending_diagnostic": self.pending_diagnostic}
        self.observations.append(record)
        return dict(record)

    def consume_diagnostic_request(self) -> bool:
        """Acknowledge the request; training decides when to sample its diagnostic."""
        pending = self.pending_diagnostic
        self.pending_diagnostic = False
        return pending

    def state_dict(self) -> dict[str, Any]:
        return copy.deepcopy({"identity": self.identity, "observations": self.observations,
            "best_rolling_score": self.best_rolling_score,
            "consecutive_rebound_events": self.consecutive_rebound_events,
            "episode_triggered": self.episode_triggered,
            "pending_diagnostic": self.pending_diagnostic})

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore monitoring only; changing fixed score definitions must fail."""
        if state.get("identity") != self.identity:
            raise ValueError("rebound monitor identity changed on resume")
        # Replaying the immutable observations validates schedule and criteria.
        replay = ReboundMonitor(source_denominators=self.identity["source_denominators"],
                                source_score=self.identity["source_score"])
        observations = state.get("observations", [])
        for row in observations:
            replay.observe(row["epoch"], row["score"])
        for key in ("best_rolling_score", "consecutive_rebound_events", "episode_triggered"):
            if state.get(key) != getattr(replay, key):
                raise ValueError(f"invalid rebound monitor recovery {key}")
        if not isinstance(state.get("pending_diagnostic"), bool):
            raise ValueError("invalid rebound pending diagnostic recovery flag")
        self.observations = copy.deepcopy(observations)
        self.best_rolling_score = replay.best_rolling_score
        self.consecutive_rebound_events = replay.consecutive_rebound_events
        self.episode_triggered = replay.episode_triggered
        self.pending_diagnostic = state["pending_diagnostic"]

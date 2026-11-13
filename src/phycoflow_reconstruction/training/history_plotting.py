"""Shared visual language for restart-safe training-history figures."""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence

HISTORY_TEXT_COLOR = "#202733"
HISTORY_MUTED_TEXT_COLOR = "#667085"
HISTORY_GRID_COLOR = "#D9DEE7"
HISTORY_SPINE_COLOR = "#7A8493"
HISTORY_FAMILY_COLORS = (
    "#3B6EA8",
    "#D95F59",
    "#B58900",
    "#5B8E7D",
    "#8C6BB1",
)
HISTORY_FAMILY_LINESTYLES = ("--", "-.", ":")
_NAMED_FAMILY_STYLES = {
    "global_distribution": ("#D95F59", "--"),
    "cross_spectrum": ("#3B6EA8", "-."),
    "topology": ("#B58900", ":"),
    "multiscale_spectrum": ("#356b9a", "-"),
}


def history_family_style(family: str, index: int = 0) -> tuple[str, str]:
    """Return a stable family color and line pattern across training figures."""
    if family in _NAMED_FAMILY_STYLES:
        return _NAMED_FAMILY_STYLES[family]
    return (
        HISTORY_FAMILY_COLORS[index % len(HISTORY_FAMILY_COLORS)],
        HISTORY_FAMILY_LINESTYLES[index % len(HISTORY_FAMILY_LINESTYLES)],
    )


def style_history_axis(axis, values: Sequence[float], *, x_max: float | None = None) -> None:
    """Apply shared history styling, scale, and an epoch range starting at zero."""
    finite = [float(value) for value in values if math.isfinite(float(value))]
    right = max(1.0, float(x_max)) if x_max is not None else None
    axis.set_xlim(0, right + max(0.5, right * 0.015)) if right is not None else axis.set_xlim(
        left=0
    )
    nonzero = [abs(value) for value in finite if value != 0.0]
    wide = bool(nonzero) and max(nonzero) / min(nonzero) >= 100.0
    if finite and all(value > 0.0 for value in finite) and wide:
        axis.set_yscale("log")
    elif finite and min(finite) < 0.0 < max(finite) and wide:
        axis.set_yscale("symlog", linthresh=max(min(nonzero), max(nonzero) / 1000.0))
    elif finite:
        axis.set_yscale("linear")
        lo, hi = min(finite), max(finite)
        padding = max((hi - lo) * 0.08, abs(lo) * 0.015, abs(hi) * 0.015, 1e-12)
        axis.set_ylim(lo - padding, hi + padding)
        from matplotlib.ticker import MaxNLocator, ScalarFormatter

        axis.yaxis.set_major_locator(MaxNLocator(nbins=5))
        formatter = ScalarFormatter(useOffset=False, useMathText=True)
        formatter.set_powerlimits((-3, 4))
        axis.yaxis.set_major_formatter(formatter)
    axis.set_axisbelow(True)
    axis.grid(
        True,
        axis="y",
        which="major",
        color=HISTORY_GRID_COLOR,
        linewidth=0.65,
        linestyle="--",
        alpha=0.58,
    )
    axis.grid(
        True,
        axis="y",
        which="minor",
        color=HISTORY_GRID_COLOR,
        linewidth=0.4,
        linestyle=":",
        alpha=0.34,
    )
    axis.tick_params(axis="both", colors=HISTORY_TEXT_COLOR, labelsize=8.8, pad=3)
    for name, spine in axis.spines.items():
        spine.set_color(HISTORY_SPINE_COLOR)
        spine.set_linewidth(0.8)
        if name in {"top", "right"}:
            spine.set_visible(False)


def plot_recorded_trend(axis, epochs, values, *, color, label=None):
    """Display every recorded value and a labeled centered median when useful."""
    axis.plot(
        epochs,
        values,
        color=color,
        linewidth=0.8,
        alpha=0.40 if len(values) >= 8 else 0.9,
        marker="o" if len(values) <= 12 else None,
        markersize=3,
        label=label or "Recorded",
    )
    if len(values) >= 8:
        window = max(3, round(len(values) * 0.02))
        if window % 2 == 0:
            window += 1
        radius = window // 2
        trend = [
            statistics.median(values[max(0, i - radius) : i + radius + 1])
            for i in range(len(values))
        ]
        axis.plot(
            epochs,
            trend,
            color=color,
            linewidth=1.6,
            label=f"Centered median ({window} observations)",
        )


def mark_calibration_freeze(axis, freeze_epoch, *, x_max):
    """Mark a recorded or configured calibration transition within the plotted range."""
    if freeze_epoch is not None and x_max >= freeze_epoch:
        axis.axvline(freeze_epoch, color="#7a8493", linestyle=":", linewidth=0.9, alpha=0.8)


def find_calibration_freeze(rows, config=None):
    """Return the first recorded calibration freeze, or the declared epoch-10 schedule."""
    for row in rows:
        if not isinstance(row, dict):
            continue
        frozen = row.get("calibration_frozen") is True or any(
            isinstance(value, dict) and value.get("calibration_frozen") is True
            for value in row.values()
        )
        if frozen:
            epoch = row.get("epoch")
            if (
                isinstance(epoch, (int, float))
                and not isinstance(epoch, bool)
                and math.isfinite(epoch)
            ):
                return float(epoch)
    if (
        (config or {})
        .get("coherence", {})
        .get("scalable_anchored_coherence", {})
        .get("enabled", False)
    ):
        return 10.0
    return None

"""Shared visual language for restart-safe training-history figures."""

from __future__ import annotations

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


def style_history_axis(axis, values: Sequence[float]) -> None:
    """Apply the common loss-history axis style and an honest adaptive scale."""
    finite = [float(value) for value in values]
    axis.set_xlim(left=0)
    if finite and all(value > 0.0 for value in finite):
        axis.set_yscale("log")
    elif finite and min(finite) < 0.0 < max(finite):
        nonzero = sorted(abs(value) for value in finite if value != 0.0)
        linear_threshold = nonzero[max(0, len(nonzero) // 10 - 1)] if nonzero else 1.0e-12
        axis.set_yscale("symlog", linthresh=max(linear_threshold, 1.0e-12))
    axis.set_axisbelow(True)
    axis.grid(
        True,
        which="major",
        color=HISTORY_GRID_COLOR,
        linewidth=0.75,
        linestyle="--",
        alpha=0.8,
    )
    axis.grid(
        True,
        which="minor",
        color=HISTORY_GRID_COLOR,
        linewidth=0.45,
        linestyle=":",
        alpha=0.55,
    )
    axis.tick_params(axis="both", colors=HISTORY_TEXT_COLOR, labelsize=9.5)
    for spine in axis.spines.values():
        spine.set_color(HISTORY_SPINE_COLOR)
        spine.set_linewidth(0.8)

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
_NAMED_FAMILY_STYLES = {
    "global_distribution": ("#D95F59", "--"),
    "cross_spectrum": ("#3B6EA8", "-."),
    "topology": ("#B58900", ":"),
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
    finite = [float(value) for value in values]
    right = max(1.0, float(x_max)) if x_max is not None else None
    if right is None:
        axis.set_xlim(left=0)
    else:
        axis.set_xlim(0, right + max(0.5, right * 0.015))
    if finite and all(value > 0.0 for value in finite):
        axis.set_yscale("log")
    elif finite and min(finite) < 0.0 < max(finite):
        nonzero = sorted(abs(value) for value in finite if value != 0.0)
        linear_threshold = nonzero[max(0, len(nonzero) // 10 - 1)] if nonzero else 1.0e-12
        axis.set_yscale("symlog", linthresh=max(linear_threshold, 1.0e-12))
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

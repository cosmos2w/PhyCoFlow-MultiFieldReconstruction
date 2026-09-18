"""Config contract of the global-distribution family."""

from __future__ import annotations

from ...family_schema import FamilySchema

CONFIG_SCHEMA = FamilySchema(
    keys=frozenset(),
    components={
        "self": frozenset({"enabled", "weight", "channel_weights"}),
        "mutual": frozenset({"enabled", "weight", "pairs", "directions", "seed"}),
        "cross": frozenset({
            "enabled", "weight", "directions", "top_fraction", "seed", "include_axes", "qmc",
        }),
    },
)

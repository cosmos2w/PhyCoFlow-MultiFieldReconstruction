"""Config contract of the persistence-matching topology family."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ...family_schema import FamilySchema, reject_unknown_keys


def _check(family: Mapping[str, Any], compute: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    geometry = family.get("geometry", {})
    reject_unknown_keys(
        geometry,
        {"grid_shape", "axes", "neighbors", "power", "periodic", "allow_projected_collisions"},
        "topology.geometry",
    )
    if bool(geometry.get("periodic", False)):
        # Creator separations are measured in the plain box, so a periodic raster
        # would be silently wrong rather than merely unsupported.
        raise ValueError("topology supports only nonperiodic geometry")
    reject_unknown_keys(
        family.get("filtration", {}),
        {"dimensions", "directions", "smoothing_sigma"},
        "topology.filtration",
    )
    matching = family.get("matching", {})
    retired = sorted({"solver", "solver_tolerance"} & set(matching))
    if retired:
        raise ValueError(
            f"topology matching keys {retired} were removed: the "
            "assignment is always solved exactly from device-built costs"
        )
    reject_unknown_keys(
        matching,
        {"order", "lambda_spatial", "spatial_mode", "min_persistence"},
        "topology.matching",
    )
    if float(matching.get("order", 1.0)) <= 0 or float(matching.get("lambda_spatial", 1.0)) < 0:
        raise ValueError("topology matching requires order>0 and lambda_spatial>=0")
    if not 0.0 <= float(matching.get("min_persistence", 0.01)) < 1.0:
        raise ValueError("topology matching.min_persistence must lie in [0, 1)")


CONFIG_SCHEMA = FamilySchema(
    keys=frozenset({"geometry", "filtration", "matching"}),
    components={
        "self": frozenset({"enabled", "weight"}),
        "mutual": frozenset({
            "enabled", "weight", "pairs", "lines", "angle_margin", "line_sampling", "seed",
        }),
    },
    requires_fixed_shared=True,
    check=_check,
)

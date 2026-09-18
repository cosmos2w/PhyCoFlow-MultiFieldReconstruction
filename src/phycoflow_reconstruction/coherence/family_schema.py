"""What a coherence family's post-training config block may contain.

Every family declares one :class:`FamilySchema` next to its registration, and
:func:`phycoflow_reconstruction.config.validate.validate_config` validates the
``coherence.families`` mapping from those declarations alone. A family that is
absent from the registry is therefore also absent from validation, which is what
lets an optional family package be dropped without leaving a stale schema behind.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

#: A family-specific check over ``(family_block, compute_budget, whole_config)``.
#: It raises ``ValueError`` on a bad configuration and returns nothing otherwise.
FamilyCheck = Callable[[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]], None]


@dataclass(frozen=True)
class FamilySchema:
    #: Keys allowed on the family block beyond the common ones (`enabled`, `weight`,
    #: `target_use`, `units`, `fields`, `reference_bank`, `components`).
    keys: frozenset[str]
    #: Component name -> the keys its settings mapping may carry.
    components: Mapping[str, frozenset[str]]
    #: Components that run only when `enabled: true` is written explicitly.
    opt_in_components: frozenset[str] = field(default_factory=frozenset)
    #: Whether an enabled instance needs `compute_budget.query_policy: fixed_shared`.
    requires_fixed_shared: bool = False
    #: Structural and numeric checks specific to this family, if any.
    check: FamilyCheck | None = None

    def component_enabled(self, name: str, settings: Mapping[str, Any]) -> bool:
        return bool(settings.get("enabled", name not in self.opt_in_components))


def reject_unknown_keys(mapping: Mapping[str, Any], allowed: set[str], path: str) -> None:
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise ValueError(f"unknown {path} keys: {unknown}")

"""Construct case-owned physics and diagnostic providers through CaseSpec."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..contracts import DataSpec
from ..data.normalization import FieldNormalizer
from ..registry import CASE_REGISTRY


def _case_spec(case_name: str):
    return CASE_REGISTRY.build(case_name)


def build_case_physics(
    case_name: str,
    settings: Mapping[str, Any],
    data_spec: DataSpec,
    normalizer: FieldNormalizer,
):
    spec = _case_spec(case_name)
    if spec.physics_factory is None:
        raise ValueError(f"case {case_name!r} has no verified PhysicsProvider")
    return spec.physics_factory(dict(settings), data_spec, normalizer)


def build_case_diagnostics(
    case_name: str,
    data_spec: DataSpec,
    normalizer: FieldNormalizer,
):
    spec = _case_spec(case_name)
    if spec.diagnostics_factory is None:
        return None
    return spec.diagnostics_factory(data_spec, normalizer)

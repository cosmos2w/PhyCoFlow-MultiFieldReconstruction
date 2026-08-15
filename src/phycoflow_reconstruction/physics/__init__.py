"""Shared differentiable operators and case-provider construction."""

from .factory import build_case_diagnostics, build_case_physics

__all__ = ["build_case_diagnostics", "build_case_physics"]

"""Brusselator metadata registered against the shared case contract."""

from phycoflow_reconstruction.contracts import CaseSpec
from phycoflow_reconstruction.registry import CASE_REGISTRY


def _physics_factory(settings, data_spec, normalizer):
    from physics import build_physics_provider

    return build_physics_provider(settings, data_spec, normalizer)


def _diagnostics_factory(data_spec, normalizer):
    from physics import build_diagnostics_provider

    return build_diagnostics_provider(data_spec, normalizer)

CASE_SPEC = CaseSpec(
    name="brusselator",
    display_name="Two-field Brusselator reaction-diffusion system",
    field_names=("u", "v"),
    field_units=("dimensionless", "dimensionless"),
    reconstruction_unit="snapshot",
    mesh_type="structured_periodic",
    grid_shape=(192, 192),
    physics_factory=_physics_factory,
    diagnostics_factory=_diagnostics_factory,
    metadata={"split_unit": "trajectory", "initial_role": "canonical_benchmark"},
)
CASE_SPEC.validate()
if CASE_SPEC.name not in CASE_REGISTRY.names():
    CASE_REGISTRY.register(CASE_SPEC.name, lambda: CASE_SPEC)

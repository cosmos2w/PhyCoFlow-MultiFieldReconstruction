"""Kuramoto--Sivashinsky space-time reconstruction case metadata."""

from phycoflow_reconstruction.contracts import CaseSpec
from phycoflow_reconstruction.registry import CASE_REGISTRY


def _diagnostics_factory(data_spec, normalizer):
    from diagnostics import build_diagnostics_provider

    return build_diagnostics_provider(data_spec, normalizer)

CASE_SPEC = CaseSpec(
    name="ks",
    display_name="Kuramoto--Sivashinsky quasi-super-resolution",
    field_names=("u",),
    field_units=("dimensionless",),
    reconstruction_unit="space_time_trajectory",
    mesh_type="structured_periodic_space_time",
    grid_shape=(401, 256),
    diagnostics_factory=_diagnostics_factory,
    metadata={
        "split_unit": "trajectory",
        "task": "joint spatial-temporal reconstruction",
        "forecasting": False,
    },
)
CASE_SPEC.validate()
if CASE_SPEC.name not in CASE_REGISTRY.names():
    CASE_REGISTRY.register(CASE_SPEC.name, lambda: CASE_SPEC)

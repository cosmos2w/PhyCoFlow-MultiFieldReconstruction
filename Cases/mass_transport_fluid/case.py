"""Mass-transport fixture metadata; this tiny demo is not a benchmark."""

from phycoflow_reconstruction.contracts import CaseSpec
from phycoflow_reconstruction.registry import CASE_REGISTRY


def _diagnostics_factory(data_spec, normalizer):
    from diagnostics import build_diagnostics_provider

    return build_diagnostics_provider(data_spec, normalizer)

CASE_SPEC = CaseSpec(
    name="mass_transport_fluid",
    display_name="Coupled mass transport and fluid-flow integration fixture",
    field_names=("u_x", "u_y", "concentration"),
    field_units=("unknown", "unknown", "unknown"),
    reconstruction_unit="snapshot",
    mesh_type="structured_nonperiodic",
    grid_shape=(32, 32),
    benchmark_eligible=False,
    diagnostics_factory=_diagnostics_factory,
    metadata={"split_unit": "stored", "initial_role": "integration_fixture"},
)
CASE_SPEC.validate()
if CASE_SPEC.name not in CASE_REGISTRY.names():
    CASE_REGISTRY.register(CASE_SPEC.name, lambda: CASE_SPEC)

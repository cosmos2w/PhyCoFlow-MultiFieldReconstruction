"""Register the canonical periodic Kolmogorov-flow reconstruction case."""

from phycoflow_reconstruction.contracts import CaseSpec
from phycoflow_reconstruction.registry import CASE_REGISTRY


def _diagnostics_factory(data_spec, normalizer):
    from diagnostics import build_diagnostics_provider

    return build_diagnostics_provider(data_spec, normalizer)

CASE_SPEC = CaseSpec(
    name="kolmogorov",
    display_name="Three-field two-dimensional Kolmogorov flow",
    field_names=("u", "v", "p"),
    field_units=("dimensionless", "dimensionless", "dimensionless"),
    reconstruction_unit="snapshot",
    mesh_type="structured_periodic",
    grid_shape=(256, 256),
    diagnostics_factory=_diagnostics_factory,
    metadata={
        "split_unit": "trajectory",
        "initial_role": "canonical_benchmark",
        "periodic_axes": ("x", "y"),
        "pressure_gauge": "zero_spatial_mean",
        "auxiliary_fields": ("vorticity",),
    },
)
CASE_SPEC.validate()
if CASE_SPEC.name not in CASE_REGISTRY.names():
    CASE_REGISTRY.register(CASE_SPEC.name, lambda: CASE_SPEC)

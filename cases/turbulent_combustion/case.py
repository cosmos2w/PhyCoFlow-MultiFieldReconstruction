"""Turbulent-combustion benchmark metadata and legacy dataset distinction."""

from phycoflow_reconstruction.contracts import CaseSpec
from phycoflow_reconstruction.registry import CASE_REGISTRY


def _diagnostics_factory(data_spec, normalizer):
    from diagnostics import build_diagnostics_provider

    return build_diagnostics_provider(data_spec, normalizer)

CASE_SPEC = CaseSpec(
    name="turbulent_combustion",
    display_name="Sparse turbulent-combustion multi-field reconstruction",
    field_names=("CH4", "CO", "T", "U_1", "p"),
    field_units=("unknown",) * 5,
    reconstruction_unit="snapshot",
    mesh_type="structured_permuted_cartesian",
    grid_shape=(100, 403),
    diagnostics_factory=_diagnostics_factory,
    metadata={
        "split_unit": "chronological_frame",
        "frame_split": (0.8, 0.1, 0.1),
        "demo50_dataset_fields": ("CO", "T", "U_0", "U_1", "p"),
        "coordinate_reorder": "lexicographic_yx",
    },
)
CASE_SPEC.validate()
if CASE_SPEC.name not in CASE_REGISTRY.names():
    CASE_REGISTRY.register(CASE_SPEC.name, lambda: CASE_SPEC)

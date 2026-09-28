"""Active-emulsion phase and velocity reconstruction metadata."""

from phycoflow_reconstruction.contracts import CaseSpec
from phycoflow_reconstruction.registry import CASE_REGISTRY

CASE_SPEC = CaseSpec(
    name="active_emulsion",
    display_name="Active-emulsion phase and velocity reconstruction",
    field_names=("phi", "vx", "vy"),
    field_units=("dimensionless",) * 3,
    reconstruction_unit="snapshot",
    mesh_type="structured",
    grid_shape=(128, 128),
    metadata={"split_unit": "source_simulation_run", "periodic": True},
)
CASE_SPEC.validate()
if CASE_SPEC.name not in CASE_REGISTRY.names():
    CASE_REGISTRY.register(CASE_SPEC.name, lambda: CASE_SPEC)

"""CoastalBench ocean-point snapshot reconstruction contract."""

from phycoflow_reconstruction.contracts import CaseSpec
from phycoflow_reconstruction.registry import CASE_REGISTRY

CASE_SPEC = CaseSpec(
    name="coastal_bench",
    display_name="CoastalBench sparse ocean reconstruction",
    field_names=("u", "v", "temp", "salt"),
    field_units=("unknown",) * 4,
    reconstruction_unit="snapshot",
    mesh_type="point",
    grid_shape=(30542,),
    metadata={"split_unit": "stored_trajectory", "coordinate_axes": ["longitude", "latitude"]},
)
CASE_SPEC.validate()
if CASE_SPEC.name not in CASE_REGISTRY.names():
    CASE_REGISTRY.register(CASE_SPEC.name, lambda: CASE_SPEC)

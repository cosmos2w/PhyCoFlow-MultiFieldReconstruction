"""Register the Shu et al. 2D Kolmogorov vorticity case (FlowReconBench data)."""

from phycoflow_reconstruction.contracts import CaseSpec
from phycoflow_reconstruction.registry import CASE_REGISTRY

CASE_SPEC = CaseSpec(
    name="kolmogorov_shu",
    display_name="Shu et al. 2D Kolmogorov vorticity (FlowReconBench split)",
    field_names=("vorticity",),
    field_units=("dimensionless",),
    reconstruction_unit="snapshot",
    mesh_type="structured_periodic",
    grid_shape=(256, 256),
    metadata={
        "split_unit": "frame block",
        "periodic_axes": ("x", "y"),
        "source": "FlowReconBench data/kolmogorov2d/Kolmogorov2D_shu_stride4.h5",
    },
)
CASE_SPEC.validate()
if CASE_SPEC.name not in CASE_REGISTRY.names():
    CASE_REGISTRY.register(CASE_SPEC.name, lambda: CASE_SPEC)

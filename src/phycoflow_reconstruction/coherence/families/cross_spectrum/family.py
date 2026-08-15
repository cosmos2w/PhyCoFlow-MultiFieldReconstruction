"""Non-computational contract marker for the reserved cross-spectrum family."""

from dataclasses import dataclass


@dataclass(frozen=True)
class CrossSpectrumFamilyRequirements:
    required_context: tuple[str, ...] = (
        "grid_topology",
        "physical_spacing",
        "boundary_periodicity",
        "wavenumber_bands",
    )
    implementation_status: str = "reserved_unimplemented"

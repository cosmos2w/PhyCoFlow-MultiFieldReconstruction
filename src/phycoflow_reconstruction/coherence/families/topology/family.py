"""Non-computational contract marker for the reserved topology family."""

from dataclasses import dataclass


@dataclass(frozen=True)
class TopologyFamilyRequirements:
    required_context: tuple[str, ...] = (
        "physical_domain",
        "connectivity",
        "valid_domain_mask",
        "filtration_or_threshold_policy",
    )
    implementation_status: str = "reserved_unimplemented"

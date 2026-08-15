"""Reusable sparse multi-field reconstruction package."""

from .contracts import (
    CaseSpec,
    DifferentiableFlowModel,
    DifferentiableReconstructionModel,
    FamilyResult,
    FieldSample,
    LossBundle,
    ModelCapabilities,
    ObservationBatch,
    ReconstructionBatch,
    TermResult,
)

__all__ = [
    "CaseSpec",
    "DifferentiableFlowModel",
    "DifferentiableReconstructionModel",
    "FamilyResult",
    "FieldSample",
    "LossBundle",
    "ModelCapabilities",
    "ObservationBatch",
    "ReconstructionBatch",
    "TermResult",
]

__version__ = "0.1.0"

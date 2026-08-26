"""Canonical point-cloud model namespace.

Use :mod:`.core` for tensor-only model implementations, :mod:`.runtime` for
construction/training/reconstruction helpers, and :mod:`.adapters` for the
project ``ObservationBatch`` model interfaces.
"""

from .core import (
    ConditionalPointHybridLocalGlobalRBF,
    ConditionalPointHybridLocalGlobalRBFCQ,
    GL_rbf_CQ,
    GL_rbf_ENH,
    GL_rbf_ENH_CQ,
    GLRbfCore,
    GLRbfEnhanced,
    IIDGaussianPrior,
    PersistentTopKGeometryCache,
    PointCloudFFM,
    RFFGaussianPrior,
    build_persistent_topk_geometry_cache,
    validate_persistent_topk_geometry_cache,
)
from .runtime import (
    ModelEMA,
    PublicModelIdentity,
    ReconstructionConfig,
    ResolvedCheckpointState,
    build_pointcloud_model,
    checkpoint_model_state,
    load_public_config,
    reconstruct_from_tensors,
    rectified_flow_loss,
    rectified_flow_loss_microbatched,
    resolve_checkpoint_state,
    resolve_model_identity,
)

__all__ = [
    "ConditionalPointHybridLocalGlobalRBF",
    "ConditionalPointHybridLocalGlobalRBFCQ",
    "GLRbfCQ",
    "GLRbfCore",
    "GLRbfEnhanced",
    "GL_rbf_CQ",
    "GL_rbf_ENH",
    "GL_rbf_ENH_CQ",
    "IIDGaussianPrior",
    "ModelEMA",
    "PersistentTopKGeometryCache",
    "PointCloudFFM",
    "ProjectGLRbfCQ",
    "ProjectGL_rbf_CQ",
    "ProjectPointCloudFFM",
    "PublicModelIdentity",
    "RFFGaussianPrior",
    "ReconstructionConfig",
    "ResolvedCheckpointState",
    "build_persistent_topk_geometry_cache",
    "build_pointcloud_model",
    "checkpoint_model_state",
    "load_public_config",
    "reconstruct_from_tensors",
    "rectified_flow_loss",
    "rectified_flow_loss_microbatched",
    "resolve_checkpoint_state",
    "resolve_model_identity",
    "validate_persistent_topk_geometry_cache",
]


def __getattr__(name: str):
    """Lazily expose adapters without making tensor-core imports project-bound."""
    if name in {"GLRbfCQ", "ProjectGLRbfCQ", "ProjectGL_rbf_CQ", "ProjectPointCloudFFM"}:
        from .adapters import GL_rbf_CQ, GLRbfCQ, PointCloudFFM

        return {
            "GLRbfCQ": GLRbfCQ,
            "ProjectGLRbfCQ": GLRbfCQ,
            "ProjectGL_rbf_CQ": GL_rbf_CQ,
            "ProjectPointCloudFFM": PointCloudFFM,
        }[name]
    raise AttributeError(name)

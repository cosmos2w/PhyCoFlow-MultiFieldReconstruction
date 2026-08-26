"""Low-level tensor-only point-cloud model cores and geometry primitives.

The classes in this package intentionally know nothing about cases,
``ObservationBatch`` objects, or project training loops.  They implement the
frozen GL-RBF/CQ architecture and its tensor-level caches.
"""

from .flow_priors import (
    IIDGaussianPrior as FlowIIDGaussianPrior,
)
from .flow_priors import (
    RFFGaussianPrior as FlowRFFGaussianPrior,
)
from .geometry import (
    PersistentTopKGeometryCache,
    build_persistent_topk_geometry_cache,
    cache_tensors,
    validate_persistent_topk_geometry_cache,
)
from .gl_rbf_core import GLRbfCore
from .gl_rbf_cq_core import (
    GL_rbf_CQ,
    GL_rbf_ENH_CQ,
)
from .gl_rbf_cq_core import (
    GLRbfCQ as CoreGLRbfCQ,
)
from .gl_rbf_enh_core import GL_rbf_ENH, GLRbfEnhanced
from .observation import (
    OBS_CONSISTENCY_MODES,
    apply_endpoint_observation_consistency,
    build_pointwise_observation_maps,
    build_smooth_observation_maps,
    normalize_obs_consistency_mode,
    observation_consistency_metrics,
    scatter_observed_values,
)
from .portable_core import (
    CompactLatentReadout,
    ConditionalPointHybridLocalGlobalRBF,
    ConditionalPointHybridLocalGlobalRBFCQ,
    CrossAttentionBlock,
    FeedForward,
    FourierPositionalEncoding,
    PointCloudFFM,
    SelfAttentionBlock,
    batched_gather_2d,
    batched_gather_3d,
    make_mlp,
)
from .priors import IIDGaussianPrior, RFFGaussianPrior

__all__ = [
    "OBS_CONSISTENCY_MODES",
    "CompactLatentReadout",
    "ConditionalPointHybridLocalGlobalRBF",
    "ConditionalPointHybridLocalGlobalRBFCQ",
    "CoreGLRbfCQ",
    "CrossAttentionBlock",
    "FeedForward",
    "FlowIIDGaussianPrior",
    "FlowRFFGaussianPrior",
    "FourierPositionalEncoding",
    "GLRbfCore",
    "GLRbfEnhanced",
    "GL_rbf_CQ",
    "GL_rbf_ENH",
    "GL_rbf_ENH_CQ",
    "IIDGaussianPrior",
    "PersistentTopKGeometryCache",
    "PointCloudFFM",
    "RFFGaussianPrior",
    "SelfAttentionBlock",
    "apply_endpoint_observation_consistency",
    "batched_gather_2d",
    "batched_gather_3d",
    "build_persistent_topk_geometry_cache",
    "build_pointwise_observation_maps",
    "build_smooth_observation_maps",
    "cache_tensors",
    "make_mlp",
    "normalize_obs_consistency_mode",
    "observation_consistency_metrics",
    "scatter_observed_values",
    "validate_persistent_topk_geometry_cache",
]

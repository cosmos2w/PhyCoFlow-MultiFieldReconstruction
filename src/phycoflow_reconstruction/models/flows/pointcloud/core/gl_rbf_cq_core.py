"""Public CQ aliases; no module or state-dict keys are renamed."""

from .portable_core import ConditionalPointHybridLocalGlobalRBFCQ

GLRbfCQ = ConditionalPointHybridLocalGlobalRBFCQ
GL_rbf_CQ = ConditionalPointHybridLocalGlobalRBFCQ
GL_rbf_ENH_CQ = ConditionalPointHybridLocalGlobalRBFCQ

__all__ = [
    "ConditionalPointHybridLocalGlobalRBFCQ",
    "GLRbfCQ",
    "GL_rbf_CQ",
    "GL_rbf_ENH_CQ",
]

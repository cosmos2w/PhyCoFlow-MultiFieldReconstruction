"""Compatibility import for the project GL-RBF/CQ adapter.

The implementation lives in :mod:`.pointcloud.adapters.gl_rbf_cq_adapter`;
this module keeps the established project-level import path stable while the
legacy second-package boundary is removed.
"""

from .pointcloud.adapters.gl_rbf_cq_adapter import GL_rbf_CQ, GLRbfCQ

__all__ = ["GLRbfCQ", "GL_rbf_CQ"]

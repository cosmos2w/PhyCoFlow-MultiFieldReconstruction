"""Project lifecycle for the full portable GL_rbf_ENH velocity backbone.

Shares the portable RF, query-microbatch, EMA and reconstruction lifecycle with
CQ, while constructing the original full-width decoder. This is distinct from
the simplified ``pointcloud_ffm/backbone=gl_rbf_enh`` adapter.
"""

from __future__ import annotations

from typing import Any

from .gl_rbf_cq_adapter import GLRbfCQ


class GLRbfENH(GLRbfCQ):
    """Full enhanced decoder; no CQ measurement/support LayerNorm exists here."""

    portable_model_name = "GL_rbf_ENH"

    def __init__(
        self, coordinate_dim: int, num_fields: int, logical_shape: tuple[int, ...], **config: Any
    ) -> None:
        if str(config.get("backbone", "GL_rbf_ENH")).lower() != "gl_rbf_enh":
            raise ValueError("gl_rbf_enh requires backbone=GL_rbf_ENH")
        unsupported = sorted(key for key in config if key.startswith("cq_"))
        if unsupported:
            raise ValueError(f"CQ-only settings cannot configure GL_rbf_ENH: {unsupported}")
        super().__init__(coordinate_dim, num_fields, logical_shape, **config)

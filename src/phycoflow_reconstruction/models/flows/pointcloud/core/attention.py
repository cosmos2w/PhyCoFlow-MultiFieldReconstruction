"""Attention primitives used by the tensor-only GL-RBF/CQ core."""

from .portable_core import (
    CompactLatentReadout,
    CrossAttentionBlock,
    FeedForward,
    SelfAttentionBlock,
)

__all__ = [
    "CompactLatentReadout",
    "CrossAttentionBlock",
    "FeedForward",
    "SelfAttentionBlock",
]

"""Seed Python, NumPy, and PyTorch consistently for repeatable comparisons."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def configure_deterministic_cublas_workspace() -> str:
    """Set or verify the CUDA workspace before CUDA work is initialized."""
    workspace = os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if workspace not in (":4096:8", ":16:8"):
        raise ValueError("deterministic CUDA requires CUBLAS_WORKSPACE_CONFIG=:4096:8 or :16:8")
    return workspace


def seed_everything(seed: int, deterministic: bool = True, *, strict_algorithms: bool = False) -> None:
    if strict_algorithms and not deterministic:
        raise ValueError("strict deterministic algorithms require deterministic runtime")
    if strict_algorithms:
        # CUDA reads this workspace setting at initialization, so set it before
        # availability probes, seeding, or any subsequent CUDA work.
        configure_deterministic_cublas_workspace()
        torch.use_deterministic_algorithms(True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

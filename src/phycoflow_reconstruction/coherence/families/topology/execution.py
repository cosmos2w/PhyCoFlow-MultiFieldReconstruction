"""Where topology work runs: the host thread pool and the device/host backend choice.

Both topology families reduce fields on detached values, either through GUDHI on the
host or through the merge trees of :mod:`.merge_tree` on an accelerator, and both
solve their host half on one shared pool. The two environment switches are

- ``PHYCOFLOW_TOPOLOGY_WORKERS``: pool size, ``1`` keeps everything inline;
- ``PHYCOFLOW_TOPOLOGY_PAIRING``: ``auto`` runs on the device wherever the fields
  already live on an accelerator, ``tensor`` and ``host`` force one path.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import TypeVar

import torch

_T = TypeVar("_T")
_R = TypeVar("_R")

_WORKER_POOL: ThreadPoolExecutor | None = None
_WORKER_POOL_LOCK = threading.Lock()
_WORKER_COUNT: int | None = None


def worker_count() -> int:
    """Pool size: `PHYCOFLOW_TOPOLOGY_WORKERS`, else `min(8, usable CPUs)`; 1 disables.

    The affinity mask is what Slurm and cgroups actually constrain, so it -- not
    `cpu_count` -- is the honest number of CPUs this process may use on a shared
    allocation. The cap is there because these batches stop scaling well before it.
    """
    global _WORKER_COUNT
    if _WORKER_COUNT is None:
        requested = os.environ.get("PHYCOFLOW_TOPOLOGY_WORKERS")
        count = None
        if requested:
            try:
                count = max(1, int(requested))
            except ValueError:
                count = None
        if count is None:
            try:
                usable = len(os.sched_getaffinity(0))
            except (AttributeError, OSError):  # pragma: no cover - platform dependent
                usable = os.cpu_count() or 1
            count = max(1, min(8, usable))
        _WORKER_COUNT = count
    return _WORKER_COUNT


def worker_pool() -> ThreadPoolExecutor:
    global _WORKER_POOL
    if _WORKER_POOL is None:
        with _WORKER_POOL_LOCK:
            if _WORKER_POOL is None:
                _WORKER_POOL = ThreadPoolExecutor(
                    max_workers=worker_count(), thread_name_prefix="phycoflow-topology"
                )
    return _WORKER_POOL


def _reset_worker_pool_after_fork() -> None:
    """A forked child inherits the pool object but none of the threads behind it.

    Dataloader workers fork while the parent may hold the construction lock, so both
    are replaced outright; the child rebuilds a pool on demand if it ever needs one.
    """
    global _WORKER_POOL, _WORKER_POOL_LOCK
    _WORKER_POOL = None
    _WORKER_POOL_LOCK = threading.Lock()


if hasattr(os, "register_at_fork"):  # pragma: no cover - exercised only across a fork
    os.register_at_fork(after_in_child=_reset_worker_pool_after_fork)


def parallel_map(fn: Callable[[_T], _R], items: Sequence[_T]) -> list[_R]:
    """`[fn(x) for x in items]` on the shared pool, order preserved.

    GUDHI's reduction and SciPy's assignment both release the GIL for essentially
    their whole runtime, so independent units genuinely overlap. `fn` must touch
    only detached NumPy data: it runs off the main thread, where building autograd
    graph would be unsafe. A single worker or a trivial batch runs inline.
    """
    if len(items) < 2 or worker_count() < 2:
        return [fn(item) for item in items]
    return list(worker_pool().map(fn, items))


PAIRING_MODES = ("auto", "tensor", "host")


def pairing_mode() -> str:
    mode = os.environ.get("PHYCOFLOW_TOPOLOGY_PAIRING", "auto").strip().lower()
    if mode not in PAIRING_MODES:
        raise ValueError(
            f"PHYCOFLOW_TOPOLOGY_PAIRING must be one of {PAIRING_MODES}, got {mode!r}"
        )
    return mode


def tensor_pairing_selected(device: torch.device) -> bool:
    """Whether the device backend runs for fields on `device` under the current mode."""
    mode = pairing_mode()
    return mode == "tensor" or (mode == "auto" and device.type != "cpu")

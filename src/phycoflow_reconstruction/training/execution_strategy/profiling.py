"""Bounded restored-state profiling for whole-plan measurements."""

from __future__ import annotations

import copy
import hashlib
import random
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch


def _clone_state(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: _clone_state(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_clone_state(item) for item in value)
    if isinstance(value, list):
        return [_clone_state(item) for item in value]
    return copy.deepcopy(value)


def _tensor_bytes(value: torch.Tensor) -> bytes:
    tensor = value.detach()
    if tensor.layout != torch.strided:
        tensor = tensor.to_dense()
    tensor = tensor.contiguous().to(device="cpu")
    return tensor.reshape(-1).view(torch.uint8).numpy().tobytes()


def state_fingerprint(model: Any) -> str:
    """Hash named model parameters and buffers without reading a checkpoint."""

    if model is None:
        return hashlib.sha256(b"no-model").hexdigest()
    digest = hashlib.sha256()
    if hasattr(model, "state_dict"):
        state = model.state_dict()
        entries = sorted(state.items(), key=lambda item: str(item[0]))
    else:
        entries = []
        if hasattr(model, "named_parameters"):
            entries.extend((f"parameter:{name}", value) for name, value in model.named_parameters())
        if hasattr(model, "named_buffers"):
            entries.extend((f"buffer:{name}", value) for name, value in model.named_buffers())
        entries.sort(key=lambda item: str(item[0]))
    for name, value in entries:
        digest.update(str(name).encode("utf-8"))
        if isinstance(value, torch.Tensor):
            digest.update(str(value.dtype).encode("ascii"))
            digest.update(repr(tuple(value.shape)).encode("ascii"))
            digest.update(_tensor_bytes(value))
        else:
            digest.update(repr(value).encode("utf-8"))
    return digest.hexdigest()


@contextmanager
def preserve_profile_state(
    model: Any = None,
    *,
    generators: Sequence[Any] = (),
    preserve_global_rng: bool = True,
) -> Iterator[dict[str, Any]]:
    """Restore model tensors, training mode, and owned/global RNG after a probe."""

    model_state = (
        _clone_state(model.state_dict())
        if model is not None and hasattr(model, "state_dict")
        else None
    )
    model_mode = getattr(model, "training", None) if model is not None else None
    model_identity = state_fingerprint(model)
    generator_states = [
        _clone_state(generator.get_state()) if hasattr(generator, "get_state") else None
        for generator in generators
    ]
    python_state = random.getstate() if preserve_global_rng else None
    numpy_state = copy.deepcopy(np.random.get_state()) if preserve_global_rng else None
    torch_cpu_state = torch.get_rng_state().clone() if preserve_global_rng else None
    torch_cuda_states = None
    if preserve_global_rng and torch.cuda.is_initialized():
        torch_cuda_states = [state.clone() for state in torch.cuda.get_rng_state_all()]
    snapshot = {
        "source_state_fingerprint": model_identity,
        "global_rng_preserved": preserve_global_rng,
        "owned_generators": len(generators),
    }
    try:
        yield snapshot
    finally:
        if model_state is not None:
            model.load_state_dict(model_state, strict=True)
        if model is not None and model_mode is not None:
            model.train(bool(model_mode))
        for generator, state in zip(generators, generator_states):
            if state is not None:
                generator.set_state(state)
        if preserve_global_rng:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            torch.set_rng_state(torch_cpu_state)
            if torch_cuda_states is not None:
                torch.cuda.set_rng_state_all(torch_cuda_states)
        restored_identity = state_fingerprint(model)
        if restored_identity != model_identity:
            raise RuntimeError("profiling failed to restore the SOURCE model state")


@dataclass(frozen=True)
class ProfileMeasurement:
    name: str
    candidate_index: int
    work_seconds: float
    charged_seconds: float
    source_identity: str
    task_hash: str
    source_state_fingerprint: str
    restored_state_fingerprint: str
    succeeded: bool
    error: str = ""
    value: Any = field(default=None, repr=False, compare=False)


class ProfileBudget:
    """At most six candidate probes, with all setup and probe time charged."""

    HARD_MAX_CANDIDATES = 6

    def __init__(self, *, max_candidates: int = HARD_MAX_CANDIDATES) -> None:
        if isinstance(max_candidates, bool) or int(max_candidates) != max_candidates:
            raise ValueError("max_candidates must be an integer")
        if not 1 <= max_candidates <= self.HARD_MAX_CANDIDATES:
            raise ValueError("profile budget allows between one and six candidates")
        self.max_candidates = int(max_candidates)
        self.measurements: list[ProfileMeasurement] = []
        self.charged_overhead_s = 0.0

    @property
    def candidates_used(self) -> int:
        return len(self.measurements)

    def measure(
        self,
        name: str,
        model: Any,
        fn: Callable[[], Any],
        *,
        source_identity: str = "",
        task_hash: str = "",
        generators: Sequence[Any] = (),
        preserve_global_rng: bool = True,
    ) -> ProfileMeasurement:
        if not name:
            raise ValueError("profile candidate name cannot be empty")
        if not source_identity or not task_hash:
            raise ValueError("profile candidate must bind the SOURCE and frozen task hash")
        if self.candidates_used >= self.max_candidates:
            raise RuntimeError(f"profile candidate limit ({self.max_candidates}) has been reached")
        candidate_index = self.candidates_used + 1
        charged_start = time.perf_counter()
        work_seconds = 0.0
        value: Any = None
        error_message = ""
        succeeded = False
        before = state_fingerprint(model)
        try:
            with preserve_profile_state(
                model,
                generators=generators,
                preserve_global_rng=preserve_global_rng,
            ):
                work_start = time.perf_counter()
                value = fn()
                work_seconds = time.perf_counter() - work_start
            succeeded = True
        except Exception as error:
            error_message = f"{type(error).__name__}: {error}"
            raise
        finally:
            charged_seconds = time.perf_counter() - charged_start
            self.charged_overhead_s += charged_seconds
            after = state_fingerprint(model)
            measurement = ProfileMeasurement(
                name=name,
                candidate_index=candidate_index,
                work_seconds=work_seconds,
                charged_seconds=charged_seconds,
                source_identity=source_identity,
                task_hash=task_hash,
                source_state_fingerprint=before,
                restored_state_fingerprint=after,
                succeeded=succeeded,
                error=error_message,
                value=value,
            )
            self.measurements.append(measurement)
            if after != before:
                raise RuntimeError("profile candidate changed the model's restored source identity")
        return measurement

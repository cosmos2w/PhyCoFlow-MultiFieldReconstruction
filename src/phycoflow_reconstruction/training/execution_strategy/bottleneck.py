"""One-update, low-payload PyTorch profiler for execution bottleneck screening.

The helper records one caller-supplied update. It keeps no trace, shapes,
stacks, or memory snapshots. Scope times are inclusive; do not sum them when
they overlap. Operator rows report self time and can be compared separately.
"""

from __future__ import annotations

import inspect
import math
import time
from collections import defaultdict
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from typing import Any

import torch
from torch import nn

_MAX_TOP_OPERATORS = 12
_MAX_SCOPE_LABEL_LENGTH = 128
_SCOPE_PREFIX = "phycoflow.execution_profile."


class _ScopeRecorder:
    """Small callback-facing API for reductions and optimizer substeps."""

    @contextmanager
    def __call__(self, name: str) -> Iterator[None]:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("profile scope name must be a non-empty string")
        label = name.strip()
        if len(label) > _MAX_SCOPE_LABEL_LENGTH:
            raise ValueError("profile scope name is too long")
        with torch.profiler.record_function(f"{_SCOPE_PREFIX}scope.{label}"):
            yield


def _time_us(event: Any, *names: str) -> float:
    for name in names:
        value = getattr(event, name, None)
        if value is not None:
            try:
                result = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(result):
                return max(0.0, result)
    return 0.0


def _scope_name(name: str) -> str | None:
    if not name.startswith(_SCOPE_PREFIX):
        return None
    return name[len(_SCOPE_PREFIX) :]


def _profile_callback(update_fn: Callable[..., Any], scope: _ScopeRecorder) -> Any:
    """Pass the optional scope helper only when the callback declares it."""

    try:
        signature = inspect.signature(update_fn)
    except (TypeError, ValueError):
        return update_fn()
    parameters = signature.parameters
    if "scope" in parameters or any(
        item.kind is inspect.Parameter.VAR_KEYWORD for item in parameters.values()
    ):
        return update_fn(scope=scope)
    return update_fn()


def profile_one_update(
    model: nn.Module,
    families: Mapping[str, Any],
    update_fn: Callable[..., Any],
    device: torch.device | str,
) -> dict[str, Any]:
    """Profile exactly one callback update on CPU and, when selected, CUDA.

    The caller supplies a real TRAIN batch through ``update_fn`` and owns the
    SOURCE/model/RNG snapshot and restoration around this helper. Top-level
    model and family ``forward`` calls receive automatic inclusive scopes.
    A callback that accepts a ``scope`` keyword can mark dense fidelity,
    backward, optimizer, dual, data, or geometry work with the same recorder.
    """

    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    if not isinstance(families, Mapping):
        raise TypeError("families must map stable family names to family objects")
    if not callable(update_fn):
        raise TypeError("update_fn must be callable")
    selected_device = torch.device(device)
    if selected_device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA profiling was requested but CUDA is unavailable")
        torch.cuda.synchronize(selected_device)

    activities = [torch.profiler.ProfilerActivity.CPU]
    if selected_device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    contexts: dict[int, list[Any]] = defaultdict(list)
    handles: list[Any] = []

    def attach(module: nn.Module, label: str) -> None:
        module_key = id(module)

        def enter(_module: nn.Module, _inputs: tuple[Any, ...]) -> None:
            context = torch.profiler.record_function(f"{_SCOPE_PREFIX}{label}")
            context.__enter__()
            contexts[module_key].append(context)

        def exit_scope(
            _module: nn.Module,
            _inputs: tuple[Any, ...],
            _output: Any,
        ) -> None:
            stack = contexts[module_key]
            if stack:
                stack.pop().__exit__(None, None, None)

        handles.append(module.register_forward_pre_hook(enter))
        handles.append(module.register_forward_hook(exit_scope, always_call=True))

    attach(model, "shared_model_rollout_forward")
    seen_modules = {id(model)}
    for family_name, family in families.items():
        if not isinstance(family_name, str) or not family_name.strip():
            raise ValueError("family names must be non-empty strings")
        if isinstance(family, nn.Module) and id(family) not in seen_modules:
            attach(family, f"family.{family_name}")
            seen_modules.add(id(family))

    scope = _ScopeRecorder()
    profiler = torch.profiler.profile(
        activities=activities,
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
        with_modules=False,
    )
    started = time.perf_counter()
    result: Any = None
    try:
        with profiler:
            result = _profile_callback(update_fn, scope)
            if selected_device.type == "cuda":
                torch.cuda.synchronize(selected_device)
    finally:
        for handle in reversed(handles):
            handle.remove()
        for stack in contexts.values():
            while stack:
                stack.pop().__exit__(None, None, None)
        if selected_device.type == "cuda":
            torch.cuda.synchronize(selected_device)
    measured_wall_seconds = time.perf_counter() - started

    averages = profiler.key_averages()
    inclusive_scopes: dict[str, dict[str, float | int]] = {}
    for event in averages:
        name = _scope_name(str(event.key))
        if name is None:
            continue
        inclusive_scopes[name] = {
            "calls": int(getattr(event, "count", 0)),
            "inclusive_cpu_seconds": _time_us(event, "cpu_time_total") / 1_000_000.0,
            "inclusive_cuda_seconds": _time_us(event, "device_time_total", "cuda_time_total")
            / 1_000_000.0,
        }

    operators = [
        event
        for event in averages
        if str(event.key).startswith("aten::") and not _scope_name(str(event.key))
    ]

    def top_operators(*, cuda: bool) -> list[dict[str, Any]]:
        self_names = (
            ("self_device_time_total", "self_cuda_time_total") if cuda else ("self_cpu_time_total",)
        )
        total_names = ("device_time_total", "cuda_time_total") if cuda else ("cpu_time_total",)
        ranked = sorted(operators, key=lambda event: _time_us(event, *self_names), reverse=True)
        rows: list[dict[str, Any]] = []
        for event in ranked[:_MAX_TOP_OPERATORS]:
            self_seconds = _time_us(event, *self_names) / 1_000_000.0
            total_seconds = _time_us(event, *total_names) / 1_000_000.0
            if cuda and self_seconds == 0.0 and total_seconds == 0.0:
                continue
            rows.append(
                {
                    "operator": str(event.key),
                    "calls": int(getattr(event, "count", 0)),
                    "self_seconds": self_seconds,
                    "total_seconds": total_seconds,
                }
            )
        return rows

    return {
        "schema_version": 1,
        "device": str(selected_device),
        "measured_update_wall_seconds": measured_wall_seconds,
        "callback_result_type": type(result).__name__,
        "scopes": inclusive_scopes,
        "top_operators_self_cpu": top_operators(cuda=False),
        "top_operators_self_cuda": top_operators(cuda=True)
        if selected_device.type == "cuda"
        else [],
        "measurement_limits": {
            "profiled_update_count": 1,
            "top_operator_count": _MAX_TOP_OPERATORS,
            "record_shapes": False,
            "profile_memory": False,
            "with_stack": False,
            "trace_exported": False,
            "scope_times_are_inclusive_and_must_not_be_summed": True,
            "family_forward_includes_generated_and_reference_work_together": True,
            "unavailable_callback_scopes_are_not_inferred": True,
        },
    }

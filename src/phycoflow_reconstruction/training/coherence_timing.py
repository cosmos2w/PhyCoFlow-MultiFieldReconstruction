"""Exclusive host-side timing for coherence training and its callbacks.

These are wall-clock host spans, not CUDA kernel durations; no synchronization is
introduced. Whole-update wall time remains the trainer's own update measurement.
"""

from __future__ import annotations

from contextlib import contextmanager
from functools import wraps
from time import perf_counter_ns


class HostTiming:
    """Accumulate disjoint named spans, subtracting nested spans from parents."""

    def __init__(self, *, clock=perf_counter_ns):
        self._clock = clock
        self._stack = []
        self._totals = {}
        self._calls = {}

    @contextmanager
    def scope(self, category):
        # Callback owns its entire subtree, even when instrumented operations run inside it.
        if self._stack and self._stack[-1][0] == "callback":
            yield
            return
        frame = [str(category), self._clock(), 0]
        self._stack.append(frame)
        try:
            yield
        finally:
            elapsed = self._clock() - frame[1]
            if self._stack.pop() is not frame:
                raise RuntimeError("timing scopes exited out of order")
            exclusive = elapsed - frame[2]
            self._totals[frame[0]] = self._totals.get(frame[0], 0) + exclusive
            self._calls[frame[0]] = self._calls.get(frame[0], 0) + 1
            if self._stack:
                self._stack[-1][2] += elapsed

    def wrap_callback(self, function):
        @wraps(function)
        def timed(*args, **kwargs):
            with self.scope("callback"):
                return function(*args, **kwargs)

        return timed

    def report(self):
        names = sorted(self._totals)
        return {
            "clock": "perf_counter_ns host spans; no CUDA synchronization",
            "totals_seconds": {name: self._totals[name] / 1e9 for name in names},
            "calls": {name: self._calls[name] for name in names},
            "mean_seconds": {name: self._totals[name] / self._calls[name] / 1e9 for name in names},
        }

    def state_dict(self):
        return self.report()

    def reset(self):
        if self._stack:
            raise RuntimeError("cannot reset active timing scopes")
        self._totals.clear()
        self._calls.clear()


@contextmanager
def instrument_training_timing(timer):
    """Temporarily instrument training functions; always restore original methods."""
    from torch.optim import AdamW

    from ..coherence.families.global_distribution.conditional import ConditionalQuantileFairFamily
    from ..coherence.families.global_distribution.family import GlobalDistributionFamily
    from ..coherence.families.multiscale_spectrum.family import MultiscaleSpectrumFamily
    from ..coherence.families.topology.family import TopologyFamily
    from . import post_training
    from .fidelity_budget import ExternalPolicy
    from .monitoring import TrainingMonitor

    targets = (
        (post_training, "differentiable_reconstruction", "reconstruction"),
        (GlobalDistributionFamily, "forward", "state"),
        (ConditionalQuantileFairFamily, "forward", "state"),
        (MultiscaleSpectrumFamily, "forward", "scale"),
        (TopologyFamily, "forward", "shape"),
        (ExternalPolicy, "update", "backward_config"),
        (ExternalPolicy, "update_with_coherence_aggregation", "backward_config"),
        (AdamW, "step", "optimizer"),
        (TrainingMonitor, "record", "logging"),
        (TrainingMonitor, "finish_step", "logging"),
        (TrainingMonitor, "close", "logging"),
    )
    installed = []
    try:
        for owner, name, category in targets:
            original = getattr(owner, name)

            def make_wrapper(function, label):
                @wraps(function)
                def wrapped(*args, **kwargs):
                    with timer.scope(label):
                        return function(*args, **kwargs)

                return wrapped

            installed.append((owner, name, original))
            setattr(owner, name, make_wrapper(original, category))
        yield timer
    finally:
        for owner, name, original in reversed(installed):
            setattr(owner, name, original)

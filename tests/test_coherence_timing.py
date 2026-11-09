"""Verify disjoint host accounting and instrumentation cleanup."""

import pytest

from phycoflow_reconstruction.training.coherence_timing import (
    HostTiming,
    instrument_training_timing,
)


def test_nested_spans_and_callback_suppression():
    class Clock:
        now = 0

        def __call__(self):
            return self.now

    clock = Clock()
    timer = HostTiming(clock=clock)
    with timer.scope("preparation"):
        clock.now = 10
        with timer.scope("state"):
            clock.now = 30
        clock.now = 40

        def callback():
            clock.now = 50
            with timer.scope("shape"):
                clock.now = 80
            clock.now = 90

        timer.wrap_callback(callback)()
        clock.now = 100
    report = timer.report()
    assert report["totals_seconds"] == {"callback": 50e-9, "preparation": 30e-9, "state": 20e-9}
    assert report["calls"] == {"callback": 1, "preparation": 1, "state": 1}
    assert sum(report["totals_seconds"].values()) == pytest.approx(100e-9)
    timer.reset()
    assert not timer.report()["calls"]


def test_instrumentation_restores_on_failure():
    from phycoflow_reconstruction.training import post_training

    original = post_training.differentiable_reconstruction
    with pytest.raises(RuntimeError, match="failure"), instrument_training_timing(HostTiming()):
        assert post_training.differentiable_reconstruction is not original
        raise RuntimeError("failure")
    assert post_training.differentiable_reconstruction is original

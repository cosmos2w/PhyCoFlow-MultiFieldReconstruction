import pytest

from scripts.posttraining.lease_car_round8 import acquire, admitted_ceiling


def test_preparation_includes_failures_and_active_reservations():
    state = {"closed": [{"token": "failed", "phase": "preparation", "seconds": 21000}],
             "active": {}}
    device = {"uuid": "runtime-selected"}
    with pytest.raises(ValueError, match="inclusive"):
        acquire(state, token="next", device=device, phase="preparation", duration=601,
                now=0, ceiling=21600)
    acquire(state, token="next", device=device, phase="preparation", duration=600,
            now=0, ceiling=21600)
    with pytest.raises(ValueError, match="inclusive"):
        acquire(state, token="extra", device={"uuid": "another-runtime-device"},
                phase="assessment", duration=60, now=0, ceiling=21600)


def test_learning_requires_a_budgeted_sealed_question():
    assert admitted_ceiling(None) == 21600
    with pytest.raises(ValueError, match="sealed"):
        admitted_ceiling({"admitted": True})
    decision = {"status": "sealed", "admitted": True, "test_locked": True,
                "total_gpu_ceiling_seconds": 86400, "question": "position utility",
                "forecast": {"total_seconds": 80000}}
    assert admitted_ceiling(decision) == 86400
    state = {"closed": [], "active": {}}
    with pytest.raises(ValueError, match="not been admitted"):
        acquire(state, token="formal", device={"uuid": "runtime"}, phase="learning",
                duration=600, now=0, ceiling=21600)


def test_expansion_keeps_preparation_cap_and_total_cost():
    state = {"closed": [{"phase": "preparation", "seconds": 21580},
                        {"phase": "learning", "seconds": 63800}], "active": {}}
    with pytest.raises(ValueError, match="preparation"):
        acquire(state, token="prep", device={"uuid": "runtime"}, phase="preparation",
                duration=60, now=0, ceiling=86400)
    with pytest.raises(ValueError, match="inclusive"):
        acquire(state, token="assessment", device={"uuid": "runtime"}, phase="assessment",
                duration=1021, now=0, ceiling=86400)

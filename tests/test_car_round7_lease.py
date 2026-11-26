import pytest

from scripts.posttraining.lease_car_round7 import CAP_SECONDS, acquire


def test_inclusive_cap_charges_closed_failures_and_active_reservations():
    state = {"closed": [{"seconds": 3600, "status": "failed"}],
             "active": {"prior": {"gpu_uuid": "GPU-prior", "max_seconds": 3600}}}
    with pytest.raises(ValueError, match="inclusive"):
        acquire(state, token="new", device={"uuid": "GPU-next"}, phase="learning",
                duration=CAP_SECONDS-7199, now=0)
    acquire(state, token="new", device={"uuid": "GPU-next"}, phase="learning",
            duration=CAP_SECONDS-7200, now=0)


def test_preparation_target_does_not_close_scientific_round():
    state = {"closed": [{"seconds": 5*3600, "phase": "preparation"}], "active": {}}
    acquire(state, token="correction", device={"uuid": "GPU-runtime"},
            phase="preparation", duration=3600, now=0)
    assert state["active"]["correction"]["phase"] == "preparation"


def test_overlapping_device_is_rejected_without_changing_ledger():
    state = {"closed": [], "active": {"owned": {"gpu_uuid": "GPU-runtime", "max_seconds": 60}}}
    with pytest.raises(ValueError, match="already leased"):
        acquire(state, token="new", device={"uuid": "GPU-runtime"}, phase="assessment",
                duration=60, now=0)
    assert list(state["active"]) == ["owned"]

import pytest

from scripts.posttraining.run_car_round6 import acquire, CAP_SECONDS, PREPARATION_SECONDS


def test_round6_budget_sums_devices_and_failed_preparation():
    state = {'closed': [{'phase': 'preparation', 'seconds': 1200, 'status': 'failed'}], 'active': {}}
    acquire(state, token='first', device={'uuid': 'runtime-device-1'}, phase='preparation', duration=600, now=0)
    with pytest.raises(ValueError, match='preparation'):
        acquire(state, token='second', device={'uuid': 'runtime-device-2'}, phase='preparation', duration=PREPARATION_SECONDS, now=0)
    acquire(state, token='second', device={'uuid': 'runtime-device-2'}, phase='formal_training', duration=CAP_SECONDS-1800, now=0)
    with pytest.raises(ValueError, match='30 occupied'):
        acquire(state, token='third', device={'uuid': 'runtime-device-3'}, phase='assessment', duration=1, now=0)


def test_round6_rejects_two_leases_on_same_device():
    state = {'closed': [], 'active': {}}
    acquire(state, token='first', device={'uuid': 'runtime-device-1'}, phase='preparation', duration=60, now=0)
    with pytest.raises(ValueError, match='already leased'):
        acquire(state, token='second', device={'uuid': 'runtime-device-1'}, phase='assessment', duration=60, now=0)

"""Persistent fixed-epoch rebound alarms request diagnostics, never intervention."""

import copy
import json

import pytest
import torch

from phycoflow_reconstruction.training.rebound_monitor import ReboundMonitor


def _monitor():
    return ReboundMonitor(source_denominators={"A": .2, "B": .3, "C": .4})


def test_fixed_window_two_events_and_one_request_per_episode():
    monitor = _monitor()
    scores = [1., .9, .9, .99, .99, .99, .99, .9, .99, .99, .99]
    records = [monitor.observe(epoch, score) for epoch, score in zip(range(0, 110, 10), scores)]
    assert records[3]["rolling_score"] == pytest.approx((.9 + .99) / 2)
    assert records[4]["consecutive_rebound_events"] == 1
    assert records[5]["diagnostic_triggered"]
    assert not records[6]["diagnostic_triggered"]
    assert records[-1]["diagnostic_triggered"]
    assert [record["score"] for record in monitor.observations] == scores
    assert monitor.consume_diagnostic_request()
    assert not monitor.consume_diagnostic_request()
    assert monitor.identity["action"] == "request_sparse_train_diagnostic_only"


def test_eighty_percent_erasure_can_alarm_below_source_corridor():
    monitor = _monitor()
    monitor.observe(10, .7)
    monitor.observe(20, .95)
    first = monitor.observe(30, .95)
    second = monitor.observe(40, .95)
    assert first["early_gain_erased"]
    assert not second["returned_to_source_corridor"]
    assert second["diagnostic_triggered"]


def test_no_alarm_without_useful_gain_and_missing_cadence_resets_persistence():
    monitor = _monitor()
    assert not any(monitor.observe(epoch, .97)["diagnostic_triggered"]
                   for epoch in (10, 20, 30))
    monitor = _monitor()
    for epoch, value in ((10, .9), (20, .9), (30, 1.), (40, 1.)):
        record = monitor.observe(epoch, value)
    assert record["consecutive_rebound_events"] == 1
    assert monitor.observe(60, 1.)["consecutive_rebound_events"] == 1
    assert monitor.observe(70, 1.)["diagnostic_triggered"]


def test_json_resume_preserves_pending_request_and_next_observation_exactly():
    monitor = _monitor()
    for epoch, score in ((10, .9), (20, .9), (30, 1.), (40, 1.), (50, 1.)):
        monitor.observe(epoch, score)
    state = json.loads(json.dumps(monitor.state_dict()))
    restored = _monitor()
    restored.load_state_dict(state)
    assert restored.consume_diagnostic_request() == monitor.consume_diagnostic_request()
    assert restored.observe(60, .9) == monitor.observe(60, .9)
    assert restored.state_dict() == monitor.state_dict()
    changed = ReboundMonitor(source_denominators={"A": .21, "B": .3, "C": .4})
    with pytest.raises(ValueError, match="identity changed"):
        changed.load_state_dict(state)
    corrupt = copy.deepcopy(state)
    corrupt["consecutive_rebound_events"] += 1
    with pytest.raises(ValueError, match="invalid rebound"):
        _monitor().load_state_dict(corrupt)


@pytest.mark.parametrize("epoch,score", [(25, .9), (True, .9), (-10, .9), (10, float("nan")), (10, -.1)])
def test_invalid_inputs_fail(epoch, score):
    with pytest.raises(ValueError):
        _monitor().observe(epoch, score)


def test_alarm_is_report_only_and_preserves_rng():
    state = torch.random.get_rng_state().clone()
    monitor = _monitor()
    for epoch, score in ((10, .8), (20, .8), (30, 1.), (40, 1.), (50, 1.)):
        monitor.observe(epoch, score)
    monitor.consume_diagnostic_request()
    assert torch.equal(state, torch.random.get_rng_state())

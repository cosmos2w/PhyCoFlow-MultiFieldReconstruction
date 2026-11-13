import importlib.util
from pathlib import Path

import pytest
import torch

from phycoflow_reconstruction.evaluation.adaptive_assessment import _tensor_identity

spec = importlib.util.spec_from_file_location('car_event_redesign', Path('scripts/posttraining/freeze_car_event.py'))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def fixture():
    truth = torch.zeros((800, 20, 3))
    truth[..., 1:] = 1
    truth[:, 8, 0] = 1e-8
    truth[:, 9, 0] = .002
    truth[:, 10:, 0] = 1
    mass = torch.ones(20, dtype=torch.float64)
    ids = [f'id{i}' for i in range(800)]
    original = {'field_names': ['CH4', 'CO', 'T'], 'sample_ids': ids, 'population': 8000,
        'split': 'train', 'data_identity': 'data', 'truth_sha256': _tensor_identity(truth),
        'native_masses_sha256': _tensor_identity(mass.expand(800, -1)), 'sha256': 'old',
        'temperature_q75': .5, 'co_q90': .5, 'ch4_q25': 0, 'units': 'unknown'}
    return truth, mass, ids, original


def test_selects_positive_train_threshold_and_preserves_cuts():
    truth, mass, ids, old = fixture()
    receipt, area = mod.redesign(truth, mass, old, ids, 8000, data_identity='data')
    assert receipt['definition']['CH4']['quantile'] == .5
    assert receipt['definition']['T']['threshold'] == .5
    assert receipt['definition']['CO']['threshold'] == .5
    assert torch.all(area > 0) and receipt['original_task_sha256'] == 'old'
    assert receipt['sha256'] == mod.redesign(truth, mass, old, ids, 8000, data_identity='data')[0]['sha256']


def test_rejects_changed_panel_or_truth():
    truth, mass, ids, old = fixture()
    with pytest.raises(ValueError, match='exact frozen'):
        mod.redesign(truth, mass, old, ids[::-1], 8000, data_identity='data')
    truth[0, 0, 0] = .1
    with pytest.raises(ValueError, match='source identity'):
        mod.redesign(truth, mass, old, ids, 8000, data_identity='data')


def test_frozen_redesigned_task_replays_and_rejects_tampering():
    truth, mass, ids, old = fixture()
    receipt, areas = mod.redesign(truth, mass, old, ids, 8000, data_identity='data')
    event = mod.FrozenRedesignedEvent(receipt)
    predicted = truth[0:1].unsqueeze(0).repeat(4, 1, 1, 1)
    result = event.score(predicted, truth[0:1], mass, torch.arange(20.).unsqueeze(-1))
    torch.testing.assert_close(result['truth_area_fraction'], areas[:1])
    assert result['task_sha256'] == receipt['sha256']
    receipt['definition']['CH4']['threshold'] = 0
    with pytest.raises(ValueError, match='invalid frozen'):
        mod.FrozenRedesignedEvent(receipt)

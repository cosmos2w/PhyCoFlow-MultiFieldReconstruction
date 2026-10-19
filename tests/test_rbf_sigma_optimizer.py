"""Verify separate bandwidth updates, legacy equivalence, and resumable AdamW groups."""
import copy
import math
from pathlib import Path

import pytest
import torch

from phycoflow_reconstruction.config import load_config, validate_config
from phycoflow_reconstruction.models.flows.pointcloud.runtime.ema import ModelEMA
from phycoflow_reconstruction.training.base_training import (
    _build_base_optimizer,
    _rbf_sigma_metrics,
)

CONFIG = Path(__file__).resolve().parents[1] / (
    'cases/kolmogorov_shu/configs/base/'
    'gl_rbf_cq_w296_range_singlefield_fix_b64_sigma_lr1e5_val1pct.yaml'
)


class BandwidthModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float64))
        self.log_rbf_sigma = torch.nn.Parameter(torch.tensor(math.log(.05), dtype=torch.float64))


def step(model, optimizer):
    optimizer.zero_grad()
    (model.weight + model.log_rbf_sigma).backward()
    optimizer.step()


def test_legacy_optimizer_matches_original_adamw_bitwise():
    model = BandwidthModel()
    reference = copy.deepcopy(model)
    settings = {'lr': 1e-4, 'weight_decay': 1e-6}
    optimizer = _build_base_optimizer(model, settings)
    original = torch.optim.AdamW(reference.parameters(), **settings)
    for _ in range(4):
        step(model, optimizer)
        step(reference, original)
    assert len(optimizer.param_groups) == 1
    for actual, expected in zip(model.parameters(), reference.parameters()):
        assert torch.equal(actual, expected)
    assert _rbf_sigma_metrics(model, optimizer) == {}


def test_bandwidth_log_updates_are_tenfold_slower_and_other_weights_match():
    model = BandwidthModel()
    original = copy.deepcopy(model)
    settings = {'lr': 1e-4, 'weight_decay': 0., 'rbf_sigma_lr': 1e-5}
    optimizer = _build_base_optimizer(model, settings)
    baseline = _build_base_optimizer(original, {'lr': 1e-4})
    initial = model.log_rbf_sigma.detach().clone()
    for _ in range(10):
        step(model, optimizer)
        step(original, baseline)
    assert torch.equal(model.weight, original.weight)
    ratio = float((model.log_rbf_sigma - initial) / (original.log_rbf_sigma - initial))
    assert ratio == pytest.approx(.1, abs=1e-10)
    ids = [id(p) for group in optimizer.param_groups for p in group['params']]
    assert len(ids) == len(set(ids)) == len(list(model.parameters()))
    assert [group['lr'] for group in optimizer.param_groups] == [1e-4, 1e-5]


def test_optimizer_roundtrip_preserves_next_update_and_ema_metrics():
    model = BandwidthModel()
    model._ema = ModelEMA(model)
    settings = {'lr': 1e-4, 'weight_decay': 1e-6, 'rbf_sigma_lr': 1e-5}
    optimizer = _build_base_optimizer(model, settings)
    for _ in range(3):
        step(model, optimizer)
        model._ema.update(model)
    restored = BandwidthModel()
    restored.load_state_dict(model.state_dict(), strict=True)
    restored._ema = ModelEMA(restored)
    restored._ema.load_state_dict(model._ema.state_dict())
    other = _build_base_optimizer(restored, settings)
    other.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    step(model, optimizer)
    step(restored, other)
    for actual, expected in zip(model.parameters(), restored.parameters()):
        assert torch.equal(actual, expected)
    metrics = _rbf_sigma_metrics(model, optimizer)
    assert metrics['rbf_sigma_lr'] == 1e-5
    assert metrics['rbf_sigma'] < .05
    assert metrics['rbf_sigma_ema'] > metrics['rbf_sigma']


def test_separate_lr_rejects_models_without_a_trainable_bandwidth():
    with pytest.raises(ValueError, match='trainable log_rbf_sigma'):
        _build_base_optimizer(torch.nn.Linear(1, 1), {'rbf_sigma_lr': 1e-5})


@pytest.mark.parametrize('value', [0, -1e-5, float('nan'), float('inf'), True, '1e-5'])
def test_invalid_sigma_lr_fails_config_validation(value):
    config = load_config(CONFIG)
    config['optimization']['rbf_sigma_lr'] = value
    with pytest.raises(ValueError, match='rbf_sigma_lr'):
        validate_config(config)


def test_sigma_lr_rejects_fixed_bandwidth_and_accepts_prepared_config():
    config = load_config(CONFIG)
    validate_config(config)
    config['model']['learnable_rbf_sigma'] = False
    with pytest.raises(ValueError, match='learnable_rbf_sigma'):
        validate_config(config)

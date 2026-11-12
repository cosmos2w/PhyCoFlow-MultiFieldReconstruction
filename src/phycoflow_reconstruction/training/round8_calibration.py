"""Checkpointable 10-epoch, eight-block internal gradient calibration."""
from __future__ import annotations
import math
import statistics
import torch

VERSION = 'round8_internal_gradient_calibration_v1'
BLOCKS = {'scale': ('signed', 'joint', 'log_power_self', 'rho'), 'state': ('marginal', 'mixed'), 'shape': ('self', 'mutual')}
PROBES = (1, 3, 6, 8, 10)


def gradient_norm(loss, parameters):
    """Measure one normalized block on the exact declared trainable parameters."""
    if not loss.requires_grad:
        return 0.0
    gradients = torch.autograd.grad(loss, tuple(parameters), retain_graph=True, allow_unused=True)
    return math.sqrt(sum(float(g.detach().double().square().sum()) for g in gradients if g is not None))


def state_channel_contribution(channel_squared, common_squared, weight):
    """Exact channel chain-rule contribution with zero-distance convention."""
    positive = common_squared > 0
    denom = torch.sqrt(torch.where(positive, common_squared, torch.ones_like(common_squared)))
    factor = torch.where(positive, weight / (2 * denom), torch.zeros_like(denom)).detach()
    return factor * channel_squared


class InternalCalibration:
    def __init__(self, priorities, units, active, numerical_floor=1e-12):
        keys = {f'{f}.{b}' for f, blocks in BLOCKS.items() for b in blocks}
        if set(priorities) != keys or set(units) != keys or set(active) != keys:
            raise ValueError('all eight semantic blocks must be declared')
        if numerical_floor <= 0 or any(not math.isfinite(float(priorities[k])) or priorities[k] <= 0 or not math.isfinite(float(units[k])) or units[k] <= 0 for k in keys):
            raise ValueError('priorities, units and floor must be positive finite')
        self.priorities, self.units, self.active = dict(priorities), dict(units), dict(active)
        self.floor = float(numerical_floor)
        self.ema, self.records, self.coefficients = {}, [], None

    def observe(self, epoch, norms):
        if self.coefficients is not None or epoch not in PROBES or (self.records and epoch <= self.records[-1]['epoch']):
            raise ValueError('probes must occur once in order before freeze')
        expected = {k for k, v in self.active.items() if v}
        if set(norms) != expected or any(not math.isfinite(float(v)) or v < 0 for v in norms.values()):
            raise ValueError('norms must contain exactly active blocks and be finite nonnegative')
        for k,v in norms.items():
            self.ema[k] = float(v) if k not in self.ema else .8*self.ema[k] + .2*float(v)
        self.records.append({'epoch': epoch, 'norms': dict(norms)})

    def freeze(self, completed_epoch):
        if completed_epoch != 10 or tuple(x['epoch'] for x in self.records) != PROBES:
            raise ValueError('freeze requires all five probes and completed epoch 10')
        result = {k: 1. for k in self.active}
        for family, blocks in BLOCKS.items():
            names = [f'{family}.{b}' for b in blocks if self.active[f'{family}.{b}']]
            positive = [self.ema[k] for k in names if self.ema[k] > self.floor]
            if not positive:
                continue
            median = statistics.median(positive)
            floor = max(self.floor, .1 * median)
            for k in names:
                # Unresolved near-zero blocks remain unamplified.
                if self.ema[k] >= floor:
                    result[k] = min(2., max(.5, median / max(self.ema[k], floor)))
        self.coefficients = result
        return dict(result)

    def state_dict(self):
        return {'version': VERSION, 'priorities': self.priorities.copy(), 'units': self.units.copy(), 'active': self.active.copy(), 'floor': self.floor, 'ema': self.ema.copy(), 'records': [{'epoch': r['epoch'], 'norms': r['norms'].copy()} for r in self.records], 'coefficients': None if self.coefficients is None else self.coefficients.copy()}

    @classmethod
    def from_state_dict(cls, state):
        if state['version'] != VERSION:
            raise ValueError('calibration version mismatch')
        obj = cls(state['priorities'], state['units'], state['active'], state['floor'])
        for record in state['records']:
            obj.observe(record['epoch'], record['norms'])
        if obj.ema != state['ema']:
            raise ValueError('inconsistent calibration EMA')
        if state['coefficients'] is not None:
            if obj.freeze(10) != state['coefficients']:
                raise ValueError('inconsistent frozen coefficients')
        return obj

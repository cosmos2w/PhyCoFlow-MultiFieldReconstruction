"""Frozen TRAIN task and finite TRAIN-only output-policy selection.

Inputs use decoded native-support fields and raw native coordinates. This module
never performs inference, chooses dataset samples, or accesses held-out data.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass

import torch

from ..coherence.physical_risk import blend_risk_quadratic

FRONTIER = (0., .125, .25, .375, .5, .625, .75, .875, 1.)


def _mass(masses, batch, nodes, device):
    mass = torch.as_tensor(masses, dtype=torch.float64, device=device)
    mass = torch.broadcast_to(mass, (batch, nodes))
    if not torch.isfinite(mass).all() or (mass < 0).any() or (mass.sum(-1) <= 0).any():
        raise ValueError('physical masses must be finite, nonnegative and nonempty per sample')
    return mass


def _quantile(values, weights, q):
    order = torch.argsort(values)
    values, weights = values[order], weights[order]
    position = torch.searchsorted(weights.cumsum(0), weights.sum() * q).clamp(max=len(values) - 1)
    return float(values[position])


def _identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _tensor_identity(tensor):
    tensor = torch.as_tensor(tensor).detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode())
    digest.update(str(tuple(tensor.shape)).encode())
    digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class FrozenEventTask:
    """Quantiles fitted over pooled TRAIN truths with physical nodal weights."""
    field_names: tuple[str, ...]
    temperature_q75: float
    co_q90: float
    ch4_q25: float
    sample_ids: tuple[str, ...]
    units: str
    population: int
    data_identity: str
    truth_sha256: str = ''
    native_masses_sha256: str = ''

    @classmethod
    def fit(cls, truth, masses, *, field_names, sample_ids, population, units, data_identity, split):
        truth = torch.as_tensor(truth)
        names, ids = tuple(field_names), tuple(sample_ids)
        if split != 'train' or truth.ndim != 3 or truth.shape[-1] != len(names):
            raise ValueError('TRAIN truths must have shape [B,N,C] in declared field order')
        if len(set(names)) != len(names) or not {'T', 'CO', 'CH4'} <= set(names):
            raise ValueError('unique CH4, CO and T fields required')
        if len(ids) != len(truth) or len(set(ids)) != len(ids) or not all(ids):
            raise ValueError('unique sample IDs must match panel')
        if population < len(ids) or len(ids) != max(1, math.ceil(.1 * population)):
            raise ValueError('task fitting requires declared approximately 10% TRAIN panel')
        if not data_identity or not units or not torch.isfinite(truth).all():
            raise ValueError('finite truth, data identity and declared units required')
        mass = _mass(masses, *truth.shape[:2], truth.device).flatten()
        def quantile(name, q):
            return _quantile(truth[..., names.index(name)].double().flatten(), mass, q)
        return cls(names, quantile('T', .75), quantile('CO', .90), quantile('CH4', .25),
                   ids, units, int(population), data_identity, _tensor_identity(truth),
                   _tensor_identity(mass.reshape(truth.shape[:2])))

    def receipt(self):
        value = {'schema': 'car_event_task_v1', 'split': 'train', **asdict(self)}
        return {**value, 'sha256': _identity(value)}

    def masks(self, fields):
        fields = torch.as_tensor(fields)
        if fields.shape[-1] != len(self.field_names) or not torch.isfinite(fields).all():
            raise ValueError('fields must be finite in the frozen field order')
        names = self.field_names
        hot = fields[..., names.index('T')] > self.temperature_q75
        co = fields[..., names.index('CO')] > self.co_q90
        low_ch4 = fields[..., names.index('CH4')] < self.ch4_q25
        return hot, co, low_ch4, hot & co & low_ch4

    def score(self, members, truth, masses, coordinates):
        """Return area CRPS and separate per-member realization diagnostics."""
        members, truth = torch.as_tensor(members), torch.as_tensor(truth)
        if members.ndim != 4 or len(members) != 4 or members.shape[1:] != truth.shape:
            raise ValueError('four matched members [4,B,N,C] and truth [B,N,C] required')
        coords = torch.as_tensor(coordinates, dtype=torch.float64, device=truth.device)
        if coords.ndim == 2:
            coords = coords.unsqueeze(0).expand(len(truth), -1, -1)
        if coords.ndim != 3 or coords.shape[:2] != truth.shape[:2] or not torch.isfinite(coords).all():
            raise ValueError('native coordinates must align with samples and nodes')
        mass = _mass(masses, *truth.shape[:2], truth.device)
        total = mass.sum(-1)
        true_masks = self.masks(truth)
        member_masks = self.masks(members)
        def area(mask):
            return (mask * mass).sum(-1) / total
        event, target = member_masks[-1], true_masks[-1]
        member_area, target_area = area(event), area(target)
        fair = (member_area - target_area).abs().mean(0)
        fair = fair - sum((member_area[a] - member_area[b]).abs() for a in range(4) for b in range(a + 1, 4)) / 12
        return {
            'task_sha256': self.receipt()['sha256'], 'members': 4,
            'area_crps': fair,
            'truth_area_fraction': target_area, 'member_area_fraction': member_area,
            'truth_unary_area_fractions': {name: area(mask) for name, mask in zip(('T', 'CO', 'CH4'), true_masks[:3])},
            'member_unary_area_fractions': {name: area(mask) for name, mask in zip(('T', 'CO', 'CH4'), member_masks[:3])},
            'truth_first_moment': (target[..., None] * mass[..., None] * coords).sum(-2),
            'member_first_moment': (event[..., None] * mass[..., None] * coords).sum(-2),
            'overlap_fraction': area(event & target.unsqueeze(0)),
            'missing_fraction': area(~event & target.unsqueeze(0)),
            'excess_fraction': area(event & ~target.unsqueeze(0)),
            'moment_semantics': 'unnormalized native physical mass times raw coordinate',
        }


def select_train_alpha(source, child, truth, masses, *, sample_ids, population,
                       source_hash, child_hash, coupling_identity, source_core,
                       core_observer, source_inference_cost, child_inference_cost, blend_inference_cost,
                       split, frontier=FRONTIER):
    """Freeze finite TRAIN feasibility and core utility on actual paired blends.

    `core_observer` receives [J,B,N,C] and returns three complete, common-core
    scalar scores in State, Scale, Shape order; no reserved task is accepted.
    """
    source, child, truth = map(torch.as_tensor, (source, child, truth))
    ids = tuple(sample_ids)
    if split != 'train' or source.shape != child.shape or source.ndim != 4 or source.shape[1:] != truth.shape:
        raise ValueError('matched TRAIN members and truth are required')
    if len(source) != 4 or len(ids) != len(truth) or len(set(ids)) != len(ids):
        raise ValueError('J4 and unique matched IDs required')
    if len(ids) != max(1, math.ceil(.1 * population)):
        raise ValueError('frontier requires declared approximately 10% TRAIN panel')
    if not all((source_hash, child_hash, coupling_identity)):
        raise ValueError('source, checkpoint and coupling identities required')
    grid = tuple(float(a) for a in frontier)
    if grid != FRONTIER:
        raise ValueError('frontier must be the declared fixed grid')
    denom = torch.as_tensor(source_core, dtype=torch.float64).detach().cpu().reshape(-1)
    if denom.numel() != 3 or not torch.isfinite(denom).all() or (denom <= 0).any():
        raise ValueError('three positive frozen SOURCE denominators required')
    if not all(math.isfinite(x) and x >= 0 for x in (source_inference_cost, child_inference_cost, blend_inference_cost)):
        raise ValueError('finite measured nonnegative inference costs required')
    mass = _mass(masses, truth.shape[0], truth.shape[1], truth.device)
    r0, cross, square = blend_risk_quadratic(source, child, truth, mass)
    r0, cross, square = (x.double().mean(dim=(0, 1)) for x in (r0, cross, square))
    scores = []
    for alpha in grid:
        risk = r0 + 2 * alpha * cross + alpha * alpha * square
        feasible = bool(torch.all(risk <= 1.1 * r0))
        core = torch.as_tensor(core_observer(source + alpha * (child - source)), dtype=torch.float64).detach().cpu().reshape(-1)
        if core.numel() != 3 or not torch.isfinite(core).all():
            raise ValueError('complete common observer must return three finite core scores')
        if alpha == 0 and not torch.allclose(core, denom, rtol=1e-5, atol=1e-8):
            raise ValueError('SOURCE denominators must match the common observer on this panel')
        cost = float(source_inference_cost if alpha == 0 else child_inference_cost if alpha == 1 else blend_inference_cost)
        scores.append({'alpha': alpha, 'risk': risk.tolist(), 'feasible': feasible,
                       'core': core.tolist(), 'utility': float((core / denom).sum()), 'inference_cost': cost})
    qualified = [row for row in scores if row['feasible']]
    if not qualified:
        raise ValueError('SOURCE must be feasible at alpha zero')
    selected = min(qualified, key=lambda row: (row['utility'], row['inference_cost'], row['alpha']))
    receipt = {'schema': 'car_alpha_frontier_v1', 'split': split, 'sample_ids': list(ids),
               'population': population, 'source_sha256': source_hash, 'child_sha256': child_hash,
               'coupling_identity': coupling_identity, 'source_core_denominators': denom.tolist(),
               'source_endpoints_sha256': _tensor_identity(source),
               'child_endpoints_sha256': _tensor_identity(child),
               'truth_sha256': _tensor_identity(truth), 'native_masses_sha256': _tensor_identity(mass),
               'frontier': list(grid), 'rows': scores, 'selected_alpha': selected['alpha'],
               'selection_rule': 'minimum sum core/source; ties lower inference cost then lower alpha'}
    return {**receipt, 'sha256': _identity(receipt)}

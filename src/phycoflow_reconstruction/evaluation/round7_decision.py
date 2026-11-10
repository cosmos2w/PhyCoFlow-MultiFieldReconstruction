"""Prospective Round 7 decisions on already saved, matched reductions.

This module performs no model evaluation. Ratios are fieldwise MSE/SOURCE,
not L2 ratios. Callers must bind inputs to the frozen observer and cohort.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

SCALE_UNITS = 0.1585698355665164
MAX_ATTEMPTS = 46214
MAX_QUALIFICATION = 64
MAX_REPLAY = 650
MAX_GPU_SECONDS = 36 * 3600


@dataclass(frozen=True)
class Observation:
    epoch: int
    scale: float
    state_delta: float  # change in frozen SOURCE units
    shape_delta: float
    q_mse: Mapping[str, float]  # primary member MSE / SOURCE MSE
    train_mse: Mapping[str, float] | None = None
    native_mse: Mapping[str, float] | None = None
    protective_stop: bool = False


@dataclass(frozen=True)
class Arm:
    name: str
    observations: Mapping[int, Observation]
    update_seconds: float
    unchanged_terms: bool = False


def _valid_ratios(ratios: Mapping[str, float] | None, limit: float) -> bool:
    return bool(ratios) and all(math.isfinite(v) and 0 <= v <= limit for v in ratios.values())


def _common(o: Observation, limit: float = .02, fidelity: float = 1.10) -> bool:
    return (not o.protective_stop and all(math.isfinite(x) for x in
            (o.scale, o.state_delta, o.shape_delta)) and o.state_delta <= limit
            and o.shape_delta <= limit and _valid_ratios(o.q_mse, fidelity))


def _endpoint(o: Observation, source_scale: float, *, efficacy: bool = True) -> bool:
    return (_common(o) and _valid_ratios(o.train_mse, 1.10)
            and _valid_ratios(o.native_mse, 1.10)
            and set(o.q_mse) == set(o.train_mse or {}) == set(o.native_mse or {})
            and (not efficacy or o.scale <= .99 * source_scale))


def eligible(arm: Arm, epoch: int, source_scale: float, window: Sequence[int]) -> bool:
    if not math.isfinite(source_scale) or source_scale <= 0:
        raise ValueError('positive finite VALIDATION SOURCE Scale required')
    o = arm.observations.get(epoch)
    return bool(o and o.epoch == epoch and _endpoint(o, source_scale) and len(window) == 3 and
                len(set(window)) == 3 and epoch in window and
                sum(arm.observations[e].epoch == e and _common(arm.observations[e]) and
                    arm.observations[e].scale <= source_scale + .02 * SCALE_UNITS
                    for e in window if e in arm.observations) >= 2)


def _safe_research(arm: Arm, source_scale: float) -> bool:
    o = arm.observations.get(500)
    p = arm.observations.get(300)
    if not o or not p or not _common(o, .05, 1.15):
        return False
    if not (_valid_ratios(o.train_mse, 1.15) and _valid_ratios(o.native_mse, 1.15)
            and set(o.q_mse) == set(o.train_mse or {}) == set(o.native_mse or {})
            and o.scale <= 1.02 * source_scale and _valid_ratios(p.q_mse, float('inf'))
            and set(p.q_mse) == set(o.q_mse)):
        return False
    return (p.scale - o.scale >= .005 * source_scale or
            (max(p.q_mse.values()) - max(o.q_mse.values()) >= .02 and o.scale <= p.scale))


def _safe_comparator(arm: Arm, source_scale: float) -> bool:
    o = arm.observations.get(500)
    return bool(o and _common(o, .05, 1.15) and _valid_ratios(o.train_mse, 1.15)
                and _valid_ratios(o.native_mse, 1.15)
                and set(o.q_mse) == set(o.train_mse or {}) == set(o.native_mse or {})
                and o.scale <= 1.02 * source_scale)


def _prefer(arms: Sequence[Arm], epoch: int, source_scale: float) -> Arm:
    best = min(a.observations[epoch].scale for a in arms)
    tied = [a for a in arms if a.observations[epoch].scale <= best + .005 * source_scale]
    if epoch == 1000:
        return min(tied, key=lambda a: (a.name != 'B', a.update_seconds, not a.unchanged_terms, a.name))
    return min(tied, key=lambda a: (a.update_seconds, not a.unchanged_terms, a.name != 'B', a.name))


def validate_field_coverage(arms: Sequence[Arm], expected_fields: Sequence[str]) -> None:
    """Require declared fields on every available support before making a decision."""
    expected = set(expected_fields)
    if not expected or len(expected) != len(expected_fields):
        raise ValueError('unique expected field names required')
    for arm in arms:
        for o in arm.observations.values():
            for support in (o.q_mse, o.train_mse, o.native_mse):
                if support is not None and set(support) != expected:
                    raise ValueError(f'incomplete field coverage: {arm.name} epoch {o.epoch}')


def decide_500(arms: Sequence[Arm], source_scale: float) -> dict:
    by_name = {a.name: a for a in arms}
    if 'B' not in by_name or len(by_name) != len(arms):
        raise ValueError('one uniquely named B arm required')
    good = [a for a in arms if eligible(a, 500, source_scale, (300, 400, 500))]
    if good:
        winner = _prefer(good, 500, source_scale)
        if winner.name == 'B':
            alternatives = [a for a in arms if a.name != 'B' and _safe_comparator(a, source_scale)]
            comparator = min(alternatives, key=lambda a: a.observations[500].scale).name if alternatives else None
            return {'mode': 'eligible', 'selected': 'B', 'continue': ['B'] + ([comparator] if comparator else [])}
        if by_name['B'].observations.get(500) is None or by_name['B'].observations[500].protective_stop:
            return {'mode': 'baseline_unsafe', 'selected': winner.name, 'continue': []}
        return {'mode': 'eligible', 'selected': winner.name, 'continue': ['B', winner.name]}
    research = [a for a in arms if a.name != 'B' and _safe_research(a, source_scale)]
    # B must meet the same safety boundaries; it need not exhibit the candidate trend.
    if research and _safe_comparator(by_name['B'], source_scale):
        winner = min(research, key=lambda a: a.observations[500].scale)
        return {'mode': 'research', 'selected': winner.name, 'continue': ['B', winner.name]}
    return {'mode': 'close', 'selected': 'SOURCE', 'continue': []}


def decide_1000(arms: Sequence[Arm], source_scale: float) -> dict:
    good = [a for a in arms if eligible(a, 1000, source_scale, (800, 900, 1000))]
    if not good:
        return {'selected': 'SOURCE', 'qualified': []}
    winner = _prefer(good, 1000, source_scale)
    return {'selected': winner.name, 'qualified': [a.name for a in good]}


def extension_allowed(arm: Arm, source_scale: float, *, fits_budget: bool) -> bool:
    a, b = arm.observations.get(800), arm.observations.get(1000)
    return bool(fits_budget and a and b and _endpoint(b, source_scale, efficacy=False)
                and b.scale <= source_scale + .02 * SCALE_UNITS
                and a.scale - b.scale >= .005 * source_scale)


def test_disposition(selected: str, loss_ratios: Mapping[str, float] | None,
                     native_ratios: Mapping[str, float] | None) -> str:
    if selected == 'SOURCE':
        return 'SOURCE'
    return selected if _valid_ratios(loss_ratios, 1.10) and _valid_ratios(native_ratios, 1.10) else 'SOURCE'


def gpu_seconds(intervals: Sequence[tuple[str, float, float]]) -> float:
    """Sum per-device unions; concurrent workers on a device count once."""
    groups: dict[str, list[tuple[float, float]]] = {}
    for device, start, end in intervals:
        if not device or not all(math.isfinite(x) for x in (start, end)) or end < start:
            raise ValueError('invalid occupied interval')
        groups.setdefault(device, []).append((start, end))
    total = 0.0
    for group in groups.values():
        last_start, last_end = sorted(group)[0]
        for start, end in sorted(group)[1:]:
            if start <= last_end:
                last_end = max(last_end, end)
            else:
                total += last_end - last_start
                last_start, last_end = start, end
        total += last_end - last_start
    return total


def validate_budget(*, qualification: int, replay: int, planned: int, intervals: Sequence[tuple[str, float, float]]) -> None:
    if min(qualification, replay, planned) < 0 or qualification > MAX_QUALIFICATION or replay > MAX_REPLAY or planned > 45500 or qualification + replay + planned > MAX_ATTEMPTS or gpu_seconds(intervals) > MAX_GPU_SECONDS:
        raise ValueError('Round 7 budget exceeded')


def extension_stop(previous: Observation, current: Observation, *,
                   preceding_check: Observation | None, other_views_material_gain: bool,
                   next_block_fits: bool) -> bool:
    """Apply fixed block stops; material-gain determination remains explicit."""
    if not next_block_fits or current.protective_stop:
        return True
    if current.scale - previous.scale > .02 * SCALE_UNITS and not other_views_material_gain:
        return True
    if preceding_check is not None:
        bad_now = {k for k, v in current.q_mse.items() if not math.isfinite(v) or v > 1.10}
        bad_before = {k for k, v in preceding_check.q_mse.items() if not math.isfinite(v) or v > 1.10}
        if bad_now & bad_before:
            return True
    return False


def choose_resource_design(*, three_arm_seconds: float, bt_seconds: float,
                           bp_seconds: float, available_seconds: float,
                           observer_qualified: bool, t_qualified: bool) -> list[str]:
    """Pre-learning forecast including protected continuation and final costs."""
    values = (three_arm_seconds, bt_seconds, bp_seconds, available_seconds)
    if not all(math.isfinite(v) and v >= 0 for v in values):
        raise ValueError('invalid resource forecast')
    if observer_qualified and t_qualified and three_arm_seconds <= available_seconds:
        return ['B', 'P', 'T']
    if observer_qualified and t_qualified and bt_seconds <= available_seconds:
        return ['B', 'T']
    if bp_seconds <= available_seconds:
        return ['B', 'P']
    raise ValueError('no matched design fits protected reserve')


def forecast_matched(arms: Sequence[str], update_seconds: Mapping[str, float], *,
                     validation_seconds: float, training_panel_seconds: float,
                     native_panel_seconds: float, recovery_attempts: int = 650) -> float:
    """Conservative full 500 and paired-1000 forecast, including final panels.

    Fixed-panel timings must be measured for the full authorized cohorts. The
    candidate pair is charged at the slowest candidate; no results are used.
    """
    if 'B' not in arms or len(arms) < 2 or len(set(arms)) != len(arms):
        raise ValueError('a baseline and at least one unique treatment are required')
    values = [update_seconds[a] for a in arms] + [validation_seconds, training_panel_seconds, native_panel_seconds]
    if any(not math.isfinite(v) or v <= 0 for v in values) or not 0 <= recovery_attempts <= 650:
        raise ValueError('positive measured times and bounded recovery are required')
    b = update_seconds['B']
    slowest = max(update_seconds[a] for a in arms if a != 'B')
    model = 6500 * (sum(update_seconds[a] for a in arms) + b + slowest)
    # Five 100-epoch checks per screening arm, six continuation checks including 750,
    # plus the final three-model TEST. TRAIN endpoints include a selected final check.
    panels = ((5 * len(arms) + 12 + 3) * validation_seconds
              + (len(arms) + 2 + 1) * training_panel_seconds
              + (len(arms) + 2 + 3) * native_panel_seconds)
    return model + panels + recovery_attempts * max(update_seconds[a] for a in arms)

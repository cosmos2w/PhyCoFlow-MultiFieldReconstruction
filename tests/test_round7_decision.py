from dataclasses import replace
import pytest
from phycoflow_reconstruction.evaluation.round7_decision import (
    Arm, Observation, decide_500, decide_1000, eligible, extension_allowed,
    gpu_seconds, test_disposition as disposition, validate_budget, SCALE_UNITS,
)


def obs(epoch, scale=.98, state=0, shape=0, q=1.1, train=1.1, native=1.1):
    return Observation(epoch, scale, state, shape, {'a': q}, {'a': train}, {'a': native})


def arm(name, scale=.98, **kwargs):
    return Arm(name, {e: obs(e, scale, **kwargs) for e in (300, 400, 500, 800, 900, 1000)}, 2, name == 'B')


def test_mse_guards_mature_window_and_supports():
    a = arm('B')
    assert eligible(a, 500, 1, (300, 400, 500))
    # 1.10 is an MSE ratio, not an L2 ratio.
    assert not eligible(arm('B', q=1.10001), 500, 1, (300, 400, 500))
    assert not eligible(arm('B', native=1.10001), 500, 1, (300, 400, 500))
    a.observations[300] = replace(a.observations[300], state_delta=.02001)
    assert eligible(a, 500, 1, (300, 400, 500))
    a.observations[400] = replace(a.observations[400], shape_delta=.02001)
    assert not eligible(a, 500, 1, (300, 400, 500))
    a = arm('B')
    a.observations[300] = replace(a.observations[300], scale=1 + .02001 * SCALE_UNITS)
    a.observations[400] = replace(a.observations[400], scale=1 + .02001 * SCALE_UNITS)
    assert not eligible(a, 500, 1, (300, 400, 500))


def test_b_tie_and_safe_comparator():
    b, p, t = arm('B', .98), arm('P', .976), arm('T', 1.01)
    assert decide_500([b, p, t], 1)['continue'] == ['B', 'P']
    assert decide_500([b, p, t], 1)['selected'] == 'B'
    assert decide_1000([b, p], 1)['selected'] == 'B'
    assert decide_500([b, arm('P', .974)], 1)['selected'] == 'P'
    cheaper = replace(p, update_seconds=1)
    assert decide_500([b, cheaper], 1)['selected'] == 'P'
    assert decide_1000([b, cheaper], 1)['selected'] == 'B'


def test_expected_fields_are_enforced():
    from phycoflow_reconstruction.evaluation.round7_decision import validate_field_coverage
    validate_field_coverage([arm('B')], ['a'])
    with pytest.raises(ValueError, match='incomplete field coverage'):
        validate_field_coverage([arm('B')], ['a', 'b'])


def test_research_requires_trend_and_baseline_safety():
    b, p = arm('B', 1.0), arm('P', .999)
    p.observations[300] = obs(300, 1.01)
    assert decide_500([b, p], 1)['mode'] == 'research'
    p.observations[300] = obs(300, 1.0)
    assert decide_500([b, p], 1)['mode'] == 'close'
    p.observations[300] = obs(300, 1.01)
    b.observations[500] = obs(500, 1, native=1.16)
    assert decide_500([b, p], 1)['mode'] == 'close'


def test_extension_fixed_scale_units_and_final_test():
    a = arm('B', 1 + .02 * SCALE_UNITS)
    a.observations[800] = obs(800, a.observations[1000].scale + .006)
    assert extension_allowed(a, 1, fits_budget=True)
    assert not extension_allowed(a, 1, fits_budget=False)
    a.observations[1000] = replace(a.observations[1000], scale=1 + .0201 * SCALE_UNITS)
    assert not extension_allowed(a, 1, fits_budget=True)
    assert disposition('P', {'a': 1.1}, {'a': 1.1}) == 'P'
    assert disposition('P', {'a': 1.11}, {'a': 1}) == 'SOURCE'
    assert disposition('P', {'a': 1}, None) == 'SOURCE'


def test_union_and_budget():
    assert gpu_seconds([('g0', 0, 3), ('g0', 1, 4), ('g1', 1, 2)]) == 5
    validate_budget(qualification=64, replay=650, planned=45500, intervals=[])
    with pytest.raises(ValueError):
        validate_budget(qualification=65, replay=0, planned=0, intervals=[])
    with pytest.raises(ValueError):
        validate_budget(qualification=0, replay=0, planned=0, intervals=[('g0', 0, 129601)])


def test_resource_forecast_and_extension_stop():
    from phycoflow_reconstruction.evaluation.round7_decision import choose_resource_design, extension_stop, forecast_matched
    assert choose_resource_design(three_arm_seconds=40, bt_seconds=30, bp_seconds=20, available_seconds=35, observer_qualified=True, t_qualified=True) == ['B', 'T']
    assert choose_resource_design(three_arm_seconds=40, bt_seconds=30, bp_seconds=20, available_seconds=35, observer_qualified=True, t_qualified=False) == ['B', 'P']
    with pytest.raises(ValueError):
        choose_resource_design(three_arm_seconds=40, bt_seconds=30, bp_seconds=20, available_seconds=19, observer_qualified=True, t_qualified=True)
    previous = obs(1000, .98)
    current = obs(1250, .98 + .021 * SCALE_UNITS)
    assert extension_stop(previous, current, preceding_check=None, other_views_material_gain=False, next_block_fits=True)
    assert not extension_stop(previous, current, preceding_check=None, other_views_material_gain=True, next_block_fits=True)
    assert extension_stop(previous, obs(1250, q=1.11), preceding_check=obs(1200, q=1.11), other_views_material_gain=True, next_block_fits=True)
    assert forecast_matched(['B', 'T'], {'B': 2, 'T': 3}, validation_seconds=10,
        training_panel_seconds=20, native_panel_seconds=30) == 6500 * (2 + 3 + 2 + 3) + (25 * 10 + 5 * 20 + 7 * 30) + 650 * 3

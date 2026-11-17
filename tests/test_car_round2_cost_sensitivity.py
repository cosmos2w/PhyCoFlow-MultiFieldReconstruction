import importlib.util
from pathlib import Path

import pytest

path = Path(__file__).resolve().parents[1]/'scripts/analysis/sensitivity_car_round2_costs.py'
spec = importlib.util.spec_from_file_location('sensitivity', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def inputs(family='senseiver'):
    prefix = 'FFM' if family == 'ffm' else 'Senseiver'
    full = f'{prefix}_Selected_Bank_Uniform_FullModel'
    residual = f'{prefix}_Selected_Bank_Residual_Repair'
    costs = {'schema':'car_round2_cost_reconciliation_v1','family':family,'complete':False,
             'missing':['Original Senseiver topology-stable allocator cost input timing missing'],
             'arms':{full:{'observed_serial_cost_seconds':100.}, residual:{'observed_serial_cost_seconds':109.}},
             'allocator_final':{'status':'measured','overhead_seconds':4.,'candidate_training_seconds':100.}}
    decision = {'schema':'car_round2_final_decision_v1','family':family,'crossings':{
        full:{'crossing_epoch':100,'crossing_cost_bounds_seconds':[100.,100.]},
        residual:{'crossing_epoch':250,'crossing_cost_bounds_seconds':[79.,79.]}}}
    clocks = {full:{'epochs':{'100':{'recorded_unix':200.}}}, residual:{'epochs':{'250':{'recorded_unix':200.}}}}
    sense = {'bound_not_measured':True,'lower_seconds':0.,'upper_seconds':6.717}
    sense_source = {'external_start_unix':100.,'external_result_unix':110.}
    ffm = {'bound_not_measured':True,'lower_seconds':0.,'upper_seconds':.334,
           'external_start_unix':100.,'external_result_unix':110.}
    return {'family':family,'costs':costs,'decision':decision,'clocks':clocks,
            'sense_bound':sense,'sense_source':sense_source,'ffm_bound':ffm}


def test_joint_endpoints_can_reverse_all_senseiver_cost_gates():
    result = module.sensitivity(**inputs())
    assert result['conditional'] and result['adoption_unchanged'] and not result['measured_cost_complete']
    assert result['gate_statuses'] == {'quality_cost_le_1_10':'sensitive', 'allocator_burden_le_0_05':'sensitive',
                                       'efficiency_cost_le_0_80':'sensitive'}
    assert result['endpoint_scenarios'][0]['quality_cost_le_1_10']['status'] == 'pass'
    assert result['endpoint_scenarios'][-1]['quality_cost_le_1_10']['status'] == 'fail'


def test_ffm_shared_amount_is_correlated_and_never_policy_overhead():
    args = inputs('ffm'); _full, res = args['costs']['arms'].values()
    res['observed_serial_cost_seconds'] = 109.99
    args['decision']['crossings'][next(iter(args['decision']['crossings']))]['crossing_cost_bounds_seconds']=[100.,100.]
    result = module.sensitivity(**args)
    low, high = result['endpoint_scenarios'][0], result['endpoint_scenarios'][1]
    assert low['quality_cost_le_1_10']['status'] == high['quality_cost_le_1_10']['status'] == 'pass'
    assert high['allocator_burden_le_0_05']['difference_seconds'] == [-1.,-1.]
    assert high['efficiency_cost_le_0_80']['difference_seconds'][0] == pytest.approx(-.9666)
    assert module.correlated_shared_prefixes(10.,20.,15.,15.,2.) == [(0.,0.), (1.,1.)]
    assert module.correlated_shared_prefixes(10.,20.,14.,16.,2.) == [(0.,0.), (0.,1.), (1.,1.)]


def test_unrelated_missing_or_missing_crossing_is_never_assumed_zero():
    args=inputs();args['costs']['missing'].append('unmeasured PDF montage')
    result=module.sensitivity(**args)
    assert result['unaddressed_missing']==['unmeasured PDF montage']
    assert set(result['gate_statuses'].values())=={'unavailable'}
    args=inputs();args['decision']['crossings'].clear()
    assert module.sensitivity(**args)['gate_statuses']['efficiency_cost_le_0_80']=='unavailable'
    args=inputs();args['clocks'].clear()
    assert module.sensitivity(**args)['gate_statuses']['efficiency_cost_le_0_80']=='unavailable'


def test_recorded_crossing_bounds_can_leave_threshold_unresolved():
    args=inputs();res=list(args['decision']['crossings'])[1]
    args['decision']['crossings'][res]['crossing_cost_bounds_seconds']=[79.,81.]
    assert module.sensitivity(**args)['endpoint_scenarios'][0]['efficiency_cost_le_0_80']['status']=='unresolved'


def test_invalid_or_measured_bound_is_rejected():
    args=inputs();args['ffm_bound']['bound_not_measured']=False
    with pytest.raises(ValueError,match='unmeasured'):module.sensitivity(**args)
    args=inputs();args['sense_bound']['upper_seconds']=-1
    with pytest.raises(ValueError,match='invalid source bounds'):module.sensitivity(**args)

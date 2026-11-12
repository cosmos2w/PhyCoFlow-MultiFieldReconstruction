import copy
from pathlib import Path
import pytest
import yaml
from phycoflow_reconstruction.config.validate import validate_config
from phycoflow_reconstruction.training.fidelity_budget import FidelityBudget


@pytest.fixture
def portable_config():
    from test_fidelity_budget_config import _config
    from test_multiscale_relation import config as relation_config
    config = _config()
    config['coherence']['families']['multiscale_spectrum'] = relation_config()
    config['coherence']['compute_budget']['query_policy'] = 'fixed_shared'
    config['optimization']['multitask']['fidelity_budget'] = {
        'relative_increase': .10, 'absolute_allowance': 0., 'denominator_floor': 1e-8}
    config['optimization']['multitask']['external_policy']['round7_diagnostics'] = False
    return config


@pytest.fixture
def actual_round8_config():
    path=Path(__file__).resolve().parents[1]/'cases/turbulent_combustion/runs/coherence_compute_round8/configs/qualification.yaml'
    if not path.exists(): pytest.skip('local resolved Round 8 qualification config unavailable')
    config=yaml.safe_load(path.read_text())
    assert 'relation_extension' in config['coherence']['families']['multiscale_spectrum']
    return config


def reference(config, *, names=None, relative=.10, absolute=0., floor=1e-8):
    names=tuple(names or config['dataset']['field_names'])
    return FidelityBudget.from_reference(names,[.1]*len(names),1.,relative_increase=relative,absolute_allowance=absolute,denominator_floor=floor).state_dict()


def test_portable_config_requires_and_accepts_checked_source_reference(portable_config):
    with pytest.raises(ValueError,match='reference_artifact'):
        validate_config(portable_config)
    validate_config(portable_config,precomputed_fidelity_reference=reference(portable_config))


def test_actual_resolved_config_with_precomputed_reference(actual_round8_config):
    validate_config(actual_round8_config,precomputed_fidelity_reference=reference(actual_round8_config))


@pytest.mark.parametrize('change,match', [
    ('field_order','field order'),('relative','relative_increase'),('absolute','absolute_allowance'),
    ('floor','denominator_floor'),('derived','invalid precomputed'),('both','mutually exclusive')])
def test_reject_wrong_or_conflicting_reference(portable_config,change,match):
    config=copy.deepcopy(portable_config)
    if change=='field_order':
        state=reference(config,names=list(reversed(config['dataset']['field_names'])))
    else:
        state=reference(config,relative=.2 if change=='relative' else .1,
                        absolute=.01 if change=='absolute' else 0.,floor=1e-5 if change=='floor' else 1e-8)
    if change=='derived': state['upper_bounds'][0] *= 2
    if change=='both':
        config['optimization']['multitask']['fidelity_budget'].update(reference_artifact='shared.json',reference_sha256='a'*64)
    with pytest.raises(ValueError,match=match): validate_config(config,precomputed_fidelity_reference=state)


def test_disabled_round7_diagnostics_is_explicit_boolean(portable_config):
    config=copy.deepcopy(portable_config)
    validate_config(config,precomputed_fidelity_reference=reference(config))
    config['optimization']['multitask']['external_policy']['round7_diagnostics']='false'
    with pytest.raises(TypeError,match='boolean'):
        validate_config(config,precomputed_fidelity_reference=reference(config))
    config['optimization']['multitask']['external_policy']['round7_diagnostics']=False
    config['optimization']['multitask']['external_policy']['round7_projection_scales']={}
    with pytest.raises(ValueError,match='requires round7_diagnostics'):
        validate_config(config,precomputed_fidelity_reference=reference(config))


def test_gram_extension_accepts_only_the_verified_reference(portable_config):
    config=copy.deepcopy(portable_config)
    family=config['coherence']['families']['multiscale_spectrum']
    family.pop('relation_extension')
    family['gram_extension']={'definition':'self_mutual_gram_v1','weight':1.,'std_floor':.001,'relative_std_floor':.05}
    with pytest.raises(ValueError,match='reference_artifact'):
        validate_config(config)
    validate_config(config,precomputed_fidelity_reference=reference(config))
    with pytest.raises(ValueError,match='relative_increase'):
        validate_config(config,precomputed_fidelity_reference=reference(config,relative=.2))

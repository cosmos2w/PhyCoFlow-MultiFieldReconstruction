import copy
import pytest
from scripts.posttraining.assess_car_round4 import ARMS, TEACHER, NATIVE, decision


def fixture():
    def candidate(regional, core=-0.06):
        return {'normalized_core_differences':{'state':core,'scale':0.,'shape':0.},
                'fidelity':{'primary_pass':True,'legacy_member_pass':True},
                'reused_tasks':{'regional_normalized_improvement':regional,'event_normalized_difference':0.}}
    return {'epoch':500, 'task_scope':'reused development tasks at endpoint',
            'candidates':{TEACHER:candidate(.10), NATIVE:candidate(.06)}}


def test_adopt_and_inclusive_cost_failure():
    assessment = fixture()
    assert decision(assessment, 100, 220)['round4_decision']['recommendation'] == TEACHER
    result = decision(assessment, 120, 220)['round4_decision']
    assert result['recommendation'] == NATIVE
    assert result['scientifically_useful_but_cost_failed']
    assert 'scientific_eligible' not in assessment['candidates'][TEACHER]


def test_tie_and_failed_field_fallback():
    assessment = fixture()
    assessment['candidates'][TEACHER]['reused_tasks']['regional_normalized_improvement'] = .07
    assert decision(assessment, 100, 220)['round4_decision']['recommendation'] == NATIVE
    assessment['candidates'][NATIVE]['fidelity']['primary_pass'] = False
    assert decision(assessment, 100, 220)['round4_decision']['recommendation'] == 'retain SOURCE/C8'


def test_review_does_not_authorize_terminal_decision():
    assessment = fixture()
    assessment['epoch'] = 100
    assessment['task_scope'] = 'both tasks excluded'
    with pytest.raises(ValueError): decision(assessment, 100, 220)
    with pytest.raises(ValueError): decision(fixture(), 100, 0)

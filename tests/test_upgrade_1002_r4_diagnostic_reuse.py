"""Exact diagnostic reuse contracts: preserved updates, graphs and telemetry.

CPU synthetic contracts only; never recalibrate SOURCE or load scientific data.
"""
import ast
import copy
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from phycoflow_reconstruction.training import gradient_balance as balance


@pytest.fixture(autouse=True)
def restore_public_cpu_rng():
    state = (torch.get_rng_state(), np.random.get_state(), random.getstate())
    yield
    torch.set_rng_state(state[0]); np.random.set_state(state[1]); random.setstate(state[2])


class Model(torch.nn.Module):
    def __init__(self, complex_parameters):
        super().__init__()
        self.x = torch.nn.Parameter(torch.tensor([.2, -.4], dtype=torch.float64))
        if complex_parameters:
            self.z = torch.nn.Parameter(torch.tensor([.3+.6j, -.2+.4j], dtype=torch.complex128))
        else:
            self.z = torch.nn.Parameter(torch.tensor([.3, -.2, .6, .4], dtype=torch.float64))
        self.unused = torch.nn.Parameter(torch.tensor([.7], dtype=torch.float64))
        self.forward_count = 0

    def forward(self):
        self.forward_count += 1
        parts = [self.x, self.z.real, self.z.imag] if self.z.is_complex() else [self.x, self.z]
        value = torch.cat(parts)
        return value + .01*torch.rand_like(value)

def flat(parameters, gradients=False):
    values = [p.grad if gradients else p.detach() for p in parameters]
    return torch.cat([(torch.view_as_real(v) if v.is_complex() else v).reshape(-1) for v in values]).clone()

def freeze(value):
    if isinstance(value, torch.Tensor): return value.detach().clone()
    if isinstance(value, dict): return {k: freeze(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return type(value)(freeze(v) for v in value)
    return copy.deepcopy(value)

def assert_equal(a, b):
    if isinstance(a, torch.Tensor): assert torch.equal(a, b)
    elif isinstance(a, np.ndarray): assert np.array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a: assert_equal(a[k], b[k])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b): assert_equal(x, y)
    else: assert a == b

def primed(complex_parameters):
    model = Model(complex_parameters)
    opt = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=.01)
    for p in model.parameters(): p.grad = torch.full_like(p, .07)
    opt.step(); opt.zero_grad(set_to_none=True)
    return freeze(model.state_dict()), freeze(opt.state_dict())

def losses(model, active_count):
    v = model()
    native = (v-.1).square().mean()
    endpoint = v.square().mean()
    families = {'A': .5*(v[0]+v[2]-.1).square(),
                'B': (v[1]-v[3]+.2).square(),
                'C': .3*((v[4]-.8).square()+(v[5]+.1).square())}
    zero_native_risk = v[4]-v[4].detach()
    zero_endpoint_risk = v[0]-v[0].detach()
    terms = {'total': (.2 if active_count == 2 else 0.)*zero_endpoint_risk,
             'CH4': 0.*endpoint, 'CO': 0.*endpoint, 'T': 0.*endpoint,
             'U_1': 0.*endpoint, 'p': 0.*endpoint,
             'native': (.3 if active_count >= 1 else 0.)*zero_native_risk}
    return native, endpoint, families, terms

def execute(module, state, optimizer_state, complex_parameters, *, adaptive=True,
            active_count=1, reuse=False, callback=False, explicit_none=False):
    torch.manual_seed(90261); np.random.seed(90261); random.seed(90261)
    model = Model(complex_parameters); model.load_state_dict(state)
    opt = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=.01)
    opt.load_state_dict(copy.deepcopy(optimizer_state))
    native, endpoint, families, terms = losses(model, active_count)
    params = list(model.parameters())
    assigned, calls = [], {'grad': 0}
    original_assign, original_grad = module._assign_flat_gradient, torch.autograd.grad
    def capture_assign(parameters, vector):
        assigned.append(vector.detach().clone()); return original_assign(parameters, vector)
    def count_grad(*a, **kw):
        calls['grad'] += 1; return original_grad(*a, **kw)
    module._assign_flat_gradient, torch.autograd.grad = capture_assign, count_grad
    try:
        if adaptive:
            active = tuple(name for name in terms if (name == 'native' and active_count >= 1)
                           or (name == 'total' and active_count == 2))
            extra = {'diagnostic_active_constraints': active, 'diagnostic_endpoint_loss': endpoint} if reuse else ({'diagnostic_active_constraints': None, 'diagnostic_endpoint_loss': None} if explicit_none else {})
            report = module.coherence_primal_dual_update(model, opt, families, sum(terms.values()),
                method='weighted_sum', grad_clip=.2, diagnostics=True, constraint_losses=terms,
                native_loss=native, execution='scalar', **extra)
        else:
            coherence = sum(families.values())
            extra = {}
            if explicit_none: extra['diagnostic_callback'] = None
            if callback:
                extra['diagnostic_callback'] = module.make_shared_family_diagnostic_callback(params, families,
                    data_weight=.1, coherence_weight=1.)
            report = module.two_objective_update(model, opt, native, coherence, mode='config',
                data_weight=.1, coherence_weight=1., grad_clip=.2, execution='scalar', diagnostics=True, **extra)
    finally:
        module._assign_flat_gradient, torch.autograd.grad = original_assign, original_grad
    return {'preclip': assigned[0], 'assigned': flat(params, gradients=True),
            'model': freeze(model.state_dict()), 'optimizer': freeze(opt.state_dict()),
            'displacement': flat(params)-flat([torch.nn.Parameter(state[name]) for name, _ in model.named_parameters()]),
            'rng': (torch.get_rng_state().clone(), np.random.get_state(), random.getstate()),
            'calls': calls['grad'], 'report': report, 'forward_count': model.forward_count}

def parity(a, b):
    for key in ['preclip', 'assigned', 'model', 'optimizer', 'displacement', 'rng']:
        assert_equal(a[key], b[key])
    assert a['forward_count'] == b['forward_count'] == 1
    return {'preclip_gradient_max_abs_difference': float((a['preclip']-b['preclip']).abs().max()),
            'postclip_gradient_max_abs_difference': float((a['assigned']-b['assigned']).abs().max()),
            'actual_displacement_max_abs_difference': float((a['displacement']-b['displacement']).abs().max()),
            'model_AdamW_RNG_bitwise_equal': True, 'model_existing_tolerance': 1e-6}

@pytest.mark.parametrize('complex_parameters', [False, True], ids=['real', 'complex_unused'])
@pytest.mark.parametrize('active_count', [0, 1, 2], ids=['zero_coefficients', 'active_at_zero_value', 'multiple_active'])
def test_pressure_reuse_preserves_primed_adamw_gradient_update_and_rng(complex_parameters, active_count):
    state, optimizer_state = primed(complex_parameters)
    oracle = execute(balance, state, optimizer_state, complex_parameters, active_count=active_count)
    actual = execute(balance, state, optimizer_state, complex_parameters, active_count=active_count, reuse=True)
    parity(oracle, actual)
    assert oracle['calls'] == 12 and actual['calls'] == (8 if active_count == 2 else 6)
    report = actual['report']
    assert report['gradient/fidelity/CH4/norm'] == report['update/actual_dot/fidelity/CH4'] == 0
    assert report['gradient/fidelity/CH4/native/cosine'] is None
    assert report['gradient/fidelity/CH4/native/cosine/undefined_reason'] == 'zero_gradient'
    assert report['gradient/endpoint.raw/norm'] > 0
    if active_count:
        assert report['gradient/fidelity/native/norm'] > 0  # coefficient>0, scalar value exactly0
    for key, reference in oracle['report'].items():
        if key in report and (key.endswith('/norm') or key.endswith('/cosine') or key.startswith('update/actual_dot/')):
            if reference is None: assert report[key] is None
            else: assert report[key] == pytest.approx(reference)


@pytest.mark.parametrize('complex_parameters', [False, True])
def test_shared_family_diagnostics_reuse_graphs_without_changing_adamw(complex_parameters):
    state, optimizer_state = primed(complex_parameters)
    oracle = execute(balance, state, optimizer_state, complex_parameters, adaptive=False)
    actual = execute(balance, state, optimizer_state, complex_parameters, adaptive=False, callback=True)
    parity(oracle, actual)
    assert oracle['calls'] == 2 and actual['calls'] == 5  # cached native/combined +three existing family graphs
    diagnostic = actual['report']['family_gradient_diagnostics']
    assert diagnostic['diagnostic_forwards'] == 0 and diagnostic['diagnostic_family_backward_requests'] == 3
    assert diagnostic['SOURCE_calibration_changed'] is False
    assert 'reconstructed' in diagnostic['diagnostic_gradient_identity']
    torch.manual_seed(90261)
    model = Model(complex_parameters); model.load_state_dict(state)
    native, _, family_losses, _ = losses(model, 1)
    parameters = tuple(model.parameters())
    assert diagnostic['native_data_gradient_norm'] == pytest.approx(float(balance._flat_gradient(native, parameters).double().norm()))
    for name, loss in family_losses.items():
        assert diagnostic['family_gradient_norms'][name] == pytest.approx(float(balance._flat_gradient(loss, parameters).double().norm()))


@pytest.mark.parametrize('adaptive', [False, True])
def test_optional_diagnostic_defaults_keep_legacy_api_and_reports(adaptive):
    state, optimizer_state = primed(True)
    default = execute(balance, state, optimizer_state, True, adaptive=adaptive)
    explicit = execute(balance, state, optimizer_state, True, adaptive=adaptive, explicit_none=True)
    parity(default, explicit)
    assert default['report'] == explicit['report']
    assert default['calls'] == explicit['calls'] == (12 if adaptive else 2)


@pytest.mark.parametrize('complex_parameters', [False, True])
def test_shared_caller_keeps_unweighted_family_graphs_with_nonunit_outer_weights(complex_parameters):
    # Exercise the actual caller extraction on shared context, without invoking
    # case/model/data machinery. Single-family combined results were weighted;
    # shared per-family results are already unweighted and must retain identity.
    source = Path(balance.__file__).with_name('post_training.py')
    assignments = [n for n in ast.walk(ast.parse(source.read_text())) if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == 'raw_family_losses' for t in n.targets)]
    assert len(assignments) == 1
    model = Model(complex_parameters)
    native, _, raw, _ = losses(model, 1)
    outer = {'A': .3, 'B': 2.7, 'C': 4.1}
    context = {'family_results': {n: SimpleNamespace(scalar_loss=v) for n,v in raw.items()}}
    extracted = eval(compile(ast.fix_missing_locations(ast.Expression(assignments[0].value)), '<shared-caller>', 'eval'),
                     {'step_context': context, 'families': {n: SimpleNamespace(family_weight=w) for n,w in outer.items()}})
    assert all(extracted[n] is raw[n] for n in raw)
    parameters = tuple(model.parameters())
    native_gradient = balance._flat_gradient(.7*native, parameters)
    combined_gradient = balance._flat_gradient(1.3*sum(outer[n]*v for n,v in raw.items()), parameters)
    diagnostic = balance.make_shared_family_diagnostic_callback(parameters, extracted, data_weight=.7, coherence_weight=1.3)(native_gradient, combined_gradient)
    assert model.forward_count == 1
    for name, value in raw.items():
        assert diagnostic['family_gradient_norms'][name] == pytest.approx(float(balance._flat_gradient(value, parameters).double().norm()))


def test_packed_diagnostics_match_real_complex_euclidean_oracle_and_undefined_zero():
    # An independent real/imaginary dot-product oracle avoids the historical
    # calibration complex-to-double cast; SOURCE calibration remains untouched.
    z = torch.tensor([1+2j, -.7+.5j], dtype=torch.complex128)
    a = torch.view_as_real(z).reshape(-1)
    b = torch.tensor([.2, -.3, 1.1, .4], dtype=torch.float64)
    vectors = {'a': a, 'b': b, 'alias': a, 'zero': None}
    packed = balance._packed_diagnostic_gram(vectors)
    assert packed['unique_vectors'] == 2
    for left in ['a','b','alias']:
        for right in ['a','b','alias']:
            dot = float(torch.dot(vectors[left], vectors[right]))
            assert packed['dot'](left,right) == pytest.approx(dot)
            expected = dot/(float(vectors[left].norm())*float(vectors[right].norm()))
            assert packed['cosines'][(left,right)] == pytest.approx(expected)
    assert packed['norms']['zero'] == packed['dot']('zero','a') == 0
    assert packed['cosines'][('zero','a')] is None
    delta = torch.tensor([.1,.2,-.1,.3], dtype=torch.float64)
    products = balance._packed_actual_displacement_products(packed,delta)
    assert products['zero'] == 0
    for name in ['a','b','alias']: assert products[name] == pytest.approx(float(torch.dot(vectors[name],delta)))


def test_invalid_activity_and_reconstructed_weight_contracts_fail_before_update():
    for active in [('native','native'), ('unknown',)]:
        model = Model(True); optimizer = torch.optim.AdamW(model.parameters(),lr=.003)
        before = freeze(model.state_dict())
        native, endpoint, family_losses, terms = losses(model,1)
        with pytest.raises(ValueError,match='unique declared term names'):
            balance.coherence_primal_dual_update(model,optimizer,family_losses,sum(terms.values()),method='weighted_sum',
                grad_clip=.2,diagnostics=True,constraint_losses=terms,native_loss=native,execution='scalar',
                diagnostic_active_constraints=active,diagnostic_endpoint_loss=endpoint)
        assert_equal(before,model.state_dict()); assert not optimizer.state
    with pytest.raises(ValueError,match='positive finite weights'):
        balance.make_shared_family_diagnostic_callback(tuple(Model(True).parameters()),{},data_weight=0.,coherence_weight=1.)

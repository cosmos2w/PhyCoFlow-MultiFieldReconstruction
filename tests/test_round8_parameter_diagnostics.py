import pytest
import torch
from torch import nn

from phycoflow_reconstruction.training.fidelity_budget import ExternalPolicy, FidelityBudget
from phycoflow_reconstruction.training.multitask import OuterAggregator


@pytest.mark.parametrize('constant_zero_channel', [False, True])
def test_actual_adamw_step_records_eight_components_and_family_products(constant_zero_channel):
    pytest.importorskip('torchjd'); pytest.importorskip('cooper')
    model = nn.Linear(4, 1, bias=False)
    optimizer = torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=0.05)
    x = model.weight.reshape(-1)
    blocks = {'state.marginal':.25*x[0], 'state.mixed':.75*x[0],
              'scale.signed':.1*x[1], 'scale.joint':.2*x[1], 'scale.log_power_self':.3*x[1], 'scale.rho':.4*x[1],
              'shape.self':.6*x[2], 'shape.mutual':.4*x[2]}
    if constant_zero_channel:
        blocks['shape.self'], blocks['shape.mutual'] = x[2], torch.tensor(0.)
    groups={'global_distribution':'state.', 'multiscale_spectrum':'scale.', 'topology':'shape.'}
    losses={family:sum(value for name,value in blocks.items() if name.startswith(prefix)) for family,prefix in groups.items()}
    risk=torch.stack([(x[3]+2).square()]); before=model.weight.detach().clone()
    policy=ExternalPolicy({'method':'lagrangian'},FidelityBudget.from_reference(['a'],[1.],1.),'cpu')
    info=policy.update_with_coherence_aggregation(model,optimizer,losses,risk,1.,OuterAggregator({'method':'torchjd_config'}),diagnostic_context={'schema':'round8_actual_step_v1','blocks':blocks,'identity':{'epoch':100}})
    report=info['round8_actual_step']; delta=(model.weight.detach()-before).reshape(-1).double()
    assert report['extra_gradient_evaluations']==8 and report['identity']['epoch']==100
    for i, family in enumerate(groups):
        assert report['family_negative_gradient_dot_displacement'][family] == pytest.approx(float(-delta[i]))
    for key, factor, i in [('state.marginal',.25,0),('scale.log_power_self',.3,1),('shape.mutual',0. if constant_zero_channel else .4,2)]:
        assert report['block_gradient_norms'][key] == pytest.approx(factor)
        assert report['block_negative_gradient_dot_displacement'][key] == pytest.approx(float(-factor*delta[i]), abs=1e-10)
    assert report['actual_displacement_norm'] == pytest.approx(float(delta.norm()))


def test_control_state_blocks_match_full_bank_gradient():
    from test_conditional_quantile_fair import family, ctx
    from phycoflow_reconstruction.training.round8_runtime import Round8Runtime
    torch.manual_seed(5)
    f=family(); a=torch.randn(1,4,2,dtype=torch.double,requires_grad=True)
    b=torch.randn(1,4,2,dtype=torch.double,requires_grad=True); target=torch.randn_like(a)
    context=ctx(conditional_draws=(b,))
    exact=f(a,target,context=context).scalar_loss
    cached,blocks=Round8Runtime.__new__(Round8Runtime)._state_control(f,a,target,context)
    torch.testing.assert_close(cached.scalar_loss,exact,atol=1e-10,rtol=1e-8)
    left=torch.autograd.grad(exact,(a,b),retain_graph=True)
    right=torch.autograd.grad(sum(blocks.values()),(a,b))
    for x,y in zip(left,right): torch.testing.assert_close(x,y,atol=1e-12,rtol=1e-10)


def test_control_shape_and_scale_live_blocks_reproduce_family_gradients():
    from test_topology_conditional_fair import setup
    from test_multiscale_relation import config
    from test_multiscale_gram_extension import _fitted, _states
    from phycoflow_reconstruction.training.round8_runtime import Round8Runtime
    runtime=Round8Runtime.__new__(Round8Runtime)
    runtime.mode='control'; runtime.manages_queries=False; runtime.epoch=100; runtime.step=99; runtime.steps_per_epoch=1
    runtime.fields=('a','b'); runtime.blocks={}
    shape, x, y, coords=setup()
    first=x.detach().requires_grad_(); second=(x*.8+.1).detach().requires_grad_()
    actual=runtime.score_family('topology',shape,first,y,coordinates=coords,context={'conditional_draws':(second,)})
    part=runtime.blocks['shape.self']+runtime.blocks['shape.mutual']
    a=torch.autograd.grad(actual.scalar_loss,(first,second),retain_graph=True)
    b=torch.autograd.grad(part,(first,second))
    for left,right in zip(a,b): torch.testing.assert_close(left,right,rtol=1e-6,atol=1e-7)
    states=_states(); scale, points, _ = _fitted(config(),states)
    first=states[:2].clone().requires_grad_(); second=states[2:4].clone().requires_grad_(); target=states[4:6]
    actual=runtime.score_family('multiscale_spectrum',scale,first,target,coordinates=points[None].expand(2,-1,-1),context={'conditional_draws':(second,)})
    part=sum(runtime.blocks[k] for k in ('scale.signed','scale.joint','scale.log_power_self','scale.rho'))
    a=torch.autograd.grad(actual.scalar_loss,(first,second),retain_graph=True)
    b=torch.autograd.grad(part,(first,second))
    for left,right in zip(a,b): torch.testing.assert_close(left,right,rtol=1e-6,atol=1e-7)


def test_candidate_diagnostic_uses_frozen_block_coefficients_and_family_scales():
    from types import SimpleNamespace
    from phycoflow_reconstruction.training.round8_runtime import Round8Runtime
    from phycoflow_reconstruction.training.round8_calibration import BLOCKS
    from phycoflow_reconstruction.coherence.round8_queries import SharedQuerySampler
    runtime=Round8Runtime.__new__(Round8Runtime)
    runtime.mode='candidate';runtime.manages_queries=True;runtime.epoch=500;runtime.step=499;runtime.steps_per_epoch=1
    runtime.fields=('a','b');runtime.artifacts={'test':1};runtime.sampler=SharedQuerySampler(runtime.fields);runtime.batch=runtime.sampler.draw()
    runtime.families={name:SimpleNamespace(family_weight=2.) for name in ('global_distribution','multiscale_spectrum','topology')}
    keys=[f'{f}.{b}' for f, names in BLOCKS.items() for b in names]
    runtime.coefficients={k:1.5 for k in keys};runtime.calibration=SimpleNamespace(active={k:True for k in keys},units={k:1. for k in keys})
    x=torch.tensor(2.,requires_grad=True);runtime.blocks={k:x for k in keys}
    names=('global_distribution','multiscale_spectrum','topology')
    result=runtime.diagnostic_context({k:x for k in names},family_scales={k:3. for k in names})
    assert result['blocks']['state.marginal'].item()==12.
    assert result['blocks']['scale.signed'].item()==18.
    assert result['blocks']['shape.self'].item()==18.
    runtime.epoch=501
    assert runtime.diagnostic_context({},family_scales={}) is None

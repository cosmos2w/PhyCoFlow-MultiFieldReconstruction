import itertools
import pytest
import torch

from phycoflow_reconstruction.training.round8_policy import physical_risk, quadratic, alpha_cap, blend, select_train_policy
from phycoflow_reconstruction.training.round8_calibration import BLOCKS, InternalCalibration, state_channel_contribution
from phycoflow_reconstruction.coherence.round8_state import select_directions, fair_score, common_distance


def test_physical_estimator_and_quadratic():
    torch.manual_seed(1)
    source = torch.randn(1, 2, 2, 4, dtype=torch.float64)
    endpoint = source + torch.randn_like(source) * .5
    target = torch.randn(1, 2, 4, dtype=torch.float64)
    weights = torch.tensor([.1, .2, .3, .4], dtype=torch.float64)
    native = physical_risk(source, target, weights)
    subsets = list(itertools.combinations(range(4), 2))
    estimates = [physical_risk(source[..., idx], target[..., idx], weights[list(idx)], design='without_replacement', probability=.5) for idx in subsets]
    assert torch.allclose(torch.stack(estimates).mean(0), native)
    p = torch.tensor([.1, .2, .3, .4], dtype=torch.float64)
    single = [physical_risk(source[..., [i]], target[..., [i]], weights[[i]], design='with_replacement', probability=p[i]) for i in range(4)]
    assert torch.allclose(sum(p[i] * single[i] for i in range(4)), native)
    r, b, a = quadratic(source, endpoint, target, weights)
    for alpha in (0., .25, .5, 1.):
        assert torch.allclose(physical_risk(source + alpha * (endpoint-source), target, weights), r+2*alpha*b+alpha**2*a)
    cap = alpha_cap(r,b,a)
    assert torch.all(r+2*cap*b+cap**2*a <= 1.08*r + 1e-12)
    assert alpha_cap(torch.tensor([0.]), torch.tensor([1.]), torch.tensor([1.])) == 0
    assert alpha_cap(torch.tensor([1.]), torch.tensor([0.]), torch.tensor([0.])) == 1


def test_coupling_and_train_selection():
    identity = {k:'x' for k in ('conditions','observations','latent','prior_buffers','solver','normalizer','support','clamp')}
    x = torch.zeros(1,2,1,2)
    y = torch.ones_like(x); y[...,0] = 0
    assert torch.equal(blend(x,y,.5,source_identity=identity,endpoint_identity=identity)[...,0], x[...,0])
    with pytest.raises(ValueError):
        blend(x,y,.5,source_identity=identity,endpoint_identity={**identity,'prior_buffers':'other'})
    scores = {'scale':1.,'shape':1.,'state':1.}
    p = select_train_policy(1.,torch.ones(1),{.25:torch.ones(1),.5:torch.ones(1)},scores,{'scale':1.,'shape':.8,'state':1.},{.25:{'scale':1.,'shape':.9,'state':1.},.5:{'scale':1.,'shape':.9,'state':1.}},scores)
    assert p.alpha == .5 and p.families == ('shape',)


def test_state_bank_fair_oracle_and_chain_rule():
    torch.manual_seed(2)
    dirs = torch.randn(32,5,dtype=torch.float64)
    bank = select_directions(dirs,torch.arange(32.),list('abcde'),17)
    assert len(set(bank['selected_ids'])) == 8
    assert not select_directions(torch.ones(2,1),torch.ones(2),['a'],1)['mixed_active']
    m = torch.tensor([[0.,1.],[2.,1.]],dtype=torch.float64)
    ref = torch.tensor([1.,1.],dtype=torch.float64)
    score, attraction, separation = fair_score(m,ref)
    assert torch.allclose(score, torch.tensor(0.,dtype=torch.float64))
    assert attraction == 1 and separation == 1
    x = torch.tensor([2.,3.],dtype=torch.float64,requires_grad=True)
    s, t = x[0]**2, x[1]**2
    total = 2*s+3*t
    actual = torch.autograd.grad(torch.sqrt(total), x, retain_graph=True)[0]
    calibrated = state_channel_contribution(s,total,2.) + state_channel_contribution(t,total,3.)
    assert torch.allclose(torch.autograd.grad(calibrated,x)[0],actual)
    zero = torch.tensor([0.],requires_grad=True)
    d = common_distance(zero,None,zero,None)
    assert torch.autograd.grad(d.sum(),zero)[0].item() == 0


def test_calibration_resume_freeze():
    from phycoflow_reconstruction.training.round8_calibration import gradient_norm
    assert gradient_norm(torch.tensor(0.), (torch.nn.Parameter(torch.tensor(1.)),)) == 0.
    keys = [f'{f}.{b}' for f,bs in BLOCKS.items() for b in bs]
    active = {k:k != 'state.mixed' for k in keys}
    cal = InternalCalibration(dict.fromkeys(keys,1.),dict.fromkeys(keys,1.),active)
    norms = {k: float(i+1) for i,k in enumerate(keys) if active[k]}
    for epoch in (1,3,6): cal.observe(epoch,norms)
    resumed = InternalCalibration.from_state_dict(cal.state_dict())
    for epoch in (8,10):
        cal.observe(epoch,norms); resumed.observe(epoch,norms)
    assert cal.freeze(10) == resumed.freeze(10)
    assert InternalCalibration.from_state_dict(cal.state_dict()).coefficients == cal.coefficients
    with pytest.raises(ValueError): cal.observe(10,norms)


def test_state_adapter_common_embedding_and_probe_gradients():
    from phycoflow_reconstruction.coherence.round8_state import score_family, fit_source_hardness
    torch.manual_seed(9)
    directions = torch.tensor([[1., 0.], [0., 1.], [1., 1.], [1., -1.]], dtype=torch.float64)
    members = torch.randn(2, 2, 6, 2, dtype=torch.float64, requires_grad=True)
    reference = torch.randn(2, 6, 2, dtype=torch.float64)
    masses = torch.tensor([[.1,.2,.1,.2,.1,.3],[.2,.2,.1,.1,.1,.3]],dtype=torch.float64)
    hardness = fit_source_hardness([(members.detach().transpose(0,1), reference, masses, masses)],directions)
    bank = select_directions(directions,hardness,['a','b'],4)
    result, channels = score_family(members,reference,masses,masses,bank,marginal_coefficient=1.3,mixed_coefficient=.7)
    total_grad = torch.autograd.grad(result.scalar_loss,members,retain_graph=True)[0]
    block_grad = torch.autograd.grad(channels['state.marginal']+channels['state.mixed'],members)[0]
    assert torch.allclose(total_grad,block_grad,atol=1e-12)
    assert result.diagnostics['score_definition'] == bank['version']


def test_exact_common_embedding_matches_weighted_w2():
    from phycoflow_reconstruction.coherence.round8_state import exact_common_embeddings
    from phycoflow_reconstruction.coherence.base import weighted_w2_columns
    torch.manual_seed(32)
    x = torch.randn(1,7,3,dtype=torch.float64)
    y = torch.randn(1,5,3,dtype=torch.float64)
    z = torch.randn(1,4,3,dtype=torch.float64)
    wx, wy, wz = (torch.rand(1,n,dtype=torch.float64) for n in (7,5,4))
    dirs = torch.randn(8,3,dtype=torch.float64); dirs = dirs / dirs.norm(dim=-1,keepdim=True)
    channel = torch.tensor([.1,.2,.7],dtype=torch.float64)
    reps = exact_common_embeddings([x,y,z],[wx,wy,wz],dirs,channel)
    for i,j,aa,bb,ma,mb in [(0,1,x,y,wx,wy),(0,2,x,z,wx,wz),(1,2,y,z,wy,wz)]:
        marginal = (reps[i][0]-reps[j][0]).square().sum()
        mixed = (reps[i][1]-reps[j][1]).square().sum()
        assert torch.allclose(marginal,(weighted_w2_columns(aa[0],bb[0],ma[0],mb[0])*channel).sum(),atol=1e-12)
        assert torch.allclose(mixed,weighted_w2_columns(aa[0]@dirs.T,bb[0]@dirs.T,ma[0],mb[0]).mean(),atol=1e-12)


def test_full_bank_matches_existing_conditional_state():
    from phycoflow_reconstruction.coherence import build_coherence_family
    from phycoflow_reconstruction.coherence.round8_state import score_family, VERSION
    from phycoflow_reconstruction.config import load_config
    from phycoflow_reconstruction.contracts import DataSpec
    from phycoflow_reconstruction.data.normalization import FieldNormalizer
    config = load_config('configs/coherence/state_conditional_fair.yaml')['coherence']['families']['global_distribution']
    family = build_coherence_family('global_distribution',config,DataSpec(('a','b','c'),('1','1','1'),2,(8,)),FieldNormalizer.identity(3))
    torch.manual_seed(192)
    members = torch.randn(2,2,8,3,dtype=torch.float64,requires_grad=True)
    reference = torch.randn(2,8,3,dtype=torch.float64)
    w = torch.rand(2,8,dtype=torch.float64)
    old = family(members[0],reference,context={'generated_weights':w,'reference_weights':w,'conditional_draws':[members[1]]})
    dirs = family.components_by_key['cross_joint_topk_swd'].directions
    bank = {'version':VERSION,'directions':dirs,'sha256':'test','mixed_active':True}
    new, _ = score_family(members,reference,w,w,bank,channel_weights=family.components_by_key['self_marginal_w2'].channel_weights, marginal_weight=family.component_weights['self_marginal_w2'], mixed_weight=family.component_weights['cross_joint_topk_swd'])
    assert torch.allclose(old.per_sample_cost,new.per_sample_cost,atol=1e-7,rtol=1e-7)
    g1 = torch.autograd.grad(old.scalar_loss,members,retain_graph=True)[0]
    g2 = torch.autograd.grad(new.scalar_loss,members)[0]
    assert torch.allclose(g1,g2,atol=1e-7,rtol=1e-7)


def test_full_bank_state_gradient_parity_with_distinct_nonuniform_endpoint_measures():
    from test_conditional_quantile_fair import family
    from phycoflow_reconstruction.coherence.round8_state import score_family, VERSION
    torch.manual_seed(451)
    original=family()
    members=torch.randn(2,1,4,2,dtype=torch.double,requires_grad=True)
    reference=torch.randn(1,4,2,dtype=torch.double)
    generated_mass=torch.tensor([[.02,.18,.3,.5]],dtype=torch.double)
    reference_mass=torch.tensor([[.4,.1,.2,.3]],dtype=torch.double)
    old=original(members[0],reference,context={'generated_weights':generated_mass,'reference_weights':reference_mass,'conditional_draws':(members[1],)})
    bank={'version':VERSION,'directions':original.components_by_key['cross_joint_topk_swd'].directions,'sha256':'test','mixed_active':True}
    new,blocks=score_family(members,reference,generated_mass,reference_mass,bank,channel_weights=original.components_by_key['self_marginal_w2'].channel_weights,marginal_weight=original.component_weights['self_marginal_w2'],mixed_weight=original.component_weights['cross_joint_topk_swd'])
    torch.testing.assert_close(old.scalar_loss,new.scalar_loss,atol=1e-10,rtol=1e-8)
    a=torch.autograd.grad(old.scalar_loss,members,retain_graph=True)[0]
    b=torch.autograd.grad(new.scalar_loss,members,retain_graph=True)[0]
    c=torch.autograd.grad(sum(blocks.values()),members)[0]
    torch.testing.assert_close(a,b,atol=1e-9,rtol=1e-8)
    torch.testing.assert_close(b,c,atol=1e-10,rtol=1e-10)


def test_runtime_full32_cached_state_matches_retained_value_and_parameter_gradient():
    from test_conditional_quantile_fair import config, family
    from phycoflow_reconstruction.training.round8_runtime import Round8Runtime
    cfg=config(3); cfg['components']['cross']['directions']=32
    original=family(3,cfg)
    torch.manual_seed(214)
    theta=torch.randn(3,3,dtype=torch.double,requires_grad=True)
    x=torch.randn(2,6,3,dtype=torch.double); y=torch.randn(2,6,3,dtype=torch.double)
    first=x@theta; second=y@theta; reference=torch.randn(2,6,3,dtype=torch.double)
    w=torch.rand(2,6,dtype=torch.double);v=torch.rand(2,6,dtype=torch.double)
    context={'generated_weights':w,'reference_weights':v,'conditional_draws':(second,)}
    old=original(first,reference,context=context)
    runtime=Round8Runtime.__new__(Round8Runtime)
    cached,blocks=runtime._state_control(original,first,reference,context)
    assert cached.diagnostics['retained_direction_count']==32
    assert cached.diagnostics['score_definition']==original.definition
    assert cached.diagnostics['computation_identity']=='shared_exact_weighted_quantile_cache_v1'
    torch.testing.assert_close(old.per_sample_cost,cached.per_sample_cost,rtol=1e-8,atol=1e-9)
    go=torch.autograd.grad(old.scalar_loss,theta,retain_graph=True)[0]
    gn=torch.autograd.grad(cached.scalar_loss,theta,retain_graph=True)[0]
    gb=torch.autograd.grad(sum(blocks.values()),theta)[0]
    torch.testing.assert_close(go,gn,rtol=1e-8,atol=1e-9)
    torch.testing.assert_close(gn,gb,rtol=1e-9,atol=1e-10)


def test_alpha_cap_rounds_to_feasible_strict_guard():
    import numpy as np
    from phycoflow_reconstruction.training.round8_policy import alpha_cap
    rng = np.random.default_rng(472)
    for _ in range(2000):
        r = np.exp(rng.uniform(-8, 8, 5))
        a = np.exp(rng.uniform(-8, 8, 5))
        b = np.sqrt(r * a) * rng.uniform(-1, 1, 5)
        cap = alpha_cap(r, b, a)
        assert np.all(r + 2 * cap * b + cap * cap * a <= 1.08 * r)

import copy

import pytest
import torch
from torch import nn

from phycoflow_reconstruction.training.multitask import OuterAggregator, preferred_config_direction
from phycoflow_reconstruction.training.round7_step_diagnostics import (
    displacement, projected_benefits, shadow_adamw, snapshot,
)
from phycoflow_reconstruction.training.gradient_balance import _assign_flat_gradient
from phycoflow_reconstruction.training.gradients import stable_clip_grad_norm_


def test_preference_is_norm_matched_and_records_applied_products():
    pytest.importorskip('torchjd')
    g = torch.tensor([[1., .1, .0], [.2, 1., .1], [.1, .2, 1.]], dtype=torch.float64)
    h0, _ = OuterAggregator({'method': 'torchjd_config'})(g)
    h, info = preferred_config_direction(g, h0, torch.tensor([1., 1.2, 1.], dtype=torch.float64))
    torch.testing.assert_close(h.norm(), h0.norm())
    products = g @ h
    assert (products > torch.tensor(info['positive_product_tolerance'])).all()
    torch.testing.assert_close(products / g.norm(dim=1), (products / g.norm(dim=1))[0] * torch.tensor([1., 1.2, 1.], dtype=torch.float64))


def test_shadow_exactly_replays_adamw_and_preserves_live_state_and_rng():
    p = nn.Parameter(torch.tensor([1., -2., 3.], dtype=torch.float32))
    opt = torch.optim.AdamW([p], lr=5e-5, weight_decay=1e-6)
    p.grad = torch.tensor([.2, -.3, .1]); opt.step(); opt.zero_grad()
    original, state, rng = p.detach().clone(), copy.deepcopy(opt.state_dict()), torch.get_rng_state().clone()
    direction = torch.tensor([2., -1., .5], dtype=torch.float64)
    delta, _ = shadow_adamw(opt, [p], direction, 1.)
    assert torch.equal(original, p.detach()) and torch.equal(rng, torch.get_rng_state())
    assert torch.equal(state['state'][0]['exp_avg'], opt.state_dict()['state'][0]['exp_avg'])
    before = snapshot([p]); _assign_flat_gradient([p], direction); stable_clip_grad_norm_([p], 1.); opt.step()
    assert torch.equal(delta, displacement([p], before))
    assert projected_benefits(direction[None], delta, [1.])[0] > 0


def test_policy_round7_probe_records_real_and_shadow_steps_and_recovers_count():
    pytest.importorskip('torchjd'); pytest.importorskip('cooper')
    from phycoflow_reconstruction.training.fidelity_budget import FidelityBudget, ExternalPolicy
    model = nn.Linear(4, 1, bias=False)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-6)
    x = model.weight.reshape(-1)
    losses = dict(zip(['global_distribution', 'multiscale_spectrum', 'topology'], [x[0] + 2, x[1] + 2, x[2] + 2]))
    risk = torch.stack([(x[3] + 2).square()])
    components = {k: c * x[1] for k, c in {'signed': .4, 'joint': .25, 'self': .2, 'rho': .15,
                                             'joint_attraction': .3, 'joint_separation': -.05}.items()}
    policy = ExternalPolicy({'method': 'lagrangian', 'round7_diagnostics': True},
                            FidelityBudget.from_reference(['T'], [1.], 1.), 'cpu')
    result = policy.update_with_coherence_aggregation(model, opt, losses, risk, 1.,
        OuterAggregator({'method': 'torchjd_config'}), diagnostic_context={
            'projection_scales': [1., 1., 1.], 'components': components, 'temperature_field': 'T',
            'identity': {'sample_ids': ['sample-1']}})
    data = result['round7_actual_step']
    assert data['shadow_count_total'] == 2
    assert set(data['shadows']) == {'preferred', 'scale_only'}
    assert data['identity']['sample_ids'] == ['sample-1']
    assert data['extra_gradient_evaluations'] == 8
    restored = ExternalPolicy({'method': 'lagrangian', 'round7_diagnostics': True}, policy.reference, 'cpu')
    restored.load_state_dict(policy.state_dict())
    assert restored.round7_shadow_count == 2


def test_parent_linked_binding_rejects_nonshape_changes(tmp_path):
    import hashlib, json
    from phycoflow_reconstruction.training.fidelity_budget import FidelityBudget
    from phycoflow_reconstruction.training.fidelity_reference import serialize_shared_reference
    from phycoflow_reconstruction.training.round7_recipe_binding import bind_child, SOURCE_SHA, C0, ORDER
    budget = FidelityBudget.from_reference(['T'], [1.], C0)
    binding = {'source_hashes': {'checkpoint': SOURCE_SHA}, 'field_names': ['T'],
               'fidelity_budget': {'relative_increase': .05, 'absolute_allowance': 0., 'denominator_floor': 1e-8},
               'family_scales': {'global_distribution': 1., 'multiscale_spectrum': 1., 'topology': 1.},
               'baseline_config_anchor': {'coherence': {'families': {'topology': {'persistence': {'definition': 'finite_primary_v1'}}}}}}
    raw = serialize_shared_reference({'schema': 'shared_fidelity_reference_v1', 'binding': binding,
        'reference': budget.state_dict(), 'origin': {'calibration_report_sha256': 'a' * 64}})
    path = tmp_path / 'parent.json'; path.write_bytes(raw); sha = hashlib.sha256(raw).hexdigest()
    pref = {'family_order': ORDER, 'vector': [1., 1.2, 1.], 'norm_match': True}
    _, read, record = bind_child(str(path), sha, binding, binding, arm='P', preference=pref)
    assert read == raw and record['reference'] == budget.state_dict()
    child = copy.deepcopy(binding)
    child['baseline_config_anchor']['coherence']['families']['topology']['persistence']['definition'] = 'conditional_fixed_transform_fair_v1'
    child['family_scales']['topology'] = 2.
    bind_child(str(path), sha, binding, child, arm='T', preference=None)
    for key, value in [('resolution', 128), ('weight', .4), ('query_bank', ['new'])]:
        changed_shape = copy.deepcopy(child)
        changed_shape['baseline_config_anchor']['coherence']['families']['topology']['persistence'][key] = value
        with pytest.raises(ValueError): bind_child(str(path), sha, binding, changed_shape, arm='T', preference=None)
    child['family_scales']['multiscale_spectrum'] = 2.
    with pytest.raises(ValueError): bind_child(str(path), sha, binding, child, arm='T', preference=None)


def test_scale_live_parts_match_true_provider_gradients():
    from test_multiscale_relation import config
    from test_multiscale_gram_extension import _fitted, _states
    from phycoflow_reconstruction.coherence.families.multiscale_relation import SELF_PATH, RHO_PATH
    from phycoflow_reconstruction.training.round7_step_diagnostics import component_gradients
    from phycoflow_reconstruction.training.gradient_balance import _flat_gradient
    states = _states(); family, coords, _ = _fitted(config(), states)
    x = nn.Parameter(states[0:2].clone()); y = nn.Parameter(states[2:4].clone())
    capture = {}
    result = family(x, states[4:6], coordinates=coords[None].expand(2, -1, -1),
                    context={'conditional_draws': (y,), 'round7_scale_components': capture})
    components = result.component_results
    capture.update({
        'signed': components['multiscale_spectrum.signed_band_energy_score.conditional'].scalar_loss,
        'joint': .25 * components['multiscale_spectrum.joint_spectral_mean_energy_score.conditional'].scalar_loss,
        'self': .2 * components[SELF_PATH].scalar_loss, 'rho': .3 * components[RHO_PATH].scalar_loss})
    gradients = component_gradients(capture, [x, y], _flat_gradient(result.scalar_loss, [x, y]))
    assert all(torch.isfinite(g).all() for g in gradients.values())

"""CPU contracts for the versioned conditional Shape observer."""
from copy import deepcopy
import pytest
import torch
pytest.importorskip('gudhi')
from test_topology_finite_primary import config, family, coords, provenance


def setup():
    settings = config()
    settings['persistence']['definition'] = 'conditional_fixed_transform_fair_v1'
    model = family(settings)
    gen = torch.Generator().manual_seed(15)
    y = torch.randn(2, 16, 2, generator=gen)
    x = y + .2 * torch.randn(y.shape, generator=gen)
    points = coords(2)
    target = model.fair_prepare_grid(y, coordinates=points)
    source = model.fair_prepare_grid(x, coordinates=points)
    obj = model.spatial_objective
    obj.accumulate_transform(target[:1])
    obj.accumulate_transform(target[1:])
    obj.freeze_transform(floors=[1e-4, 1e-4], provenance={'split': 'train', 'sample_ids':['train:0','train:1']})
    obj.accumulate_calibration(source[:1], target[:1])
    obj.accumulate_calibration(source[1:], target[1:])
    obj.freeze_calibration(provenance=provenance(2))
    return model, x, y, points


def test_fair_identity_symmetry_and_member_gradients():
    model, x, y, points = setup()
    first = x.detach().requires_grad_()
    second = (x.detach() * .8 + .1).requires_grad_()
    def score(a, b):
        return model(a, y, coordinates=points, context={'second_generated': b})
    result = score(first, second)
    reverse = score(second, first)
    torch.testing.assert_close(result.scalar_loss, reverse.scalar_loss)
    assert score(y, y).scalar_loss == 0
    g1, g2 = torch.autograd.grad(result.scalar_loss, (first, second))
    assert torch.isfinite(g1).all() and torch.isfinite(g2).all()
    assert g1.abs().sum() > 0 and g2.abs().sum() > 0
    assert result.diagnostics['predicted_endpoint_filtrations_per_condition'] == 32
    torch.testing.assert_close(torch.tensor(result.diagnostics['attraction_per_sample']) - torch.tensor(result.diagnostics['separation_per_sample']), result.per_sample_cost.detach())


def test_versioned_restore_and_fixed_domain():
    model, x, y, points = setup()
    artifact = deepcopy(model.state_artifact())
    restored = family(artifact['config'])
    restored.load_state_artifact(artifact)
    actual = restored(x, y, coordinates=points, context={'second_generated': x})
    expect = model(x, y, coordinates=points, context={'second_generated': x})
    torch.testing.assert_close(actual.scalar_loss, expect.scalar_loss, rtol=0, atol=0)
    bad = deepcopy(artifact)
    bad['transform']['scale'][0] *= 2
    with pytest.raises(ValueError, match='transform artifact contents'):
        family(artifact['config']).load_state_artifact(bad)
    with pytest.raises(ValueError, match='identity mismatch'):
        family().load_state_artifact(artifact)
    mask = torch.ones(4, 4, dtype=torch.bool)
    mask[:, 1] = False
    obj = model.spatial_objective
    grid = model.fair_prepare_grid(y, coordinates=points)
    with pytest.raises(ValueError, match='fixed-domain raster mask'):
        obj(grid, grid, second_generated=grid, valid_mask=mask)


def test_selected_complete_and_duplicate_weights_reuse_diagrams(monkeypatch):
    from dataclasses import replace
    from phycoflow_reconstruction.training.adaptive_coverage import QueryDraw, QuerySelection
    from phycoflow_reconstruction.coherence.families.topology import conditional_fair
    model, x, y, points = setup()
    obj = model.spatial_objective
    a = model.fair_prepare_grid(x, coordinates=points).requires_grad_()
    b = (a.detach() * .8).requires_grad_()
    z = model.fair_prepare_grid(y, coordinates=points)
    queries = []
    for component, _, ids, signs, line in obj.units:
        kind = 'self' if component.startswith('self') else 'mutual'
        fields = tuple(obj.fields[i] for i in ids)
        weight = 1 / 4 if line is None else float(obj.line_weights[line]) / 4
        queries.append(QueryDraw((kind, fields, signs, line), kind, fields, signs, line, 1., weight, 1, weight))
    dense = obj(a, z, second_generated=b)
    selected = obj(a, z, second_generated=b, query_selection=QuerySelection(tuple(queries),4,12))
    torch.testing.assert_close(dense.scalar_loss, selected.scalar_loss)
    for grad_dense, grad_selected in zip(torch.autograd.grad(dense.scalar_loss, (a,b)), torch.autograd.grad(selected.scalar_loss, (a,b))):
        torch.testing.assert_close(grad_dense, grad_selected)
    calls = []
    original = conditional_fair.cubical_diagrams
    def counting(tensor, **kwargs):
        calls.append(tensor.shape[0])
        return original(tensor, **kwargs)
    monkeypatch.setattr(conditional_fair, 'cubical_diagrams', counting)
    q = queries[0]
    # Multiplicity is carried by the existing estimator weight; the query is extracted once.
    twice = replace(q, multiplicity=2, estimator_weight=2*q.estimator_weight)
    subset = obj(a, z, second_generated=b, query_selection=QuerySelection((twice,), 2, 0))
    single = obj(a, z, second_generated=b, query_selection=QuerySelection((q,), 1, 0))
    torch.testing.assert_close(subset.scalar_loss, 2 * single.scalar_loss)
    assert calls == [2, 2, 2, 2, 2, 2]  # target and both members, batch size two


def test_old_artifact_source_binding_and_draw_routing():
    from phycoflow_reconstruction.coherence.families.topology.family import _ROUND6_SOURCE
    from phycoflow_reconstruction.training.coherence_sampling import _family_context, _conditional_draw_count
    model, x, y, points = setup()
    assert _conditional_draw_count({'topology':model}) == 2
    context = _family_context(model, {}, [x])
    routed = model(x, y, coordinates=points, context=context)
    explicit = model(x, y, coordinates=points, context={'second_generated':x})
    torch.testing.assert_close(routed.scalar_loss, explicit.scalar_loss)
    old = family()
    old.fit_calibration(x, y, coordinates=points, provenance=provenance(2))
    artifact = deepcopy(old.state_artifact())
    artifact['scientific_source']['source_sha256'].update(_ROUND6_SOURCE)
    restored = family()
    restored.load_state_artifact(artifact)
    torch.testing.assert_close(old(x,y,coordinates=points).scalar_loss, restored(x,y,coordinates=points).scalar_loss, rtol=0, atol=0)


def test_finite_and_essential_distance_domain_contract():
    from phycoflow_reconstruction.coherence.families.topology.persistence import Diagram
    model, _, _, _ = setup()
    obj = model.spatial_objective
    def diagram(bars, births):
        return Diagram(finite=torch.tensor(bars, dtype=torch.float32).reshape(-1, 2),
                       essential=torch.tensor(births, dtype=torch.float32))
    a = diagram([[0., 2.]], [0.])
    b = diagram([[0., 1.]], [2.])
    ab = obj._distance([a, a], [b, b], (1, 1, 4, 4), 16)
    ba = obj._distance([b, b], [a, a], (1, 1, 4, 4), 16)
    torch.testing.assert_close(ab, ba)
    assert ab[0, 0, 0, 0] > 0 and ab[0, 0, 0, 1] == 2
    with pytest.raises(ValueError, match='essential class counts differ'):
        obj._distance([a], [diagram([], [0., 1.])], (1, 1, 4, 4), 16)

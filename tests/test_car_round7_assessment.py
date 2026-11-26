import json
from types import SimpleNamespace

import pytest

from scripts.posttraining.assess_car_round7 import generate, ref, tasks, verify_bank


def test_banks_bind_support_panel_and_source(tmp_path):
    path = tmp_path / 'receipt.json'
    payload = {'status':'completed', 'split':'validation', 'sample_ids':['condition'], 'members':4,
               'panel_sha256':'panel', 'source_sha256':'source', 'source_bank_sha256':'native-source',
               'support':'selected', 'files_sha256':['row']}
    path.write_text(json.dumps(payload))
    spec = {'split':'validation', 'panel':{'sha256':'panel'}, 'source_checkpoint':{'sha256':'source'},
            'source_native_receipt':{'sha256':'native-source'}}
    panel = {'sample_ids':['condition']}
    assert verify_bank(path, spec, panel, {}, 'selected')[1] == payload
    with pytest.raises(ValueError, match='incompatible'):
        verify_bank(path, spec, panel, {}, 'native')
    payload['source_bank_sha256'] = 'different'
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match='incompatible'):
        verify_bank(path, spec, panel, {}, 'selected')


def test_unsealed_task_release_rejected_before_scoring(tmp_path):
    source = tmp_path / 'source.json'; source.write_text('{}')
    child = tmp_path / 'child.json'
    child.write_text(json.dumps({'status':'completed', 'split':'validation','sample_ids':['condition'],
        'members':4,'panel_sha256':'panel','source_sha256':'source','source_bank_sha256':ref(source)['sha256'],
        'support':'native','files_sha256':['row'],'checkpoint_sha256':'checkpoint'}))
    decision = tmp_path / 'decision.json'; decision.write_text(json.dumps({'status':'draft'}))
    spec = {'split':'validation','panel':{'sha256':'panel'},'source_checkpoint':{'sha256':'source'},
            'source_native_receipt':ref(source)}
    args = SimpleNamespace(child_native=child, decision=decision)
    with pytest.raises(ValueError, match='sealed'):
        tasks(args,spec,{'sample_ids':['condition']},{})


def test_generate_requires_lease_before_creating_bank(tmp_path, monkeypatch):
    monkeypatch.delenv('CAR_ROUND7_LEASE_TOKEN', raising=False)
    with pytest.raises(ValueError, match='lease token'):
        generate(SimpleNamespace(output=tmp_path / 'bank'), {}, {}, {})
    assert not (tmp_path / 'bank').exists()


def test_shape_components_reconstruct_compiled_score():
    import torch
    from test_frozen_coherence_observer import make_observer
    observer, members, _, _ = make_observer('conditional')
    with torch.no_grad():
        default = observer.score_tensors(members)
        detailed = observer.score_tensors(members, include_components=True)
    assert set(default) == {'state', 'scale', 'shape'}
    for key in default:
        torch.testing.assert_close(detailed[key], default[key])
    torch.testing.assert_close(detailed['shape.self'] + detailed['shape.mutual'], detailed['shape'])
    for kind in ('self', 'mutual'):
        reconstructed = sum(detailed[f'shape.{kind}.{category}.attraction'] - detailed[f'shape.{kind}.{category}.separation']
                            for category in ('finite', 'essential'))
        torch.testing.assert_close(reconstructed, detailed[f'shape.{kind}'])
    with pytest.raises(ValueError, match='compiled inference'):
        observer.score_tensors(members, compiled=False, include_components=True)


def test_compiled_target_round_trip_and_binding(tmp_path, monkeypatch):
    import torch
    from phycoflow_reconstruction.evaluation.frozen_coherence import FrozenCoherenceObserver
    from test_frozen_coherence_observer import make_observer
    original, members, families, artifact = make_observer('conditional')
    target = original.export_compiled_target()
    path = tmp_path / 'target.pt'; torch.save(target, path)
    target = torch.load(path, weights_only=True)
    def fail(*args, **kwargs):
        raise AssertionError('target diagrams were recomputed')
    monkeypatch.setattr(type(original.adapter.shape), 'compile_reference', fail)
    kwargs = dict(expected_provenance=artifact['provenance'], reference=original.reference,
                  coordinates=original.coordinates, sample_ids=original.receipt['sample_ids'],
                  member_weights=original.member_weights, reference_weights=original.reference_weights,
                  mode='conditional', query_seed=original.receipt['query_seed'], query_batches=len(original.batches))
    restored = FrozenCoherenceObserver(artifact, families, compiled_target=target, **kwargs)
    with torch.no_grad():
        old = original.score_tensors(members, include_components=True)
        new = restored.score_tensors(members, include_components=True)
    for key in old: torch.testing.assert_close(old[key], new[key])
    target['scale'][0].add_(1)
    with pytest.raises(ValueError, match='cache changed'):
        FrozenCoherenceObserver(artifact, families, compiled_target=target, **kwargs)


@pytest.mark.parametrize('member_count', [1, 2, 4])
def test_shape_default_matches_literal_parent_value_and_gradient(member_count):
    import ast
    import hashlib
    import subprocess
    import torch
    from phycoflow_reconstruction.coherence.observer_compatibility import _INTERMEDIATE_SHAPE
    from phycoflow_reconstruction.coherence.query_runtime import _shape_grid
    from test_frozen_coherence_observer import make_observer
    path = 'src/phycoflow_reconstruction/coherence/shape_compute.py'
    parent = subprocess.check_output(['git', 'show', f'b5bbf8f:{path}'])
    assert hashlib.sha256(parent).hexdigest() == _INTERMEDIATE_SHAPE
    parsed = ast.parse(parent)
    cls = next(item for item in parsed.body if isinstance(item, ast.ClassDef) and item.name == 'LazyPooledShape')
    method = next(item for item in cls.body if isinstance(item, ast.FunctionDef) and item.name == 'score_compiled')
    namespace = {'torch':torch}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), '<literal parent score_compiled>', 'exec'), namespace)
    original = namespace['score_compiled']
    observer, members, _, _ = make_observer('conditional')
    if member_count == 4: members = torch.cat((members, members + .173), 0)
    members = members[:member_count].clone().requires_grad_()
    family = observer.families['topology']
    grids = torch.stack([_shape_grid(family, member, observer.reference) for member in members])
    mask = family.domain_mask if family.native_masked else None
    args = (grids, observer.shape_queries, observer.shape_reference)
    old = original(observer.adapter.shape, *args, valid_mask=mask)
    new = observer.adapter.shape.score_compiled(*args, valid_mask=mask)
    detailed = observer.adapter.shape.score_compiled(*args, valid_mask=mask, include_components=True)
    for kind in ('self','mutual'):
        torch.testing.assert_close(new[kind], old[kind], rtol=0, atol=0)
        old_grad = torch.autograd.grad(old[kind].sum(), members, retain_graph=True)[0]
        new_grad = torch.autograd.grad(new[kind].sum(), members, retain_graph=True)[0]
        torch.testing.assert_close(new_grad, old_grad, rtol=0, atol=0)
        reconstructed = sum(detailed[f'{kind}.{category}.attraction'] - detailed[f'{kind}.{category}.separation']
                            for category in ('finite','essential'))
        # Separate A/R reductions reorder float32 arithmetic.
        torch.testing.assert_close(reconstructed, new[kind], rtol=2e-6, atol=2e-7)


def test_pooled_source_allows_only_three_pinned_shape_revisions():
    from copy import deepcopy
    from phycoflow_reconstruction.coherence.families.topology.family import _pooled_source
    from phycoflow_reconstruction.coherence.observer_compatibility import _INTERMEDIATE_SHAPE, _PAIRS, compatible_pooled_source
    current = _pooled_source()
    assert current['source_sha256']['shape_compute.py'] == _PAIRS['pooled']['shape_compute.py'][1]
    for revision in (*_PAIRS['pooled']['shape_compute.py'], _INTERMEDIATE_SHAPE):
        saved = deepcopy(current); saved['source_sha256']['shape_compute.py'] = revision
        assert compatible_pooled_source(saved,current)
    unknown = deepcopy(current); unknown['source_sha256']['shape_compute.py'] = '0' * 64
    assert not compatible_pooled_source(unknown,current)
    wrong_current = deepcopy(current); wrong_current['source_sha256']['shape_compute.py'] = '0' * 64
    old = deepcopy(current); old['source_sha256']['shape_compute.py'] = _INTERMEDIATE_SHAPE
    assert not compatible_pooled_source(old,wrong_current)

import pytest

from phycoflow_reconstruction.evaluation.adaptive_campaign import (
    CAP_SECONDS,
    RESERVE_SECONDS,
    begin,
    end,
    forecast,
    immutable_json,
    panel,
    spent,
)


def test_panel_is_reproducible_and_limited():
    ids = [f'id_{i:04}' for i in range(1000)]
    a = panel(ids, 'train', 8)
    assert len(a['sample_ids']) == 100 and len(set(a['sample_ids'])) == 100
    assert a == panel(list(reversed(ids)), 'train', 8)
    assert a['members'] == 4 and a['sensor_count'] == 256
    with pytest.raises(ValueError):
        panel(ids, 'test', 8)


def test_additive_device_occupancy_and_closeout_reserve():
    state = {'closed': [], 'active': {}}
    begin(state, 'a', 'GPU-a', 'training', now=0)
    begin(state, 'b', 'GPU-b', 'training', now=10)
    assert spent(state, now=20) == 30
    with pytest.raises(ValueError):
        begin(state, 'c', 'GPU-a', 'training', now=20)
    end(state, 'a', now=30)
    end(state, 'b', now=30)
    assert spent(state, now=30) == 50
    assert forecast(state, 1, 0, 0, now=30)['fits']
    assert not forecast(state, (CAP_SECONDS - RESERVE_SECONDS) / 26000, 0, 0, now=30)['fits']


def test_immutable_identity(tmp_path):
    path = tmp_path / 'receipt.json'
    immutable_json(path, {'id': 1})
    immutable_json(path, {'id': 1})
    with pytest.raises(ValueError):
        immutable_json(path, {'id': 2})


def test_execution_snapshot_is_content_bound(tmp_path):
    from phycoflow_reconstruction.evaluation.adaptive_campaign import freeze_execution_code
    repo = tmp_path / 'repo'
    (repo / 'src/pkg').mkdir(parents=True)
    (repo / 'scripts').mkdir()
    source = repo / 'src/pkg/module.py'
    source.write_text('VALUE = 1\n')
    (repo / 'scripts/run.py').write_text('import pkg.module\n')
    first = freeze_execution_code(repo, tmp_path / 'execution', ['scripts/run.py'])
    assert freeze_execution_code(repo, tmp_path / 'execution', ['scripts/run.py']) == first
    source.write_text('VALUE = 2\n')
    second = freeze_execution_code(repo, tmp_path / 'execution', ['scripts/run.py'])
    assert second != first
    assert (first / 'src/pkg/module.py').read_text() == 'VALUE = 1\n'


def test_owner_constraints_exclude_device_and_pause_backend(tmp_path, monkeypatch):
    from phycoflow_reconstruction.evaluation.adaptive_campaign import (
        atomic_json,
        enforce_execution_constraints,
    )
    runtime = tmp_path / 'runtime.json'
    atomic_json(runtime, {'schema':'car_execution_runtime_v1', 'gpu_uuid':'GPU-one',
                          'excluded_gpu_uuids':['GPU-zero'], 'paused_source_families':['edm']})
    monkeypatch.setenv('CAR_EXECUTION_RUNTIME', str(runtime))
    enforce_execution_constraints(tmp_path, 'GPU-one', 'ffm')
    with pytest.raises(ValueError, match='excluded'):
        enforce_execution_constraints(tmp_path, 'GPU-zero', 'ffm')
    with pytest.raises(ValueError, match='backend paused'):
        enforce_execution_constraints(tmp_path, 'GPU-one', 'edm')


def test_snapshot_validates_identity_and_never_changes_source(tmp_path, monkeypatch):
    import torch

    import phycoflow_reconstruction.evaluation.adaptive_campaign as campaign
    source = tmp_path / 'old'
    (source / 'checkpoints').mkdir(parents=True)
    checkpoint = source / 'checkpoints' / 'last.pt'
    torch.save({'model': {}, 'training_aux_state': {'model_ema': {}}, 'global_step': 126,
                'model_name': 'sample', 'config_sha256': 'a'}, checkpoint)
    for name in ('resolved_config.yaml', 'run_manifest.json', 'status.json'):
        (source / name).write_text('{}')
    source_hash = campaign.digest(checkpoint)
    monkeypatch.setattr(campaign, 'source_ownership', lambda _: [])
    result = campaign.snapshot_source(source, tmp_path / 'snapshot', source_hash, 2, 126)
    assert result['global_step'] == 126
    assert campaign.digest(checkpoint) == source_hash
    assert campaign.digest(tmp_path / 'snapshot/checkpoints/last.pt') == source_hash
    with pytest.raises(ValueError, match='hash mismatch'):
        campaign.snapshot_source(source, tmp_path / 'another', 'wrong', 2, 126)


def test_budget_stop_includes_active_time_and_margin(tmp_path, monkeypatch):
    import phycoflow_reconstruction.evaluation.adaptive_campaign as campaign
    path = tmp_path / 'ledger.json'
    campaign.atomic_json(path, {'active': {}, 'closed': [{'seconds': CAP_SECONDS - RESERVE_SECONDS - 20}]})
    assert not campaign.budget_stop(path, safety_seconds=10)
    assert campaign.budget_stop(path, safety_seconds=20)


def test_budget_pause_resume_matches_uninterrupted(tmp_path):
    import json
    from pathlib import Path

    import torch
    from helpers.coherence import _base_config, _post_config, _write_fixture

    from phycoflow_reconstruction.config import load_config
    from phycoflow_reconstruction.config.load import deep_merge
    from phycoflow_reconstruction.training.base_training import run_base_training
    from phycoflow_reconstruction.training.post_training import run_post_training
    from phycoflow_reconstruction.training.run_store import load_project_checkpoint
    pytest.importorskip('gudhi')
    data = tmp_path / 'fixture.h5'
    _write_fixture(data)
    base = _base_config(data)
    base['dataset'].update(coordinate_dim=2)
    base['model'] = {'name': 'coordinate_mlp', 'hidden_dim': 8, 'fourier_bands': 2}
    source = run_base_training(base, case_dir=tmp_path / 'case')
    post = _post_config(data, source)
    post['model'] = base['model']
    post['dataset'].update(coordinate_dim=2, physical_measure='native_rectilinear_area')
    post['optimization'].update(epochs=12, steps_per_epoch=1, sampling='full_pass')
    post['coherence']['families'] = {}
    post = deep_merge(post, load_config(Path('configs/defaults/scalable_anchored_coherence.yaml')))
    post['coherence']['compute_budget'].update(point_count=16, batch_size=1)
    post['coherence']['families']['multiscale_spectrum']['bands']['count'] = 1
    post['coherence']['families']['topology']['geometry']['grid_shape'] = [4, 4]
    post['coherence']['families']['topology']['persistence']['projections'] = 4
    post['evaluation'].update(query_points=16, batch_size=1, split='train', preview={'enabled': False})
    post['observation_consistency'] = {'mode': 'hard', 'final_clamp': True}
    post['checkpointing'] = {'every_epochs': 1, 'epochs': [9, 12],
                             'panel_milestones_only': True, 'selection_metric': 'reconstruction_mse'}
    post['optimization']['multitask'] = {
        'family_scalarization': {'method': 'fixed_sum'}, 'outer_aggregation': None,
        'external_policy': {'method': 'lagrangian', 'vector_reduction': 'sum',
                            'lagrangian': {'dual_lr': .01, 'penalty_interval_updates': 1}},
        'fidelity_budget': {'metric': 'physical_member_mse_model_units',
                            'calibration_batches': 2, 'calibration_draws': 2},
    }
    post['runtime']['progress'] = False
    child = run_post_training(post, case_dir=tmp_path / 'case', max_steps=12,
                              budget_stop_requested=lambda step, run_dir: step == 9)
    assert json.loads((child / 'status.json').read_text())['status'] == 'budget_paused'
    assert not (child / 'evaluation/after.json').exists()
    assert load_project_checkpoint(child / 'checkpoints/epoch_009.pt')['global_step'] == 9
    assert not (child / 'evaluation/train_core_epoch_009.json').exists()
    from phycoflow_reconstruction.training.run_store import file_sha256
    paused_manifest = json.loads((child / 'run_manifest.json').read_text())
    for name, expected in paused_manifest['coherence_family_state_sha256s'].items():
        path = child / 'artifacts' / f'{name}_family.pt'
        assert file_sha256(path) == expected
        saved = torch.load(path, map_location='cpu', weights_only=True)
        assert saved['frozen_training_context']['split'] == 'train'
        assert saved['frozen_training_context']['field_names'] == list(base['dataset']['field_names'])
    journal = child / 'metrics/training_coupling.jsonl'
    with journal.open('a') as stream:
        stream.write(json.dumps({'step': 10, 'stale_future_record': True}) + '\n')
    run_post_training(post, case_dir=tmp_path / 'case', resume=child, max_steps=3)
    assert json.loads((child / 'evaluation/train_core_epoch_009.json').read_text())['global_step'] == 9
    whole = run_post_training(post, case_dir=tmp_path / 'case', max_steps=12)
    a = load_project_checkpoint(child / 'checkpoints/last.pt')
    b = load_project_checkpoint(whole / 'checkpoints/last.pt')
    def equal(x, y):
        if isinstance(x, torch.Tensor):
            assert torch.equal(x, y)
        elif isinstance(x, dict):
            assert x.keys() == y.keys()
            for key in x:
                equal(x[key], y[key])
        elif isinstance(x, (list, tuple)):
            assert len(x) == len(y)
            for first, second in zip(x, y):
                equal(first, second)
        else:
            assert x == y
    for key in ('model', 'optimizer', 'rng_state', 'scalable_runtime', 'external_policy'):
        equal(a[key], b[key])
    resumed_coupling = [json.loads(line) for line in journal.read_text().splitlines()]
    whole_coupling = [json.loads(line) for line in (whole / 'metrics/training_coupling.jsonl').read_text().splitlines()]
    assert resumed_coupling == whole_coupling
    assert len(resumed_coupling) == 12 and all(len(row['member_seeds']) == 1 for row in resumed_coupling)
    assert list((journal.parent / 'recovery').glob('training_coupling.*.jsonl'))
    final_manifest = json.loads((child / 'run_manifest.json').read_text())
    for name, expected in final_manifest['coherence_family_state_sha256s'].items():
        path = child / 'artifacts' / f'{name}_family.pt'
        assert file_sha256(path) == expected
        assert torch.load(path, map_location='cpu', weights_only=True)['frozen_training_context']['split'] == 'train'


def test_assessment_amendment_checks_inventory_and_preserves_training_identity(tmp_path, monkeypatch):
    import hashlib
    import json

    from phycoflow_reconstruction.evaluation import adaptive_campaign as c

    def capsule(files):
        hashes = {k:hashlib.sha256(v.encode()).hexdigest() for k,v in files.items()}
        identity = hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
        root = tmp_path / 'execution' / identity
        for name, body in files.items():
            p = root/name
            p.parent.mkdir(parents=True,exist_ok=True)
            p.write_text(body)
        (root/'execution_receipt.json').write_text(json.dumps({'schema':'car_execution_code_v1','sha256':identity,'files':hashes}))
        return root, hashes

    names = ('scripts/posttraining/assess_car_checkpoints.py','src/phycoflow_reconstruction/evaluation/adaptive_campaign.py')
    original, old = capsule({names[0]:'old driver',names[1]:'old runtime','src/other.py':'same'})
    revised, new = capsule({names[0]:'new driver',names[1]:'new runtime','src/other.py':'same'})
    campaign = tmp_path/'campaign'; campaign.mkdir()
    formal = {'schema':'car_freeze_receipt_v1','manifest':{'status':'frozen','code_root':str(original),
              'assessment_driver_sha256':old[names[0]],'event_task_sha256':'unchanged'}}
    frozen = campaign/'formal_freeze.json'; frozen.write_text(json.dumps(formal))
    body = {'schema':'car_device_amendment_v1','formal_freeze_sha256':c.digest(frozen),
            'training_code_root':str(original),'training_capsule_sha256':original.name,
            'original_assessment_driver_sha256':old[names[0]],'assessment_code_root':str(revised),
            'assessment_capsule_sha256':revised.name,'assessment_driver_sha256':new[names[0]],
            'changed_files':{k:{'old':old[k],'new':new[k]} for k in sorted(names)},
            'artifact_namespace':'assessment/device_v1'}
    amendment = campaign/'amendment.json'
    def save():
        amendment.write_text(json.dumps({**body,'sha256':c._content_hash(body)}))
    save(); monkeypatch.setenv('CAR_ASSESSMENT_AMENDMENT',str(amendment))
    effective = c.load_assessment_formal(campaign)['manifest']
    assert effective['code_root'] == str(original)
    assert effective['assessment_code_root'] == str(revised)
    assert effective['assessment_driver_sha256'] == new[names[0]]
    assert effective['event_task_sha256'] == 'unchanged'
    assert c.assessment_root(campaign) == campaign/'assessment/device_v1'
    altered = json.loads(amendment.read_text()); altered['artifact_namespace'] = 'assessment/other'
    amendment.write_text(json.dumps(altered))
    with pytest.raises(ValueError,match='amendment identity'): c.load_assessment_formal(campaign)
    save()
    body['formal_freeze_sha256'] = '0'*64; save()
    with pytest.raises(ValueError,match='frozen formal identity'): c.load_assessment_formal(campaign)
    body['formal_freeze_sha256'] = c.digest(frozen); save()
    (revised/'src/other.py').write_text('tampered')
    with pytest.raises(ValueError,match='capsule file changed'): c.load_assessment_formal(campaign)


def test_runtime_requires_selected_nonexcluded_gpu_and_paused_backend(tmp_path, monkeypatch):
    import json

    from phycoflow_reconstruction.evaluation import adaptive_campaign as c
    runtime = tmp_path/'runtime.json'
    values = {'schema':'car_execution_runtime_v1','gpu_uuid':'GPU-arbitrary',
              'excluded_gpu_uuids':['GPU-excluded'],'paused_source_families':['edm']}
    runtime.write_text(json.dumps(values)); monkeypatch.setenv('CAR_EXECUTION_RUNTIME',str(runtime))
    c.enforce_execution_constraints(tmp_path,'GPU-arbitrary','ffm')
    values['gpu_uuid'] = 'GPU-excluded'; runtime.write_text(json.dumps(values))
    with pytest.raises(ValueError,match='excluded'): c.enforce_execution_constraints(tmp_path,'GPU-excluded','ffm')
    values['gpu_uuid'] = 'GPU-arbitrary'; runtime.write_text(json.dumps(values))
    with pytest.raises(ValueError,match='paused'): c.enforce_execution_constraints(tmp_path,'GPU-arbitrary','edm')

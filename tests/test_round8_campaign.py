import json

import numpy as np
import pytest

from phycoflow_reconstruction.evaluation.round8_campaign import (
    COUPLING_KEYS, assert_matched, cache_identity, file_hash, forecast,
    freeze_manifest, load_prediction_cache, occupied_seconds, validate_t2000_receipt,
    write_prediction_cache,
)


def test_scoring_writer_prepares_nested_output_and_preserves_existing(tmp_path):
    from phycoflow_reconstruction.evaluation.round8_inference import _freeze_json
    path = tmp_path / 'policy/train_scores/alpha_1.json'
    _freeze_json(path, {'score': 1.})
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        _freeze_json(path, {'score': 2.})
    assert path.read_bytes() == original
    invalid = tmp_path / 'invalid.json'
    with pytest.raises(ValueError):
        _freeze_json(invalid, {'score': float('nan')})
    assert not invalid.exists()


def test_manifest_stratifies_and_avoids_known_exposure():
    train = [{'id': str(i)} for i in range(8000)]
    heldout = [{'id': str(i), 'source': source} for source in ('validation', 'test') for i in range(1000)]
    exposed = {(source, str(i)) for source in ('validation', 'test') for i in range(100)}
    a = freeze_manifest(train, heldout, exposed=exposed)
    assert a == freeze_manifest(train, heldout, exposed=exposed)
    assert len(a['TRAIN']) == 800 and len(a['ASSESS']) == 200
    assert sum(x['source'] == 'test' for x in a['ASSESS']) == 100
    assert not any((x['source'], x['id']) in exposed for x in a['ASSESS'])


def test_receipt_rejects_incomplete_and_changed_checkpoint(tmp_path):
    run = tmp_path / 'run'
    (run / 'checkpoints').mkdir(parents=True)
    ckpt = run / 'checkpoints/last.pt'
    ckpt.write_bytes(b'checkpoint')
    authorization = tmp_path / 'authority.json'
    authorization.write_text('{}')
    receipt = tmp_path / 'receipt.json'
    payload = dict(status='completed', epoch=2000, run=str(run), last_checkpoint_sha256=file_hash(ckpt),
                   user_continuation=str(authorization), user_continuation_sha256=file_hash(authorization))
    receipt.write_text(json.dumps(payload))
    assert validate_t2000_receipt(receipt)['epoch'] == 2000
    ckpt.write_bytes(b'changed')
    with pytest.raises(ValueError, match='endpoint differs'):
        validate_t2000_receipt(receipt)
    payload['epoch'] = 1750
    receipt.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match='completed epoch-2000'):
        validate_t2000_receipt(receipt)


def test_prediction_cache_coupling_and_tamper(tmp_path):
    coupling = {key: 'a' * 64 for key in COUPLING_KEYS}
    coupling.update(population='ASSESS', member_count=2)
    left = cache_identity(coupling, 'a' * 64)
    right = cache_identity(coupling, 'b' * 64)
    assert_matched(left, right)
    predictions = np.ones((2, 2, 3, 4))
    receipt = write_prediction_cache(tmp_path / 'cache.npz', identity=left, predictions=predictions,
                                     targets=np.ones((2, 3, 4)), weights=np.ones((2, 4)) / 4)
    data, actual = load_prediction_cache(receipt)
    assert np.array_equal(data['predictions'], predictions) and actual == left
    changed = dict(coupling, prior_buffers_sha256='c' * 64)
    with pytest.raises(ValueError, match='coupling differs'):
        assert_matched(left, cache_identity(changed, 'b' * 64))
    with open(receipt['path'], 'ab') as f:
        f.write(b'corruption')
    with pytest.raises(ValueError, match='changed'):
        load_prediction_cache(receipt)


def test_additive_gpu_time_and_forecast_reserve():
    assert occupied_seconds([{'device_uuid': '1', 'start': 0, 'end': 10},
                             {'device_uuid': '2', 'start': 0, 'end': 10}], 10) == 20
    with pytest.raises(ValueError, match='overlapping'):
        occupied_seconds([{'device_uuid': '1', 'start': 0, 'end': 10},
                          {'device_uuid': '1', 'start': 9, 'end': 11}], 11)
    result = forecast(spent_seconds=0, update_seconds=1, callback_seconds=0, stage0_seconds=0,
                      qualification_seconds=0, final_seconds=100, storage_bytes=1)
    assert result['closeout_reserved_seconds'] == 14400 and result['fits']
    assert result['formal_unique_updates'] == 25870


def test_inference_streams_actual_initial_fields_and_ht_risk(tmp_path, monkeypatch):
    """Exercise a real condition and observation assembler with a synthetic CPU model."""
    import yaml
    import torch
    from pathlib import Path
    from phycoflow_reconstruction.data.factory import open_field_dataset
    from phycoflow_reconstruction.evaluation import round8_inference as inference

    root = Path(__file__).resolve().parents[1]
    path = root / 'cases/turbulent_combustion/runs/coherence_compute_round8/artifacts/manifest.json'
    if not path.exists():
        pytest.skip('local case manifest not available')
    manifest = json.loads(path.read_text())
    config = yaml.safe_load(Path(manifest['inputs']['base_config']['path']).read_text())
    if not Path(config['dataset']['path']).exists():
        pytest.skip('local case dataset not available')
    manifest['TRAIN'][1]['id'] = 'force-stop-after-one-condition'
    test_manifest = tmp_path / 'manifest.json'
    test_manifest.write_text(json.dumps(manifest))
    dataset = open_field_dataset(config['dataset'], split='train')

    class SyntheticFlow(torch.nn.Module):
        capabilities = type('Capabilities', (), {'differentiable_rollout': True})()
        def sample_source(self, batch, *, generator):
            return torch.randn((*batch.query_coords.shape[:2], 5), generator=generator)
        def velocity(self, batch, state, time):
            return torch.zeros_like(state)

    monkeypatch.setattr(inference, '_load_model', lambda *args: (SyntheticFlow(), dataset, 'a'*64, 'b'*64, None))
    output = tmp_path / 'output'
    with pytest.raises(ValueError, match='sample identity'):
        inference.infer_panel(config, test_manifest, output, population='TRAIN', device='cpu')
    with np.load(output / 'condition_00000.npz', allow_pickle=False) as shard:
        assert shard['draws_model'].shape == (2, 4096, 5)
        assert shard['inclusion_probability'] == 4096 / 40300
        assert shard['physical_weights'].sum() < 1
        assert not np.array_equal(shard['draws_model'][0], shard['draws_model'][1])
        assert np.array_equal(shard['native_ids'], inference.fixed_query_indices(40300, 4096, seed=100045).numpy())


def test_panel_matching_checks_realized_prior_and_member_fields(tmp_path):
    from phycoflow_reconstruction.evaluation.round8_inference import validate_matched_panels
    shard = tmp_path / 'shared.npz'
    shard.write_bytes(b'bytes')
    conditions = [{'sample_id': str(i), 'source': 'train', 'dataset_index': i,
                   'path': shard.name, 'sha256': file_hash(shard), 'coupling_sha256': 'c'*64,
                   'initial_field_sha256': ['d'*64, 'e'*64]} for i in range(800)]
    panel = {'schema': 'round8_endpoint_panel_v1', 'status': 'completed', 'endpoint': 'SOURCE',
             'population': 'TRAIN', 'manifest_sha256': 'a'*64, 'source_checkpoint_sha256': 'b'*64,
             'prior_sha256': 'c'*64, 'solver_sha256': 'd'*64, 'clamp_sha256': 'e'*64,
             'normalizer_sha256': 'f'*64, 'field_names': ['T'], 'risk_design': 'HT_uniform_without_replacement',
             'inclusion_probability': 4096/40300, 'conditions': conditions}
    a, b = tmp_path / 'source.json', tmp_path / 'endpoint.json'
    a.write_text(json.dumps(panel))
    panel['endpoint'] = 'C8'
    b.write_text(json.dumps(panel))
    validate_matched_panels(a, b)
    panel['conditions'][0]['initial_field_sha256'][0] = 'f'*64
    b.write_text(json.dumps(panel))
    with pytest.raises(ValueError, match='initial_field_sha256'):
        validate_matched_panels(a, b)


def test_formal_child_must_match_exact_fork_receipt(tmp_path, monkeypatch):
    import json
    import phycoflow_reconstruction.training.round8_campaign as c
    root=tmp_path/'child'; (root/'checkpoints').mkdir(parents=True)
    cp=root/'checkpoints/last.pt'; cp.write_bytes(b'first')
    parent='f'*64
    manifest={'round8_fork_parent_sha256':parent,'round8_mode':'control','config_sha256':'cfg'}
    (root/'run_manifest.json').write_text(json.dumps(manifest))
    receipt={'schema':'round8_common_warmup_fork_v1','qualification_only':False,'fork_epoch':10,'shared_updates':130,'parent_checkpoint_sha256':parent,'arms':{'B8':{'run':str(root),'exact_shared_state':True,'config_sha256':'cfg','checkpoint_sha256':c.file_sha256(cp)}}}
    path=tmp_path/'fork.json';path.write_text(json.dumps(receipt))
    monkeypatch.setattr(c,'load_project_checkpoint',lambda _: {'global_step':130})
    assert c.validate_formal_child(path,'B8',root)['global_step']==130
    with pytest.raises(ValueError,match='run does not match'):
        c.validate_formal_child(path,'B8',tmp_path/'other')
    cp.write_bytes(b'different')
    with pytest.raises(ValueError,match='initial child'):
        c.validate_formal_child(path,'B8',root)
    monkeypatch.setattr(c,'load_project_checkpoint',lambda _: {'global_step':260})
    assert c.validate_formal_child(path,'B8',root)['global_step']==260


def test_round8_training_callback_skips_legacy_metric_guard(monkeypatch,tmp_path):
    import phycoflow_reconstruction.training.round8_execution as ex
    captured={}
    def fake_train(config, **kwargs):
        captured.update(kwargs)
        return tmp_path
    monkeypatch.setattr(ex,'run_post_training',fake_train)
    monkeypatch.setattr(ex,'Round8Runtime',lambda artifacts,mode:object())
    cfg={'runtime':{'plot_every_steps':10}}
    ex.execute_training(cfg,case_dir=tmp_path,artifacts={},phase='formal',charge_attempt=lambda _:None)
    assert captured['evaluation_callback'](global_step=0) is None
    ex.execute_training(cfg,case_dir=tmp_path,artifacts={},phase='qualification',charge_attempt=lambda _:None,max_steps=1)
    assert captured['evaluation_callback'](global_step=1) is None


def test_formal_resume_replays_boundary_callback_before_training(monkeypatch,tmp_path):
    import phycoflow_reconstruction.training.round8_execution as ex
    calls=[]
    monkeypatch.setattr(ex,'Round8Runtime',lambda artifacts,mode:object())
    monkeypatch.setattr(ex,'load_project_checkpoint',lambda path:{'global_step':1000*13})
    monkeypatch.setattr(ex,'run_post_training',lambda config,**kw:calls.append('train') or tmp_path)
    ex.execute_training({'runtime':{'plot_every_steps':10}},case_dir=tmp_path,artifacts={},phase='formal',charge_attempt=lambda _:None,resume=tmp_path,mode='candidate',checkpoint_callback=lambda run,path,step:calls.append(('observe',step)))
    assert calls==[('observe',13000),'train']
    calls.clear()
    monkeypatch.setattr(ex,'load_project_checkpoint',lambda path:{'global_step':130})
    ex.execute_training({'runtime':{'plot_every_steps':10}},case_dir=tmp_path,artifacts={},phase='formal',charge_attempt=lambda _:None,resume=tmp_path,mode='candidate',checkpoint_callback=lambda run,path,step:calls.append(('observe',step)))
    assert calls==['train']

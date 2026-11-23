"""Real trainer restart: consumed rejected batch then accepted batch, CPU fixture."""
import json
import os
from pathlib import Path
import subprocess
import sys
import torch
import yaml
from helpers.coherence import _base_config, _post_config, _write_fixture
from phycoflow_reconstruction.config import load_config
from phycoflow_reconstruction.config.load import deep_merge
from phycoflow_reconstruction.cli import _load_case_config
from phycoflow_reconstruction.training.base_training import run_base_training
from phycoflow_reconstruction.training.run_store import load_project_checkpoint


def test_direct_forced_reject_fresh_process_resume(tmp_path):
    path = tmp_path / 'fixture.h5'
    _write_fixture(path)
    base = _base_config(path)
    base['dataset']['coordinate_dim'] = 2
    source = run_base_training(base, case_dir=tmp_path / 'case')
    post = _post_config(path, source)
    post['dataset'].update(coordinate_dim=2, physical_measure='native_rectilinear_area')
    post['optimization'].update(epochs=10, steps_per_epoch=1, sampling='full_pass')
    post['coherence']['families'] = {}
    post = deep_merge(post, load_config(Path('configs/defaults/scalable_anchored_coherence.yaml')))
    post['optimization'].pop('multitask', None)
    post['optimization']['direct_supervision'] = {'mode': 'safeguarded', 'units': {'native': 2., 'state': 1., 'scale': 1., 'shape': 1.}, 'source_member_mse': [1e6, 1e6], 'field_tolerance': [0., 0.]}
    post['coherence']['compute_budget'].update(point_count=16, batch_size=1)
    post['coherence']['families']['multiscale_spectrum']['bands']['count'] = 1
    post['coherence']['families']['topology']['geometry']['grid_shape'] = [4, 4]
    post['coherence']['families']['topology']['persistence']['projections'] = 4
    post['evaluation'].update(query_points=16, batch_size=1, external_assessment=True, preview={'enabled': False})
    post['observation_consistency'] = {'mode': 'hard', 'final_clamp': True}
    post['checkpointing'] = {'every_epochs': 1, 'save_epoch_zero': True, 'selection_metric': 'native_validation_loss'}
    post['runtime']['progress'] = False
    post_path = tmp_path / 'posttrain.yaml'
    post_path.write_text(yaml.safe_dump(post))
    post = _load_case_config(post_path, tmp_path / 'case', 'fixture', [])
    post_path.write_text(yaml.safe_dump(post))
    runner = tmp_path / 'worker.py'
    runner.write_text('''import sys, torch\nfrom phycoflow_reconstruction.config import load_config\nfrom phycoflow_reconstruction.training.post_training import run_post_training\nimport phycoflow_reconstruction.training.direct_supervision as ds\nconfig=load_config(sys.argv[1])\nif sys.argv[4] == "force":\n original=ds.finite_field_action\n count=[0]\n def forced(*args, **kwargs):\n  count[0]+=1\n  if count[0] == 1:\n   args=list(args)\n   args[6]=lambda: torch.full_like(args[3], 1e9)\n  return original(*args, **kwargs)\n ds.finite_field_action=forced\nrun=run_post_training(config, case_dir=sys.argv[2], resume=None if sys.argv[3] == "new" else sys.argv[3], max_steps=int(sys.argv[5]))\nopen(sys.argv[6], "w").write(str(run))\n''')
    env = dict(os.environ, KEOPS_CACHE_FOLDER='/tmp/keops-round5')
    def run(resume, force, steps, tag, config_path=post_path):
        result = tmp_path / (tag + '.txt')
        subprocess.run([sys.executable, str(runner), str(config_path), str(tmp_path/'case'), str(resume), force, str(steps), str(result)], check=True, env=env, cwd=Path.cwd())
        return Path(result.read_text())
    resumed = run('new', 'force', 1, 'initial')
    definitions_before = (resumed/'logs/recording_definitions.json').read_bytes()
    with (resumed/'logs/direct_updates.jsonl').open('a') as stream:
        stream.write(json.dumps({'step': 2, 'interrupted_attempt': True}) + '\n')
    run(resumed, 'normal', 1, 'resume')
    whole = run('new', 'force', 2, 'whole')
    a = load_project_checkpoint(resumed/'checkpoints/last.pt')
    b = load_project_checkpoint(whole/'checkpoints/last.pt')
    def equal(x, y):
        if isinstance(x, torch.Tensor): assert torch.equal(x, y)
        elif isinstance(x, dict):
            assert x.keys() == y.keys()
            for k in x: equal(x[k], y[k])
        elif isinstance(x, (tuple, list)):
            assert len(x) == len(y)
            for p, q in zip(x, y): equal(p, q)
        else: assert x == y
    for key in ('model', 'optimizer', 'native_stream', 'rng_state'):
        equal(a[key], b[key])
    for key in ('step', 'sampler', 'probe_sampler', 'sampler_completed_updates'):
        equal(a['scalable_runtime'][key], b['scalable_runtime'][key])
    assert a['global_step'] == b['global_step'] == 2
    assert a['optimizer']['param_groups'][0]['lr'] == b['optimizer']['param_groups'][0]['lr']
    rows = [json.loads(x) for x in (resumed/'logs/scalable_updates.jsonl').read_text().splitlines()]
    assert [x['update_accepted'] for x in rows] == [False, True]
    assert (resumed/'logs/recording_definitions.json').read_bytes() == definitions_before
    definitions = json.loads(definitions_before)
    assert definitions == json.loads((whole/'logs/recording_definitions.json').read_text())
    assert 'attempted updates including consumed rejections' in definitions['aggregation']
    columns = {item['name']: item for item in definitions['columns']}
    assert columns['native_data_loss']['units'] == 'native_objective_units'
    assert columns['direct_scalar']['units'] == 'normalized_scalar'
    assert columns['field_excess_pre']['units'] == 'model_field_units_squared'
    assert 'field order' in columns['field_excess_pre']['meaning']
    assert columns['action_dot']['units'] == 'normalized_scalar'
    assert columns['accepted_actions']['units'] == 'actions'
    assert columns['accepted_objective_condition_visits']['units'] == 'condition_visits'
    assert columns['field_post_ensemble_checks']['units'] == 'ensemble_checks_per_attempt'
    direct_rows = [json.loads(x) for x in (resumed/'logs/direct_updates.jsonl').read_text().splitlines()]
    whole_rows = [json.loads(x) for x in (whole/'logs/direct_updates.jsonl').read_text().splitlines()]
    assert [r['update_accepted'] for r in direct_rows] == [False, True]
    assert list((resumed/'logs/recovery').glob('direct_updates.*.jsonl'))
    for left, right in zip(direct_rows, whole_rows, strict=True):
        for row in (left, right):
            row.pop('update_seconds')
            row.pop('data_wait_seconds')
        assert left == right
    history = [json.loads(x) for x in (resumed/'logs/history.jsonl').read_text().splitlines()]
    assert [row['field_source_ensemble_checks'] for row in history] == [1, 1]
    assert [row['field_post_ensemble_checks'] for row in history] == [2, 1]
    for row in history:
        assert row['data_wait_seconds'] >= 0
        assert row['total'] == row['direct_scalar']
        assert row['data_update_weight'] == .5
        assert row['native_data_loss'] == row['data_loss']
    assert json.loads((resumed/'run_manifest.json').read_text())['effective_child_ema_enabled'] is False

    native = deep_merge({}, post)
    native['optimization']['direct_supervision']['mode'] = 'native'
    native['coherence']['scalable_anchored_coherence']['enabled'] = False
    native_path = tmp_path / 'native.yaml'
    native_path.write_text(yaml.safe_dump(native))
    native_run = run('new', 'normal', 1, 'native', native_path)
    native_row = json.loads((native_run/'logs/history.jsonl').read_text().splitlines()[0])
    assert native_row['data_wait_seconds'] >= 0
    assert native_row['data_update_weight'] == .5
    assert native_row['total'] == native_row['native_data_loss'] / 2
    assert native_row['total_semantics'] == 'normalized_native_scalar'
    native_attempt = json.loads((native_run/'logs/direct_updates.jsonl').read_text().splitlines()[0])
    assert native_attempt['native_coupling'] == direct_rows[0]['native_coupling']
    assert native_attempt['native_data_loss'] == direct_rows[0]['native_data_loss']

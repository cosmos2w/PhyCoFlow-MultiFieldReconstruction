"""Ordering and failure behavior of the bounded assessment sequencer."""
import argparse
import importlib.util
import json
from pathlib import Path

import pytest

path = Path(__file__).resolve().parents[1] / 'scripts/posttraining/sequence_car_round2_assessment.py'
spec = importlib.util.spec_from_file_location('round2_assessment_sequence', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def args(tmp_path):
    return argparse.Namespace(campaign=tmp_path, family='senseiver', full_run=tmp_path/'full',
        residual_run=tmp_path/'residual', core_run=tmp_path/'core', train_source_bank=tmp_path/'train_source',
        gpu_uuid='GPU-test', generation_seconds=100., deployment_seconds=80., save_margin_seconds=5., warmup=3,
        phase='train', final_assessment=False)


def test_training_freeze_then_validation_use_expected_paths(tmp_path):
    options = args(tmp_path)
    artifacts = {'contract':'contract', 'event':'event', 'reserved':'reserved'}
    train = module.build_jobs(options, artifacts)
    assert len(train) == 37
    assert all(job['name'].startswith('train_') for job in train[:36])
    assert train[-1]['name'] == 'freeze_train_nominations'
    assert all('--final-assessment' not in job['command'] for job in train)
    options.phase, options.final_assessment = 'validation', True
    validation = module.build_jobs(options, artifacts)
    assert len(validation) == 26
    assert [job['name'] for job in validation[:2]] == [
        'freeze_source_reserved_normalizer', 'validation_source_generate']
    assert all(job['name'].startswith('validation_') for job in validation[2:])
    for job in validation:
        command = job['command']
        assert ('--final-assessment' in command) == (job['kind'] == 'gpu')
        assert 'test' not in command and 'edm' not in command
        if job['name'].endswith('_score'):
            split = job['name'].split('_', 1)[0]
            output = Path(command[command.index('--output') + 1])
            assert output.parts[-4:-2] == ('assessment', split)
    validation_scores = [job for job in validation if job['name'].startswith('validation_') and job['name'].endswith('_score')]
    assert len(validation_scores) == 12


def test_child_failure_stops_sequence_and_preserves_receipts(tmp_path):
    jobs = [{'name':str(i), 'kind':'cpu' if i == 1 else 'gpu', 'command':['command', str(i)]} for i in range(4)]
    called, children = [], []
    def runner(command, **kwargs):
        called.append(command)
        return argparse.Namespace(returncode=4 if command[-1] == '1' else 0)
    with pytest.raises(RuntimeError, match='1 failed'):
        module.execute_jobs(jobs, tmp_path, {}, children, runner=runner)
    assert [cmd[-1] for cmd in called] == ['0', '1']
    assert len(children) == 2
    first = json.loads((tmp_path / '0.json').read_text())
    second = json.loads((tmp_path / '1.json').read_text())
    assert first['returncode'] == 0 and not first['independent_cpu']
    assert second['returncode'] == 4 and second['independent_cpu']
    assert not (tmp_path / '2.json').exists()

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


def test_train_batch_only_runs_requested_arm_and_epochs_without_nominations(tmp_path):
    options = args(tmp_path)
    options.phase = 'train-batch'
    options.arm = 'Senseiver_Selected_Bank_Residual_Repair'
    options.run = tmp_path/'formal_run'
    options.epochs = [0, 250]
    options.full_run = options.residual_run = None
    jobs = module.build_jobs(options, {'contract':'contract', 'event':'event', 'reserved':'reserved'})
    assert len(jobs) == 6
    assert [j['name'].rsplit('_', 1)[-1] for j in jobs] == [
        'generate', 'deployment', 'score', 'generate', 'deployment', 'score']
    assert all('Residual_Repair' in job['name'] for job in jobs)
    assert not any('nomination' in job['name'] or 'validation' in job['name'] for job in jobs)
    generate = [j for j in jobs if j['name'].endswith('_generate')]
    assert [j['command'][j['command'].index('--epoch')+1] for j in generate] == ['0', '250']
    target = generate[1]['command'][generate[1]['command'].index('--output') + 1]
    assert Path(target) == tmp_path/'assessment/banks/train'/options.arm/'epoch_250'
    for job in jobs:
        assert '--final-assessment' not in job['command']
    options.final_assessment = True
    closeout = module.build_jobs(options, {'contract':'contract', 'event':'event', 'reserved':'reserved'})
    assert all(('--final-assessment' in job['command']) == (job['kind'] == 'gpu') for job in closeout)


def test_train_batch_binds_existing_checkpoint_without_terminal_run(tmp_path):
    import hashlib
    import yaml

    options = args(tmp_path)
    options.arm = 'Senseiver_Selected_Bank_Uniform_FullModel'
    options.run = tmp_path/'runs/formal'/options.arm/'attempt'
    options.epochs = [0, 100]
    options.full_run = options.residual_run = None
    options.run.mkdir(parents=True)
    source = tmp_path/'sources/senseiver'
    (source/'checkpoints').mkdir(parents=True)
    (source/'checkpoints/last.pt').write_bytes(b'source')
    source_sha = hashlib.sha256(b'source').hexdigest()
    cfg = {'source_run':str(source), 'runtime':{'device':'cuda:0'}}
    frozen = tmp_path/'frozen.yaml'
    frozen.write_text(yaml.safe_dump(cfg))
    (options.run/'resolved_config.yaml').write_text(yaml.safe_dump(cfg))
    (options.run/'checkpoints').mkdir()
    for epoch in options.epochs:
        (options.run/'checkpoints'/f'epoch_{epoch:03d}.pt').write_bytes(f'checkpoint-{epoch}'.encode())
    options.train_source_bank.mkdir()
    (options.train_source_bank/'receipt.json').write_text(json.dumps({
        'schema':'car_round2_native_bank_v1', 'status':'completed', 'kind':'source',
        'split':'train', 'family':'senseiver', 'source_sha256':source_sha}))
    config_sha = hashlib.sha256(frozen.read_bytes()).hexdigest()
    formal = {'manifest':{'arms':{options.arm:{'config':{'path':str(frozen), 'sha256':config_sha}}}}}
    class Driver:
        @staticmethod
        def check_binding(binding, campaign):
            path = Path(binding['path'])
            assert hashlib.sha256(path.read_bytes()).hexdigest() == binding['sha256']
            return path
    class Assess:
        @staticmethod
        def same_scientific_config(left, right):
            return left == right
    result = module.validate_train_batch(options, tmp_path, formal, Assess, Driver)
    assert set(result['checkpoints']) == {'0', '100'}
    assert result['source_sha256'] == source_sha
    assert not (options.run/'status.json').exists()
    options.epochs = [0, 0]
    with pytest.raises(ValueError, match='unique prescribed'):
        module.validate_train_batch(options, tmp_path, formal, Assess, Driver)
    options.epochs = [250]
    with pytest.raises(FileNotFoundError):
        module.validate_train_batch(options, tmp_path, formal, Assess, Driver)
    options.epochs = [0]
    receipt_path = options.train_source_bank/'receipt.json'
    bank_receipt = json.loads(receipt_path.read_text())
    bank_receipt['source_sha256'] = 'wrong'
    receipt_path.write_text(json.dumps(bank_receipt))
    with pytest.raises(ValueError, match='same frozen SOURCE'):
        module.validate_train_batch(options, tmp_path, formal, Assess, Driver)
    bank_receipt['source_sha256'] = source_sha
    receipt_path.write_text(json.dumps(bank_receipt))
    (options.run/'resolved_config.yaml').write_text(yaml.safe_dump({**cfg, 'changed':True}))
    with pytest.raises(ValueError, match='differs from frozen configuration'):
        module.validate_train_batch(options, tmp_path, formal, Assess, Driver)


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

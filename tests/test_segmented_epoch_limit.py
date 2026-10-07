"""Public bounded segmented execution preserves horizon and recovery identity."""
import json
from types import SimpleNamespace

import pytest

from phycoflow_reconstruction.training import segmented


def fixture_launcher(tmp_path, monkeypatch):
    case = tmp_path / 'case'
    case.mkdir()
    (case / 'run.py').write_text('# fixture\n')
    run = case / 'runs/Test_1002/R6_sibling/child'
    (run / 'metrics').mkdir(parents=True)
    (run / 'status.json').write_text(json.dumps({'post_training_seconds': 1, 'peak_cuda_memory_bytes': 0}))
    config = {'case': 'fixture', 'stage': 'post_training',
              'output': {'experiment_name': 'Test_1002/R6_sibling'},
              'optimization': {'epochs': 5000}}
    contexts, commands = [], []
    monkeypatch.setattr(segmented, 'load_config', lambda *a: config)

    def load(*args, **kwargs):
        contexts.append(kwargs)
        return config

    monkeypatch.setattr(segmented, '_load_case_config', load)
    monkeypatch.setattr(segmented, 'configured_epoch_steps', lambda c: 38)
    age = [0]
    monkeypatch.setattr(segmented, 'select_run', lambda *a: (run if age[0] else None, age[0] * 38, False))

    def execute(command, **kwargs):
        commands.append(command)
        age[0] = int(command[command.index('--until-epoch') + 1])
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(segmented.subprocess, 'run', execute)
    argv = ['--config', str(tmp_path / 'fixture.yaml'), '--case-dir', str(case),
            '--segment-epochs', '1', '--allocation-hours', '100']
    return case, run, config, contexts, commands, age, argv


def test_bounded_horizon_source_then_own_resume(tmp_path, monkeypatch, capsys):
    case, run, config, contexts, commands, age, argv = fixture_launcher(tmp_path, monkeypatch)
    segmented.main(argv + ['--until-epoch', '1'])
    assert age[0] == 1 and len(commands) == 1
    assert '--resume' not in commands[0] and '--max-steps' not in commands[0]
    segmented.main(argv + ['--until-epoch', '2'])
    assert age[0] == 2 and len(commands) == 2
    assert commands[1][commands[1].index('--resume') + 1] == str(run)
    assert contexts == [{'invocation_until_epoch': 1}, {'invocation_until_epoch': 2}]
    assert config['optimization']['epochs'] == 5000
    assert not (case / 'runs/long_jobs').exists()
    summary = json.loads((run.parent / '_launcher/segmented.json').read_text())
    assert summary['execution_limit_epoch'] == 2 and summary['configured_steps'] == 5000 * 38
    assert not summary['completed']
    assert 'Reached execution limit at epoch 2 of 5000' in capsys.readouterr().out


def test_clean_stop_after_segment_then_unchanged_own_resume(tmp_path, monkeypatch):
    _, run, config, _, commands, age, argv = fixture_launcher(tmp_path, monkeypatch)
    stop = tmp_path / 'stop'
    original_execute = segmented.subprocess.run

    def execute(command, **kwargs):
        result = original_execute(command, **kwargs)
        stop.touch()
        return result

    monkeypatch.setattr(segmented.subprocess, 'run', execute)
    same_argv = argv + ['--until-epoch', '2', '--stop-file', str(stop)]
    segmented.main(same_argv)
    assert age[0] == 1 and len(commands) == 1
    stop.unlink()
    segmented.main(same_argv)
    assert age[0] == 2 and len(commands) == 2
    assert commands[1][commands[1].index('--resume') + 1] == str(run)
    assert config['optimization']['epochs'] == 5000


@pytest.mark.parametrize('limit', [0, -1, 5001])
def test_invalid_limit_rejects_before_spawn(tmp_path, monkeypatch, limit):
    *_, commands, age, argv = fixture_launcher(tmp_path, monkeypatch)
    with pytest.raises(SystemExit):
        segmented.main(argv + ['--until-epoch', str(limit)])
    assert commands == [] and age[0] == 0


def test_resume_cannot_move_limit_backwards(tmp_path, monkeypatch):
    *_, commands, age, argv = fixture_launcher(tmp_path, monkeypatch)
    age[0] = 2
    with pytest.raises(ValueError, match='beyond'):
        segmented.main(argv + ['--until-epoch', '1'])
    assert commands == []

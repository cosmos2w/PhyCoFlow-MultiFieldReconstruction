"""Exercise serial orchestration without starting model processes."""
import importlib.util
import json
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


def load():
    spec = importlib.util.spec_from_file_location('sequence_test', ROOT / 'scripts/posttraining/sequence_car_round2_qualification.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('failed_child', [None, 2])
def test_serial_fresh_resume_and_failure_receipts(tmp_path, monkeypatch, failed_child):
    module = load()
    campaign = tmp_path.resolve()
    paths = {}
    names = ['FFM_Selected_Bank_Uniform_FullModel', 'FFM_Selected_Bank_Residual_Repair']
    for arm in names:
        path = campaign / f'{arm}.yaml'
        path.write_text(yaml.safe_dump({'optimization': {'epochs': 1000}, 'checkpointing': {}, 'output': {}}))
        paths[arm] = {'path': str(path)}
    manifest = campaign / 'arms.json'
    manifest.write_text(json.dumps({'schema': 'car_round2_assembled_arms_v1', 'arms': paths}))
    driver = SimpleNamespace(check_code_capsule=lambda _: 'capsule',
                             check_binding=lambda record, _: Path(record['path']),
                             check_config=lambda *_: None, check_source_pair=lambda *_: None)
    class Loader:
        def exec_module(self, obj):
            obj.__dict__.update(vars(driver))
    monkeypatch.setattr(module.importlib.util, 'spec_from_file_location', lambda *_: SimpleNamespace(loader=Loader()))
    monkeypatch.setattr(module.importlib.util, 'module_from_spec', lambda _: SimpleNamespace())
    commands = []

    class Process:
        pid = 123
        def __init__(self, command, **_):
            self.command = command
            commands.append(command)
        def wait(self):
            if len(commands) == failed_child:
                return 1
            command = self.command
            if 'run' in command:
                arm = command[command.index('--arm') + 1]
                phase = 'pause' if '--pause-at-epoch' in command else 'resume' if '--resume' in command else 'uninterrupted'
                run = Path(command[command.index('--resume') + 1]) if phase == 'resume' else campaign / f'{arm}_{phase}'
                run.mkdir(exist_ok=True)
                config = Path(command[command.index('--config') + 1])
                if phase != 'resume':
                    (run / 'resolved_config.yaml').write_text(config.read_text())
                launch = Path(command[command.index('--launch-receipt-output') + 1])
                token = f'{arm}_{phase}'
                launch.write_text(json.dumps({'token': token}))
                completions = campaign / 'operations/completions'
                completions.mkdir(parents=True, exist_ok=True)
                (completions / f'{token}.json').write_text(json.dumps({'run_dir': str(run), 'status': 'budget_paused' if phase == 'pause' else 'completed', 'global_step': 108 if phase == 'pause' else 144}))
            else:
                Path(command[command.index('--output') + 1]).write_text('{}')
            return 0

    monkeypatch.setattr(module.subprocess, 'Popen', Process)
    args = Namespace(campaign=campaign, output=campaign / 'group.json', family='ffm', device='GPU-example',
                     arm_configs=manifest, max_seconds=1200., save_margin_seconds=100., initial_stage_bound_seconds=400.)
    if failed_child:
        with pytest.raises(RuntimeError, match='exit code'):
            module.sequence(args)
    else:
        module.sequence(args)
    receipt = json.loads(args.output.read_text())
    assert receipt['status'] == ('failed' if failed_child else 'completed')
    assert len(receipt['children']) == (failed_child if failed_child else 10)
    if not failed_child:
        resumed = [cmd for cmd in commands if '--resume' in cmd]
        assert len(resumed) == 2
        for cmd in resumed:
            assert cmd[cmd.index('--config') + 1] == str(Path(cmd[cmd.index('--resume') + 1]) / 'resolved_config.yaml')

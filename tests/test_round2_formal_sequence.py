"""Formal pair orchestration stops on failures and never overlaps children."""
import importlib.util
import json
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load():
    spec = importlib.util.spec_from_file_location('formal_sequence_test', ROOT / 'scripts/posttraining/sequence_car_round2_formal.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('failure', [None, 'exit', 'paused', 'incomplete'])
def test_formal_pair_receipts_and_stop(tmp_path, monkeypatch, failure):
    module = load()
    names = ['FFM_Selected_Bank_Uniform_FullModel', 'FFM_Selected_Bank_Residual_Repair',
             'Senseiver_Selected_Bank_Uniform_FullModel', 'Senseiver_Selected_Bank_Residual_Repair']
    configs = {}
    for arm in names:
        path = tmp_path / f'{arm}.yaml'
        path.write_text('formal')
        configs[arm] = {'config': {'path': str(path)}}
    forecast = tmp_path / 'forecast.json'
    forecast.write_text(json.dumps({'schema': 'car_round2_forecast_v2', 'fits': True,
                                    'per_arm_update_seconds': [1, 1, 1, 1],
                                    'per_arm_callback_seconds_per_epoch': [1, 1, 1, 1],
                                    'per_arm_outside_epoch_seconds': [2, 2, 2, 2],
                                    'per_arm_cold_train_excess_seconds': [1, 1, 1, 1]}))
    inventory = tmp_path / 'inventory.json'
    inventory.write_text(json.dumps({'schema': 'car_execution_code_v1', 'sha256': 'capsule'}))
    manifest = {'arms': configs, 'artifacts': {'forecast': {'path': str(forecast)},
                                               'code_inventory': {'path': str(inventory)}}}
    freeze = tmp_path / 'configs/formal_freeze.json'
    freeze.parent.mkdir()
    freeze.write_text(json.dumps({'schema': 'car_round2_freeze_receipt_v1', 'manifest': manifest}))
    driver = SimpleNamespace(ARMS=names, check_code_capsule=lambda _: 'capsule',
                             check_binding=lambda record, _: Path(record['path']), check_freeze=lambda value, _: value)
    class Loader:
        def exec_module(self, obj):
            obj.__dict__.update(vars(driver))
    monkeypatch.setattr(module.importlib.util, 'spec_from_file_location', lambda *_: SimpleNamespace(loader=Loader()))
    monkeypatch.setattr(module.importlib.util, 'module_from_spec', lambda _: SimpleNamespace())
    commands = []

    class Process:
        pid = 10
        def __init__(self, command, **_):
            self.command = command
            commands.append(command)
        def wait(self):
            command = self.command
            token = str(len(commands))
            launch = Path(command[command.index('--launch-receipt-output') + 1])
            launch.write_text(json.dumps({'token': token}))
            completion = tmp_path / 'operations/completions' / f'{token}.json'
            completion.parent.mkdir(parents=True, exist_ok=True)
            arm = command[command.index('--arm') + 1]
            completion.write_text(json.dumps({'token': token, 'run_dir': str(tmp_path / 'runs/formal' / arm / 'attempt'),
                                               'status': 'budget_paused' if failure == 'paused' else 'completed',
                                               'global_step': 11999 if failure == 'incomplete' else 12000}))
            return 1 if failure == 'exit' else 0
    monkeypatch.setattr(module.subprocess, 'Popen', Process)
    args = Namespace(campaign=tmp_path, output=tmp_path / 'group.json', family='ffm', device='GPU-example',
                     full_max_seconds=20000., residual_max_seconds=20000., full_epoch_seconds=13., residual_epoch_seconds=13.,
                     initial_stage_bound_seconds=300., save_margin_seconds=100.)
    if failure:
        with pytest.raises(RuntimeError, match='did not complete'):
            module.sequence(args)
    else:
        module.sequence(args)
    record = json.loads(args.output.read_text())
    assert record['status'] == ('failed' if failure else 'completed')
    assert len(record['children']) == (1 if failure else 2)
    launch = json.loads(Path(record['launch_receipt']).read_text())
    assert [x['completed_epochs'] for x in launch['luna_sparse_schedule']] == [500, 1000, 500, 1000]
    assert [cmd[cmd.index('--arm') + 1] for cmd in commands] == names[:len(commands)]

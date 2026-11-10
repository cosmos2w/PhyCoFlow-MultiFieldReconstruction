"""Wrapper contracts only; fixtures do not qualify a real native model endpoint."""
import json
import random
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from phycoflow_reconstruction.evaluation import round7_native as native


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(1, 1)


def make_runtime(tmp_path):
    model = Model()
    model.train()
    model.layer.eval()
    checkpoint = tmp_path / 'checkpoint'
    checkpoint.write_bytes(b'fixture')
    return SimpleNamespace(model=model, device='cpu', checkpoint_path=checkpoint)


def fake_infer(runtime, dataset, indices, output_dir, endpoint, *, panel_seed_base, draws, retain_payloads, consumer):
    assert draws == 2 and retain_payloads is False
    random.random(); np.random.random(); torch.rand(1)
    runtime.model.eval()
    root = Path(output_dir)
    rows = []
    arrays = {'physical_mask': np.ones(40300, dtype=bool), 'physical_weights': np.ones(40300),
              'draws_physical': np.ones((2, 40300, 2)), 'target_physical': np.zeros((40300, 2))}
    for i in indices:
        row = {'sample_id': f's:{i}', 'dataset_index': i, 'point_count': 40300,
               'valid_count': 40299, 'physical_measure': 'physical_area', 'draw_seeds': [i, i + 100],
               'contract_sha256': str(i), 'payload': None}
        consumer(arrays, row)
        rows.append(row)
    mse = np.full((100, 2, 2), dataset.multiplier, dtype=float)
    power = np.full((100, 2), 4.)
    np.savez_compressed(root / 'field_risk.npz', member_mse_physical=mse,
                        reference_power_physical=power, sample_ids=[r['sample_id'] for r in rows],
                        field_names=['a', 'b'])
    manifest = {'schema': 'native_coherence_inference_v1', 'query_policy': 'full_native',
                'sampler': {'path': 'full_query_differentiable_reconstruction', 'effective_weights':'live',
                            'draws': draws, 'panel_seed_base':panel_seed_base},
                'sensor_protocol': {}, 'sensor_manifest_sha256': 'c'*64, 'field_names': ['a', 'b'],
                'field_units': ['m', 'K'], 'normalizer_sha256':'d'*64,
                'dataset_split':'validation', 'conditions': rows, 'checkpoint_sha256':native.file_sha256(runtime.checkpoint_path),
                'wall_seconds':1.}
    path = root / 'manifest.json'
    path.write_text(json.dumps(manifest))
    return path


def test_native_wrapper_aggregates_and_strict_matching(tmp_path, monkeypatch):
    monkeypatch.setattr(native, 'infer_native_panel', fake_infer)
    monkeypatch.setattr(native, '_manifest', lambda *args: SimpleNamespace(digest=lambda: 'c'*64))
    runtime = make_runtime(tmp_path)
    endpoint = {'sensor_manifest':'frozen.json', 'weight_selection':'live', 'steps':1, 'solver':'euler', 'observation_config':{}}
    dataset = SimpleNamespace(split_name='validation', multiplier=1., checkpoint='a'*64, __len__=lambda:100)
    # special methods are looked up on the class, not instance
    class Dataset:
        split_name = 'validation'
        def __init__(self, multiplier, checkpoint):
            self.multiplier, self.checkpoint = multiplier, checkpoint
        def __len__(self): return 100
    a, b = Dataset(1., 'a'*64), Dataset(2., 'b'*64)
    random.seed(3); np.random.seed(4); torch.manual_seed(5)
    py_state, np_state, t_state = random.getstate(), np.random.get_state(), torch.random.get_rng_state()
    receipt_path = native.evaluate_round7_native(runtime, a, range(100), tmp_path/'a', endpoint, panel_seed_base=77)
    assert random.getstate() == py_state
    np.testing.assert_array_equal(np.random.get_state()[1], np_state[1])
    assert torch.equal(torch.random.get_rng_state(), t_state)
    assert runtime.model.training and not runtime.model.layer.training
    receipt = json.loads(receipt_path.read_text())
    assert receipt['primary_mse_physical'] == [1., 1.]
    assert receipt['primary_rmse_physical'] == [1., 1.]
    assert receipt['pooled_relative_l2_physical'] == [.5, .5]
    assert receipt['coverage'][0]['valid_count'] == 40299
    native.evaluate_round7_native(runtime, b, range(100), tmp_path/'b', endpoint, panel_seed_base=77)
    comparison = native.compare_round7_native(tmp_path/'a', tmp_path/'b')
    assert comparison['primary_mse_ratio'] == [2., 2.]
    manifest_path = tmp_path/'b'/'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['conditions'][0]['contract_sha256'] = 'different'
    manifest_path.write_text(json.dumps(manifest))
    # Update manifest hash so the stricter underlying matched contract is exercised.
    receipt_path = tmp_path/'b'/'round7_native_receipt.json'
    receipt = json.loads(receipt_path.read_text())
    receipt['manifest_sha256'] = native.file_sha256(manifest_path)
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match='condition contract differs'):
        native.compare_round7_native(tmp_path/'a', tmp_path/'b')


def test_rejects_incomplete_panel_and_cleans_failed_staging(tmp_path, monkeypatch):
    class Dataset:
        split_name = 'validation'
        def __len__(self): return 100
    endpoint = {'sensor_manifest':'frozen.json', 'weight_selection':'live', 'steps':1, 'solver':'euler', 'observation_config':{}}
    with pytest.raises(ValueError, match='exactly 100'):
        native.evaluate_round7_native(make_runtime(tmp_path), Dataset(), range(99), tmp_path/'bad', endpoint, panel_seed_base=1)
    monkeypatch.setattr(native, '_manifest', lambda *args: SimpleNamespace(digest=lambda: 'c'*64))
    def fail(*args, **kwargs):
        random.random(); np.random.random(); torch.rand(1)
        arrays = {'physical_mask': np.ones(40300, dtype=bool), 'physical_weights': np.ones(40300),
                  'draws_physical': np.ones((2, 40300, 2)), 'target_physical': np.zeros((40300, 2))}
        row = {'sample_id': 's:0', 'dataset_index': 0, 'point_count': 40300, 'valid_count': 40300,
               'physical_measure': 'physical_area', 'draw_seeds': [0, 100], 'contract_sha256': 'identity'}
        kwargs['consumer'](arrays, row)
        raise RuntimeError('failure')
    monkeypatch.setattr(native, 'infer_native_panel', fail)
    random.seed(8); np.random.seed(9); torch.manual_seed(10)
    state = torch.random.get_rng_state().clone()
    model = make_runtime(tmp_path)
    state = torch.random.get_rng_state().clone()
    with pytest.raises(RuntimeError, match='failure'):
        native.evaluate_round7_native(model, Dataset(), range(100), tmp_path/'bad', endpoint, panel_seed_base=1)
    assert torch.equal(torch.random.get_rng_state(), state)
    assert model.model.training and not model.model.layer.training
    stages = list(tmp_path.glob('.bad.*'))
    assert len(stages) == 1
    failure = json.loads((stages[0] / "failure_receipt.json").read_text())
    assert failure["status"] == "incomplete" and not (tmp_path / "bad").exists()
    assert failure['completed_sample_ids'] == ['s:0']
    partial = json.loads((stages[0] / 'partial_conditions.jsonl').read_text())
    assert partial['primary_mse_physical'] == pytest.approx([1., 1.])
    assert partial['endpoint_identity']['sensor_manifest_sha256'] == 'c'*64

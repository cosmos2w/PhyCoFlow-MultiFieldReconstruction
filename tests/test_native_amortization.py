"""Fresh-process native continuation and fixed-observation contract."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import torch

from phycoflow_reconstruction.contracts import DataSpec, FieldSample, LossBundle
from phycoflow_reconstruction.data.manifest import SensorManifest, dataset_fingerprint
from phycoflow_reconstruction.data.normalization import FieldNormalizer
from phycoflow_reconstruction.data.sensor_protocols import SensorProtocol
from phycoflow_reconstruction.training.native_amortization import run_native_amortization


class TinyDataset:
    def __init__(self, path):
        self.path = path
        self.normalizer = FieldNormalizer(torch.tensor([10.]), torch.tensor([2.]))
        self.data_spec = DataSpec(('T',), ('K',), 1, (256,))
    def __len__(self): return 3
    def __getitem__(self, i):
        if i >= len(self): raise IndexError
        coords = torch.linspace(0, 1, 256).unsqueeze(1)
        return FieldSample(values=torch.full((256, 1), float(i)), coordinates=coords,
            coordinates_raw=coords, time=torch.tensor(0.), trajectory_id=f'case{i}',
            time_index=0, conditions=torch.zeros(1), field_names=('T',), logical_shape=(256,))


class TinyFlow(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.5))
    def training_loss(self, batch):
        # Fresh prior/time analogues use the same global RNG as the real RF adapter.
        assert batch.target_fields.dtype == torch.float32
        noise = torch.randn_like(batch.target_fields)
        t = torch.rand((len(batch.sample_ids), 1, 1))
        loss = ((self.weight * (t + batch.query_coords) - (batch.target_fields - noise)) ** 2).mean()
        return LossBundle(loss, {'rectified_flow_mse': loss})


def invoke(base, name, epochs, resume, kind='teacher'):
    dataset = TinyDataset(base / 'dataset')
    manifest = SensorManifest(dataset_path='dataset', dataset_fingerprint=dataset_fingerprint(dataset.path),
        split='train', protocol=SensorProtocol(field_counts={'T': 256}).to_dict(),
        indices={f'case{i}:0': [[q, 0] for q in range(256)] for i in range(3)})
    config_path = base / name / 'resolved_config.json'
    config = json.loads(config_path.read_text()) if resume else {
        'seed': 42, 'epochs': epochs, 'device': 'cpu', 'query_points': 4,
        'deterministic': True, 'strict_algorithms': True,
        'temperature_field_index': 0,
        'identity': {'target_kind': kind, 'source_sha256': 'source',
            'membership_sha256': hashlib.sha256(json.dumps([f'case{i}:0' for i in range(3)], separators=(',', ':')).encode()).hexdigest(), 'manifest_sha256': manifest.digest(), 'bank_sha256': 'bank'}}
    config['epochs'] = epochs
    def reader(ids, atoms, query):
        # Return physical units: normalized target equals case + atom / 2.
        return torch.stack([torch.full((query.shape[1], 1), 10 + 2*int(s[4]) + int(a), dtype=torch.float64)
                            for s, a in zip(ids, atoms)])
    run_native_amortization(model=TinyFlow(), dataset=dataset, manifest=manifest,
        config=config, run_dir=base / name, target_reader=reader if kind == 'teacher' else None, resume=resume,
        inference_config={'model': {'name': 'fixture'}})


def test_fresh_process_resume(tmp_path):
    (tmp_path / 'dataset').write_text('dataset identity')
    env = dict(os.environ)
    for name, epochs, resume, kind in [('whole', 2, 0, 'teacher'), ('split', 1, 0, 'teacher'),
                                        ('split', 2, 1, 'teacher'), ('control', 2, 0, 'truth')]:
        subprocess.run([sys.executable, __file__, str(tmp_path), name, str(epochs), str(resume), kind],
                       check=True, env=env)
    full = torch.load(tmp_path / 'whole/checkpoint.pt', weights_only=False)
    resumed = torch.load(tmp_path / 'split/checkpoint.pt', weights_only=False)
    control = torch.load(tmp_path / 'control/checkpoint.pt', weights_only=False)
    assert torch.equal(full['model']['weight'], resumed['model']['weight'])
    assert full['visit_counts'] == resumed['visit_counts']
    assert torch.equal(full['condition_rng'], resumed['condition_rng'])
    assert torch.equal(full['query_rng'], resumed['query_rng'])
    assert torch.equal(full['atom_rng'], resumed['atom_rng'])
    for a, b in zip(full['history'], resumed['history']):
        timings = {'epoch_seconds', 'data_seconds', 'update_seconds'}
        assert {k:v for k,v in a.items() if k not in timings} == {k:v for k,v in b.items() if k not in timings}
    for teacher_row, control_row in zip(full['history'], control['history']):
        for teacher_batch, control_batch in zip(teacher_row['batch_stream'], control_row['batch_stream']):
            assert teacher_batch['condition_indices'] == control_batch['condition_indices']
            assert teacher_batch['query_seed'] == control_batch['query_seed']
            assert len(teacher_batch['teacher_atom_indices']) == len(teacher_batch['condition_indices'])
            assert control_batch['teacher_atom_indices'] is None
    assert (tmp_path / 'split/logs/native_loss.pdf').is_file()
    assert len((tmp_path / 'split/logs/history.jsonl').read_text().splitlines()) == 2
    exported = torch.load(tmp_path / 'split/checkpoints/epoch_2.pt', weights_only=True)
    assert exported['model_name'] == 'fixture'
    assert torch.equal(exported['model']['weight'], resumed['model']['weight'])
    assert exported['normalization']['method'] == 'mean_std'
    assert exported['source_hashes'] == {'checkpoint': 'source'}
    assert exported['config_sha256'] == exported['inference_config_sha256']
    assert exported['config_sha256_role'] == 'inherited inference configuration'
    assert exported['native_scientific_definitions'] == resumed['scientific']
    assert resumed['scientific']['numerical_settings']['deterministic_algorithms_enabled'] is True
    assert resumed['scientific']['numerical_settings']['cudnn_deterministic'] is True
    assert resumed['scientific']['numerical_settings']['cudnn_benchmark'] is False


if __name__ == '__main__':
    invoke(Path(sys.argv[1]), sys.argv[2], int(sys.argv[3]), bool(int(sys.argv[4])), sys.argv[5])

from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

from phycoflow_reconstruction.training import round2_prefit


@pytest.mark.parametrize('reordered', [False, True])
def test_matched_panel_accepts_tuple_ids_but_rejects_real_reordering(monkeypatch, reordered):
    @dataclass
    class Protocol:
        seed: int = 0

    ids = ['train:1', 'train:2']
    reference = torch.ones(2, 3, 1)
    coordinates = torch.zeros(2, 3, 2)
    masses = torch.ones(2, 3)
    query = torch.arange(3).expand(2, -1)
    batch = SimpleNamespace(sample_ids=tuple(reversed(ids)) if reordered else tuple(ids),
                            target_fields=reference, query_coords=coordinates,
                            metadata={'query_indices': query})
    batch.to = lambda _device: batch
    panel = {'schema': 'car_source_panel_v1', 'provenance': {'sample_ids': ids},
             'sample_ids': ids, 'sample_indices': [0, 1],
             'observation_manifest': {ident: [] for ident in ids},
             'reference': reference, 'coordinates': coordinates, 'masses': masses,
             'query_indices': query}
    class Dataset:
        split_name = 'train'

        def __getitem__(self, index):
            return ids[index]

    monkeypatch.setattr(round2_prefit, 'sensor_protocol_from_config', lambda _config: Protocol())
    monkeypatch.setattr(round2_prefit, 'build_observation_batch', lambda *_args, **_kwargs: batch)
    monkeypatch.setattr(round2_prefit, 'attach_batch_measure', lambda _dataset, value: value)
    monkeypatch.setattr(round2_prefit, 'batch_measure_context', lambda _batch: {'generated_weights': masses})
    runtime = SimpleNamespace(artifacts={'provenance': panel['provenance']})
    values = round2_prefit.matched_panel_batches(runtime, Dataset(), {}, panel, device='cpu')
    if reordered:
        with pytest.raises(ValueError, match='sample ordering changed'):
            list(values)
    else:
        assert list(values) == [(0, batch)]

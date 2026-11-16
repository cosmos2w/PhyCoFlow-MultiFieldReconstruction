import copy
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('car_formal_driver', Path('scripts/posttraining/run_coherence_adaptive_repair.py'))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_formal_pair_binds_common_config_and_adaptive_bank(tmp_path):
    bank = {'version':'adaptive_measured_train_bank_v1', 'source_fit_binding_sha256':'bound',
            'policy_qualified':False, 'directions':[[1.,1.]], 'triples':[]}
    bank['sha256'] = mod.hashlib.sha256(json.dumps(bank, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    path = tmp_path/'bank.json'
    path.write_text(json.dumps(bank))
    base = {'coherence': {'scalable_anchored_coherence': {'common_core': {'sha256':'core'}}},
            'output': {'experiment_name':'B1'}, 'runtime': {'seed':47},
            'optimization': {'multitask': {'fidelity_budget': {'reference_artifact': {'sha256':'risk'}}}}}
    candidate = copy.deepcopy(base)
    candidate['output']['experiment_name'] = 'A1'
    candidate['coherence']['scalable_anchored_coherence']['adaptive'] = {
        'enabled':True, 'bank_artifact':{'path':str(path),'sha256':mod.digest(path)}}
    pair = {'FFM_B1':base,'FFM_A1':candidate}
    mod.validate_formal_pair(pair)
    candidate['runtime']['seed'] = 48
    with pytest.raises(ValueError, match='differ outside'):
        mod.validate_formal_pair(pair)
    candidate['runtime']['seed'] = 47
    path.write_text('{}')
    with pytest.raises(ValueError, match='bank file hash'):
        mod.validate_formal_pair(pair)


def test_train_panel_ids_match_declared_order(tmp_path, monkeypatch):
    from phycoflow_reconstruction.data import factory
    class Dataset:
        trajectory_ids = ('trajectory_000000',)
        _items = tuple((0, i) for i in range(20))
        def __len__(self): return len(self._items)
        def close(self): pass
    monkeypatch.setattr(factory, 'open_field_dataset', lambda *args, **kwargs: Dataset())
    (tmp_path/'panels').mkdir()
    panel = {'split':'train','population':20,'sample_ids':['trajectory_000000:3','trajectory_000000:12'], 'sensor_count':256,'members':4}
    (tmp_path/'panels/train.json').write_text(json.dumps(panel))
    config = {'dataset':{}, 'evaluation':{'split':'train','max_samples':2,'members':4,
              'sensor_counts':{'T':256},'sensor_seed':2047,'sample_indices':[3,12]}}
    mod.validate_train_panel(config,tmp_path)
    config['evaluation']['sample_indices'] = [12,3]
    with pytest.raises(ValueError, match='IDs differ'):
        mod.validate_train_panel(config,tmp_path)
    config['evaluation']['sample_indices'] = [3,12]
    config['evaluation']['sensor_seed'] = 2048
    with pytest.raises(ValueError, match='sensor seed'):
        mod.validate_train_panel(config,tmp_path)


def test_code_capsule_detects_engine_mutation(tmp_path):
    source = tmp_path/'work'
    (source/'src').mkdir(parents=True)
    (source/'src'/'engine.py').write_text('value = 1\n')
    from phycoflow_reconstruction.evaluation.adaptive_campaign import freeze_execution_code
    capsule = freeze_execution_code(source, tmp_path/'capsules', [])
    assert mod.validate_code_capsule(capsule) == capsule.name
    changed = capsule/'src/engine.py'
    changed.chmod(0o644)
    changed.write_text('value = 2\n')
    with pytest.raises(ValueError, match='frozen execution file changed'):
        mod.validate_code_capsule(capsule)

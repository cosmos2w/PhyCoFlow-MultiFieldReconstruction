"""CPU-only setup assembly preserves the paired TRAIN-derived configuration."""
import importlib.util
import json
from argparse import Namespace
from pathlib import Path

import pytest
import yaml


def test_prepare_reuses_matched_scales_and_refuses_overwrite(tmp_path, monkeypatch):
    script = Path(__file__).resolve().parents[1] / 'scripts/analysis/assemble_car_round2.py'
    spec = importlib.util.spec_from_file_location('round2_assembly', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def prepared(path, family, campaign):
        return {'family': family, 'matched_train_support_sha256': 'matched',
                'normalizer_sha256': 'normalizer', 'model_field_scales': [1.5, 2.5],
                'config': {'dataset': {'coordinate_dim': 2}, 'optimization': {}}}

    monkeypatch.setattr(module, 'inspect_prepared', prepared)
    args = Namespace(campaign=tmp_path, output_dir=tmp_path / 'setup',
                     ffm_prepared_run=tmp_path / 'ffm', senseiver_prepared_run=tmp_path / 'senseiver')
    module.prepare(args)
    receipt = json.loads((args.output_dir / 'setup_references.json').read_text())
    assert receipt['no_model_scores_computed'] is True
    assert receipt['wall_seconds'] >= 0
    for family in ('ffm', 'senseiver'):
        config = yaml.safe_load((args.output_dir / f'provisional_residual_probe_{family}.yaml').read_text())
        assert config['residual']['field_scales'] == [1.5, 2.5]
        assert config['residual']['enabled'] is True
        if family == 'ffm':
            assert config['optimization']['rollout_execution']['checkpointing'] is True
    with pytest.raises(FileExistsError):
        module.prepare(args)


def test_checkpoint_probe_changes_only_execution_flag(tmp_path):
    script = Path(__file__).resolve().parents[1] / 'scripts/analysis/assemble_car_round2.py'
    spec = importlib.util.spec_from_file_location('round2_assembly_checkpoint', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = tmp_path / 'source.yaml'
    initial = {'model': {'name': 'gl_rbf_cq'}, 'residual': {'enabled': True},
               'optimization': {'lr': .1, 'rollout_execution': {'context_cache': True}}}
    source.write_text(yaml.safe_dump(initial))
    output = tmp_path / 'checkpointed.yaml'
    args = Namespace(campaign=tmp_path, config=source, output=output)
    module.checkpoint_probe(args)
    result = yaml.safe_load(output.read_text())
    assert result['optimization']['rollout_execution'] == {'context_cache': True, 'checkpointing': True}
    assert result['optimization']['lr'] == initial['optimization']['lr']
    assert yaml.safe_load(source.read_text()) == initial
    with pytest.raises(FileExistsError):
        module.checkpoint_probe(args)


def test_arm_prefit_binds_actual_probe_configuration_and_source_artifacts():
    script = Path(__file__).resolve().parents[1] / 'scripts/analysis/assemble_car_round2.py'
    spec = importlib.util.spec_from_file_location('round2_assembly_bindings', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = {'source_checkpoint_sha256':'source', 'source_panel_sha256':'panel',
              'core_sha256':'core', 'fidelity_sha256':'fidelity'}
    prefit = {'version':'common_prefit_calibration_v1', 'binding':{'source_checkpoint_sha256':'source'},
              'inputs':{'residual_config_sha256':'config', 'source_panel_sha256':'panel',
                        'scalable_artifact_sha256':'core', 'fidelity_reference_sha256':'fidelity'},
              'scale_joint_ratio':2, 'scale_probe':{'schema':'car_round2_scale_probe_v1', 'reserved_task_used':False, 'decision':{'chosen_ratio':2}}}
    module.validate_prefit_inputs(prefit, source, {'sha256':'config'})
    with pytest.raises(ValueError, match='residual config'):
        module.validate_prefit_inputs(prefit, source, {'sha256':'different'})
    prefit['inputs']['source_panel_sha256'] = 'different'
    with pytest.raises(ValueError, match='SOURCE'):
        module.validate_prefit_inputs(prefit, source, {'sha256':'config'})

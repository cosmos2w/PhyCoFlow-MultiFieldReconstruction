import copy
import hashlib
import json

import pytest
import torch
from torch import nn

from phycoflow_reconstruction.models.flows.pointcloud.runtime.ema import ModelEMA
from phycoflow_reconstruction.training import scale_only_supplement as s6
from phycoflow_reconstruction.training.fidelity_budget import FidelityBudget
from phycoflow_reconstruction.training.fidelity_reference import (
    serialize_shared_reference,
    shared_reference_binding,
)
from phycoflow_reconstruction.training.post_training import (
    _add_source_relative_coherence,
    _fidelity_guard_report,
    _scheduled_model_lr,
)


def configs():
    fields = ["CH4", "CO", "T", "U_1", "p"]
    baseline = {
        "dataset": {"field_names": fields, "path": "/synthetic/train.h5"},
        "model": {"model_ema_enabled": True, "model_ema_eval": False,
                  "model_ema_decay": 0.999},
        "observations": {"seed": 42},
        "observation_consistency": {"mode": "endpoint_smooth"},
        "rollout": {"steps": 2},
        "trainable": {"scope": "full_model"},
        "runtime": {"seed": 43, "device": "cpu"},
        "optimization": {
            "epochs": 200, "batch_size": 32, "steps_per_epoch": 13,
            "lr": 5e-5, "weight_decay": 1e-6, "grad_clip": 1.0,
            "model_mode": "eval",
            "lr_warmup": {"enabled": True, "start_lr": 5e-6, "updates": 76},
            "multitask": {
                "family_scalarization": {"method": "none"},
                "coherence_aggregation": {"method": "torchjd_config"},
                "external_policy": {"method": "lagrangian"},
                "fidelity_budget": {"relative_increase": 0.1},
            },
        },
        "coherence": {
            "compute_budget": {"point_count": 4096, "query_seed": 100045},
            "family_balance": {"mode": "none"}, "schedule": {"start_epoch": 1},
            "families": {
                "global_distribution": {"enabled": True, "weight": 1.0},
                "multiscale_spectrum": {"enabled": True, "fields": fields, "weight": 1.0,
                    "relation_extension": {"rho_weight": 0.2},
                    "objective": {"kind": "signed"}},
                "topology": {"enabled": True, "weight": 1.0},
            },
            "query_coverage": {"enabled": True, "relation_seed": 600001},
            "frozen_family_artifacts": {
                "multiscale_spectrum": {"path": "/synthetic/scale.pt", "sha256": s6.SCALE_SHA},
                "topology": {"path": "/synthetic/shape.pt", "sha256": "a" * 64},
            },
        },
    }
    active = copy.deepcopy(baseline)
    for name in ("global_distribution", "topology"):
        active["coherence"]["families"][name]["enabled"] = False
    active["coherence"]["query_coverage"]["enabled"] = False
    artifacts = active["coherence"]["frozen_family_artifacts"]
    active["coherence"]["frozen_family_artifacts"] = {"multiscale_spectrum": artifacts["multiscale_spectrum"]}
    active["optimization"]["multitask"].pop("coherence_aggregation")
    active["optimization"]["multitask"]["family_scalarization"] = {"method": "fixed_sum"}
    active["optimization"]["epochs"] = 100
    return baseline, active


def test_recipe_rejects_unauthorized_changes():
    baseline, active = configs()
    s6.validate_recipe(active, baseline)
    broken = copy.deepcopy(active)
    broken["coherence"]["families"]["multiscale_spectrum"]["relation_extension"]["rho_weight"] *= 2
    with pytest.raises(ValueError, match="frozen family recipe"):
        s6.validate_recipe(broken, baseline)
    broken = copy.deepcopy(active)
    broken["optimization"]["multitask"]["coherence_aggregation"] = {"method": "torchjd_config"}
    with pytest.raises(ValueError, match="optimization"):
        s6.validate_recipe(broken, baseline)


def test_parent_reference_binds_original_recipe_and_rejects_false_parent(tmp_path, monkeypatch):
    baseline, active = configs()
    baseline = s6.validate_recipe(active, baseline)
    source_hashes = {"checkpoint": s6.SOURCE_SHA}
    arguments = {"field_names": baseline["dataset"]["field_names"],
                 "normalizer_sha256": "b" * 64, "dataset_fingerprint": "c" * 64,
                 "train_selection": {"split": "train", "sample_count": 2}}
    scales = {name: 1.0 for name in ("global_distribution", "multiscale_spectrum", "topology")}
    parent = shared_reference_binding(baseline, source_hashes=source_hashes,
                                      family_scales=scales, **arguments)
    reference = FidelityBudget.from_reference(arguments["field_names"], [1., 2., 3., 4., 5.],
                                               s6.C0, relative_increase=0.1)
    raw = serialize_shared_reference({"schema": "shared_fidelity_reference_v1", "binding": parent,
                                      "reference": reference.state_dict(),
                                      "origin": {"calibration_report_sha256": "d" * 64}})
    path = tmp_path / "reference.json"
    path.write_bytes(raw)
    monkeypatch.setattr(s6, "REFERENCE_SHA", hashlib.sha256(raw).hexdigest())
    loaded, unchanged, binding = s6.bind_reference(active, baseline, reference_path=path,
        source_hashes=source_hashes, binding_arguments=arguments)
    assert loaded.state_dict() == reference.state_dict()
    assert unchanged == raw == path.read_bytes()
    assert binding["parent_shared_reference_sha256"] == s6.REFERENCE_SHA
    false_baseline = copy.deepcopy(baseline)
    false_baseline["rollout"]["steps"] = 3
    with pytest.raises(ValueError, match="binding mismatch"):
        s6.bind_reference(active, false_baseline, reference_path=path,
                          source_hashes=source_hashes, binding_arguments=arguments)


def test_model_only_initialization_resets_ema_and_preserves_fresh_optimizer(tmp_path, monkeypatch):
    class Toy(nn.Module):
        def __init__(self):
            super().__init__(); self.w = nn.Parameter(torch.tensor([3.]))
            self._ema_eval = False; self._ema = ModelEMA(self)
    model = Toy(); target = Toy(); target.w.data.fill_(7)
    path = tmp_path / 'init.pt'
    torch.save({'model': target.state_dict(), 'optimizer': {'invalid': True},
                'training_aux_state': {'model_ema': {'invalid': True}}}, path)
    monkeypatch.setattr(s6, 'INIT_SHA', s6.file_sha256(path))
    monkeypatch.setattr(s6, 'MODEL_SHA', s6.model_state_digest(target))
    s6.initialize_live_model(model, path)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-6)
    assert model.w.item() == 7 and not optimizer.state
    assert model._ema.num_updates == 0 and torch.equal(model._ema.shadow['w'], model.w)
    assert model._ema_eval is False
    monkeypatch.setattr(s6, 'MODEL_SHA', '0' * 64)
    with pytest.raises(ValueError, match='model hash'):
        s6.initialize_live_model(model, path)


def test_warmup_reaches_base_on_update_76():
    _, config = configs()
    assert _scheduled_model_lr(config['optimization'], 0) == pytest.approx(5e-6)
    assert _scheduled_model_lr(config['optimization'], 75) == pytest.approx(5e-5)
    assert _scheduled_model_lr(config['optimization'], 76) == pytest.approx(5e-5)


def test_pinned_source_metrics_drive_guard_and_relative_coherence(tmp_path, monkeypatch):
    fields = ['CH4', 'CO', 'T', 'U_1', 'p']
    names = ('global_distribution', 'multiscale_spectrum', 'topology')
    row = {'field_error_sums': [1.] * 5, 'field_error_counts': [1.] * 5,
           'losses': {f'{name}.total': 2. for name in names},
           'family_weights': {name: {'weight': 1., 'calibration_scale': 1.} for name in names}}
    payload = {'manifest_sha256': s6.EVALUATION_MANIFEST_ID,
               'weight_identity_kind': 'model_state_sha256', 'weight_sha256': s6.SOURCE_MODEL_SHA,
               'split': 'validation', 'rows': [row] * 100}
    path = tmp_path / 'source.json'
    path.write_text(json.dumps(payload))
    monkeypatch.setattr(s6, 'SOURCE_VALIDATION_SHA', hashlib.sha256(path.read_bytes()).hexdigest())
    source = s6.load_source_validation_metrics(path, fields)
    initial_a200 = {'mse_normalized': 3., 'coherence': {'families': {
        'multiscale_spectrum': {'total': 9.}}}}
    after = {'mse_normalized': 1.2, 'coherence': {'families': {
        'multiscale_spectrum': {'total': 4.}}}}
    _add_source_relative_coherence(source, after)
    guard = _fidelity_guard_report(source, after, {'posttrain_fidelity': {
        'max_relative_mse_increase': 0.1}})
    assert after['coherence']['families']['multiscale_spectrum']['source_total'] == 2.
    assert guard['source_mse_normalized'] == 1. and guard['exceeded']
    assert initial_a200['mse_normalized'] == 3.
    payload['weight_sha256'] = '0' * 64
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match='hash mismatch'):
        s6.load_source_validation_metrics(path, fields)

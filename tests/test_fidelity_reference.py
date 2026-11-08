"""The shared SOURCE reference is exact, opt-in and recovery-bound."""

import hashlib
import json
from copy import deepcopy
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from phycoflow_reconstruction.training.fidelity_budget import FidelityBudget
from phycoflow_reconstruction.training.fidelity_reference import (
    create_shared_reference,
    load_shared_reference,
    serialize_shared_reference,
    shared_reference_binding,
    validate_shared_reference_recovery,
)


def _digest(raw):
    return hashlib.sha256(raw).hexdigest()


def _case(tmp_path):
    config = {
        "dataset": {"path": "/data/train.h5", "physical_measure": "native_area"},
        "model": {"name": "example", "mode": "matched"},
        "trainable": {"scope": "full_model"},
        "observations": {"seed": 42},
        "observation_consistency": {"mode": "endpoint_smooth", "final_clamp": True},
        "rollout": {"steps": 2, "solver": "euler"},
        "coherence": {"compute_budget": {"point_count": 4096}, "families": {
            "global_distribution": {"weight": 1},
            "multiscale_spectrum": {"objective": {"kind": "signed", "joint_weight": .25}},
            "topology": {"weight": 1},
        }},
        "optimization": {"epochs": 50, "batch_size": 32, "lr": .00005, "model_mode": "eval",
            "multitask": {"external_policy": {"method": "lagrangian"},
                "fidelity_budget": {"metric": "endpoint_mse_model_units", "relative_increase": .1,
                    "absolute_allowance": 0., "denominator_floor": 1e-8,
                    "calibration_batches": 8, "calibration_draws": 2, "calibration_seed": 700103}}},
        "runtime": {"seed": 43, "device": "cuda:0", "deterministic": True,
                    "data_strategy": "async_cpu", "num_workers": 0},
    }
    arguments = {"field_names": ("T", "CO"), "normalizer_sha256": "a" * 64,
                 "source_hashes": {"checkpoint": "b" * 64, "resolved_config": "c" * 64},
                 "dataset_fingerprint": "d" * 64,
                 "train_selection": {"split": "train", "strategy": "temporal", "frame_indices": [0, 2],
                                     "trajectory_indices": [0], "sample_count": 2, "training_subset_sha256": None},
                 "family_scales": {"global_distribution": 1., "multiscale_spectrum": 1., "topology": 1.}}
    reference = FidelityBudget.from_reference(arguments["field_names"], [0.5, 2.], 3.25,
                                               relative_increase=.1, absolute_allowance=0.)
    report = {"settings": deepcopy(config["optimization"]["multitask"]["fidelity_budget"]),
              "reference": reference.state_dict()}
    report_path = tmp_path / "original_reference.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    return config, arguments, reference, report_path


def _artifact(tmp_path):
    config, arguments, reference, report_path = _case(tmp_path)
    report_sha = _digest(report_path.read_bytes())
    artifact = create_shared_reference(report_path, report_sha, config, **arguments)
    raw = serialize_shared_reference(artifact)
    path = tmp_path / "shared.json"
    path.write_bytes(raw)
    return config, arguments, reference, report_path, artifact, path, _digest(raw)


def test_shared_baseline_and_gram_extension_have_identical_fixed_reference(tmp_path):
    config, arguments, reference, _, artifact, path, digest = _artifact(tmp_path)
    baseline = shared_reference_binding(config, **arguments)
    assert load_shared_reference(path, digest, baseline)[0] == reference
    candidate = deepcopy(config)
    candidate["coherence"]["families"]["multiscale_spectrum"]["gram_extension"] = {
        "definition": "self_mutual_gram_v1", "weight": .02, "std_floor": .001}
    candidate["optimization"]["epochs"] = 500
    candidate["optimization"]["lr"] = .001
    candidate["runtime"]["seed"] = 999
    candidate["runtime"]["plot_every_steps"] = 100
    candidate["runtime"]["vram_dataset_threshold_gb"] = 99
    candidate["output"] = {"experiment_name": "different_formal_run"}
    candidate["coherence"]["family_balance"] = {"mode": "none", "calibration_batches": 99,
                                                  "gradient_diagnostics_every_epochs": 20}
    candidate["optimization"]["multitask"]["fidelity_budget"].update(
        reference_artifact=str(path), reference_sha256=digest)
    assert shared_reference_binding(candidate, **arguments) == baseline
    loaded, _ = load_shared_reference(path, digest, shared_reference_binding(candidate, **arguments))
    assert loaded.reference_risks == reference.reference_risks
    assert loaded.objective_scale == 3.25
    assert artifact["origin"]["calibration_report_sha256"] == _digest((tmp_path / "original_reference.json").read_bytes())
    with pytest.raises(ValueError, match="unaugmented baseline"):
        create_shared_reference(tmp_path / "original_reference.json", artifact["origin"]["calibration_report_sha256"], candidate, **arguments)


def test_conditional_state_shares_only_the_exact_v3_reference(tmp_path):
    config, arguments, _, _, _, path, digest = _artifact(tmp_path)
    state = config["coherence"]["families"]["global_distribution"]
    state.update(definition="marginal_mixed_weighted_v3", components={"cross": {"seed": 1234, "weight": 1.0}})
    # Rebind the fixture to this explicitly versioned reference.
    artifact = json.loads(path.read_text())
    artifact["binding"] = shared_reference_binding(config, **arguments)
    raw = serialize_shared_reference(artifact)
    path.write_bytes(raw)
    digest = _digest(raw)
    candidate = deepcopy(config)
    conditional = candidate["coherence"]["families"]["global_distribution"]
    conditional.update(definition="conditional_quantile_fair_v1", base_definition="marginal_mixed_weighted_v3", conditional_draws=2)
    load_shared_reference(path, digest, shared_reference_binding(candidate, **arguments))
    for key, value in (("base_definition", "other"), ("conditional_draws", 4)):
        tampered = deepcopy(candidate)
        tampered["coherence"]["families"]["global_distribution"][key] = value
        with pytest.raises(ValueError, match="exact v3 State base and J2"):
            shared_reference_binding(tampered, **arguments)
    for key, value in (("seed", 1235), ("weight", .9)):
        tampered = deepcopy(candidate)
        tampered["coherence"]["families"]["global_distribution"]["components"]["cross"][key] = value
        with pytest.raises(ValueError, match="binding mismatch"):
            load_shared_reference(path, digest, shared_reference_binding(tampered, **arguments))


@pytest.mark.parametrize("change", ["source", "normalizer", "field_order", "dataset", "train_split", "scales", "endpoint", "observations", "query", "budget", "scale_anchor", "other_family", "model", "estimator"])
def test_binding_rejects_incompatible_contracts(tmp_path, change):
    config, arguments, _, _, artifact, path, digest = _artifact(tmp_path)
    if change == "source":
        arguments["source_hashes"]["checkpoint"] = "e" * 64
    elif change == "normalizer":
        arguments["normalizer_sha256"] = "e" * 64
    elif change == "field_order":
        arguments["field_names"] = ("CO", "T")
    elif change == "dataset":
        arguments["dataset_fingerprint"] = "e" * 64
    elif change == "train_split":
        arguments["train_selection"]["frame_indices"] = [1, 2]
    elif change == "scales":
        arguments["family_scales"]["topology"] = 2.
    elif change == "endpoint":
        config["rollout"]["steps"] = 3
    elif change == "observations":
        config["observations"]["seed"] += 1
    elif change == "query":
        config["coherence"]["compute_budget"]["point_count"] = 2048
    elif change == "budget":
        config["optimization"]["multitask"]["fidelity_budget"]["calibration_seed"] += 1
    elif change == "scale_anchor":
        config["coherence"]["families"]["multiscale_spectrum"]["objective"]["joint_weight"] = .3
    elif change == "other_family":
        config["coherence"]["families"]["topology"]["weight"] = .9
    elif change == "model":
        config["model"]["mode"] = "different"
    elif change == "estimator":
        artifact["binding"]["estimator"] = "different_estimator"
        raw = serialize_shared_reference(artifact)
        path.write_bytes(raw)
        digest = _digest(raw)
    with pytest.raises(ValueError, match="binding mismatch"):
        load_shared_reference(path, digest, shared_reference_binding(config, **arguments))


def test_original_calibration_provenance_and_consistency_are_required(tmp_path):
    config, arguments, _, report_path = _case(tmp_path)
    with pytest.raises(ValueError, match="calibration report hash mismatch"):
        create_shared_reference(report_path, "f" * 64, config, **arguments)
    report = json.loads(report_path.read_text())
    report["settings"]["calibration_seed"] += 1
    report_path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="calibration settings differ"):
        create_shared_reference(report_path, _digest(report_path.read_bytes()), config, **arguments)


def test_effective_seed_fallback_and_train_mode_are_bound(tmp_path):
    config, arguments, _, _ = _case(tmp_path)
    baseline = shared_reference_binding(config, **arguments)
    # Explicit observation and query seeds make an eval-mode runtime seed unrelated.
    changed = deepcopy(config)
    changed["runtime"]["seed"] = 999
    assert shared_reference_binding(changed, **arguments) == baseline
    del config["observations"]["seed"]
    fallback = shared_reference_binding(config, **arguments)
    changed = deepcopy(config)
    changed["runtime"]["seed"] += 1
    assert shared_reference_binding(changed, **arguments) != fallback
    config["observations"]["seed"] = 42
    config["optimization"]["model_mode"] = "train"
    train_mode = shared_reference_binding(config, **arguments)
    changed = deepcopy(config)
    changed["runtime"]["seed"] += 1
    assert shared_reference_binding(changed, **arguments) != train_mode


def test_recovery_rejects_tampered_copy_manifest_checkpoint_or_reference(tmp_path):
    config, arguments, reference, _, _, path, digest = _artifact(tmp_path)
    loaded, raw = load_shared_reference(path, digest, shared_reference_binding(config, **arguments))
    local = tmp_path / "run_local.json"
    local.write_bytes(raw)
    checkpoint = {"shared_fidelity_reference_sha256": digest, "external_policy": {"reference": reference.state_dict()}}
    manifest = {"shared_fidelity_reference_sha256": digest}
    validate_shared_reference_recovery(loaded, digest, checkpoint, manifest, local)
    for key in ("checkpoint", "manifest", "reference", "copy"):
        changed_checkpoint = deepcopy(checkpoint)
        changed_manifest = deepcopy(manifest)
        if key == "checkpoint":
            changed_checkpoint["shared_fidelity_reference_sha256"] = "e" * 64
        elif key == "manifest":
            changed_manifest["shared_fidelity_reference_sha256"] = "e" * 64
        elif key == "reference":
            changed_checkpoint["external_policy"]["reference"]["c0"] += 1
        else:
            local.write_text("tampered")
        with pytest.raises(ValueError, match="artifact or checkpoint mismatch"):
            validate_shared_reference_recovery(loaded, digest, changed_checkpoint, changed_manifest, local)
        local.write_bytes(raw)
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        load_shared_reference(path, "f" * 64, shared_reference_binding(config, **arguments))


def test_checkpoint_payload_binds_only_opt_in_reference(monkeypatch):
    from phycoflow_reconstruction.training import post_training

    @dataclass
    class Spec:
        name: str = "synthetic"

    class Dataset:
        data_spec = Spec()
        normalizer = SimpleNamespace(state_dict=lambda: {"scale": 1})

        def __len__(self):
            return 2

    monkeypatch.setattr(post_training, "checkpoint_model_state", lambda model: {"synthetic": True})
    monkeypatch.setattr(post_training, "add_training_aux_state", lambda payload, model: payload)
    monkeypatch.setattr(post_training, "sample_exposure", lambda *args: {})
    monkeypatch.setattr(post_training, "source_checkpoint_path", lambda config: "source.pt")
    policy = SimpleNamespace(state_dict=lambda: {"reference": "saved"},
                             shared_fidelity_reference_sha256="e" * 64)
    kwargs = {"global_step": 0, "config": {"source_run": "source", "optimization": {"batch_size": 1}},
              "train_dataset": Dataset(), "families": {}, "source_hashes_before": {"checkpoint": "b" * 64},
              "index_generator": SimpleNamespace(get_state=lambda: "rng"),
              "device": SimpleNamespace(type="cpu"), "config_sha256": "config",
              "coherence_calibration_sha256": "family", "family_scales": {}, "policy": policy}
    checkpoint = post_training._post_checkpoint_payload(
        object(), SimpleNamespace(state_dict=lambda: {"synthetic": "optimizer state"}), **kwargs
    )
    assert checkpoint["external_policy"] == {"reference": "saved"}
    assert checkpoint["shared_fidelity_reference_sha256"] == "e" * 64
    del policy.shared_fidelity_reference_sha256
    legacy = post_training._post_checkpoint_payload(
        object(), SimpleNamespace(state_dict=dict), **kwargs
    )
    assert "shared_fidelity_reference_sha256" not in legacy

"""Check parsing gaps that can distort resumed epoch/native evidence."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

_path = (
    Path(__file__).resolve().parents[1] / "scripts/visualization/upgrade_1002_r2_campaign_report.py"
)
_spec = importlib.util.spec_from_file_location("r2_campaign_report", _path)
report = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(report)
_audit_path = Path(__file__).resolve().parents[1] / "scripts/evaluation/audit_upgrade_1002_r2.py"
_audit_spec = importlib.util.spec_from_file_location("r2_native_fields", _audit_path)
audit = importlib.util.module_from_spec(_audit_spec)
_audit_spec.loader.exec_module(audit)


def test_partial_epoch_merge_uses_metric_finite_counts_not_batch_counts():
    rows = [
        {
            "step": 19,
            "batches": 19,
            "native_data_loss": 1.0,
            "gradient/sparse": 0.1,
            "native_monitor/train/ratio": 1.02,
            "finite_counts": {
                "native_data_loss": 19,
                "gradient/sparse": 1,
                "native_monitor/train/ratio": 1,
            },
        },
        {
            "step": 38,
            "batches": 19,
            "native_data_loss": 3.0,
            "gradient/sparse": 0.3,
            "finite_counts": {"native_data_loss": 19, "gradient/sparse": 5},
        },
        {"step": 76, "batches": 38, "native_data_loss": None, "total_loss": 0.01},
    ]
    epochs = report.epoch_means(rows, 38)
    assert epochs[0]["epoch"] == 1 and epochs[0]["epoch_complete"]
    assert epochs[0]["native_data_loss"] == 2.0
    assert epochs[0]["gradient/sparse"] == pytest.approx((0.1 + 0.3 * 5) / 6)
    assert epochs[0]["native_monitor/train/ratio"] == 1.02
    assert "native_data_loss" not in epochs[1]


def test_read_run_matches_child_epoch_source_fields_and_sparse_native(tmp_path):
    (tmp_path / "metrics").mkdir()
    (tmp_path / "evaluation").mkdir()
    (tmp_path / "run_manifest.json").write_text(json.dumps({"steps_per_epoch": 38}))
    (tmp_path / "resolved_config.yaml").write_text(
        yaml.safe_dump(
            {
                "runtime": {"plot_format": "pdf"},
                "checkpointing": {"selection_metric": "coherence_with_fidelity"},
            }
        )
    )
    (tmp_path / "evaluation/before.json").write_text(
        json.dumps({"mse_normalized": 2.0, "per_field_mse_normalized": {"p": 2.0}})
    )
    (tmp_path / "metrics/history.jsonl").write_text(
        json.dumps({"step": 3800, "batches": 38, "native_data_loss": 1.0}) + "\n"
    )
    selection = {
        "step": 3800,
        "metric": 0.8,
        "eligible": False,
        "mse": 4.0,
        "metrics": {
            "per_field_mse_normalized": {"p": 4.0},
            "per_field_relative_l2_physical": {"p": 0.2},
        },
    }
    source_row = {
        "step": 0,
        "metric": 1.0,
        "eligible": True,
        "mse": 2.0,
        "metrics": {"per_field_mse_normalized": {"p": 2.0}},
    }
    (tmp_path / "metrics/coherence_validation.jsonl").write_text(
        json.dumps(selection) + "\n" + json.dumps(source_row) + "\n"
    )
    (tmp_path / "evaluation/native_epoch_100.json").write_text(
        json.dumps({"validation": {"ratio": 1.04}})
    )
    parsed = report.read_run("arm", tmp_path)
    assert parsed["selection"][0]["epoch"] == 0 and parsed["selection"][0]["is_source_baseline"]
    assert (
        parsed["selection"][1]["epoch"] == 100 and not parsed["selection"][1]["is_source_baseline"]
    )
    assert parsed["selection"][1]["per_field_relative_change"]["p"] == 1.0
    assert parsed["selection"][1]["metric"] == parsed["selection"][1]["pareto_metric"] == 0.8
    assert parsed["native"] == [{"epoch": 100, "validation": {"ratio": 1.04}}]
    summary = report.summarize(parsed)
    assert summary["mature_checkpoint_evidence"]["best_eligible_mature"] is None
    assert summary["epoch_series"]["native_data_loss"]["windows"]["epochs_1_25"]["median"] is None
    parsed["epochs"][0]["epoch_complete"] = False
    assert not report.summarize(parsed)["minimum_horizon_observed"]


def test_legacy_reader_preserves_raw_selector_and_derives_own_definition_diagnostics(tmp_path):
    (tmp_path / "metrics").mkdir()
    (tmp_path / "evaluation").mkdir()
    config = {
        "checkpointing": {"selection_metric": "topology_with_fidelity"},
        "coherence": {"families": {"topology": {"components": {"mutual": {"lines": 3}}}}},
    }
    (tmp_path / "resolved_config.yaml").write_text(yaml.safe_dump(config))
    (tmp_path / "run_manifest.json").write_text(json.dumps({"steps_per_epoch": 38}))
    source = {
        "mse_normalized": 2.0,
        "per_field_mse_normalized": {"p": 2.0},
        "coherence": {
            "families": {
                name: {"total": value} for name, value in zip(report.FAMILIES, (2.0, 5.0, 4.0))
            }
        },
    }
    (tmp_path / "evaluation/before.json").write_text(json.dumps(source))
    rows = [{"step": 0, "eligible": True, "metric": 4.0, "mse": 2.0, "metrics": source}]
    for epoch, totals in [(100, (1.0, 5.0, 3.0)), (110, (4.0, 10.0, 2.0))]:
        metrics = {
            **source,
            "coherence": {
                "families": {name: {"total": value} for name, value in zip(report.FAMILIES, totals)}
            },
        }
        rows.append(
            {
                "step": epoch * 38,
                "eligible": True,
                "metric": totals[-1],
                "mse": 2.0,
                "failed_fidelity_fields": [],
                "metrics": metrics,
            }
        )
    (tmp_path / "metrics/topology_validation.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n"
    )
    (tmp_path / "metrics/coherence_validation.jsonl").write_text(
        json.dumps({"step": 3800, "metric": 999.0}) + "\n"
    )
    parsed = report.read_run("legacy", tmp_path)
    baseline, first, latest = parsed["selection"]
    assert baseline["is_source_baseline"] and baseline["epoch"] == 0
    assert baseline["metric"] == 4.0 and baseline["pareto_metric"] == 1.0
    assert first["metric"] == 3.0 and latest["metric"] == 2.0
    assert first["family_ratios"] == dict(zip(report.FAMILIES, (0.5, 1.0, 0.75)))
    assert first["pareto_metric"] == 0.75 and latest["pareto_metric"] == 1.5
    assert latest["family_ratio_role"] == "within_definition_reporting_only"
    assert (
        report.summarize(parsed)["mature_checkpoint_evidence"]["best_eligible_mature"]["epoch"]
        == 110
    )
    assert first["failed_fidelity_fields"] == [] and first["per_field_mse"] == {"p": 2.0}
    assert parsed["effective_topology_bank"] == "fixed 3-line bank"
    assert report.topology_bank_label(config, training=True) == "fixed 3-line bank"
    modern = {
        "coherence": {
            "families": {
                "topology": {
                    "components": {"mutual": {"line_bank_size": 16, "training_subset_size": 4}}
                }
            }
        }
    }
    assert report.topology_bank_label(modern) == "full 16-line bank"
    assert report.topology_bank_label(modern, training=True) == "sampled 4/16-line bank"
    assert report.family_mean_diagnostic({"topology": 0.5}) is None


def test_native_fields_reject_immature_out_of_split_or_repeated_probes():
    with pytest.raises(ValueError, match="mature"):
        audit.validate_native_field_indices([32, 437], 1000, 99)
    with pytest.raises(ValueError, match="unique"):
        audit.validate_native_field_indices([32, 32], 1000, 100)
    with pytest.raises(ValueError, match="outside"):
        audit.validate_native_field_indices([32, 1000], 1000, 100)
    audit.validate_native_field_indices([32, 437, 999], 1000, 100)


def test_legacy_audit_metadata_has_fixed_bank_and_no_unobserved_pooled_B(tmp_path):
    run = {
        "run_dir": str(tmp_path / "run"),
        "manifest": {"source_hashes": {"checkpoint": "source"}},
        "config": {
            "checkpointing": {"selection_metric": "topology_with_fidelity"},
            "coherence": {"families": {"topology": {"components": {"mutual": {"lines": 3}}}}},
        },
    }
    endpoint = {
        "mse_normalized": 1.0,
        "per_field_mse_normalized": {"p": 1.0},
        "coherence": {"families": {name: {"total": 2.0} for name in report.FAMILIES}},
    }
    payload = {
        "schema": "phycoflow.upgrade_1002_r2_audit.v1",
        "run": run["run_dir"],
        "panel": "extended",
        "split": "validation",
        "epoch": 100,
        "test_locked": True,
        "dataset_indices": list(range(128)),
        "source_checkpoint_sha256": "source",
        "results": {"source_live": endpoint, "candidate_live": endpoint},
        "fidelity_coherence_audit": {
            "metric": 0.9,
            "eligible": True,
            "family_source_normalized_scores": dict(zip(report.FAMILIES, (0.6, 1.0, 1.1))),
        },
    }
    path = tmp_path / "audit.json"
    path.write_text(json.dumps(payload))
    result = report.read_audit(run, path)
    assert result["effective_topology_bank"] == "fixed 3-line bank"
    assert not result["pooled_B_present"] and "no_pooled_B" in result["group_role"]
    assert result["family_ratio_role"] == "within_definition_reporting_only"
    assert result["pareto_metric"] == pytest.approx(0.9)
    assert result["metric"] == 0.9 and result["eligible"] is True


def test_native_field_archive_has_matched_live_full_grid_and_actual_t_sensors(tmp_path):
    from test_upgrade_1002_native_audit import _Dataset, _grid_sample

    from phycoflow_reconstruction.contracts import DataSpec
    from phycoflow_reconstruction.data.normalization import FieldNormalizer
    from phycoflow_reconstruction.data.sensor_protocols import SensorProtocol

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(2.0))
            self._ema_eval = True

        def reconstruct(self, batch, *, steps, generator):
            assert batch.target_fields is None and self._ema_eval is False
            assert steps == 2 and batch.query_coords.shape[1] == 16
            values = self.weight + torch.rand(len(batch.sample_ids), 16, 2, generator=generator)
            return SimpleNamespace(prediction=values)

    spec = DataSpec(("u", "T"), ("1", "1"), 2, (4, 4))
    normalizer = FieldNormalizer.identity(2)
    dataset = _Dataset(
        tmp_path / "fixture.h5", [_grid_sample(i, spec) for i in range(4)], spec, normalizer
    )
    live, source = Model().train(), Model().eval()
    runtime = SimpleNamespace(
        model=live, dataset=dataset, device=torch.device("cpu"), generation_steps=2
    )
    torch_state = torch.get_rng_state().clone()
    result = audit.save_native_fields(
        runtime,
        source,
        [1, 3],
        {"evaluation": {"seed": 2027}},
        SensorProtocol(field_counts={"T": 3}),
        tmp_path,
        epoch=100,
    )
    assert result["full_native_grid"] and not result["runs_extra_ABC_or_PH"]
    with np.load(tmp_path / result["path"], allow_pickle=False) as payload:
        assert payload["source_live"].shape == (2, 16, 2)
        np.testing.assert_array_equal(payload["source_live"], payload["candidate_live"])
        assert payload["observation_coordinates_raw"].shape == (2, 3, 2)
        assert np.all(payload["observation_field_ids"] == 1)
        assert payload["observation_valid_mask"].all()
        assert payload["epoch"] == 100
    assert live.training and not source.training and live._ema_eval and source._ema_eval
    assert torch.equal(torch_state, torch.get_rng_state())


def test_saved_development_audits_preserve_panels_source_baseline_and_noise_role(tmp_path):
    run = {
        "run_dir": str(tmp_path / "run"),
        "manifest": {"source_hashes": {"checkpoint": "source_hash"}},
    }
    source = {
        "mse_normalized": 2.0,
        "per_field_mse_normalized": {"p": 2.0},
        "coherence": {"families": {"topology": {"total": 4.0}}},
    }
    candidate = {
        "mse_normalized": 2.1,
        "per_field_mse_normalized": {"p": 2.1},
        "per_field_relative_l2_physical": {"p": 0.2},
        "pooled_B": {"total": 0.1},
        "coherence": {"families": {"topology": {"total": 3.0}}},
        "group_reports": [
            {"sample_ids": [str(index) for index in range(32)], "mse_normalized": 2.1}
        ],
    }
    payload = {
        "schema": "phycoflow.upgrade_1002_r2_audit.v1",
        "run": run["run_dir"],
        "panel": "extended",
        "split": "validation",
        "epoch": 100,
        "scientific_maturity": True,
        "dataset_indices": list(range(128)),
        "source_checkpoint_sha256": "source_hash",
        "checkpoint_sha256": "child_hash",
        "sensor_manifest_sha256": "sensor_hash",
        "source_weight_selection": "live",
        "test_locked": True,
        "results": {"source_live": source, "candidate_live": candidate},
        "fidelity_coherence_audit": {
            "metric": 0.8,
            "eligible": False,
            "family_source_normalized_scores": {"topology": 0.75},
        },
    }
    path = tmp_path / "audit.json"
    path.write_text(json.dumps(payload))
    noise_path = tmp_path / "native_noise_check.json"
    noise_path.write_text(
        json.dumps(
            {
                "schema": "phycoflow.upgrade_1002_r2_native_noise_check.v1",
                "role": "development_noise_check_only_no_selection_controller_or_fitting",
                "epoch": 100,
                "panel": "extended",
                "source_checkpoint_sha256": "source_hash",
                "checkpoint_sha256": "child_hash",
                "sensor_manifest_sha256": "sensor_hash",
                "requested_seeds": [2027, 6027, 2027],
                "baseline_replay_precision": [
                    {"seed": 2027, "candidate_abs_difference": 0.0, "source_abs_difference": 0.0}
                ],
                "matched_native_draws": {
                    "extended": {
                        "candidate": 1.01,
                        "source": 1.0,
                        "draws": [
                            {"seed": 2027, "candidate": 1.01, "source": 1.0},
                            {"seed": 6027, "candidate": 1.02, "source": 1.0},
                            {"seed": 2027, "candidate": 1.01, "source": 1.0},
                        ],
                    }
                },
            }
        )
    )
    parsed = report.read_audit(run, path)
    assert parsed["sample_count"] == 128 and parsed["panel"] == "extended"
    assert (
        parsed["source_baseline"]["is_source_baseline"] and parsed["source_baseline"]["epoch"] == 0
    )
    assert parsed["source_baseline"]["mse"] == 2.0
    assert parsed["per_field_relative_change"]["p"] == pytest.approx(0.05)
    assert (
        parsed["groups"][0]["role"] == "candidate_live"
        and parsed["groups"][0]["sample_count"] == 32
    )
    assert parsed["candidate"]["pooled_B"]["total"] == 0.1
    noise = parsed["native_noise_check"]
    assert noise["summary"]["extended"]["unique_seed_count"] == 2
    assert not noise["summary"]["extended"]["controller_calibration"]
    assert noise["baseline_replay_precision"][0]["candidate_abs_difference"] == 0
    bad_noise = json.loads(noise_path.read_text())
    bad_noise["role"] = "controller_calibration"
    noise_path.write_text(json.dumps(bad_noise))
    with pytest.raises(ValueError, match="never controller calibration"):
        report.read_audit(run, path)
    noise_path.unlink()
    payload.update(panel="audit", dataset_indices=list(range(64)))
    path.write_text(json.dumps(payload))
    fresh = report.read_audit(run, path)
    assert (
        fresh["panel"] == "audit"
        and fresh["sample_count"] == 64
        and fresh["native_noise_check"] is None
    )
    payload["source_checkpoint_sha256"] = "other_source"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="source hash"):
        report.read_audit(run, path)
    payload["source_checkpoint_sha256"] = "source_hash"
    payload["split"] = "test"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="development panels"):
        report.read_audit(run, path)
    payload["split"] = "validation"
    payload["run"] = str(tmp_path / "other_run")
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="report label"):
        report.read_audit(run, path)


def _locked_test_recipe_fixture(tmp_path):
    run, source = tmp_path / "run", tmp_path / "source"
    for folder in (run, source):
        (folder / "checkpoints").mkdir(parents=True)
        (folder / "checkpoints/last.pt").write_bytes(b"CPU identity fixture, not a model")
    config = {
        "source_run": str(source),
        "source_checkpoint": "last",
        "evaluation": {"query_points": 4096},
        "coherence": {"families": {"topology": {"components": {"mutual": {"line_bank_size": 16}}}}},
    }
    (run / "resolved_config.yaml").write_text(yaml.safe_dump(config))
    (run / "run_manifest.json").write_text(
        json.dumps(
            {
                "steps_per_epoch": 38,
                "config_sha256": audit.config_digest(config),
                "source_hashes": {"checkpoint": audit.file_sha256(source / "checkpoints/last.pt")},
            }
        )
    )
    declaration = tmp_path / "validation_declaration.json"
    declaration.write_text(json.dumps({"test_locked": True, "split": "validation"}))
    indices = audit.selection_indices(1000, 128, 16)
    panel = {
        "schema": "phycoflow.upgrade_1002_r2_final_test_panel.v1",
        "split": "test",
        "data_role": "locked_final_test",
        "access_allowed": False,
        "dataset_size": 1000,
        "sample_count": 128,
        "dataset_indices": indices,
        "strata": 16,
        "groups": [indices[i : i + 32] for i in range(0, 128, 32)],
        "equal_sample_weights": True,
        "sensor_seed_offset": 2000,
        "generation_seed": 6027,
        "native_seeds": [6027, 7027],
        "weight_selection": "live",
        "query_count": 4096,
        "topology_bank": "full_16",
        "dataset_opened_to_declare": False,
        "model_or_recipe_chosen": False,
    }
    panel_path = tmp_path / "_audit/R2_campaign/final_test_panel_declaration.json"
    panel_path.parent.mkdir(parents=True)
    panel_path.write_text(json.dumps(panel))
    lock = {
        "schema": "phycoflow.upgrade_1002_r2_final_recipe_lock.v1",
        "locked_final_recipe": True,
        "use_test_for_selection": False,
        "use_test_for_adaptation": False,
        "run": str(run.resolve()),
        "checkpoint": str((run / "checkpoints/last.pt").resolve()),
        "source_checkpoint": str((source / "checkpoints/last.pt").resolve()),
        "checkpoint_sha256": audit.file_sha256(run / "checkpoints/last.pt"),
        "source_checkpoint_sha256": audit.file_sha256(source / "checkpoints/last.pt"),
        "config_sha256": audit.file_sha256(run / "resolved_config.yaml"),
        "run_manifest_sha256": audit.file_sha256(run / "run_manifest.json"),
        "declaration_sha256": audit.file_sha256(declaration),
        "test_panel_declaration_sha256": audit.file_sha256(panel_path),
    }
    lock_path = tmp_path / "final_recipe_lock.json"
    lock_path.write_text(json.dumps(lock))
    return run, declaration, panel_path, lock_path, lock


def _locked_fixture_payload(run, step=3800):
    manifest = json.loads((run / "run_manifest.json").read_text())
    return {
        "global_step": step,
        "config_sha256": manifest["config_sha256"],
        "source_hashes": manifest["source_hashes"],
    }


def test_final_test_cpu_lock_guards_and_campaign_wide_single_access(tmp_path, monkeypatch):
    run, declaration, panel_path, lock_path, lock = _locked_test_recipe_fixture(tmp_path)
    monkeypatch.setattr(audit, "load_project_checkpoint", lambda _: _locked_fixture_payload(run))

    def forbidden(*args, **kwargs):
        raise AssertionError("CPU guard must never open a dataset, model, or GPU runtime")

    monkeypatch.setattr(audit, "load_evaluation_runtime", forbidden)
    monkeypatch.setattr(audit, "open_field_dataset", forbidden)
    valid = audit.validate_recipe_lock(lock_path, run, "last", declaration, tmp_path)
    assert valid["epoch"] == 100 and valid["panel"]["sample_count"] == 128
    receipt = audit.begin_final_test_access(valid, tmp_path, tmp_path / "result")
    initial_bytes = Path(receipt["path"]).read_bytes()
    valid["lock"] = {**lock, "checkpoint_sha256": "different_recipe"}
    with pytest.raises(FileExistsError):
        audit.begin_final_test_access(valid, tmp_path, tmp_path / "different_result")
    assert Path(receipt["path"]).read_bytes() == initial_bytes
    for key, value in (
        ("locked_final_recipe", False),
        ("checkpoint_sha256", "changed"),
        ("config_sha256", "changed"),
        ("run_manifest_sha256", "changed"),
        ("source_checkpoint_sha256", "changed"),
        ("declaration_sha256", "changed"),
        ("test_panel_declaration_sha256", "changed"),
        ("use_test_for_adaptation", True),
        ("use_test_for_selection", True),
    ):
        lock_path.write_text(json.dumps({**lock, key: value}))
        with pytest.raises(ValueError):
            audit.validate_recipe_lock(lock_path, run, "last", declaration, tmp_path)
    lock_path.write_text(json.dumps(lock))
    monkeypatch.setattr(
        audit, "load_project_checkpoint", lambda _: _locked_fixture_payload(run, 3762)
    )
    with pytest.raises(ValueError, match="mature"):
        audit.validate_recipe_lock(lock_path, run, "last", declaration, tmp_path)
    monkeypatch.setattr(audit, "load_project_checkpoint", lambda _: _locked_fixture_payload(run))
    panel = json.loads(panel_path.read_text())
    panel["dataset_indices"][0] = 1
    panel_path.write_text(json.dumps(panel))
    lock_path.write_text(
        json.dumps({**lock, "test_panel_declaration_sha256": audit.file_sha256(panel_path)})
    )
    with pytest.raises(ValueError, match="immutable"):
        audit.validate_recipe_lock(lock_path, run, "last", declaration, tmp_path)


def test_final_test_cli_rejects_before_any_runtime_or_receipt(tmp_path, monkeypatch):
    # Redirect only the script's repository identity to a fresh CPU fixture tree.
    repository = tmp_path / "repository"
    monkeypatch.setattr(audit, "__file__", str(repository / "scripts/evaluation/audit.py"))
    fixture_root = repository / "cases/turbulent_combustion/runs/Test_1002/fixture"
    run, declaration, _, lock_path, _ = _locked_test_recipe_fixture(fixture_root)

    def forbidden(*args, **kwargs):
        raise AssertionError("rejected CLI must not open test or reserve access")

    monkeypatch.setattr(audit, "load_evaluation_runtime", forbidden)
    monkeypatch.setattr(audit, "begin_final_test_access", forbidden)
    base = [
        "--declaration",
        str(declaration),
        "--run",
        str(run),
        "--output",
        str(fixture_root / "output"),
        "--panel",
        "test",
        "--device",
        "cpu",
    ]
    with pytest.raises(ValueError, match="recipe-lock"):
        audit.main(base)
    with pytest.raises(ValueError, match="forbids"):
        audit.main(base + ["--recipe-lock", str(lock_path), "--native-field-index", "32"])
    with pytest.raises(ValueError, match="forbids"):
        audit.main(base + ["--recipe-lock", str(lock_path), "--native-noise-seed", "9027"])
    with pytest.raises(ValueError, match="forbids"):
        audit.main(base + ["--recipe-lock", str(lock_path), "--pressure-samples", "2"])


def test_locked_test_review_file_hashes_source_metadata_and_payload_association(
    tmp_path, monkeypatch
):
    run, declaration, _, lock_path, lock = _locked_test_recipe_fixture(tmp_path)
    source_root = Path(lock["source_checkpoint"]).parent.parent
    source_config = source_root / "resolved_config.yaml"
    source_manifest = source_root / "run_manifest.json"
    source_config.write_text("model:\n  model_ema_eval: true\n")
    source_manifest.write_text(json.dumps({"source": "fixture"}))
    manifest_path = run / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["source_hashes"].update(
        resolved_config=audit.file_sha256(source_config),
        run_manifest=audit.file_sha256(source_manifest),
    )
    manifest_path.write_text(json.dumps(manifest))
    lock["run_manifest_sha256"] = audit.file_sha256(manifest_path)
    lock_path.write_text(json.dumps(lock))
    monkeypatch.setattr(audit, "load_project_checkpoint", lambda _: _locked_fixture_payload(run))
    audit.validate_recipe_lock(lock_path, run, "last", declaration, tmp_path)
    for path, replacement, message in (
        (source_config, "model:\n  model_ema_eval: false\n", "source metadata"),
        (source_manifest, json.dumps({"source": "changed"}), "source metadata"),
        (manifest_path, json.dumps({**manifest, "steps_per_epoch": 1}), "artifact hash"),
    ):
        original = path.read_text()
        path.write_text(replacement)
        with pytest.raises(ValueError, match=message):
            audit.validate_recipe_lock(lock_path, run, "last", declaration, tmp_path)
        path.write_text(original)
    config_path = run / "resolved_config.yaml"
    original = config_path.read_text()
    semantic = audit.config_digest(audit.load_config(config_path))
    config_path.write_text("# unchanged YAML values, changed file bytes\n" + original)
    assert audit.config_digest(audit.load_config(config_path)) == semantic
    with pytest.raises(ValueError, match="artifact hash"):
        audit.validate_recipe_lock(lock_path, run, "last", declaration, tmp_path)
    config_path.write_text(original)
    valid_payload = _locked_fixture_payload(run)
    for key, value, message in (
        ("config_sha256", "other", "semantic config"),
        ("source_hashes", {"checkpoint": "other"}, "checkpoint source hashes"),
    ):
        monkeypatch.setattr(
            audit,
            "load_project_checkpoint",
            lambda _, key=key, value=value: {**valid_payload, key: value},
        )
        with pytest.raises(ValueError, match=message):
            audit.validate_recipe_lock(lock_path, run, "last", declaration, tmp_path)


def test_locked_test_review_rejects_foreign_or_escaping_checkpoint(tmp_path, monkeypatch):
    run, declaration, _, lock_path, lock = _locked_test_recipe_fixture(tmp_path)
    foreign = Path(lock["source_checkpoint"])
    lock.update(checkpoint=str(foreign), checkpoint_sha256=audit.file_sha256(foreign))
    lock_path.write_text(json.dumps(lock))

    def forbidden(*args, **kwargs):
        raise AssertionError("foreign lineage must fail before checkpoint loading")

    monkeypatch.setattr(audit, "load_project_checkpoint", forbidden)
    with pytest.raises(ValueError, match="belong to this run"):
        audit.validate_recipe_lock(lock_path, run, str(foreign), declaration, tmp_path)
    alias = run / "checkpoints/escaping.pt"
    alias.symlink_to(foreign)
    with pytest.raises(ValueError, match="belong to this run"):
        audit.validate_recipe_lock(lock_path, run, str(alias), declaration, tmp_path)
    checkpoint_directory = run / "checkpoints"
    checkpoint_directory.rename(run / "original_checkpoints")
    checkpoint_directory.symlink_to(foreign.parent, target_is_directory=True)
    with pytest.raises(ValueError, match="belong to this run"):
        audit.validate_recipe_lock(lock_path, run, "last", declaration, tmp_path)


def test_locked_test_review_output_cannot_consume_receipt_without_save_path(tmp_path, monkeypatch):
    run, declaration, _, lock_path, _ = _locked_test_recipe_fixture(tmp_path)
    monkeypatch.setattr(audit, "load_project_checkpoint", lambda _: _locked_fixture_payload(run))
    valid = audit.validate_recipe_lock(lock_path, run, "last", declaration, tmp_path)
    receipt = tmp_path / "_audit/R2_campaign/final_test_access.json"
    for output in (receipt, receipt / "results"):
        with pytest.raises(ValueError, match="collide"):
            audit.begin_final_test_access(valid, tmp_path, output)
        assert not receipt.exists()


def test_controller_completed_recovery_uses_true_fractions_and_recorded_counts():
    rows = [
        {
            "step": 19,
            "batches": 19,
            "gradient/native/constraint_active": True,
            "gradient/native/constraint_active_fraction": 0.0,
            "gradient/clipping_applied": True,
            "gradient/clipping_applied_fraction": 0.25,
            "gradient/native/norm": 2.0,
            "fidelity_grad_norm": 2.0,
            "gradient/coherence_raw_norm": 10.0,
            "gradient/coherence_calibrated_norm": 4.0,
            "update/actual_dot/native": -4.0,
            "gradient/final_direction_dot/native": 3.0,
            "finite_counts": {
                "gradient/native/norm": 1,
                "fidelity_grad_norm": 2,
                "gradient/coherence_raw_norm": 1,
                "gradient/coherence_calibrated_norm": 1,
                "update/actual_dot/native": 1,
                "gradient/final_direction_dot/native": 1,
            },
        },
        {
            "step": 38,
            "batches": 19,
            "gradient/native/constraint_active": True,
            "gradient/native/constraint_active_fraction": 1.0,
            "gradient/clipping_applied": True,
            "gradient/clipping_applied_fraction": 0.75,
            "gradient/native/norm": 6.0,
            "fidelity_grad_norm": 8.0,
            "gradient/coherence_raw_norm": 14.0,
            "gradient/coherence_calibrated_norm": 2.0,
            "update/actual_dot/native": -2.0,
            "gradient/final_direction_dot/native": 1.0,
            "finite_counts": {
                "gradient/native/norm": 3,
                "fidelity_grad_norm": 6,
                "gradient/coherence_raw_norm": 3,
                "gradient/coherence_calibrated_norm": 3,
                "update/actual_dot/native": 3,
                "gradient/final_direction_dot/native": 3,
            },
        },
        {
            "step": 48,
            "batches": 10,
            "gradient/native/norm": 99.0,
            "finite_counts": {"gradient/native/norm": 1},
        },
    ]
    result = report.controller_diagnostics(report.epoch_means(rows, 38))
    assert len(result) == 1 and result[0]["epoch"] == 1
    diagnostics = result[0]["diagnostics"]
    assert diagnostics["native_active_fraction"]["value"] == 0.5
    assert diagnostics["clipping_fraction"]["value"] == 0.5
    assert diagnostics["native_norm"]["value"] == 5.0
    assert diagnostics["native_norm"]["finite_count"] == 4
    assert diagnostics["native_norm"]["scope"] == "mean_over_recorded_finite_diagnostics"
    assert diagnostics["clipping_fraction"]["finite_count"] == 38
    assert diagnostics["clipping_fraction"]["scope"] == "processed_batch_fraction"
    assert diagnostics["native_actual_dot"]["value"] == -2.5
    assert diagnostics["native_proposal_dot"]["value"] == 1.5
    ratio = result[0]["fidelity_over_calibrated_coherence"]
    assert ratio["value"] == pytest.approx(6.5 / 2.5)
    assert ratio["value"] != pytest.approx((2.0 / 4.0 + 8.0 / 2.0) / 2)
    assert ratio["numerator_finite_count"] == 8 and ratio["denominator_finite_count"] == 4
    assert ratio["definition"] == "ratio_of_epoch_mean_norms_not_mean_of_ratios"


def test_controller_missing_bool_only_zero_denominator_and_unknown_diagnostic_scope():
    rows = [
        {
            "step": 38,
            "batches": 38,
            "gradient/native/constraint_active": True,
            "gradient/clipping_applied": True,
            "fidelity_grad_norm": 2.0,
            "gradient/coherence_calibrated_norm": 0.0,
            "finite_counts": {"fidelity_grad_norm": 38, "gradient/coherence_calibrated_norm": 38},
        },
        {
            "step": 76,
            "batches": 38,
            "gradient/native/norm": 2.0,
            "gradient/native/constraint_active_fraction": True,
            "gradient/clipping_applied_fraction": False,
        },
    ]
    result = report.controller_diagnostics(report.epoch_means(rows, 38))
    assert result[0]["diagnostics"]["native_active_fraction"]["value"] is None
    assert result[0]["diagnostics"]["clipping_fraction"]["value"] is None
    assert result[0]["fidelity_over_calibrated_coherence"]["value"] is None
    assert result[0]["fidelity_over_calibrated_coherence"]["undefined_reason"] == "zero_denominator"
    assert result[1]["diagnostics"]["native_norm"]["value"] is None
    assert result[1]["diagnostics"]["native_norm"]["finite_count"] is None
    assert (
        result[1]["diagnostics"]["native_norm"]["undefined_reason"]
        == "diagnostic_count_not_recorded"
    )
    assert result[1]["diagnostics"]["native_active_fraction"]["value"] is None
    assert result[1]["diagnostics"]["clipping_fraction"]["value"] is None
    assert (
        result[1]["fidelity_over_calibrated_coherence"]["undefined_reason"]
        == "numerator_or_denominator_diagnostics_missing"
    )

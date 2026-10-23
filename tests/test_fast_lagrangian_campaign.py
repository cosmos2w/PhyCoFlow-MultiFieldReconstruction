"""Bounded campaign preparation and command gates stay source-safe."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import threading
from pathlib import Path

import pytest
import yaml

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/posttraining/fast_lagrangian_campaign.py"
SPEC = importlib.util.spec_from_file_location("fast_lagrangian_campaign", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_campaign_pins_integration_and_descriptor_code_before_resume(tmp_path, monkeypatch):
    package = tmp_path / "src/phycoflow_reconstruction"
    package.mkdir(parents=True)
    integration = package / "post_training.py"
    integration.write_text("original_update = True\n")
    monkeypatch.setattr(MODULE, "REPO_ROOT", tmp_path)
    binding = MODULE._numerical_source_binding()
    MODULE._verify_numerical_source_binding(binding)
    integration.write_text("changed_update = True\n")
    with pytest.raises(MODULE.CampaignError, match="numerical source changed"):
        MODULE._verify_numerical_source_binding(binding)
    with pytest.raises(MODULE.CampaignError, match="lacks"):
        MODULE._verify_numerical_source_binding(None)


def test_goal1_stage_source_comes_from_verified_immutable_snapshot(tmp_path, monkeypatch):
    root = tmp_path / "campaign"
    snapshot_root = root / "source_snapshots/goal1_c073eba"
    source = snapshot_root / "src/phycoflow_reconstruction/training.py"
    source.parent.mkdir(parents=True)
    source.write_text("frozen numerical implementation\n", encoding="utf-8")
    files = {"src/phycoflow_reconstruction/training.py": hashlib.sha256(source.read_bytes()).hexdigest()}
    snapshot_manifest = snapshot_root / "snapshot_manifest.json"
    snapshot_manifest.write_text(
        json.dumps(
            {
                "stage": "goal1",
                "git_head": "goal1-head",
                "snapshot": str(snapshot_root.resolve()),
                "files_sha256": files,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(MODULE, "GOAL1_SNAPSHOT_MANIFEST_SHA256", hashlib.sha256(snapshot_manifest.read_bytes()).hexdigest())
    monkeypatch.setattr(MODULE, "GOAL1_SNAPSHOT_GIT_HEAD", "goal1-head")

    selected = MODULE._verify_goal1_source_snapshot(
        root,
        {"numerical_source_sha256": files},
    )
    assert selected == snapshot_root / "src"

    source.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(MODULE.CampaignError, match="snapshot file changed"):
        MODULE._verify_goal1_source_snapshot(root, {"numerical_source_sha256": files})


def test_stage_source_binding_rejects_malformed_binding_map():
    with pytest.raises(MODULE.CampaignError, match="stage source bindings are malformed"):
        MODULE._stage_source_bindings({"stage_source_bindings": None})


def _resolved_config() -> dict:
    return {
        "source_run": "/data/source-run",
        "source_checkpoint": "last.pt",
        "inherit_base_config": True,
        "dataset": {"split": "train"},
        "evaluation": {"split": "validation", "preview": {"enabled": False}},
        "optimization": {
            "epochs": 1000,
            "gradient_balance": "weighted_sum",
            "multitask": {
                "family_scalarization": {"method": "none"},
                "coherence_aggregation": {"method": "torchjd_config"},
                "external_policy": {
                    "method": "lagrangian",
                    "vector_reduction": "sum",
                    "objective_scale": "source_mean",
                    "lagrangian": {"penalty_interval_updates": 1},
                },
                "fidelity_budget": {"relative_increase": 0.05},
            },
        },
        "runtime": {"device": "cuda:0", "seed": 42},
        "output": {"experiment_name": "old/run"},
        "coherence": {"families": {"topology": {"geometry": {"grid_shape": [32, 128]}}}},
    }


def test_scalar_config_removes_recursive_config_selector_and_preserves_run11f_map():
    base = _resolved_config()
    scalar = MODULE._make_config(base, "LPCAccel_test", "G1-S", MODULE._arm_specs("p3")["G1-S"])
    config = scalar["optimization"]["multitask"]
    assert config["family_scalarization"] == {"method": "fixed_sum"}
    assert config["outer_aggregation"] is None
    assert "coherence_aggregation" not in config
    assert config["external_policy"] == base["optimization"]["multitask"]["external_policy"]
    assert config["fidelity_budget"] == base["optimization"]["multitask"]["fidelity_budget"]
    assert scalar["optimization"]["epochs"] == 100
    assert scalar["optimization"]["fast_lagrangian"] == {
        "topology_estimator": "p3",
        "family_probability": 1.0,
        "coefficient_interval": 1,
        "seed": 700211,
    }
    assert scalar["coherence"] == base["coherence"]
    assert base["optimization"]["epochs"] == 1000
    assert base["optimization"]["multitask"]["coherence_aggregation"] == {
        "method": "torchjd_config"
    }


def test_config_route_retains_config_and_campaign_specs_stay_within_limits():
    specs = MODULE._arm_specs("s2")
    assert len([name for name in specs if name.startswith("T")]) == 5
    assert len([name for name in specs if name.startswith("G1")]) == 4
    assert max(item["epochs"] for item in specs.values()) == 100
    config = MODULE._make_config(_resolved_config(), "LPCAccel_test", "G1-P", specs["G1-P"])
    assert config["optimization"]["multitask"]["coherence_aggregation"] == {
        "method": "torchjd_config"
    }
    assert config["optimization"]["fast_lagrangian"]["coefficient_interval"] == 2
    assert specs["G1-R"]["family_probability"] == 0.5
    assert specs["T0"]["topology_estimator"] == "full"
    assert (
        "fast_lagrangian"
        not in MODULE._make_config(_resolved_config(), "LPCAccel_test", "T0", specs["T0"])[
            "optimization"
        ]
    )


def test_ensure_missing_symlink_is_idempotent_and_never_replaces_existing_path(tmp_path):
    target = tmp_path / "payload"
    target.mkdir()
    link = tmp_path / "repo/runs/campaign"
    link.parent.mkdir(parents=True)
    MODULE.ensure_missing_symlink(link, target)
    MODULE.ensure_missing_symlink(link, target)
    assert link.is_symlink() and link.resolve() == target

    other = tmp_path / "other"
    other.mkdir()
    with pytest.raises(MODULE.CampaignError, match="unrelated symlink"):
        MODULE.ensure_missing_symlink(link, other)

    directory = tmp_path / "repo/runs/real"
    directory.mkdir()
    with pytest.raises(MODULE.CampaignError, match="non-symlink"):
        MODULE.ensure_missing_symlink(directory, target)


def test_timing_gate_requires_measured_correctness_and_positive_effect(tmp_path, monkeypatch):
    root = tmp_path / "campaign"
    (root / "manifests").mkdir(parents=True)
    evidence_path = root / "manifests/timing-evidence.json"
    evidence_path.write_text(
        json.dumps(
            {
                "implementation_verified": True,
                "correctness_verified": True,
                "selected_topology_estimator": "p3",
                "latency_effect": "positive",
                "profiles": ["T0", "T1", "T2"],
            }
        )
    )
    manifest = {
        "root": str(root),
        "selected_topology_estimator": "p3",
        "gates": {"timing_passed": False, "goal1_authorized": False},
    }
    monkeypatch.setattr(MODULE, "_campaign_manifest", lambda _: manifest)
    monkeypatch.setattr(MODULE, "_run_dirs", lambda _root, _arm: [tmp_path / "synthetic-run"])
    monkeypatch.setattr(MODULE, "_run_record", lambda _run: {"completed_epochs": 10.0})
    result = MODULE.authorize(root, "timing", evidence_path)
    assert result["gates"]["timing_passed"] is True
    assert (
        result["gate_evidence"]["timing"]["sha256"]
        == hashlib.sha256(evidence_path.read_bytes()).hexdigest()
    )

    bad = root / "manifests/bad-evidence.json"
    bad.write_text(
        json.dumps({**json.loads(evidence_path.read_text()), "latency_effect": "inconclusive"})
    )
    manifest["gates"]["timing_passed"] = False
    with pytest.raises(MODULE.CampaignError, match="positive timing effect"):
        MODULE.authorize(root, "timing", bad)


def test_goal1_and_goal2_remain_blocked_until_their_evidence_exists(tmp_path, monkeypatch):
    root = tmp_path / "campaign"
    (root / "manifests").mkdir(parents=True)
    evidence = root / "manifests/g1.json"
    evidence.write_text(
        json.dumps(
            {
                "goal1_ready": True,
                "evaluation_reserve_hours": 1.0,
                "selected_topology_estimator": "p3",
            }
        )
    )
    manifest = {
        "root": str(root),
        "selected_topology_estimator": "p3",
        "gates": {"timing_passed": False, "goal1_authorized": False},
    }
    monkeypatch.setattr(MODULE, "_campaign_manifest", lambda _: manifest)
    with pytest.raises(MODULE.CampaignError, match="timing gate must pass"):
        MODULE.authorize(root, "goal1", evidence)
    with pytest.raises(MODULE.CampaignError, match="unknown or unprepared arm"):
        MODULE.launch(root, "G2", gpu=0, execute=False)


def test_evaluation_command_is_validation_only_and_never_selects_test(tmp_path, monkeypatch):
    root = tmp_path / "campaign"
    run = root / "runs/G1-S/run-1"
    run.mkdir(parents=True)
    (run / "run_manifest.json").write_text(
        json.dumps({"experiment_name": "_LPCAccel/campaign/G1-S"})
    )
    manifest = {
        "root": str(root),
        "campaign_id": "campaign",
        "configs": {"G1-S": {}},
    }
    monkeypatch.setattr(MODULE, "_campaign_manifest", lambda _: manifest)
    command = MODULE.evaluate(root, "G1-S", run, execute=False)
    assert "fast_lagrangian_evaluate.py" in command
    assert "--campaign-root" in command
    assert "--max-samples 64" in command
    assert "--device cuda:0" in command
    assert ".png" not in command
    assert "--split test" not in command
    assert "--eval-set test" not in command


def test_goal2_full128_evaluation_uses_authorized_goal2_source_stage(tmp_path, monkeypatch):
    root = tmp_path / "campaign"
    run = root / "runs/G2-O/run-1"
    run.mkdir(parents=True)
    prepared = root / "configs/G2-O.yaml"
    prepared.parent.mkdir(parents=True)
    config = {
        "stage": "post_training",
        "optimization": {"epochs": 100, "lpc_ordered": MODULE.GOAL2_ORDERED_SETTINGS},
        "output": {"experiment_name": "_LPCAccel/campaign/G2-O"},
    }
    prepared.write_text(yaml.safe_dump(config), encoding="utf-8")
    (run / "resolved_config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    (run / "run_manifest.json").write_text(
        json.dumps(
            {
                "run_id": "run-1",
                "experiment_name": "_LPCAccel/campaign/G2-O",
                "source_immutable_verified": True,
                "source_hashes": {"checkpoint": "source"},
                "source_hashes_after": {"checkpoint": "source"},
            }
        ),
        encoding="utf-8",
    )
    manifest = {
        "root": str(root),
        "campaign_id": "campaign",
        "configs": {},
        "goal2_configs": {"G2-O": {"path": str(prepared), "sha256": "prepared"}},
        "gates": {"goal2_authorized": True},
    }
    monkeypatch.setattr(MODULE, "_campaign_manifest", lambda _root: manifest)
    monkeypatch.setattr(MODULE, "_goal2_formal_gate", lambda _manifest: {"status": "ready"})
    monkeypatch.setattr(MODULE, "_validate_arm", lambda *_args: {"config": prepared})
    monkeypatch.setattr(MODULE, "_run_dirs", lambda *_args: [run])
    monkeypatch.setattr(
        MODULE,
        "_run_record",
        lambda _run: {
            "status": "completed",
            "completed_epochs": 100.0,
            "completed_updates": 3800,
            "steps_per_epoch": 38,
            "configured_steps": 3800,
        },
    )

    command = MODULE.evaluate(root, "G2-O", run, checkpoint="best", max_samples=128, gpu=0)

    assert "--max-samples 128" in command
    assert "--checkpoint best" in command
    assert "CUDA_VISIBLE_DEVICES=0" in command
    assert "PHYCOFLOW_CAMPAIGN_SOURCE_STAGE=goal2" in command
    with pytest.raises(MODULE.CampaignError, match="128-sample"):
        MODULE.evaluate(root, "G2-O", run, max_samples=64, gpu=0)
    with pytest.raises(MODULE.CampaignError, match="best and last"):
        MODULE.evaluate(root, "G2-O", run, checkpoint="epoch_050", max_samples=128, gpu=0)


def test_evaluation_reserve_counts_every_completed_and_active_evaluation(tmp_path):
    root = tmp_path / "campaign"
    (root / "manifests").mkdir(parents=True)
    ledger = root / "manifests/launch_ledger.jsonl"
    ledger.write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                {"action": "evaluate-full-abt-validation", "elapsed_seconds": 12.0},
                {"action": "evaluate-native-maps", "elapsed_seconds": 25.0},
                {"action": "launch", "elapsed_seconds": 900.0},
            )
        )
        + "\n",
        encoding="utf-8",
    )

    assert MODULE._evaluation_spent_seconds(
        root,
        {
            "active_processes": [
                {"action": "evaluate-native-maps", "elapsed_seconds": 3.0},
                {"action": "launch", "elapsed_seconds": 500.0},
            ]
        },
    ) == 40.0


def _native_maps_campaign_fixture(tmp_path):
    root = tmp_path / "campaign"
    run = root / "runs/G1-C/run-c"
    run.mkdir(parents=True)
    (run / "run_manifest.json").write_text(
        json.dumps({"experiment_name": "_LPCAccel/campaign/G1-C"}), encoding="utf-8"
    )
    manifest = {
        "root": str(root),
        "campaign_id": "campaign",
        "gates": {"goal1_authorized": True},
        "gate_evidence": {"goal1": {"approved_arms": ["G1-C", "G1-R", "G1-S"]}},
    }
    status = {
        "active_processes": [],
        "runs": {
            arm: [
                {
                    "status": "completed",
                    "completed_epochs": 100.0,
                    "completed_updates": 3800,
                    "steps_per_epoch": 38,
                }
            ]
            for arm in ("G1-C", "G1-R", "G1-S")
        },
    }
    return root, run, manifest, status


def test_native_maps_uses_fixed_evaluator_entrypoint_and_requires_all_g1_complete(
    tmp_path, monkeypatch
):
    root, run, manifest, campaign_status = _native_maps_campaign_fixture(tmp_path)
    monkeypatch.setattr(MODULE, "_campaign_manifest", lambda _root: manifest)
    monkeypatch.setattr(MODULE, "_native_maps_approval_binding", lambda _root: {"schema_version": 1})
    monkeypatch.setattr(MODULE, "_evaluation_spent_seconds", lambda *_args: 17.0)
    monkeypatch.setattr(MODULE, "_goal1_remaining_hours", lambda *_args, **_kwargs: 0.0)
    monkeypatch.setattr(MODULE, "_evaluation_reserve_hours", lambda _manifest: 1.0)
    monkeypatch.setattr(MODULE, "status", lambda _root: campaign_status)
    stores = []
    monkeypatch.setattr(
        MODULE,
        "_store_native_maps_approval_binding",
        lambda _root, binding: stores.append(binding) or manifest,
    )
    calls = []
    monkeypatch.setattr(
        MODULE, "_execute_runner", lambda *args, **kwargs: calls.append((args, kwargs))
    )

    rendered = MODULE.maps(root, run, execute=False)
    assert "fast_lagrangian_evaluate.py" in rendered
    assert "--native-maps" in rendered
    assert "--run" in rendered and str(run.resolve()) in rendered
    assert "--device cuda:0" in rendered
    assert stores == [] and calls == []

    MODULE.maps(root, run, execute=True)
    assert stores == [{"schema_version": 1}]
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args[1:3] == ("G1-C", "evaluate-native-maps")
    assert "--native-maps" in args[3]
    assert "--run" in args[3] and str(run.resolve()) in args[3]
    assert kwargs["run"] == run.resolve()
    assert kwargs["gpu"] == 0
    assert kwargs["max_seconds"] == 180.0
    assert kwargs["evaluation_reserve_seconds"] == 3600.0

    campaign_status["runs"]["G1-S"][0]["completed_epochs"] = 99.0
    stores.clear()
    calls.clear()
    with pytest.raises(MODULE.CampaignError, match="100-epoch G1-S"):
        MODULE.maps(root, run, execute=True)
    assert stores == [] and calls == []


def test_campaign_launch_prints_existing_runner_command_without_execution(tmp_path, monkeypatch):
    root = tmp_path / "campaign"
    configs = root / "configs"
    configs.mkdir(parents=True)
    config = MODULE._make_config(
        _resolved_config(), "campaign", "T0", MODULE._arm_specs(None)["T0"]
    )
    config_path = configs / "T0.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    manifest = {
        "root": str(root),
        "campaign_id": "campaign",
        "selected_topology_estimator": None,
        "configs": {
            "T0": {
                "path": str(config_path),
                "sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
            }
        },
        "gates": {"timing_passed": False, "goal1_authorized": False},
    }
    monkeypatch.setattr(MODULE, "_campaign_manifest", lambda _: manifest)
    monkeypatch.setattr(MODULE, "_check_budget", lambda *_: None)
    monkeypatch.setattr(MODULE, "_validate_config_only", lambda *_: None)
    monkeypatch.setattr(
        MODULE.subprocess, "Popen", lambda *_args, **_kwargs: pytest.fail("unexpected launch")
    )
    command = MODULE.launch(root, "T0", gpu=2, execute=False)
    assert "CUDA_VISIBLE_DEVICES=2" in command
    assert "PHYCOFLOW_CAMPAIGN_GPU=2" in command
    assert "post-train" in command
    assert "--config" in command and str(config_path) in command
    assert "KEOPS_CACHE_FOLDER" in command and "PHYCOFLOW_TOPOLOGY_WORKERS" in command
    assert "runpy.run_path" in command
    assert "sys.path.insert(0, os.path.dirname(os.path.abspath(script)))" in command
    assert MODULE._environment(root)["PHYCOFLOW_CAMPAIGN_ADMITTED"] == str(root.resolve())
    assert MODULE._environment(root, 2)["PHYCOFLOW_CAMPAIGN_GPU"] == "2"


def test_pretest_exclusions_block_periodic_arms(tmp_path):
    path = tmp_path / "T4.yaml"
    path.write_text("{}\n")
    manifest = {
        "campaign_id": "campaign",
        "selected_topology_estimator": "p3",
        "configs": {
            "T4": {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        },
        "excluded_arms": MODULE.EXCLUDED_ARMS,
    }
    with pytest.raises(
        MODULE.CampaignError, match="periodic coefficients failed the TRAIN-only age probe"
    ):
        MODULE._validate_arm(manifest, "T4")


def test_short_socket_bootstrap_preserves_script_directory_imports(tmp_path):
    import subprocess
    import sys

    (tmp_path / "case.py").write_text("CASE_SPEC = 13\n")
    script = tmp_path / "entry.py"
    script.write_text("from case import CASE_SPEC\nprint(CASE_SPEC)\n")
    command = MODULE._with_short_multiprocessing_tmp([sys.executable, str(script)])
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "13"


def test_interrupted_training_is_charged_instead_of_silently_replayed():
    assert MODULE.segment_epoch_charge(2, 10, 0, 0, False) == 0
    assert MODULE.segment_epoch_charge(0, 10, 0, 10, True) == 10
    assert MODULE.segment_epoch_charge(-2, 100, 50, 55, True) == 50
    assert MODULE.segment_epoch_charge(0, 100, 50, 100, True) == 50


def test_idle_gpu_check_rejects_even_a_small_existing_compute_process(monkeypatch):
    answers = iter(["GPU-fixed, 26, 0\n", "GPU-fixed, 321\n"])
    monkeypatch.setattr(MODULE.subprocess, "check_output", lambda *a, **k: next(answers))
    with pytest.raises(MODULE.CampaignError, match="compute process"):
        MODULE._ensure_gpu1_idle()


def _runner_manifest(root: Path, arm: str = "T0") -> dict:
    manifests = root / "manifests"
    configs = root / "configs"
    manifests.mkdir(parents=True, exist_ok=True)
    configs.mkdir(parents=True, exist_ok=True)
    config_path = configs / f"{arm}.yaml"
    config_path.write_text("optimization: {epochs: 10}\n", encoding="utf-8")
    return {
        "root": str(root),
        "campaign_id": root.name,
        "selected_topology_estimator": None,
        "gates": {"timing_passed": False, "goal1_authorized": False},
        "configs": {
            arm: {
                "path": str(config_path),
                "sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
            }
        },
    }


def _mock_runner_prerequisites(monkeypatch, root: Path, lock_checks=None):
    monkeypatch.setattr(
        MODULE,
        "_status_data",
        lambda _manifest: {
            "gpu_seconds_charged": 0.0,
            "reported_gpu_hours": 0.0,
            "active_processes": [],
            "process_seconds_by_arm": {},
            "runs": {},
            "training_epochs_charged": 0.0,
            "timing_epochs_charged": 0.0,
            "timing_profile_count": 0,
        },
    )
    monkeypatch.setattr(MODULE, "_run_dirs", lambda _root, _arm: [])
    monkeypatch.setattr(MODULE, "_prepare_keops_build_folder", lambda *_args: root / "caches/keops")

    def check_idle(gpu=1):
        if lock_checks is not None:
            assert MODULE._try_campaign_lock(MODULE._gpu_lock_path(root, gpu)) is None
            lock_checks.append(True)
        return {"index": gpu, "uuid": f"GPU-{gpu}"}

    monkeypatch.setattr(MODULE, "_ensure_gpu_idle", check_idle)


def test_execute_runner_reuses_stale_lock_inode_and_charges_no_evaluation_epochs(
    tmp_path, monkeypatch
):
    import subprocess

    root = tmp_path / "campaign"
    manifest = _runner_manifest(root)
    lock_path = root / "manifests/active_process.lock"
    lock_path.write_text("stale coordinator receipt\n", encoding="utf-8")
    original_inode = lock_path.stat().st_ino
    active_path = root / "manifests/active_process.json"
    active_path.write_text(
        json.dumps({"pid": 2147483000, "config": "old.yaml"}), encoding="utf-8"
    )
    run = root / "runs/T0/evaluation-target"
    run.mkdir(parents=True)
    (run / "run_manifest.json").write_text(
        json.dumps({"last_checkpoint_epoch": 10.0}), encoding="utf-8"
    )
    output = tmp_path / "admitted_root.txt"
    command = [
        sys.executable,
        "-c",
        "import os,sys; open(sys.argv[1],'w').write(os.environ['PHYCOFLOW_CAMPAIGN_ADMITTED'])",
        str(output),
    ]
    lock_checks = []
    _mock_runner_prerequisites(monkeypatch, root, lock_checks)
    real_atomic_json = MODULE._atomic_json
    active_payloads = []

    def capture_active(path, payload):
        if path == active_path:
            active_payloads.append(dict(payload))
        real_atomic_json(path, payload)

    monkeypatch.setattr(MODULE, "_atomic_json", capture_active)
    monkeypatch.setattr(MODULE.subprocess, "Popen", subprocess.Popen)
    MODULE._execute_runner(
        manifest,
        "T0",
        "evaluate-full-abt-validation",
        command,
        reserve_gpu_seconds=0.0,
        max_seconds=5.0,
        evaluation_reserve_seconds=3600.0,
        run=run,
    )

    assert output.read_text(encoding="utf-8") == str(root.resolve())
    assert lock_checks == [True, True]
    assert lock_path.exists() and lock_path.stat().st_ino == original_inode
    assert "pid=" in lock_path.read_text(encoding="utf-8")
    assert not active_path.exists()
    assert active_payloads[0]["run"] == str(run.resolve())
    ledger = [json.loads(line) for line in (root / "manifests/launch_ledger.jsonl").read_text().splitlines()]
    assert ledger[0]["epochs_charged"] == 0.0
    assert ledger[0]["action"] == "evaluate-full-abt-validation"
    stale_lock_reopened = MODULE._try_campaign_lock(lock_path)
    assert stale_lock_reopened is not None
    MODULE._release_campaign_lock(stale_lock_reopened)


def test_execute_runner_rejects_concurrent_flock_without_starting_child(tmp_path, monkeypatch):
    root = tmp_path / "campaign"
    manifest = _runner_manifest(root)
    _mock_runner_prerequisites(monkeypatch, root)
    held_lock = MODULE._acquire_campaign_lock(root / "manifests/active_process.lock")
    monkeypatch.setattr(
        MODULE.subprocess,
        "Popen",
        lambda *_args, **_kwargs: pytest.fail("runner launched while flock was held"),
    )
    try:
        with pytest.raises(MODULE.CampaignError, match="physical GPU 1 already has an active"):
            MODULE._execute_runner(
                manifest,
                "T0",
                "launch",
                [sys.executable, "-c", "pass"],
                reserve_gpu_seconds=0.0,
            )
    finally:
        MODULE._release_campaign_lock(held_lock)
    assert (root / "manifests/active_process.lock").exists()
    assert not (root / "manifests/launch_ledger.jsonl").exists()


def test_execute_runner_stops_evaluation_at_max_seconds_and_keeps_zero_epoch_charge(
    tmp_path, monkeypatch
):
    root = tmp_path / "campaign"
    manifest = _runner_manifest(root)
    _mock_runner_prerequisites(monkeypatch, root)
    with pytest.raises(MODULE.CampaignError, match="evaluation reached its protected"):
        MODULE._execute_runner(
            manifest,
            "T0",
            "evaluate-full-abt-validation",
            [sys.executable, "-c", "import time; time.sleep(30)"],
            reserve_gpu_seconds=0.0,
            max_seconds=0.15,
            evaluation_reserve_seconds=3600.0,
        )
    ledger = json.loads(
        (root / "manifests/launch_ledger.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    assert ledger["runtime_interrupt"] is True
    assert ledger["epochs_charged"] == 0.0
    assert (root / "manifests/active_process.lock").exists()
    assert not (root / "manifests/active_process.json").exists()


def test_evaluation_execute_uses_shared_runner_with_run_binding_and_cap(tmp_path, monkeypatch):
    root = tmp_path / "campaign"
    run = root / "runs/G1-S/run-1"
    run.mkdir(parents=True)
    (run / "run_manifest.json").write_text(
        json.dumps({"experiment_name": "_LPCAccel/campaign/G1-S"}), encoding="utf-8"
    )
    manifest = {
        "root": str(root),
        "campaign_id": "campaign",
        "configs": {"G1-S": {}},
        "gates": {"goal1_authorized": False},
    }
    calls = []
    monkeypatch.setattr(MODULE, "_campaign_manifest", lambda _root: manifest)
    monkeypatch.setattr(MODULE, "status", lambda _root: {"gpu_seconds_charged": 0.0})
    monkeypatch.setattr(MODULE, "_evaluation_spent_seconds", lambda *_args: 0.0)
    monkeypatch.setattr(MODULE, "_goal1_remaining_hours", lambda *_args, **_kwargs: 0.0)
    monkeypatch.setattr(
        MODULE,
        "_execute_runner",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    MODULE.evaluate(root, "G1-S", run, max_seconds=300.0, execute=True)
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args[2] == "evaluate-full-abt-validation"
    assert kwargs["run"] == run.resolve()
    assert kwargs["max_seconds"] == 300.0
    assert kwargs["evaluation_reserve_seconds"] == 3600.0


@pytest.mark.parametrize("gpu", [0, 1, 2])
def test_stop_waits_for_campaign_child_exit_and_coordinator_unlock(tmp_path, monkeypatch, gpu):
    root = tmp_path / "campaign"
    manifests = root / "manifests"
    manifests.mkdir(parents=True)
    active_path = MODULE._gpu_active_path(root, gpu)
    lock_path = MODULE._gpu_lock_path(root, gpu)
    coordinator_released = threading.Event()
    coordinator_ready = threading.Event()
    selected_pid = 42321

    def coordinator():
        lock = MODULE._acquire_campaign_lock(lock_path)
        active_path.write_text(
            json.dumps(
                {
                    "pid": selected_pid,
                    "coordinator_pid": os.getpid(),
                    "config": "/campaign/configs/G1-S.yaml",
                    "run": "/campaign/runs/G1-S/run-1",
                    "gpu_index": gpu,
                }
            ),
            encoding="utf-8",
        )
        coordinator_ready.set()
        assert coordinator_released.wait(timeout=2.0)
        active_path.unlink(missing_ok=True)
        MODULE._release_campaign_lock(lock)

    thread = threading.Thread(target=coordinator)
    thread.start()
    assert coordinator_ready.wait(timeout=1.0)
    monkeypatch.setattr(MODULE, "_campaign_manifest", lambda _root: {"root": str(root)})
    monkeypatch.setattr(
        MODULE, "_pid_matches", lambda pid, _config, _run=None: pid == selected_pid and not coordinator_released.is_set()
    )

    def signal_campaign(pid, signum):
        assert pid == selected_pid
        assert signum == MODULE.signal.SIGINT
        threading.Timer(0.03, coordinator_released.set).start()

    monkeypatch.setattr(MODULE.os, "killpg", signal_campaign)
    result = MODULE.stop(root, gpu=gpu, timeout_seconds=1.0)
    thread.join(timeout=1.0)
    assert not thread.is_alive()
    assert "coordinator released the lock" in result
    assert not active_path.exists()
    assert lock_path.exists()
    available = MODULE._try_campaign_lock(lock_path)
    assert available is not None
    MODULE._release_campaign_lock(available)


def test_stop_reports_bounded_failure_when_owned_process_does_not_exit(tmp_path, monkeypatch):
    root = tmp_path / "campaign"
    manifests = root / "manifests"
    manifests.mkdir(parents=True)
    active_path = manifests / "active_process.json"
    lock_path = manifests / "active_process.lock"
    selected_pid = 42322
    active_path.write_text(
        json.dumps({"pid": selected_pid, "config": "/campaign/config.yaml", "run": None}),
        encoding="utf-8",
    )
    held_lock = MODULE._acquire_campaign_lock(lock_path)
    monkeypatch.setattr(MODULE, "_campaign_manifest", lambda _root: {"root": str(root)})
    monkeypatch.setattr(MODULE, "_pid_matches", lambda *_args: True)
    monkeypatch.setattr(MODULE, "_pid_is_alive", lambda _pid: True)
    monkeypatch.setattr(MODULE.os, "killpg", lambda *_args: None)
    original_sleep = MODULE.time.sleep

    def check_wait_does_not_hold_admission(seconds):
        assert MODULE._campaign_lock_available(MODULE._admission_lock_path(root))
        original_sleep(seconds)

    monkeypatch.setattr(MODULE.time, "sleep", check_wait_does_not_hold_admission)
    try:
        with pytest.raises(MODULE.CampaignError, match="stop timed out after 0.02s"):
            MODULE.stop(root, timeout_seconds=0.02)
    finally:
        MODULE._release_campaign_lock(held_lock)


def test_stop_clears_a_dead_coordinator_receipt_only_after_child_exit(tmp_path, monkeypatch):
    root = tmp_path / "campaign"
    manifests = root / "manifests"
    manifests.mkdir(parents=True)
    active_path = MODULE._gpu_active_path(root, 2)
    active_path.write_text(
        json.dumps(
            {
                "pid": 42323,
                "gpu_index": 2,
                "config": "/campaign/config.yaml",
                "run": None,
                "action": "launch",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(MODULE, "_campaign_manifest", lambda _root: {"root": str(root)})
    monkeypatch.setattr(MODULE, "_pid_matches", lambda *_args: False)
    monkeypatch.setattr(MODULE, "_pid_is_alive", lambda _pid: False)
    result = MODULE.stop(root, gpu=2, timeout_seconds=0.1)
    assert "orphaned campaign process 42323 exited" in result
    assert not active_path.exists()


def test_stop_receipt_binds_preexisting_goal2_epoch1_checkpoint_before_sigint(tmp_path, monkeypatch):
    root = tmp_path / "campaign"
    manifests = root / "manifests"
    manifests.mkdir(parents=True)
    run = root / "runs/recovery_smoke2ep/run-1"
    receipt_path = root / MODULE.GOAL2_SMOKE_EPOCH1_RECEIPT
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text('{"status":"atomic_epoch1_checkpoint_verified"}\n', encoding="utf-8")
    selected_pid = 42324
    active_path = MODULE._gpu_active_path(root, 0)
    active_path.write_text(
        json.dumps(
            {
                "arm": MODULE.GOAL2_SMOKE_ARM,
                "action": "launch",
                "pid": selected_pid,
                "coordinator_pid": 55555,
                "gpu_index": 0,
                "config": str(root / "configs/recovery_smoke2ep.yaml"),
                "run": str(run),
            }
        ),
        encoding="utf-8",
    )
    live_states = iter((True, True, False))
    monkeypatch.setattr(MODULE, "_campaign_manifest", lambda _root: {"root": str(root)})
    monkeypatch.setattr(MODULE, "_pid_matches", lambda *_args: next(live_states))
    monkeypatch.setattr(MODULE, "_pid_is_alive", lambda _pid: False)
    monkeypatch.setattr(MODULE.os, "killpg", lambda *_args: None)

    MODULE.stop(root, gpu=0, timeout_seconds=0.1)

    row = json.loads((manifests / "stop_requests.jsonl").read_text(encoding="utf-8"))
    assert row["checkpoint_receipt_path"] == str(receipt_path.resolve())
    assert row["checkpoint_receipt_sha256"] == hashlib.sha256(receipt_path.read_bytes()).hexdigest()


def test_physical_gpu_parser_and_g1_assignment_are_explicit():
    assert [MODULE._parse_gpu(value) for value in ("0", "1", "2", "physical0", "physical1", "physical2")] == [0, 1, 2, 0, 1, 2]
    assert MODULE._validate_gpu_for_arm("G1-C", 1) == 1
    assert MODULE._validate_gpu_for_arm("G1-S", 2) == 2
    assert MODULE._validate_gpu_for_arm("G1-R", 0) == 0
    with pytest.raises(MODULE.CampaignError, match="assigned to physical GPU 2"):
        MODULE._validate_gpu_for_arm("G1-S", 1)
    # Evaluation can be explicitly sent to any idle physical GPU.
    assert MODULE._validate_gpu_for_arm("G1-S", 1, enforce_assignment=False) == 1
    launch_args = MODULE.build_parser().parse_args(["launch", "G1-S", "--gpu", "physical2"])
    stop_args = MODULE.build_parser().parse_args(["stop", "--gpu", "0"])
    assert launch_args.gpu == 2
    assert stop_args.gpu == 0


def test_physical_gpu_locks_are_independent_and_same_gpu_is_exclusive(tmp_path):
    root = tmp_path / "campaign"
    (root / "manifests").mkdir(parents=True)
    gpu1 = MODULE._acquire_gpu_lock(root, 1)
    gpu2 = MODULE._acquire_gpu_lock(root, 2)
    try:
        with pytest.raises(MODULE.CampaignError, match="physical GPU 1 already has"):
            MODULE._acquire_gpu_lock(root, 1)
        assert not MODULE._campaign_lock_available(MODULE._gpu_lock_path(root, 1))
        assert not MODULE._campaign_lock_available(MODULE._gpu_lock_path(root, 2))
    finally:
        MODULE._release_campaign_lock(gpu2)
        MODULE._release_campaign_lock(gpu1)
    assert MODULE._campaign_lock_available(MODULE._gpu_lock_path(root, 1))
    assert MODULE._campaign_lock_available(MODULE._gpu_lock_path(root, 2))


def test_admission_lock_has_a_bounded_wait_and_keeps_inode(tmp_path):
    root = tmp_path / "campaign"
    (root / "manifests").mkdir(parents=True)
    lock_path = MODULE._admission_lock_path(root)
    held = MODULE._acquire_admission_lock(root)
    inode = lock_path.stat().st_ino
    started = MODULE.time.monotonic()
    try:
        with pytest.raises(MODULE.CampaignError, match="stayed busy for 0.03s"):
            MODULE._acquire_admission_lock(root, timeout_seconds=0.03)
    finally:
        MODULE._release_campaign_lock(held)
    assert MODULE.time.monotonic() - started < 0.5
    assert lock_path.stat().st_ino == inode


def test_unrepresented_admitted_child_is_rejected_as_an_orphan(tmp_path):
    import subprocess

    root = tmp_path / "campaign"
    (root / "manifests").mkdir(parents=True)
    env = dict(os.environ)
    env["PHYCOFLOW_CAMPAIGN_ADMITTED"] = str(root.resolve())
    child = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import time; time.sleep(30)",
            str(MODULE.CASE_DIR / "run.py"),
        ],
        cwd=MODULE.REPO_ROOT,
        env=env,
    )
    try:
        with pytest.raises(MODULE.CampaignError, match="without a matching per-GPU owner"):
            MODULE._reject_active_or_orphaned_process(root)
    finally:
        child.terminate()
        child.wait(timeout=5)


def test_process_ownership_requires_a_live_parent_chain_not_a_shared_group(monkeypatch):
    parents = {
        4002: 4001,
        4003: 4002,
        4010: 3999,
        4011: 4010,
        4020: 4021,
        4021: 4020,
    }
    monkeypatch.setattr(MODULE, "_proc_parent_pid", lambda pid: parents.get(pid))
    assert MODULE._is_descendant_process(4002, 4001)
    assert MODULE._is_descendant_process(4003, 4001)
    assert not MODULE._is_descendant_process(4011, 4001)
    assert not MODULE._is_descendant_process(4020, 4001)
    assert not MODULE._is_descendant_process(4001, 4001)


def test_campaign_process_gpu_binding_rejects_mismatched_or_unbound_devices(
    tmp_path, monkeypatch
):
    root = tmp_path / "campaign"
    monkeypatch.setattr(
        MODULE,
        "_campaign_process_environment",
        lambda _pid: {
            "PHYCOFLOW_CAMPAIGN_ADMITTED": str(root.resolve()),
            "CUDA_VISIBLE_DEVICES": "2",
            "PHYCOFLOW_CAMPAIGN_GPU": "2",
        },
    )
    assert MODULE._campaign_process_matches_gpu(4001, root, 2)
    assert not MODULE._campaign_process_matches_gpu(4001, root, 1)
    monkeypatch.setattr(
        MODULE,
        "_campaign_process_environment",
        lambda _pid: {
            "PHYCOFLOW_CAMPAIGN_ADMITTED": str((root / "other").resolve()),
            "CUDA_VISIBLE_DEVICES": "2",
            "PHYCOFLOW_CAMPAIGN_GPU": "2",
        },
    )
    assert not MODULE._campaign_process_matches_gpu(4001, root, 2)
    monkeypatch.setattr(
        MODULE,
        "_campaign_process_environment",
        lambda _pid: {
            "PHYCOFLOW_CAMPAIGN_ADMITTED": str(root.resolve()),
            "CUDA_VISIBLE_DEVICES": "1",
            "PHYCOFLOW_CAMPAIGN_GPU": "2",
        },
    )
    assert not MODULE._campaign_process_matches_gpu(4001, root, 1)
    monkeypatch.setattr(
        MODULE,
        "_campaign_process_environment",
        lambda _pid: {
            "PHYCOFLOW_CAMPAIGN_ADMITTED": str(root.resolve()),
            "CUDA_VISIBLE_DEVICES": "1",
        },
    )
    assert not MODULE._campaign_process_matches_gpu(4001, root, 1)
    assert MODULE._campaign_process_matches_gpu(
        4001, root, 1, allow_legacy_gpu_token=True
    )


def test_forked_admitted_worker_is_only_owned_while_descended_from_locked_run(
    tmp_path, monkeypatch
):
    import signal
    import subprocess
    import time

    root = tmp_path / "campaign"
    config = root / "configs/G1-C.yaml"
    config.parent.mkdir(parents=True)
    config.write_text("optimization: {epochs: 100}\n", encoding="utf-8")
    run = root / "runs/G1-C/run-1"
    run.mkdir(parents=True)
    (root / "manifests").mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": "1",
            "PHYCOFLOW_CAMPAIGN_ADMITTED": str(root.resolve()),
            "PHYCOFLOW_CAMPAIGN_GPU": "1",
        }
    )
    worker_code = (
        "import multiprocessing,time\n"
        "def worker():\n"
        "    while True: time.sleep(1)\n"
        "if __name__ == '__main__':\n"
        "    child=multiprocessing.get_context('fork').Process(target=worker)\n"
        "    child.start()\n"
        "    print(child.pid, flush=True)\n"
        "    while True: time.sleep(1)\n"
    )
    owner = subprocess.Popen(
        [
            sys.executable,
            "-c",
            worker_code,
            str(MODULE.CASE_DIR / "run.py"),
            "post-train",
            "--config",
            str(config),
        ],
        cwd=MODULE.REPO_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    worker_pid = None
    lock = None
    active_path = MODULE._gpu_active_path(root, 1)
    try:
        assert owner.stdout is not None
        worker_pid = int(owner.stdout.readline().strip())
        assert MODULE._is_descendant_process(worker_pid, owner.pid)
        assert os.getpgid(worker_pid) == owner.pid
        lock = MODULE._acquire_gpu_lock(root, 1)
        MODULE._atomic_json(
            active_path,
            {
                "arm": "G1-C",
                "action": "launch",
                "pid": owner.pid,
                "coordinator_pid": os.getpid(),
                "config": str(config),
                "run": str(run),
                "gpu_index": 1,
                "gpu_uuid": "GPU-test-1",
                "started_monotonic_ns": MODULE.time.monotonic_ns(),
                "watchdog_seconds": 600.0,
            },
        )

        verified = MODULE._reject_active_or_orphaned_process(root)
        assert len(verified) == 1

        with monkeypatch.context() as wrong_config:
            wrong_config.setattr(
                MODULE,
                "_pid_matches",
                lambda pid, _config, _run=None: pid == owner.pid,
            )
            with pytest.raises(
                MODULE.CampaignError,
                match="without a matching per-GPU owner",
            ):
                MODULE._reject_active_or_orphaned_process(root)

        owner.terminate()
        owner.wait(timeout=5)
        deadline = time.monotonic() + 3.0
        while MODULE._proc_parent_pid(worker_pid) == owner.pid and time.monotonic() < deadline:
            time.sleep(0.01)
        assert MODULE._proc_parent_pid(worker_pid) != owner.pid
        # The worker kept the same process group but lost its verified ancestry.
        assert os.getpgid(worker_pid) == owner.pid
        with pytest.raises(MODULE.CampaignError, match="without a matching per-GPU owner"):
            MODULE._reject_active_or_orphaned_process(root)
    finally:
        if owner.poll() is None:
            owner.terminate()
            owner.wait(timeout=5)
        try:
            os.killpg(owner.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        if owner.stdout is not None:
            owner.stdout.close()
        active_path.unlink(missing_ok=True)
        if lock is not None:
            MODULE._release_campaign_lock(lock)


def test_idle_gpu_check_queries_the_selected_physical_index(monkeypatch):
    calls = []
    answers = iter(["GPU-two, 0, 0\n", "\n"])

    def check_output(command, **kwargs):
        calls.append(command)
        return next(answers)

    monkeypatch.setattr(MODULE.subprocess, "check_output", check_output)
    snapshot = MODULE._ensure_gpu_idle(2)
    assert snapshot["index"] == 2
    assert snapshot["uuid"] == "GPU-two"
    assert calls[0][2] == "2"


def _write_active(root: Path, gpu: int, arm: str, *, pid: int = 2147483000) -> object:
    lock = MODULE._acquire_gpu_lock(root, gpu)
    record = {
        "arm": arm,
        "action": "launch",
        "pid": pid,
        "coordinator_pid": os.getpid(),
        "gpu_index": gpu,
        "gpu_uuid": f"GPU-{gpu}",
        "config": str(root / "configs" / f"{arm}.yaml"),
        "started_monotonic_ns": MODULE.time.monotonic_ns() - 2_000_000_000,
        "watchdog_seconds": 7200.0,
    }
    MODULE._atomic_json(MODULE._gpu_active_path(root, gpu), record)
    return lock


def test_an_active_arm_blocks_duplicate_on_another_gpu_but_other_arm_can_run(
    tmp_path, monkeypatch
):
    import subprocess

    root = tmp_path / "campaign"
    manifest = _runner_manifest(root)
    (root / "configs/G1-C.yaml").write_text("optimization: {epochs: 100}\n")
    held = _write_active(root, 1, "T0")
    real_status_data = MODULE._status_data
    _mock_runner_prerequisites(monkeypatch, root)
    monkeypatch.setattr(MODULE, "_status_data", real_status_data)
    try:
        with pytest.raises(MODULE.CampaignError, match="T0 already has an active process"):
            MODULE._execute_runner(
                manifest,
                "T0",
                "launch",
                [sys.executable, "-c", "pass"],
                reserve_gpu_seconds=0.0,
                gpu=2,
            )

        outputs = []
        monkeypatch.setattr(MODULE.subprocess, "Popen", subprocess.Popen)
        other_manifest = dict(manifest)
        other_manifest["selected_topology_estimator"] = "p3"
        other_manifest["configs"] = dict(manifest["configs"])
        other_manifest["configs"]["T1"] = manifest["configs"]["T0"]
        outputs.append(
            MODULE._execute_runner(
                other_manifest,
                "T1",
                "launch",
                [sys.executable, "-c", "pass"],
                reserve_gpu_seconds=0.0,
                gpu=2,
            )
        )
        assert outputs == [None]
        row = json.loads((root / "manifests/launch_ledger.jsonl").read_text().splitlines()[0])
        assert row["gpu_index"] == 2
        assert row["gpu_uuid"] == "GPU-2"
        assert row["coordinator_pid"] == os.getpid()
    finally:
        MODULE._gpu_active_path(root, 1).unlink(missing_ok=True)
        MODULE._release_campaign_lock(held)


def test_active_process_time_is_counted_once_across_ledger_and_run_receipt(tmp_path, monkeypatch):
    root = tmp_path / "campaign"
    manifest = _runner_manifest(root)
    ledger = root / "manifests/launch_ledger.jsonl"
    ledger.write_text(json.dumps({"arm": "T0", "elapsed_seconds": 100.0, "epochs_charged": 4}) + "\n")
    MODULE._atomic_json(
        MODULE._gpu_active_path(root, 0),
        {
            "arm": "T0",
            "gpu_index": 0,
            "pid": 999999,
            "started_monotonic_ns": MODULE.time.monotonic_ns() - 10_000_000_000,
        },
    )
    monkeypatch.setattr(
        MODULE,
        "_run_dirs",
        lambda _root, _arm: [root / "runs/T0/run"],
    )
    monkeypatch.setattr(
        MODULE,
        "_run_record",
        lambda _run: {"completed_epochs": 4.0, "post_training_seconds": 50.0},
    )
    current = MODULE._status_data(manifest)
    assert 109.0 <= current["process_seconds_by_arm"]["T0"] <= 111.0
    assert current["training_epochs_charged"] == 4.0
    assert current["active_processes"][0]["gpu_index"] == 0


def test_evaluation_time_does_not_consume_goal1_training_quota(tmp_path):
    root = tmp_path / "campaign"
    manifest = _runner_manifest(root, "G1-C")
    (root / "manifests/launch_ledger.jsonl").write_text(
        json.dumps(
            {
                "arm": "G1-C",
                "action": "evaluate-full-abt-validation",
                "elapsed_seconds": 3600.0,
                "epochs_charged": 0.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    MODULE._atomic_json(
        MODULE._gpu_active_path(root, 0),
        {
            "arm": "G1-C",
            "action": "evaluate-full-abt-validation",
            "gpu_index": 0,
            "pid": 999999,
            "started_monotonic_ns": MODULE.time.monotonic_ns() - 100_000_000_000,
        },
    )
    manifest["gate_evidence"] = {
        "goal1": {"approved_arms": ["G1-C"], "runtime_quotas_hours_by_arm": {"G1-C": 2.0}}
    }
    current = MODULE._status_data(manifest)
    assert current["campaign_process_seconds_by_arm"]["G1-C"] == 3600.0
    assert current["training_process_seconds_by_arm"].get("G1-C", 0.0) == 0.0
    assert MODULE._goal1_remaining_hours(manifest, current_status=current) == 2.0


def test_goal1_authorization_records_individual_two_x_quotas_and_reserves(tmp_path, monkeypatch):
    root = tmp_path / "campaign"
    manifests = root / "manifests"
    manifests.mkdir(parents=True)
    timing_path = manifests / "timing.json"
    timing_path.write_text(json.dumps({"profiles": ["T1", "T2", "T3"]}), encoding="utf-8")
    timing_record = {
        "path": str(timing_path),
        "sha256": hashlib.sha256(timing_path.read_bytes()).hexdigest(),
    }
    evidence_path = manifests / "goal1.json"
    evidence_path.write_text(
        json.dumps(
            {
                "goal1_ready": True,
                "evaluation_reserve_hours": 0.25,
                "selected_topology_estimator": "p3",
                "approved_arms": ["G1-C", "G1-S", "G1-R"],
            }
        ),
        encoding="utf-8",
    )
    manifest = {
        "root": str(root),
        "campaign_id": "campaign",
        "selected_topology_estimator": "p3",
        "gates": {"timing_passed": True, "goal1_authorized": False},
        "gate_evidence": {"timing": timing_record},
    }
    current = {
        "reported_gpu_hours": 1.1128,
        "campaign_process_seconds_by_arm": {"T1": 360.0, "T2": 720.0, "T3": 180.0},
        "trainer_reported_seconds_by_arm": {},
        "runs": {arm: [{"completed_epochs": 10.0}] for arm in ("T1", "T2", "T3")},
    }
    monkeypatch.setattr(MODULE, "_campaign_manifest", lambda _root: manifest)
    monkeypatch.setattr(MODULE, "status", lambda _root: current)
    result = MODULE.authorize(root, "goal1", evidence_path)
    gate = result["gate_evidence"]["goal1"]
    assert gate["projected_hours_by_arm"] == {"G1-C": 1.0, "G1-S": 2.0, "G1-R": 0.5}
    assert gate["runtime_quotas_hours_by_arm"] == {"G1-C": 2.0, "G1-S": 4.0, "G1-R": 1.0}
    assert gate["protected_evaluation_reserve_hours"] == 1.0
    assert gate["goal2_gpu_hours_reserved"] == 2.0
    assert gate["watchdog_shutdown_reserve_hours"] == 450.0 / 3600
    assert gate["budget_check"]["reported_gpu_hours_at_authorization"] == 1.1128
    assert gate["budget_check"]["protected_future_gpu_hours"] == 10.125
    assert 1.1128 + gate["budget_check"]["protected_future_gpu_hours"] < MODULE.MAX_GPU_SECONDS / 3600


def test_goal1_watchdogs_never_exceed_the_per_arm_quota_with_concurrent_gpu_owners(tmp_path):
    root = tmp_path / "campaign"
    (root / "manifests").mkdir(parents=True)
    manifest = {
        "root": str(root),
        "gate_evidence": {
            "goal1": {
                "approved_arms": ["G1-C", "G1-S", "G1-R"],
                "runtime_quotas_hours_by_arm": {arm: 2.0 for arm in ("G1-C", "G1-S", "G1-R")},
                "evaluation_reserve_hours": 1.0,
            }
        },
    }
    allocations = []
    current = {
        "gpu_seconds_charged": 180.0,
        "process_seconds_by_arm": {},
        "active_processes": [],
    }
    for arm, gpu in (("G1-C", 1), ("G1-S", 2), ("G1-R", 0)):
        allocation = MODULE._watchdog_seconds(
            manifest,
            arm,
            "launch",
            current,
            reserve_gpu_seconds=0.0,
            max_seconds=None,
            evaluation_reserve_seconds=None,
        )
        allocations.append(allocation)
        current["gpu_seconds_charged"] += 1.0
        current["process_seconds_by_arm"][arm] = 1.0
        current["active_processes"].append(
            {
                "arm": arm,
                "gpu_index": gpu,
                "action": "launch",
                "elapsed_seconds": 1.0,
                "watchdog_seconds": allocation,
            }
        )
    assert allocations == [7200.0, 7200.0, 7200.0]
    assert sum(allocations) <= 6 * 3600


def test_completed_goal1_arm_releases_its_unused_runtime_quota():
    manifest = {
        "root": "/campaign",
        "gate_evidence": {
            "goal1": {
                "approved_arms": ["G1-C", "G1-S"],
                "runtime_quotas_hours_by_arm": {"G1-C": 2.0, "G1-S": 3.0},
            }
        },
    }
    current = {
        "process_seconds_by_arm": {"G1-C": 500.0, "G1-S": 36000.0},
        "training_process_seconds_by_arm": {"G1-C": 500.0, "G1-S": 3600.0},
        "runs": {
            "G1-C": [{"completed_epochs": 100.0}],
            "G1-S": [{"completed_epochs": 50.0}],
        },
    }
    assert MODULE._goal1_remaining_hours(manifest, current_status=current) == 2.0


def test_goal2_residual_approval_binding_fixes_probe_plan_and_goal1_snapshot(tmp_path, monkeypatch):
    root = tmp_path / "campaign"
    snapshot_root = root / "source_snapshots/goal1_c073eba"
    snapshot_src_file = snapshot_root / "src/phycoflow_reconstruction/training.py"
    snapshot_src_file.parent.mkdir(parents=True)
    snapshot_src_file.write_text("frozen source\n", encoding="utf-8")
    snapshot_files = {
        "src/phycoflow_reconstruction/training.py": hashlib.sha256(snapshot_src_file.read_bytes()).hexdigest()
    }
    snapshot_manifest_path = snapshot_root / "snapshot_manifest.json"
    snapshot_manifest_path.write_text(
        json.dumps(
            {
                "stage": "goal1",
                "git_head": "goal1-head",
                "snapshot": str(snapshot_root.resolve()),
                "files_sha256": snapshot_files,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    snapshot_manifest_sha = hashlib.sha256(snapshot_manifest_path.read_bytes()).hexdigest()
    monkeypatch.setattr(MODULE, "GOAL1_SNAPSHOT_MANIFEST_SHA256", snapshot_manifest_sha)
    monkeypatch.setattr(MODULE, "GOAL1_SNAPSHOT_GIT_HEAD", "goal1-head")

    for relative in ("manifests", "analysis"):
        (root / relative).mkdir(parents=True, exist_ok=True)
    closeout = {
        "status": "goal1_complete_assessed",
        "goal1_arms_completed_100": ["G1-C", "G1-S", "G1-R"],
        "source_142_files_unchanged": True,
        "test_evaluation_or_selection": False,
    }
    clarification = {
        "endpoint_relative_increase": 0.05,
        "endpoint_absolute_allowance": 0.0,
        "A_bound": "fixed full SOURCE TRAIN mean without slack",
        "B_bound": "fixed full SOURCE TRAIN mean without slack",
    }
    closeout_path = root / MODULE.GOAL1_COMPLETION_ASSESSMENT
    clarification_path = root / MODULE.GOAL2_CONSTRAINT_CLARIFICATION
    closeout_path.write_text(json.dumps(closeout), encoding="utf-8")
    clarification_path.write_text(json.dumps(clarification), encoding="utf-8")
    monkeypatch.setattr(MODULE, "GOAL1_COMPLETION_ASSESSMENT_SHA256", hashlib.sha256(closeout_path.read_bytes()).hexdigest())
    monkeypatch.setattr(MODULE, "GOAL2_CONSTRAINT_CLARIFICATION_SHA256", hashlib.sha256(clarification_path.read_bytes()).hexdigest())

    script_rel = "analysis/probe.py"
    plan_rel = "analysis/probe.plan.json"
    binding_rel = "analysis/probe.binding.json"
    script_path = root / script_rel
    plan_path = root / plan_rel
    binding_path = root / binding_rel
    script_path.write_text("probe script\n", encoding="utf-8")
    plan_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "predeclared_protocol_no_results",
                "campaign_id": root.name,
                "campaign_root": str(root),
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    script_sha = hashlib.sha256(script_path.read_bytes()).hexdigest()
    plan_sha = hashlib.sha256(plan_path.read_bytes()).hexdigest()
    result_rel = "analysis/probe.result.json"
    binding = {
        "schema_version": 1,
        "status": "frozen_hash_binding",
        "script_path": str(script_path),
        "script_sha256": script_sha,
        "plan_path": str(plan_path),
        "plan_sha256": plan_sha,
        "result_path": str(root / result_rel),
        "result_schema": "goal2_residual_probe.result.v1",
        "source_snapshot_manifest_sha256": snapshot_manifest_sha,
        "code_dependencies_sha256": snapshot_files,
        "input_dependencies": {
            "goal1_closeout_sha256": MODULE.GOAL1_COMPLETION_ASSESSMENT_SHA256,
            "goal2_constraint_clarification_sha256": MODULE.GOAL2_CONSTRAINT_CLARIFICATION_SHA256,
        },
    }
    binding_path.write_text(json.dumps(binding, sort_keys=True), encoding="utf-8")
    binding_sha = hashlib.sha256(binding_path.read_bytes()).hexdigest()
    monkeypatch.setattr(
        MODULE,
        "GOAL2_RESIDUAL_APPROVAL_FILES",
        {
            "script": (script_rel, script_sha),
            "plan": (plan_rel, plan_sha),
            "binding": (binding_rel, binding_sha),
        },
    )
    monkeypatch.setattr(MODULE, "GOAL2_RESIDUAL_RESULT", result_rel)
    manifest_path = root / "manifests/campaign.json"
    manifest_path.write_text(json.dumps({"numerical_source_sha256": snapshot_files}), encoding="utf-8")

    result = MODULE._goal2_residual_approval_binding(root)
    assert result["files"]["script"]["sha256"] == script_sha
    assert result["source_snapshot_manifest_sha256"] == snapshot_manifest_sha
    assert result["goal1_closeout_sha256"] == MODULE.GOAL1_COMPLETION_ASSESSMENT_SHA256

    script_path.write_text("changed probe\n", encoding="utf-8")
    with pytest.raises(MODULE.CampaignError, match="approval file changed"):
        MODULE._goal2_residual_approval_binding(root)


def test_residual_action_prints_fixed_goal2_evaluator_and_gpu0_without_execution(tmp_path, monkeypatch):
    root = tmp_path / "campaign"
    run = root / "runs/G1-C/run-c"
    run.mkdir(parents=True)
    plan_path = root / MODULE.GOAL2_RESIDUAL_APPROVAL_FILES["plan"][0]
    plan_path.parent.mkdir(parents=True)
    plan_path.write_text(
        json.dumps({"admission_anchor": {"run_path": str(run.resolve())}}), encoding="utf-8"
    )
    manifest = {
        "root": str(root),
        "campaign_id": root.name,
        "gates": {"goal1_authorized": True},
    }
    monkeypatch.setattr(MODULE, "_campaign_manifest", lambda _root: manifest)
    monkeypatch.setattr(MODULE, "_goal1_residual_runs_complete", lambda _root: {"G1-C": run.resolve()})
    monkeypatch.setattr(MODULE, "_goal2_residual_approval_binding", lambda _root: {"schema_version": 1})

    command = MODULE.goal2_residual_probe(root, run, execute=False)
    assert "fast_lagrangian_evaluate.py" in command
    assert "--goal2-residual-probe" in command
    assert "--run" in command and str(run.resolve()) in command
    assert "CUDA_VISIBLE_DEVICES=0" in command
    assert f"PHYCOFLOW_CAMPAIGN_ADMITTED={root.resolve()}" in command
    assert "PHYCOFLOW_CAMPAIGN_SOURCE_STAGE=goal2" in command


def test_goal2_offline_diagnostic_spend_includes_fixed_reserve_live_and_ledger(tmp_path):
    root = tmp_path / "campaign"
    (root / "manifests").mkdir(parents=True)
    (root / "manifests/launch_ledger.jsonl").write_text(
        json.dumps({"action": MODULE.GOAL2_RESIDUAL_ACTION, "elapsed_seconds": 800.0})
        + "\n"
        + json.dumps({"action": "evaluate-full-abt-validation", "elapsed_seconds": 400.0})
        + "\n",
        encoding="utf-8",
    )
    spent = MODULE._goal2_offline_diagnostic_seconds(
        root,
        {
            "active_processes": [
                {"action": MODULE.GOAL2_RESIDUAL_ACTION, "elapsed_seconds": 100.0},
                {"action": "evaluate-full-abt-validation", "elapsed_seconds": 1000.0},
            ]
        },
    )
    assert spent == 180.0 + 800.0 + 100.0


def _goal2_smoke_recovery_fixture(tmp_path, monkeypatch, observed_end):
    root = tmp_path / "campaign"
    run = root / "runs/recovery_smoke2ep/run-1"
    manifests = root / "manifests"
    configs = root / "configs"
    run.mkdir(parents=True)
    manifests.mkdir(parents=True)
    configs.mkdir(parents=True)
    prepared_config = configs / "recovery_smoke2ep.yaml"
    prepared_config.write_text("profile: frozen\n", encoding="utf-8")
    (run / "resolved_config.yaml").write_text("profile: frozen\n", encoding="utf-8")
    (run / "run_manifest.json").write_text(
        json.dumps({"experiment_name": f"_LPCAccel/{root.name}/{MODULE.GOAL2_SMOKE_ARM}"}),
        encoding="utf-8",
    )
    config_sha = hashlib.sha256(prepared_config.read_bytes()).hexdigest()
    manifest = {
        "root": str(root),
        "campaign_id": root.name,
        "goal2_configs": {MODULE.GOAL2_SMOKE_ARM: {"sha256": config_sha}},
    }
    monkeypatch.setattr(MODULE, "_config_path", lambda _manifest, _arm: prepared_config)
    monkeypatch.setattr(MODULE, "_run_dirs", lambda _root, _arm: [run])
    monkeypatch.setattr(
        MODULE,
        "_run_record",
        lambda _run: {
            "status": "completed",
            "completed_epochs": 2.0,
            "completed_updates": 76,
            "steps_per_epoch": 38,
        },
    )
    monkeypatch.setattr(MODULE, "_stage_source_digest", lambda *_args: "source-binding")
    checkpoint_receipt = {
        "path": str((root / MODULE.GOAL2_SMOKE_EPOCH1_RECEIPT).resolve()),
        "sha256": "receipt-digest",
        "checkpoint_path": str((run / "checkpoints/epoch_001.pt").resolve()),
        "checkpoint_sha256": "checkpoint-digest",
        "config_sha256": "canonical-run-config",
        "reference_sha256": "reference-digest",
    }
    monkeypatch.setattr(
        MODULE,
        "_goal2_smoke_epoch1_checkpoint_receipt",
        lambda *_args: checkpoint_receipt,
    )
    monkeypatch.setattr(
        MODULE,
        "_goal2_smoke_recovery_verification",
        lambda *_args: {"status": "verified", "details": recovery_details},
    )
    launch_row = {
        "arm": MODULE.GOAL2_SMOKE_ARM,
        "action": "launch",
        "pid": 101,
        "run": str(run.resolve()),
        "gpu_index": 0,
        "source_stage": "goal2",
        "source_binding_sha256": "source-binding",
        "segment_start_epoch": 0.0,
        "segment_end_epoch": observed_end,
        "epochs_charged": 2.0,
        "return_code": -2,
    }
    resume_row = {
        **launch_row,
        "action": "resume",
        "pid": 102,
        "segment_start_epoch": 1.0,
        "segment_end_epoch": 2.0,
        "epochs_charged": 1.0,
        "return_code": 0,
    }
    (manifests / "launch_ledger.jsonl").write_text(
        json.dumps(launch_row) + "\n" + json.dumps(resume_row) + "\n", encoding="utf-8"
    )
    stop_row = {
        "arm": MODULE.GOAL2_SMOKE_ARM,
        "pid": 101,
        "run": str(run.resolve()),
        "signal": "SIGINT",
        "checkpoint_receipt_path": checkpoint_receipt["path"],
        "checkpoint_receipt_sha256": checkpoint_receipt["sha256"],
    }
    (manifests / "stop_requests.jsonl").write_text(json.dumps(stop_row) + "\n", encoding="utf-8")
    analysis = root / "analysis/goal2_smoke_pre_resume"
    analysis.mkdir(parents=True)
    metrics = {}
    rows_by_name = {
        "lpc_ordered_updates": [
            {"step": step, "ordered": {"accepted_update": step}} for step in range(1, 47)
        ],
        "fast_lagrangian_updates": [{"step": step} for step in range(1, 47)],
        "lr_warmup_updates": [{"step": step} for step in range(1, 47)],
        "history": [{"epoch": 1.0, "completed_updates": 19.5, "epoch_complete": True}],
    }
    filenames = {
        "lpc_ordered_updates": "lpc_ordered_updates.jsonl",
        "fast_lagrangian_updates": "fast_lagrangian_updates.jsonl",
        "lr_warmup_updates": "lr_warmup_updates.jsonl",
        "history": "history.jsonl",
    }
    for name, rows in rows_by_name.items():
        path = analysis / filenames[name]
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        metrics[name] = {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "complete_rows": len(rows),
            "last_complete_step": 38 if name == "history" else 46,
        }
    inventory_path = root / MODULE.GOAL2_SMOKE_INTERRUPTION_INVENTORY
    inventory_path.write_text(
        json.dumps(
            {
                "status": "stopped_before_resume",
                "run": str(run.resolve()),
                "pre_stop_epoch1_receipt": checkpoint_receipt["path"],
                "metrics": metrics,
                "discarded_accepted_steps_after_checkpoint": 8,
                "possible_additional_inflight_update_upper_bound": 1,
                "conservative_first_segment_epochs_charged": 2,
            }
        ),
        encoding="utf-8",
    )
    recovery_details = {
        "interruption_inventory_path": str(inventory_path.resolve()),
        "interruption_inventory_sha256": hashlib.sha256(inventory_path.read_bytes()).hexdigest(),
        "observed_pre_interrupt_accepted_step": 46,
        "atomic_recovery_checkpoint_epoch": 1.0,
        "progress_metadata_reported_epoch": 0.0 if observed_end == 0.0 else observed_end,
        "discarded_durable_accepted_updates": 8,
        "possible_additional_inflight_update_upper_bound": 1,
        "first_ordered_row_bitwise_identical": True,
        "all_preserved_tail_sampler_decisions_identical": True,
        "index_rng_matches_exact_38_update_advance": True,
    }
    return root, run, manifest


def test_goal2_recovery_gate_preserves_fractional_interrupted_tail_and_safe_charge(
    tmp_path, monkeypatch
):
    observed_end = 46 / 38
    root, _run, manifest = _goal2_smoke_recovery_fixture(tmp_path, monkeypatch, observed_end)

    receipt = MODULE._goal2_smoke_recovery_receipt(root, manifest)

    assert receipt["observed_first_segment_end_epoch"] == observed_end
    assert receipt["epochs_charged"] == 3.0
    assert receipt["checkpoint_receipt"]["checkpoint_sha256"] == "checkpoint-digest"
    assert receipt["recovery_verification"]["status"] == "verified"


@pytest.mark.parametrize("observed_end", [0.99, 2.0, float("nan")])
def test_goal2_recovery_gate_rejects_missing_epoch1_or_full_epoch2_tail(
    tmp_path, monkeypatch, observed_end
):
    root, _run, manifest = _goal2_smoke_recovery_fixture(tmp_path, monkeypatch, observed_end)

    with pytest.raises(MODULE.CampaignError, match="progress"):
        MODULE._goal2_smoke_recovery_receipt(root, manifest)


def test_goal2_recovery_gate_accepts_zero_reported_progress_only_with_atomic_e1_tail_proof(
    tmp_path, monkeypatch
):
    root, _run, manifest = _goal2_smoke_recovery_fixture(tmp_path, monkeypatch, 0.0)

    receipt = MODULE._goal2_smoke_recovery_receipt(root, manifest)

    assert receipt["observed_first_segment_end_epoch"] == 0.0
    assert receipt["interruption_inventory"]["atomic_recovery_checkpoint_epoch"] == 1.0
    assert receipt["interruption_inventory"]["observed_pre_interrupt_accepted_step"] == 46


def test_goal2_recovery_gate_rejects_zero_progress_if_interruption_inventory_changes(
    tmp_path, monkeypatch
):
    root, _run, manifest = _goal2_smoke_recovery_fixture(tmp_path, monkeypatch, 0.0)
    inventory_path = root / MODULE.GOAL2_SMOKE_INTERRUPTION_INVENTORY
    inventory_path.write_text(inventory_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(MODULE.CampaignError, match="interruption inventory"):
        MODULE._goal2_smoke_recovery_receipt(root, manifest)


def test_goal2_epoch1_checkpoint_receipt_binds_immutable_state_and_goal2_source(
    tmp_path, monkeypatch
):
    root = tmp_path / "campaign"
    run = root / "runs/recovery_smoke2ep/run-1"
    (run / "checkpoints").mkdir(parents=True)
    (root / "manifests").mkdir(parents=True)
    config_path = run / "resolved_config.yaml"
    config_path.write_text("profile: ordered\n", encoding="utf-8")
    checkpoint_path = run / "checkpoints/epoch_001.pt"
    checkpoint_path.write_bytes(b"immutable epoch-one checkpoint")
    source_run = root / "source"
    source_run.mkdir()
    source_checkpoint = source_run / "checkpoints/last.pt"
    source_checkpoint.parent.mkdir()
    source_checkpoint.write_bytes(b"source checkpoint")
    source_config = source_run / "resolved_config.yaml"
    source_config.write_text("source: fixed\n", encoding="utf-8")
    source_run_manifest = source_run / "run_manifest.json"
    source_run_manifest.write_text("{}\n", encoding="utf-8")
    source_hashes = {
        "checkpoint": hashlib.sha256(source_checkpoint.read_bytes()).hexdigest(),
        "resolved_config": hashlib.sha256(source_config.read_bytes()).hexdigest(),
        "run_manifest": hashlib.sha256(source_run_manifest.read_bytes()).hexdigest(),
    }
    g1s_run = root / "runs/G1-S/run-g1s"
    g1s_run.mkdir(parents=True)
    (g1s_run / "run_manifest.json").write_text(
        json.dumps({"source_hashes": source_hashes}), encoding="utf-8"
    )
    monkeypatch.setattr(
        MODULE,
        "_goal1_residual_runs_complete",
        lambda _root: {"G1-S": g1s_run},
    )
    source_files = {
        path: hashlib.sha256(path.encode()).hexdigest()
        for path in (
            "src/phycoflow_reconstruction/training/lpc_ordered.py",
            "src/phycoflow_reconstruction/training/post_training.py",
            "src/phycoflow_reconstruction/training/run_store.py",
            "src/phycoflow_reconstruction/config/validate.py",
            "src/phycoflow_reconstruction/other.py",
        )
    }
    monkeypatch.setattr(MODULE, "_verify_goal2_source_binding", lambda _manifest: {"source_binding_sha256": "pin"})
    reference = {
        "version": 1,
        "family_names": ["global_distribution", "cross_spectrum", "topology"],
        "means": {"global_distribution": 1.0, "cross_spectrum": 2.0, "topology": 3.0},
        "denominators": {"global_distribution": 1.0, "cross_spectrum": 2.0, "topology": 3.0},
        "denominator_floor": MODULE.GOAL2_ORDERED_SETTINGS["denominator_floor"],
        "source_sha256": source_hashes["checkpoint"],
        "no_regression_bounds": {"global_distribution": 1.0, "cross_spectrum": 2.0},
    }
    state = {
        "version": 1,
        "reference": reference,
        "dual_lr": MODULE.GOAL2_ORDERED_SETTINGS["dual_lr"],
        "multipliers": {"global_distribution": 0.25, "cross_spectrum": 0.0},
        "completed_updates": 38,
    }
    ordered_reference_artifact = {
        "reference": reference,
        "settings": MODULE.GOAL2_ORDERED_SETTINGS,
        "source_checkpoint_sha256": source_hashes["checkpoint"],
    }
    ordered_reference_path = run / "artifacts/lpc_ordered_reference.json"
    ordered_reference_path.parent.mkdir()
    ordered_reference_path.write_text(json.dumps(ordered_reference_artifact), encoding="utf-8")
    prepared_config_path = root / "configs/recovery_smoke2ep.yaml"
    prepared_config_path.parent.mkdir()
    prepared_config_path.write_text("profile: ordered\n", encoding="utf-8")
    canonical_config_sha = hashlib.sha256(b"canonical config digest, not raw yaml").hexdigest()
    prepared_config_file_sha = hashlib.sha256(prepared_config_path.read_bytes()).hexdigest()
    assert canonical_config_sha != prepared_config_file_sha
    (run / "run_manifest.json").write_text(
        json.dumps({"config_sha256": canonical_config_sha}), encoding="utf-8"
    )
    implementation_paths = (
        "src/phycoflow_reconstruction/training/lpc_ordered.py",
        "src/phycoflow_reconstruction/training/post_training.py",
        "src/phycoflow_reconstruction/training/run_store.py",
        "src/phycoflow_reconstruction/config/validate.py",
    )
    receipt = {
        "schema_version": 1,
        "status": "atomic_epoch1_checkpoint_verified",
        "run": str(run.resolve()),
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        "global_step": 38,
        "reporting_epoch": 1.0,
        "config_sha256": canonical_config_sha,
        "lpc_ordered_reference_sha256": hashlib.sha256(
            ordered_reference_path.read_bytes()
        ).hexdigest(),
        "lpc_ordered_implementation": {path: source_files[path] for path in implementation_paths},
        "lpc_ordered_state": state,
        "external_policy_multipliers": [0.0] * 5,
        "external_policy_completed_updates": 38,
        "rng_state_sha256": {
            "torch_cpu": "b" * 64,
            "index_generator": "c" * 64,
            "torch_cuda": ["d" * 64],
        },
        "optimizer_step_counts": [38],
        "source_hashes": source_hashes,
        "created_utc": "2026-10-23T15:10:00Z",
    }
    receipt_path = root / MODULE.GOAL2_SMOKE_EPOCH1_RECEIPT
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    manifest = {
        "root": str(root),
        "source_run": str(source_run),
        "source_checkpoint": str(source_checkpoint),
        "source_checkpoint_sha256": source_hashes["checkpoint"],
        "source_config_sha256": hashlib.sha256(b"separate Run 11f anchor config").hexdigest(),
        "goal2_configs": {
            MODULE.GOAL2_SMOKE_ARM: {
                "path": str(prepared_config_path),
                "sha256": prepared_config_file_sha,
            }
        },
        "stage_source_bindings": {"goal2": {"files_sha256": source_files}},
    }

    verified = MODULE._goal2_smoke_epoch1_checkpoint_receipt(root, manifest, run)

    assert verified["sha256"] == hashlib.sha256(receipt_path.read_bytes()).hexdigest()
    assert verified["checkpoint_sha256"] == receipt["checkpoint_sha256"]
    checkpoint_path.write_bytes(b"changed after stop")
    with pytest.raises(MODULE.CampaignError, match="receipt failed"):
        MODULE._goal2_smoke_epoch1_checkpoint_receipt(root, manifest, run)


def test_goal2_recovery_verification_binds_stop_checkpoint_resume_and_replay(tmp_path, monkeypatch):
    root = tmp_path / "campaign"
    run = root / "runs/recovery_smoke2ep/run-1"
    (root / "manifests").mkdir(parents=True)
    run.mkdir(parents=True)
    receipt = {
        "path": str((root / MODULE.GOAL2_SMOKE_EPOCH1_RECEIPT).resolve()),
        "sha256": "receipt-sha",
        "checkpoint_sha256": "checkpoint-sha",
        "config_sha256": "config-sha",
        "reference_sha256": "reference-sha",
    }
    manifest = {
        "root": str(root),
        "goal2_configs": {MODULE.GOAL2_SMOKE_ARM: {"sha256": "config-sha"}},
    }
    monkeypatch.setattr(MODULE, "_stage_source_digest", lambda *_args: "source-sha")
    checks = {
        "checkpoint_and_source_binding": True,
        "lpc_ordered_reference_and_implementation": True,
        "per_field_mu_and_external_policy_state": True,
        "optimizer_and_rng_state": True,
        "first_resumed_step_is_39": True,
        "sampler_and_replayed_batch_values_reconciled": True,
        "log_tail_has_no_duplicate_update": True,
    }
    verification = {
        "schema_version": 1,
        "status": "verified",
        "run": str(run.resolve()),
        "checkpoint_receipt_path": receipt["path"],
        "epoch1_checkpoint_receipt_sha256": receipt["sha256"],
        "checkpoint_sha256": receipt["checkpoint_sha256"],
        "source_binding_sha256": "source-sha",
        "config_sha256": "config-sha",
        "lpc_ordered_reference_sha256": "reference-sha",
        "first_resumed_step": 39,
        "final_step": 76,
        "checks": checks,
        "details": {"limitation": "raw sample IDs are not present in fast history"},
    }
    verification_path = root / MODULE.GOAL2_SMOKE_RECOVERY_VERIFICATION
    verification_path.write_text(json.dumps(verification), encoding="utf-8")

    result = MODULE._goal2_smoke_recovery_verification(root, manifest, run, receipt)

    assert result["status"] == "verified"
    assert result["first_resumed_step"] == 39
    verification["checks"]["sampler_and_replayed_batch_values_reconciled"] = False
    verification_path.write_text(json.dumps(verification), encoding="utf-8")
    with pytest.raises(MODULE.CampaignError, match="incomplete or not bound"):
        MODULE._goal2_smoke_recovery_verification(root, manifest, run, receipt)


def test_goal2_smoke_authorization_round_trips_to_launch_gate(tmp_path, monkeypatch):
    root = tmp_path / "campaign"
    (root / "manifests").mkdir(parents=True)
    evidence = root / "manifests/ready.json"
    evidence.write_text("{}\n")
    config = root / "prepared.yaml"
    config.write_text("{}\n")
    binding = {"evidence_path": "manifests/ready.json", "source_binding_sha256": "source-pin"}
    manifest = {
        "root": str(root), "gates": {"goal1_authorized": True},
        "stage_source_bindings": {"goal2": binding},
        "goal2_configs": {
            MODULE.GOAL2_FORMAL_ARM: {"sha256": "formal-config"},
            MODULE.GOAL2_SMOKE_ARM: {"sha256": "smoke-config"},
        },
    }
    monkeypatch.setattr(MODULE, "_campaign_manifest", lambda _: manifest)
    monkeypatch.setattr(MODULE, "_verify_goal2_source_binding", lambda _: binding)
    monkeypatch.setattr(MODULE, "_goal1_residual_runs_complete", lambda _: {})
    monkeypatch.setattr(MODULE, "_validate_arm", lambda *_: {"config": config})
    monkeypatch.setattr(MODULE, "_validate_config_only", lambda _: None)
    monkeypatch.setattr(MODULE, "_run_dirs", lambda *_: [])
    MODULE.authorize(root, "goal2-smoke", evidence)
    launch_gate = MODULE._goal2_smoke_gate(manifest)
    assert launch_gate["path"] == str(evidence)
    assert launch_gate["sha256"] == MODULE._sha256(evidence)
    assert launch_gate["source_binding_sha256"] == "source-pin"

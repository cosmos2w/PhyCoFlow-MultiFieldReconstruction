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
    with pytest.raises(MODULE.CampaignError, match="no implemented LPC-ordered runner"):
        MODULE.launch(root, "G2", execute=False)


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
    command = MODULE.launch(root, "T0", execute=False)
    assert "CUDA_VISIBLE_DEVICES=1" in command
    assert "post-train" in command
    assert "--config" in command and str(config_path) in command
    assert "KEOPS_CACHE_FOLDER" in command and "PHYCOFLOW_TOPOLOGY_WORKERS" in command
    assert "runpy.run_path" in command
    assert "sys.path.insert(0, os.path.dirname(os.path.abspath(script)))" in command
    assert MODULE._environment(root)["PHYCOFLOW_CAMPAIGN_ADMITTED"] == str(root.resolve())


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
        "configs": {
            arm: {
                "path": str(config_path),
                "sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
            }
        },
    }


def _mock_runner_prerequisites(monkeypatch, root: Path, lock_checks=None):
    monkeypatch.setattr(MODULE, "status", lambda _root: {"gpu_seconds_charged": 0.0})
    monkeypatch.setattr(MODULE, "_run_dirs", lambda _root, _arm: [])
    monkeypatch.setattr(MODULE, "_prepare_keops_build_folder", lambda *_args: root / "caches/keops")

    def check_idle():
        if lock_checks is not None:
            assert MODULE._try_campaign_lock(root / "manifests/active_process.lock") is None
            lock_checks.append(True)

    monkeypatch.setattr(MODULE, "_ensure_gpu1_idle", check_idle)


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
    assert lock_checks == [True]
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
        with pytest.raises(MODULE.CampaignError, match="holds the campaign lock"):
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
    monkeypatch.setattr(MODULE, "_ledger_action_seconds", lambda *_args: 0.0)
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


def test_stop_waits_for_campaign_child_exit_and_coordinator_unlock(tmp_path, monkeypatch):
    root = tmp_path / "campaign"
    manifests = root / "manifests"
    manifests.mkdir(parents=True)
    active_path = manifests / "active_process.json"
    lock_path = manifests / "active_process.lock"
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
    result = MODULE.stop(root, timeout_seconds=1.0)
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
    try:
        with pytest.raises(MODULE.CampaignError, match="stop timed out after 0.02s"):
            MODULE.stop(root, timeout_seconds=0.02)
    finally:
        MODULE._release_campaign_lock(held_lock)

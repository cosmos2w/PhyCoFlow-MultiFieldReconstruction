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

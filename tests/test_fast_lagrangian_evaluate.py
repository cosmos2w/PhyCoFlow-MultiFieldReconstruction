from __future__ import annotations

import fcntl
import hashlib
import json
import os

import pytest
import torch
import yaml
from torch import nn

from phycoflow_reconstruction.coherence import build_coherence_family
from phycoflow_reconstruction.contracts import (
    DataSpec,
    FamilyResult,
    FieldSample,
    ModelCapabilities,
    ObservationBatch,
    TermResult,
)
from phycoflow_reconstruction.data.normalization import FieldNormalizer
from phycoflow_reconstruction.data.sensor_protocols import SensorProtocol, build_observation_batch
from phycoflow_reconstruction.training.fidelity_budget import FidelityBudget
from phycoflow_reconstruction.training.lpc_ordered import (
    LPCOrderedReference,
    LPCOrderedState,
    implementation_binding,
)
from scripts.posttraining import fast_lagrangian_campaign as CAMPAIGN
from scripts.posttraining import fast_lagrangian_evaluate as EVALUATOR
from scripts.posttraining.fast_lagrangian_evaluate import (
    EvaluationError,
    _compact_evaluation,
    _evaluate_groups,
    _evaluation_filename,
    _group_slices,
    _parser,
    _ratio_tree,
    _slice_batch,
    _summarize_groups,
    _uniform_validation_indices,
    _validate_device_admission,
    _validate_experiment_binding,
    _validate_max_samples,
)

_REAL_PROC_COMMANDLINE = EVALUATOR._proc_commandline
_REAL_CONTROLLER_ENTRY_MATCHES = EVALUATOR._controller_entry_matches


def test_cuda_device_selection_requires_matching_physical_gpu_environment(tmp_path, monkeypatch):
    root = tmp_path / "LPCAccel_fixture"
    monkeypatch.delenv("PHYCOFLOW_CAMPAIGN_ADMITTED", raising=False)
    _validate_device_admission(root, "cpu")
    with pytest.raises(EvaluationError, match="admission"):
        _validate_device_admission(root, "cuda:0")
    monkeypatch.setenv("PHYCOFLOW_CAMPAIGN_ADMITTED", str(root))
    for physical in ("0", "1", "2"):
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", physical)
        monkeypatch.setenv("PHYCOFLOW_CAMPAIGN_GPU", physical)
        _validate_device_admission(root, "cuda:0")
    for invalid in ("", "3", "0,1", "-1"):
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", invalid)
        with pytest.raises(EvaluationError, match="admission"):
            _validate_device_admission(root, "cuda:0")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("PHYCOFLOW_CAMPAIGN_GPU", "1")
    with pytest.raises(EvaluationError, match="admission"):
        _validate_device_admission(root, "cuda:0")
    with pytest.raises(EvaluationError, match="explicitly reserved"):
        _validate_device_admission(root, "cuda:1")


def _controller_receipt_fixture(
    tmp_path,
    monkeypatch,
    *,
    action="evaluate-full-abt-validation",
    arm="G1-S",
    source_stage="goal1",
    mode_flag=None,
    lock_held=True,
):
    root = tmp_path / "LPCAccel_controller_fixture"
    run = root / "runs" / arm / "run-controller"
    run.mkdir(parents=True)
    config = root / "configs" / f"{arm}.yaml"
    config.parent.mkdir(parents=True)
    config.write_text("fixture: true\n", encoding="utf-8")
    manifests = root / "manifests"
    manifests.mkdir()
    active = manifests / "active.json"
    lock = manifests / "gpu.lock"
    lock.write_text("fixture lock inode\n", encoding="utf-8")
    controller_pid = os.getppid()
    evaluator_command = CAMPAIGN._with_short_multiprocessing_tmp(
        [
            str(CAMPAIGN.PYTHON),
            str(EVALUATOR.REPO_ROOT / "scripts/posttraining/fast_lagrangian_evaluate.py"),
            "--campaign-root",
            str(root),
            "--run",
            str(run),
        ]
    )
    if mode_flag is not None:
        evaluator_command.append(mode_flag)
    evaluator_command.extend(("--device", "cuda:0"))
    parent_subcommand = {
        "evaluate-full-abt-validation": "evaluate",
        "evaluate-native-maps": "maps",
        "evaluate-goal2-residual-probe": "residual",
    }[action]
    controller_command = [
        str(CAMPAIGN.PYTHON),
        str(EVALUATOR.REPO_ROOT / "scripts/posttraining/fast_lagrangian_campaign.py"),
        "--root",
        str(root),
        parent_subcommand,
    ]
    if parent_subcommand == "evaluate":
        controller_command.append(arm)
    if parent_subcommand == "evaluate":
        controller_command.extend(
            (
                "--run",
                str(run),
                "--checkpoint",
                "best",
                "--max-samples",
                "128",
                "--gpu",
                "0",
                "--execute",
            )
        )
    else:
        controller_command.extend(("--run", str(run), "--execute"))
    record = {
        "action": action,
        "arm": arm,
        "pid": os.getpid(),
        "coordinator_pid": controller_pid,
        "config": str(config),
        "run": str(run),
        "gpu_index": 0,
        "gpu_uuid": "GPU-fixture-0",
        "source_stage": source_stage,
        "source_binding_sha256": f"binding-{source_stage}",
        "command": evaluator_command,
    }
    active.write_text(json.dumps(record), encoding="utf-8")
    monkeypatch.setenv("PHYCOFLOW_CAMPAIGN_ADMITTED", str(root.resolve()))
    monkeypatch.setenv("PHYCOFLOW_CAMPAIGN_SOURCE_STAGE", source_stage)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("PHYCOFLOW_CAMPAIGN_GPU", "0")
    monkeypatch.setattr(CAMPAIGN, "_campaign_manifest", lambda _root: {"root": str(root)})
    monkeypatch.setattr(
        CAMPAIGN,
        "_stage_source_digest",
        lambda _root, _manifest, stage: f"binding-{stage}",
    )
    monkeypatch.setattr(CAMPAIGN, "_config_path", lambda _manifest, _arm: config)
    monkeypatch.setattr(CAMPAIGN, "_gpu_active_path", lambda _root, _gpu: active)
    monkeypatch.setattr(CAMPAIGN, "_gpu_lock_path", lambda _root, _gpu: lock)
    monkeypatch.setattr(
        CAMPAIGN,
        "_proc_parent_pid",
        lambda pid: controller_pid if pid == os.getpid() else None,
    )
    monkeypatch.setattr(CAMPAIGN, "_pid_is_alive", lambda pid: pid == controller_pid)
    monkeypatch.setattr(EVALUATOR, "_controller_entry_matches", lambda _pid: True)
    monkeypatch.setattr(
        EVALUATOR,
        "_proc_commandline",
        lambda _pid: controller_command,
    )
    monkeypatch.setattr(CAMPAIGN, "_pid_matches", lambda pid, *_args: pid == os.getpid())
    monkeypatch.setattr(CAMPAIGN, "_campaign_process_matches_gpu", lambda *_args: True)
    monkeypatch.setattr(EVALUATOR, "_lock_is_held_without_mutation", lambda _path: lock_held)
    return root, run, record, active


@pytest.mark.parametrize(
    ("action", "arm", "source_stage", "mode_flag"),
    [
        ("evaluate-full-abt-validation", "G1-S", "goal1", None),
        ("evaluate-native-maps", "G1-C", "goal1", "--native-maps"),
        (
            "evaluate-goal2-residual-probe",
            "G1-C",
            "goal2",
            "--goal2-residual-probe",
        ),
    ],
)
def test_campaign_inner_evaluator_requires_controller_child_and_held_gpu_lock(
    tmp_path, monkeypatch, action, arm, source_stage, mode_flag
):
    root, run, record, _active = _controller_receipt_fixture(
        tmp_path,
        monkeypatch,
        action=action,
        arm=arm,
        source_stage=source_stage,
        mode_flag=mode_flag,
    )
    assert record["command"][1] == "-c"
    assert "multiprocessing.current_process()._config['tempdir'] = short_tmp" in record["command"][2]
    assert record["command"][3] == str(
        EVALUATOR.REPO_ROOT / "scripts/posttraining/fast_lagrangian_evaluate.py"
    )
    _validate_device_admission(
        root,
        "cuda:0",
        run_path=run,
        arm=arm,
        action=action,
        source_stage=source_stage,
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("action", "launch"),
        ("run", "/another/run"),
        ("config", "/another/config.yaml"),
        ("source_stage", "goal2"),
        ("source_binding_sha256", "unbound"),
        ("gpu_index", 1),
        ("gpu_index", False),
    ],
)
def test_campaign_inner_evaluator_rejects_mismatched_active_receipt(
    tmp_path, monkeypatch, field, value
):
    root, run, record, active = _controller_receipt_fixture(tmp_path, monkeypatch)
    record[field] = value
    active.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(EvaluationError, match="active campaign receipt"):
        _validate_device_admission(
            root,
            "cuda:0",
            run_path=run,
            arm="G1-S",
            action="evaluate-full-abt-validation",
            source_stage="goal1",
        )


@pytest.mark.parametrize(
    ("index", "value"),
    [(4, "launch"), (5, "G1-C"), (3, "/another/root"), (7, "/another/run"), (13, "1")],
)
def test_campaign_inner_evaluator_rejects_wrong_live_controller_command(
    tmp_path, monkeypatch, index, value
):
    root, run, _record, _active = _controller_receipt_fixture(tmp_path, monkeypatch)
    controller_pid = os.getppid()
    command = [
        str(CAMPAIGN.PYTHON),
        str(EVALUATOR.REPO_ROOT / "scripts/posttraining/fast_lagrangian_campaign.py"),
        "--root",
        str(root),
        "evaluate",
        "G1-S",
        "--run",
        str(run),
        "--checkpoint",
        "best",
        "--max-samples",
        "128",
        "--gpu",
        "0",
        "--execute",
    ]
    command[index] = value
    monkeypatch.setattr(
        EVALUATOR,
        "_proc_commandline",
        lambda pid: command if pid == controller_pid else [],
    )
    with pytest.raises(EvaluationError, match="live child of the locked controller"):
        _validate_device_admission(
            root,
            "cuda:0",
            run_path=run,
            arm="G1-S",
            action="evaluate-full-abt-validation",
            source_stage="goal1",
        )


def test_campaign_controller_command_accepts_repo_relative_script_path(tmp_path, monkeypatch):
    root = tmp_path / "campaign"
    run = root / "runs/G2-O/run-relative"
    run.mkdir(parents=True)
    command = [
        str(CAMPAIGN.PYTHON),
        "scripts/posttraining/fast_lagrangian_campaign.py",
        "--root",
        str(root),
        "evaluate",
        "G2-O",
        "--run",
        str(run),
        "--checkpoint",
        "best",
        "--max-samples",
        "128",
        "--gpu",
        "0",
        "--execute",
    ]
    monkeypatch.setattr(EVALUATOR, "_proc_commandline", lambda _pid: command)
    assert EVALUATOR._controller_command_matches(
        os.getpid(), root, run, "G2-O", "evaluate-full-abt-validation", 0
    )
    wrapped = [command[0], "-c", "bootstrap", *command[1:]]
    monkeypatch.setattr(EVALUATOR, "_proc_commandline", lambda _pid: wrapped)
    assert not EVALUATOR._controller_command_matches(
        os.getpid(), root, run, "G2-O", "evaluate-full-abt-validation", 0
    )


def test_campaign_inner_evaluator_rejects_non_campaign_parent_process(tmp_path, monkeypatch):
    root, run, record, active = _controller_receipt_fixture(tmp_path, monkeypatch)
    record["command"][-1] = "cpu"
    active.write_text(json.dumps(record), encoding="utf-8")
    monkeypatch.setattr(EVALUATOR, "_proc_commandline", _REAL_PROC_COMMANDLINE)
    monkeypatch.setattr(
        EVALUATOR, "_controller_entry_matches", _REAL_CONTROLLER_ENTRY_MATCHES
    )
    with pytest.raises(EvaluationError, match="live child of the locked controller"):
        _validate_device_admission(
            root,
            "cpu",
            run_path=run,
            arm="G1-S",
            action="evaluate-full-abt-validation",
            source_stage="goal1",
        )


@pytest.mark.parametrize("device_name", ("cpu", "cuda:0"))
def test_campaign_inner_evaluator_rejects_environment_only_invocation(
    tmp_path, monkeypatch, device_name
):
    root, run, _record, active = _controller_receipt_fixture(tmp_path, monkeypatch)
    active.unlink()
    ticks = iter((0.0, EVALUATOR.CONTROLLER_RECEIPT_WAIT_SECONDS + 1.0))
    monkeypatch.setattr(EVALUATOR.time, "monotonic", lambda: next(ticks))
    with pytest.raises(EvaluationError, match="no matching active campaign controller receipt"):
        _validate_device_admission(
            root,
            device_name,
            run_path=run,
            arm="G1-S",
            action="evaluate-full-abt-validation",
            source_stage="goal1",
        )


def test_lock_probe_is_read_only_and_does_not_create_missing_lock(tmp_path):
    missing = tmp_path / "missing.lock"
    assert EVALUATOR._lock_is_held_without_mutation(missing) is False
    assert not missing.exists()

    lock_path = tmp_path / "held.lock"
    lock_path.write_text("unchanged\n", encoding="utf-8")
    descriptor = os.open(lock_path, os.O_RDWR)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert EVALUATOR._lock_is_held_without_mutation(lock_path) is True
        assert lock_path.read_text(encoding="utf-8") == "unchanged\n"
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)
    assert EVALUATOR._lock_is_held_without_mutation(lock_path) is False
    assert lock_path.read_text(encoding="utf-8") == "unchanged\n"


def test_campaign_inner_evaluator_rejects_an_unheld_lifetime_lock(tmp_path, monkeypatch):
    root, run, _record, _active = _controller_receipt_fixture(
        tmp_path, monkeypatch, lock_held=False
    )
    with pytest.raises(EvaluationError, match="live child of the locked controller"):
        _validate_device_admission(
            root,
            "cuda:0",
            run_path=run,
            arm="G1-S",
            action="evaluate-full-abt-validation",
            source_stage="goal1",
        )


def test_evaluation_identity_is_bound_and_output_names_are_immutable(tmp_path):
    root = tmp_path / "LPCAccel_fixture"
    name = f"_LPCAccel/{root.name}/T0"
    config = {"output": {"experiment_name": name}}
    manifest = {"experiment_name": name}
    _validate_experiment_binding(config, manifest, root, "T0")
    with pytest.raises(EvaluationError, match="another campaign arm"):
        _validate_experiment_binding(config, manifest, root, "T1")
    config["output"]["experiment_name"] = "different"
    with pytest.raises(EvaluationError, match="another campaign arm"):
        _validate_experiment_binding(config, manifest, root, "T0")
    baseline = _evaluation_filename("T0", "last", 10, 64, "a" * 64)
    assert baseline == "T0_last_epoch10_n64_aaaaaaaaaaaa_validation.json"
    assert baseline != _evaluation_filename("T0", "last", 20, 64, "a" * 64)
    assert baseline != _evaluation_filename("T0", "last", 10, 32, "a" * 64)
    assert baseline != _evaluation_filename("T0", "last", 10, 64, "b" * 64)
    args = _parser().parse_args(["--campaign-root", str(root), "--run", str(root)])
    assert args.device == "cpu"


def test_native_map_approval_binding_rejects_changed_file_and_manifest(tmp_path, monkeypatch):
    files = {}
    entries = {}
    for key in ("extractor", "plan", "clarification"):
        path = tmp_path / f"{key}.json"
        payload = f"fixed {key} payload\n".encode()
        path.write_bytes(payload)
        digest = hashlib.sha256(payload).hexdigest()
        files[key] = (path.name, digest)
        entries[key] = {"path": path.name, "sha256": digest}
    monkeypatch.setattr(EVALUATOR, "NATIVE_MAP_APPROVAL_FILES", files)

    binding = EVALUATOR._native_map_approval_binding(tmp_path)
    assert binding == {"schema_version": 1, **entries}
    EVALUATOR._verify_native_map_approval_binding(tmp_path, {"native_map_approval_binding": binding})
    with pytest.raises(EvaluationError, match="exact native-map approval binding"):
        EVALUATOR._verify_native_map_approval_binding(
            tmp_path,
            {"native_map_approval_binding": {**binding, "plan": {"sha256": "0" * 64}}},
        )

    (tmp_path / "plan.json").write_text("changed after approval\n", encoding="utf-8")
    with pytest.raises(EvaluationError, match="approval file changed"):
        EVALUATOR._native_map_approval_binding(tmp_path)


def test_native_map_readiness_rejects_one_g1_arm_below_3800_updates(tmp_path, monkeypatch):
    root = tmp_path / "campaign"
    runs_root = root / "runs"
    campaign_root = root
    campaign = {
        "root": str(root),
        "campaign_id": root.name,
        "gates": {"goal1_authorized": True},
        "eligible_arms": {"goal1": ["G1-C", "G1-S", "G1-R"]},
        "gate_evidence": {"goal1": {"approved_arms": ["G1-C", "G1-S", "G1-R"]}},
    }
    (root / "analysis").mkdir(parents=True)
    (root / "manifests").mkdir()
    (root / "manifests/campaign.json").write_text(json.dumps(campaign), encoding="utf-8")
    (runs_root / "G1-C/run-c").mkdir(parents=True)
    (runs_root / "G1-S/run-s").mkdir(parents=True)
    (runs_root / "G1-R/run-r").mkdir(parents=True)
    (root / "analysis/native_grid_field_map_predeclaration.json").write_text(
        json.dumps({"goal1_arms": ["G1-C", "G1-S", "G1-R"]}), encoding="utf-8"
    )
    for arm, run_name in (("G1-C", "run-c"), ("G1-S", "run-s"), ("G1-R", "run-r")):
        run = runs_root / arm / run_name
        (run / "run_manifest.json").write_text(
            json.dumps(
                {
                    "last_checkpoint_epoch": 100.0,
                    "last_checkpoint_step": 3800,
                    "source_immutable_verified": True,
                    "source_hashes": {"source": "same"},
                    "source_hashes_after": {"source": "same"},
                }
            ),
            encoding="utf-8",
        )
        (run / "status.json").write_text(
            json.dumps(
                {
                    "status": "completed",
                    "steps_per_epoch": 38,
                    "configured_steps": 3800,
                    "global_step": 3799 if arm == "G1-S" else 3800,
                    "source_immutable_verified": True,
                }
            ),
            encoding="utf-8",
        )
        (run / "progress.json").write_text(
            json.dumps(
                {
                    "completed_updates": 3799 if arm == "G1-S" else 3800,
                    "reporting_epoch": 100.0,
                }
            ),
            encoding="utf-8",
        )
    selected = runs_root / "G1-C/run-c"
    monkeypatch.setattr(
        EVALUATOR,
        "_validate_campaign_run",
        lambda _campaign, _run: (campaign_root, selected, "G1-C"),
    )
    monkeypatch.setattr(
        EVALUATOR,
        "_load_run_inputs",
        lambda run, _checkpoint: (
            {"optimization": {"epochs": 100}},
            EVALUATOR._read_json(run / "run_manifest.json"),
            run / "checkpoints/epoch_100.pt",
            "checkpoint-hash",
        ),
    )
    monkeypatch.setattr(EVALUATOR, "_validate_experiment_binding", lambda *_args: None)

    with pytest.raises(EvaluationError, match="completed 100-epoch / 3800-step"):
        EVALUATOR._validate_native_maps_runs(root, selected)


def test_evaluator_default_path_remains_full_abt_without_native_maps(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        EVALUATOR,
        "evaluate_campaign_run",
        lambda *args, **kwargs: calls.append((args, kwargs)) or "full-abt-result.json",
    )
    monkeypatch.setattr(
        EVALUATOR,
        "extract_native_maps_campaign",
        lambda *_args, **_kwargs: pytest.fail("default evaluator path invoked native-map extraction"),
    )
    result = EVALUATOR.main(
        ["--campaign-root", str(tmp_path), "--run", str(tmp_path / "runs/G1-C/run-c")]
    )
    assert result == 0
    assert calls == [
        (
            (tmp_path, tmp_path / "runs/G1-C/run-c"),
            {"checkpoint_name": "best", "max_samples": 64, "device_name": "cpu"},
        )
    ]


def test_goal2_residual_mode_is_opt_in_and_uses_its_fixed_outer_entrypoint(tmp_path, monkeypatch):
    parsed = _parser().parse_args(
        [
            "--campaign-root",
            str(tmp_path),
            "--run",
            str(tmp_path / "runs/G1-C/run-c"),
            "--goal2-residual-probe",
        ]
    )
    assert parsed.goal2_residual_probe is True
    assert parsed.native_maps is False

    calls = []
    monkeypatch.setattr(
        EVALUATOR,
        "run_goal2_residual_probe_campaign",
        lambda *args, **kwargs: calls.append((args, kwargs)) or tmp_path / "probe.json",
    )
    monkeypatch.setattr(
        EVALUATOR,
        "extract_native_maps_campaign",
        lambda *_args, **_kwargs: pytest.fail("residual mode dispatched to native maps"),
    )
    assert EVALUATOR.main(
        [
            "--campaign-root",
            str(tmp_path),
            "--run",
            str(tmp_path / "runs/G1-C/run-c"),
            "--goal2-residual-probe",
        ]
    ) == 0
    assert calls == [
        (
            (tmp_path, tmp_path / "runs/G1-C/run-c"),
            {"device_name": "cpu"},
        )
    ]
    assert EVALUATOR.main(
        [
            "--campaign-root",
            str(tmp_path),
            "--run",
            str(tmp_path / "runs/G1-C/run-c"),
            "--goal2-residual-probe",
            "--native-maps",
        ]
    ) == 2


def test_goal2_residual_entry_rejects_nonzero_physical_gpu(tmp_path, monkeypatch):
    root = tmp_path / "campaign"
    run = root / "runs/G1-C/run-c"
    monkeypatch.setattr(EVALUATOR, "_validate_campaign_run", lambda *_args: (root, run, "G1-C"))
    monkeypatch.setattr(
        EVALUATOR, "_validate_device_admission", lambda *_args, **_kwargs: None
    )
    monkeypatch.setenv("PHYCOFLOW_CAMPAIGN_ADMITTED", str(root))
    monkeypatch.setenv("PHYCOFLOW_CAMPAIGN_GPU", "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    monkeypatch.setenv("PHYCOFLOW_CAMPAIGN_SOURCE_STAGE", "goal2")
    with pytest.raises(EvaluationError, match="physical GPU 0"):
        EVALUATOR.run_goal2_residual_probe_campaign(root, run, device_name="cuda:0")


def test_campaign_stage_bootstrap_selects_the_goal1_snapshot_before_project_imports(
    tmp_path, monkeypatch
):
    root = tmp_path / "campaign"
    root.mkdir()
    snapshot_src = root / "source_snapshots/goal1_c073eba/src"
    snapshot_src.mkdir(parents=True)
    monkeypatch.setenv("PHYCOFLOW_CAMPAIGN_SOURCE_STAGE", "goal1")
    monkeypatch.setenv("PHYCOFLOW_CAMPAIGN_ADMITTED", str(root))
    monkeypatch.setattr(CAMPAIGN, "_campaign_manifest", lambda _root: {"root": str(root)})
    monkeypatch.setattr(CAMPAIGN, "_stage_source_root", lambda *_args: snapshot_src)

    assert EVALUATOR._campaign_source_path() == snapshot_src


def test_evaluation_source_stage_keeps_goal1_on_its_frozen_snapshot(tmp_path, monkeypatch):
    root = tmp_path / "campaign"
    root.mkdir()
    snapshot_src = root / "source_snapshots/goal1_c073eba/src"
    snapshot_src.mkdir(parents=True)
    monkeypatch.setenv("PHYCOFLOW_CAMPAIGN_SOURCE_STAGE", "goal1")
    monkeypatch.setenv("PHYCOFLOW_CAMPAIGN_ADMITTED", str(root.resolve()))
    monkeypatch.setattr(EVALUATOR, "SOURCE_PATH", snapshot_src)
    monkeypatch.setattr(CAMPAIGN, "_campaign_manifest", lambda _root: {"root": str(root.resolve())})
    monkeypatch.setattr(CAMPAIGN, "_stage_source_root", lambda *_args: snapshot_src)
    monkeypatch.setattr(CAMPAIGN, "_stage_source_digest", lambda *_args: "frozen-goal1-binding")

    binding = EVALUATOR._validate_evaluation_source_stage(root, "G1-S")

    assert binding["stage"] == "goal1"
    assert binding["source_root"] == str(snapshot_src)
    assert binding["source_binding_sha256"] == "frozen-goal1-binding"


def _goal2_evaluation_binding_fixture(tmp_path, monkeypatch):
    root = tmp_path / "campaign"
    prepared_path = root / "configs/G2-O.yaml"
    run = root / "runs/G2-O/run-1"
    g1s = root / "runs/G1-S/run-s"
    source_root = root / "current-source/src"
    for directory in (prepared_path.parent, run / "artifacts", g1s, source_root):
        directory.mkdir(parents=True, exist_ok=True)

    source_hashes = {
        "checkpoint": "a" * 64,
        "resolved_config": "b" * 64,
        "run_manifest": "c" * 64,
    }
    fidelity_settings = {
        "metric": "endpoint_mse_model_units",
        "relative_increase": 0.05,
        "absolute_allowance": 0.0,
        "denominator_floor": 1.0e-8,
        "calibration_batches": 8,
        "calibration_draws": 2,
        "calibration_seed": 700103,
    }
    config = {
        "optimization": {
            "lpc_ordered": dict(EVALUATOR.GOAL2_ORDERED_SETTINGS),
            "multitask": {
                "fidelity_budget": fidelity_settings,
                "external_policy": {"method": "lagrangian", "objective_scale": "source_mean"},
            },
        }
    }
    prepared_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    (run / "resolved_config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    (run / "status.json").write_text(
        json.dumps({"status": "completed", "source_immutable_verified": True}), encoding="utf-8"
    )
    run_manifest = {
        "run_id": "run-1",
        "experiment_name": "_LPCAccel/campaign/G2-O",
        "last_checkpoint_epoch": 100.0,
        "last_checkpoint_step": 3800,
        "source_immutable_verified": True,
        "source_hashes": source_hashes,
        "source_hashes_after": source_hashes,
    }
    (run / "run_manifest.json").write_text(json.dumps(run_manifest), encoding="utf-8")
    (g1s / "run_manifest.json").write_text(json.dumps({"source_hashes": source_hashes}), encoding="utf-8")

    means = {"global_distribution": 1.25, "cross_spectrum": 2.5, "topology": 3.75}
    ordered_reference = LPCOrderedReference.from_means(
        means,
        source_sha256=source_hashes["checkpoint"],
        denominator_floor=EVALUATOR.GOAL2_ORDERED_SETTINGS["denominator_floor"],
    )
    ordered_report = {
        "reference": ordered_reference.state_dict(),
        "settings": dict(EVALUATOR.GOAL2_ORDERED_SETTINGS),
        "source_checkpoint_sha256": source_hashes["checkpoint"],
        "calibration": "same full-descriptor SOURCE TRAIN draws as fidelity/C0; no extra forward",
    }
    (run / "artifacts/lpc_ordered_reference.json").write_text(
        json.dumps(ordered_report), encoding="utf-8"
    )
    fidelity_reference = FidelityBudget.from_reference(
        ("field",),
        (0.5,),
        sum(means.values()),
        relative_increase=0.05,
        absolute_allowance=0.0,
        denominator_floor=1.0e-8,
    )
    (run / "artifacts/fidelity_budget_reference.json").write_text(
        json.dumps({"settings": fidelity_settings, "reference": fidelity_reference.state_dict()}),
        encoding="utf-8",
    )

    campaign_manifest = {
        "root": str(root.resolve()),
        "campaign_id": root.name,
        "goal2_configs": {},
    }
    monkeypatch.setenv("PHYCOFLOW_CAMPAIGN_SOURCE_STAGE", "goal2")
    monkeypatch.setenv("PHYCOFLOW_CAMPAIGN_ADMITTED", str(root.resolve()))
    monkeypatch.setattr(EVALUATOR, "SOURCE_PATH", source_root)
    monkeypatch.setattr(CAMPAIGN, "_campaign_manifest", lambda _root: campaign_manifest)
    monkeypatch.setattr(CAMPAIGN, "_stage_source_root", lambda *_args: source_root)
    monkeypatch.setattr(CAMPAIGN, "_stage_source_digest", lambda *_args: "current-goal2-binding")
    monkeypatch.setattr(CAMPAIGN, "_goal2_formal_gate", lambda _manifest: {"status": "ready"})
    monkeypatch.setattr(CAMPAIGN, "_validate_arm", lambda *_args: {"config": prepared_path})
    monkeypatch.setattr(CAMPAIGN, "_run_dirs", lambda *_args: [run])
    monkeypatch.setattr(CAMPAIGN, "_goal1_residual_runs_complete", lambda _root: {"G1-S": g1s})
    monkeypatch.setattr(
        CAMPAIGN,
        "_run_record",
        lambda _run: {
            "status": "completed",
            "completed_epochs": 100.0,
            "completed_updates": 3800,
            "steps_per_epoch": 38,
            "configured_steps": 3800,
        },
    )
    return root, run, config, run_manifest, campaign_manifest


def test_goal2_evaluator_binds_prepared_source_profile_and_ordered_reference(tmp_path, monkeypatch):
    root, run, config, run_manifest, _campaign_manifest = _goal2_evaluation_binding_fixture(
        tmp_path, monkeypatch
    )
    source_stage = EVALUATOR._validate_evaluation_source_stage(root, "G2-O")
    binding = EVALUATOR._validate_goal2_campaign_binding(
        root, run, config, run_manifest, source_stage
    )

    assert binding["source_stage"] == "goal2"
    assert binding["source_binding_sha256"] == "current-goal2-binding"
    assert len(binding["ordered_reference"]["reference_sha256"]) == 64
    assert binding["ordered_reference"]["reference"]["source_sha256"] == "a" * 64


def test_goal2_evaluator_rejects_changed_source_and_ordered_mean_bindings(tmp_path, monkeypatch):
    root, run, config, run_manifest, _campaign_manifest = _goal2_evaluation_binding_fixture(
        tmp_path, monkeypatch
    )
    source_stage = EVALUATOR._validate_evaluation_source_stage(root, "G2-O")
    wrong_profile = yaml.safe_load(yaml.safe_dump(config))
    wrong_profile["optimization"]["lpc_ordered"]["dual_lr"] = 0.02
    ordered_report = json.loads((run / "artifacts/lpc_ordered_reference.json").read_text())
    fidelity_report = json.loads((run / "artifacts/fidelity_budget_reference.json").read_text())
    with pytest.raises(EvaluationError, match="settings differ"):
        EVALUATOR._validate_goal2_ordered_reference(
            wrong_profile, run_manifest, ordered_report, fidelity_report
        )

    changed_run_manifest = {**run_manifest, "source_hashes": {"checkpoint": "d" * 64}}
    with pytest.raises(EvaluationError, match="SOURCE immutability"):
        EVALUATOR._validate_goal2_campaign_binding(
            root, run, config, changed_run_manifest, source_stage
        )

    changed_means = {**ordered_report["reference"]["means"], "topology": 4.75}
    ordered_report["reference"] = LPCOrderedReference.from_means(
        changed_means,
        source_sha256="a" * 64,
        denominator_floor=EVALUATOR.GOAL2_ORDERED_SETTINGS["denominator_floor"],
    ).state_dict()
    with pytest.raises(EvaluationError, match="sum to fidelity C0"):
        EVALUATOR._validate_goal2_ordered_reference(
            config, run_manifest, ordered_report, fidelity_report
        )


def test_goal2_evaluator_binds_checkpoint_to_ordered_and_endpoint_references():
    source_hashes = {"checkpoint": "a" * 64, "resolved_config": "b" * 64}
    means = {"global_distribution": 1.0, "cross_spectrum": 2.0, "topology": 3.0}
    reference = LPCOrderedReference.from_means(means, source_sha256=source_hashes["checkpoint"])
    state = LPCOrderedState(reference, dual_lr=0.01)
    ordered_state = state.state_dict()
    ordered_state["completed_updates"] = 3800
    fidelity = FidelityBudget.from_reference(("field",), (0.5,), 6.0).state_dict()
    config = {
        "optimization": {
            "lpc_ordered": dict(EVALUATOR.GOAL2_ORDERED_SETTINGS),
            "multitask": {"external_policy": {"method": "lagrangian"}},
        }
    }
    run_manifest = {"source_hashes": source_hashes, "config_sha256": "config-digest"}
    binding = {"reference": reference.state_dict(), "fidelity_reference": fidelity}
    checkpoint = {
        "global_step": 3800,
        "config_sha256": "config-digest",
        "source_hashes": source_hashes,
        "lpc_ordered_reference_sha256": "reference-digest",
        "lpc_ordered_implementation": implementation_binding(),
        "lpc_ordered_state": ordered_state,
        "external_policy": {
            "reference": fidelity,
            "settings": config["optimization"]["multitask"]["external_policy"],
            "completed_updates": 3800,
        },
    }

    EVALUATOR._validate_goal2_ordered_checkpoint(
        checkpoint, config, run_manifest, binding, "reference-digest"
    )
    with pytest.raises(EvaluationError, match="ordered SOURCE/config implementation"):
        EVALUATOR._validate_goal2_ordered_checkpoint(
            {**checkpoint, "lpc_ordered_reference_sha256": "changed"},
            config,
            run_manifest,
            binding,
            "reference-digest",
        )


def test_validation_selection_gates_and_uniform_group_boundaries():
    selected = _uniform_validation_indices(101, 64)
    assert len(selected) == len(set(selected)) == 64
    assert selected[0] == 0 and selected[-1] == 100
    assert _group_slices(96) == ((0, 32), (32, 64), (64, 96))
    for invalid in (0, 31, 33, 160, True):
        with pytest.raises(EvaluationError):
            _validate_max_samples(invalid)
    with pytest.raises(EvaluationError, match="validation split"):
        _uniform_validation_indices(63, 64)
    with pytest.raises(SystemExit):
        _parser().parse_args(
            ["--campaign-root", "/tmp/campaign", "--run", "/tmp/run", "--split", "test"]
        )


def _batch() -> ObservationBatch:
    batch_size, obs_count, query_count = 3, 2, 4
    return ObservationBatch(
        obs_coords=torch.arange(batch_size * obs_count, dtype=torch.float32).reshape(
            batch_size, obs_count, 1
        ),
        obs_values=torch.ones(batch_size, obs_count, 1),
        obs_field_ids=torch.zeros(batch_size, obs_count, dtype=torch.long),
        obs_valid_mask=torch.ones(batch_size, obs_count, dtype=torch.bool),
        query_coords=torch.arange(batch_size * query_count, dtype=torch.float32).reshape(
            batch_size, query_count, 1
        ),
        query_valid_mask=torch.ones(batch_size, query_count, dtype=torch.bool),
        target_fields=torch.arange(batch_size * query_count, dtype=torch.float32).reshape(
            batch_size, query_count, 1
        ),
        sample_ids=("a:0", "b:0", "c:0"),
        obs_indices=torch.arange(batch_size * obs_count).reshape(batch_size, obs_count),
        logical_shapes=((2,), (2,), (2,)),
        metadata={
            "query_indices": torch.arange(batch_size * query_count).reshape(
                batch_size, query_count
            ),
            "protocol": {"name": "fixture"},
            "sample_context": {
                "conditions": torch.arange(batch_size).reshape(batch_size, 1),
                "time": ("t0", "t1", "t2"),
                "auxiliary": {
                    "value": torch.arange(batch_size * query_count).reshape(batch_size, query_count)
                },
            },
        },
    )


def test_fixed_batch_slicing_preserves_sensor_query_and_context_identity():
    batch = _batch()
    sliced = _slice_batch(batch, 1, 3)
    assert sliced.sample_ids == ("b:0", "c:0")
    assert sliced.target_fields.tolist() == batch.target_fields[1:3].tolist()
    assert (
        sliced.metadata["query_indices"].tolist() == batch.metadata["query_indices"][1:3].tolist()
    )
    assert sliced.metadata["sample_context"]["time"] == ("t1", "t2")
    assert sliced.metadata["sample_context"]["conditions"].tolist() == [[1], [2]]
    assert sliced.metadata["sample_context"]["auxiliary"]["value"].tolist() == (
        batch.metadata["sample_context"]["auxiliary"]["value"][1:3].tolist()
    )


def _compact_fixture(topology_estimator: str | None = None) -> dict:
    diagnostics = {"scalar_filtrations": 6}
    if topology_estimator is not None:
        diagnostics["estimator_value"] = topology_estimator
    return {
        "mse_normalized": 1.5,
        "per_field_mse_normalized": {"u": 1.0, "v": 2.0},
        "mean_relative_l2": 0.25,
        "per_field_relative_l2": {"u": 0.2, "v": 0.3},
        "worst_field_relative_l2": 0.3,
        "endpoint_mse_model_units": {"u": 3.0, "v": 4.0},
        "coherence": {
            "total": 0.5,
            "components": {"topology.self.persistence": 0.5},
            "families": {
                name: {
                    "total": 0.5,
                    "weighted_total": 0.5,
                    "component_scalars": {f"{name}.fixture": 0.5},
                    "diagnostics": diagnostics if name == "topology" else {},
                }
                for name in ("global_distribution", "cross_spectrum", "topology")
            },
        },
    }


def test_compact_metrics_requires_full_topology_and_keeps_A_B_T_components():
    result = _compact_evaluation(_compact_fixture(), ("u", "v"))
    assert result["endpoint_mse_model_units"] == {"u": 3.0, "v": 4.0}
    assert result["coherence"]["families"]["topology"]["topology_scalar_filtrations"] == 6
    assert set(result["coherence"]["families"]) == {
        "global_distribution",
        "cross_spectrum",
        "topology",
    }
    with pytest.raises(EvaluationError, match="sampled topology"):
        _compact_evaluation(_compact_fixture("T_estimate"), ("u", "v"))
    missing = _compact_fixture()
    missing["coherence"]["families"]["topology"]["diagnostics"] = {}
    with pytest.raises(EvaluationError, match="six complete"):
        _compact_evaluation(missing, ("u", "v"))


def test_group_means_weight_equal_sized_groups_and_ratios_keep_zero_baselines_missing():
    groups = [
        {"sample_ids": ("a", "b"), "metrics": {"value": 2.0, "nested": {"u": 4.0}}},
        {"sample_ids": ("c",), "metrics": {"value": 8.0, "nested": {"u": 10.0}}},
    ]
    summary = _summarize_groups(groups)
    assert summary == {"value": 4.0, "nested": {"u": 6.0}}
    assert _ratio_tree({"value": 4.0, "zero": 1.0}, {"value": 2.0, "zero": 0.0}) == {
        "value": 2.0,
        "zero": None,
    }
    opposite_worst_fields = [
        {"sample_ids": ("a",), "metrics": {"reconstruction": {
            "per_field_relative_l2": {"u": 1.0, "v": 3.0}, "worst_field_relative_l2": 3.0,
        }}},
        {"sample_ids": ("b",), "metrics": {"reconstruction": {
            "per_field_relative_l2": {"u": 4.0, "v": 1.0}, "worst_field_relative_l2": 4.0,
        }}},
    ]
    assert _summarize_groups(opposite_worst_fields)["reconstruction"]["worst_field_relative_l2"] == 2.5


class _EndpointModel(nn.Module):
    capabilities = ModelCapabilities("point", True, True, False, False)

    def __init__(self, offset: float):
        super().__init__()
        self.offset = nn.Parameter(torch.tensor(float(offset)))

    def differentiable_reconstruct(self, batch, *, steps, generator):
        return self.offset.to(batch.query_coords).expand(
            batch.query_coords.shape[0], batch.query_coords.shape[1], 2
        )


class _SimpleFamily:
    def __init__(self, family_name: str, coefficient: float):
        self.family_name = family_name
        self.family_weight = 1.0
        self.target_use = "paired_supervised"
        self.units = "model_units"
        self.component_weights = {"fixture": 1.0}
        self.coefficient = coefficient

    def __call__(self, generated, reference, *, coordinates, context=None):
        per_sample = self.coefficient * (generated - reference).square().mean((1, 2))
        term = TermResult(per_sample, per_sample.mean())
        path = f"{self.family_name}.fixture"
        return FamilyResult(
            component_results={path: term},
            per_sample_cost=per_sample,
            scalar_loss=per_sample.mean(),
            diagnostics={"family": self.family_name, "component_weights": {path: 1.0}},
        )


def _small_persistence_family():
    spec = DataSpec(("u", "v"), ("1", "1"), 2, (4, 4))
    settings = {
        "strategy": "cubical_persistence",
        "target_use": "paired_supervised",
        "units": "model_units",
        "fields": ["u", "v"],
        "geometry": {"grid_shape": [4, 4], "axes": [0, 1], "neighbors": 1},
        "filtration": {
            "dimensions": [0, 1],
            "directions": ["sublevel", "superlevel"],
            "smoothing_sigma": 0.0,
        },
        "persistence": {"distance": "sliced_wasserstein", "projections": 32},
        "components": {
            "self": {"enabled": True, "weight": 1.0},
            "mutual": {"enabled": True, "weight": 1.0, "groups": [["u", "v"]], "lines": 1},
        },
    }
    return build_coherence_family("topology", settings, spec, FieldNormalizer.identity(2))


def _small_evaluation_batch():
    axis = torch.linspace(0.0, 1.0, 4)
    coordinates = torch.stack(torch.meshgrid(axis, axis, indexing="ij"), dim=-1).reshape(-1, 2)
    samples = []
    for sample_index in range(2):
        values = torch.stack(
            (
                coordinates[:, 0] + 0.05 * sample_index,
                coordinates[:, 1].square() + 0.03 * sample_index,
            ),
            dim=-1,
        )
        samples.append(
            FieldSample(
                values=values,
                coordinates=coordinates,
                coordinates_raw=coordinates,
                time=torch.empty(0),
                trajectory_id=f"fixture-{sample_index}",
                time_index=0,
                conditions=torch.empty(0),
                field_names=("u", "v"),
                logical_shape=(4, 4),
            )
        )
    protocol = SensorProtocol(field_counts={"u": 2, "v": 2}, seed=12)
    return build_observation_batch(samples, protocol, query_points=None)


def test_group_evaluator_calls_real_full_GUDHI_topology_on_shared_cpu_batches():
    pytest.importorskip("gudhi")
    from phycoflow_reconstruction.training.post_training import _build_comparison_batch

    complete = _small_evaluation_batch()
    config = {
        "optimization": {"multitask": {"external_policy": {"method": "lagrangian"}}},
        "evaluation": {"seed": 17, "generation_steps": 1, "query_points": 16},
        "coherence": {"compute_budget": {"point_count": 16, "query_policy": "random_per_sample"}},
        "rollout": {"solver": "euler"},
        "observation_consistency": {"mode": "none", "final_clamp": False},
    }
    comparison = _build_comparison_batch(complete, config)
    families = {
        "global_distribution": _SimpleFamily("global_distribution", 0.5),
        "cross_spectrum": _SimpleFamily("cross_spectrum", 0.75),
        "topology": _small_persistence_family(),
    }
    source_groups = _evaluate_groups(
        _EndpointModel(0.1),
        complete,
        comparison,
        families,
        {},
        ("u", "v"),
        config,
        normalizer=FieldNormalizer.identity(2),
        family_scales={name: 1.0 for name in families},
        group_size=1,
    )
    child_groups = _evaluate_groups(
        _EndpointModel(0.3),
        complete,
        comparison,
        families,
        {},
        ("u", "v"),
        config,
        normalizer=FieldNormalizer.identity(2),
        family_scales={name: 1.0 for name in families},
        group_size=1,
    )

    assert len(source_groups) == len(child_groups) == 2
    assert [group["generation_seed"] for group in source_groups] == [2027, 102030]
    assert [group["generation_seed"] for group in child_groups] == [2027, 102030]
    for source_group, child_group in zip(source_groups, child_groups):
        source_topology = source_group["metrics"]["coherence"]["families"]["topology"]
        child_topology = child_group["metrics"]["coherence"]["families"]["topology"]
        assert source_topology["topology_scalar_filtrations"] == 6
        assert child_topology["topology_scalar_filtrations"] == 6
        assert "estimator" not in source_topology
        assert "estimator" not in child_topology
    assert (
        source_groups[0]["metrics"]["endpoint_mse_model_units"]
        != child_groups[0]["metrics"]["endpoint_mse_model_units"]
    )

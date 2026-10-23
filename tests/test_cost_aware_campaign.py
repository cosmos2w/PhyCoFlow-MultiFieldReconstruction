"""Focused safety/accounting tests for the foreground campaign controller."""

from __future__ import annotations

import hashlib
import json

import pytest

from scripts.posttraining import cost_aware_campaign as campaign


def _write_cursor_checkpoint(run_dir, *, global_step, config_sha256, source_run):
    import torch

    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "global_step": global_step,
            "config_sha256": config_sha256,
            "source_run": str(source_run),
            "source_hashes": {"checkpoint": "source-checkpoint-hash"},
        },
        checkpoint_dir / "last.pt",
    )


def test_duration_parser_rejects_nonfinite_and_over_cap_values():
    for value in (float("nan"), float("inf"), float("-inf"), 10.1, 0.0):
        with pytest.raises(campaign.CampaignError):
            campaign._parse_duration(value, maximum=10.0, label="test")
    assert campaign._parse_duration(10.0, maximum=10.0, label="test") == 10.0


def test_source_checkpoint_must_remain_inside_its_pinned_run(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    inside = source / "checkpoints" / "last.pt"
    inside.parent.mkdir()
    inside.write_bytes(b"source")
    assert campaign._resolve_source_checkpoint({"source_checkpoint": "last.pt"}, source) == inside

    outside = tmp_path / "outside.pt"
    outside.write_bytes(b"not-source")
    with pytest.raises(campaign.CampaignError, match="escapes its SOURCE run"):
        campaign._resolve_source_checkpoint({"source_checkpoint": str(outside)}, source)


def test_run_epoch_uses_checkpoint_cursor_when_progress_is_ahead(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    source_run = tmp_path / "source"
    source_run.mkdir()
    config_sha256 = "a" * 64
    (run / "run_manifest.json").write_text(
        json.dumps(
            {
                "config_sha256": config_sha256,
                "parent_run": str(source_run),
                "last_checkpoint_step": 20,
                "last_checkpoint_epoch": 2,
            }
        ),
        encoding="utf-8",
    )
    (run / "status.json").write_text(
        json.dumps({"global_step": 20, "steps_per_epoch": 10}), encoding="utf-8"
    )
    (run / "progress.json").write_text(
        json.dumps({"completed_updates": 30, "steps_per_epoch": 10}), encoding="utf-8"
    )
    _write_cursor_checkpoint(
        run,
        global_step=20,
        config_sha256=config_sha256,
        source_run=source_run,
    )
    assert campaign._run_epoch(run) == 2


def test_run_epoch_rejects_committed_metadata_without_last_checkpoint(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "status.json").write_text(
        json.dumps({"global_step": 10, "steps_per_epoch": 10}), encoding="utf-8"
    )
    with pytest.raises(campaign.CampaignError, match="no last.pt"):
        campaign._run_epoch(run)


def test_epoch_observation_separates_checkpointed_science_from_replayable_work(tmp_path):
    run = tmp_path / "run"
    (run / "metrics").mkdir(parents=True)
    (run / "run_manifest.json").write_text(
        json.dumps(
            {
                "config_sha256": "b" * 64,
                "parent_run": "/source",
                "last_checkpoint_step": 20,
                "last_checkpoint_epoch": 2,
            }
        ),
        encoding="utf-8",
    )
    (run / "status.json").write_text(
        json.dumps({"global_step": 20, "steps_per_epoch": 10}), encoding="utf-8"
    )
    (run / "progress.json").write_text(
        json.dumps({"completed_updates": 20, "steps_per_epoch": 10}), encoding="utf-8"
    )
    (run / "checkpoints").mkdir()
    _write_cursor_checkpoint(
        run,
        global_step=20,
        config_sha256="b" * 64,
        source_run="/source",
    )
    (run / "metrics/history.jsonl").write_text(
        '{"epoch": 2, "step": 20}\n{"epoch": 3, "step": 30}\n',
        encoding="utf-8",
    )

    observed = campaign._training_epoch_observation(run, start_epoch=0, horizon=100)

    assert observed["epoch_exposure"] == 2
    assert observed["accepted_end_epoch"] == 2
    assert observed["observed_history_end_epoch"] == 3
    assert observed["uncheckpointed_epoch_exposure_lower_bound"] == 1
    assert observed["training_complete"] is False


def test_calibration_receipt_pools_only_recorded_fidelity_and_coherence_seconds(tmp_path):
    run = tmp_path / "run"
    artifacts = run / "artifacts"
    artifacts.mkdir(parents=True)
    (artifacts / "fidelity_budget_reference.json").write_text(
        json.dumps({"calibration_seconds": 17.5}), encoding="utf-8"
    )
    (artifacts / "coherence_calibration.json").write_text(
        json.dumps({"mode": "none", "resolved_scales": {"A": 1}}), encoding="utf-8"
    )

    observed = campaign._calibration_observation(run, reserve_seconds=120)

    assert observed["status"] == "measured_artifacts"
    assert observed["fidelity_seconds"] == 17.5
    assert observed["coherence_seconds"] == 0
    assert observed["seconds"] == 17.5


def test_latest_ledger_uses_observed_profile_exposure_and_running_reserves(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    campaign._append_ledger(
        root,
        {
            "attempt_id": "profile",
            "status": "running",
            "reserved_process_seconds": 5400,
            "reserved_profile_seconds": 5400,
            "reserved_aux_epochs": 6,
        },
    )
    campaign._append_ledger(
        root,
        {
            "attempt_id": "profile",
            "status": "completed",
            "charged_process_seconds": 512,
            "charged_profile_seconds": 512,
            "charged_science_epochs": 0,
            "charged_aux_epochs": 41 / 38,
        },
    )
    campaign._append_ledger(
        root,
        {
            "attempt_id": "active",
            "status": "running",
            "reserved_process_seconds": 100,
            "reserved_closeout_seconds": 50,
            "reserved_science_epochs": 7,
        },
    )
    totals = campaign._resource_totals(root)
    assert totals["process_seconds"] == 612
    assert totals["profile_seconds"] == 512
    assert totals["closeout_seconds"] == 50
    assert totals["aux_epochs"] == pytest.approx(41 / 38)
    assert totals["science_epochs"] == 7


def test_prior_failed_profile_charge_reduces_remaining_calibration_cap(tmp_path, monkeypatch):
    class Lock:
        closed = False

        def close(self):
            self.closed = True

    root = tmp_path / "campaign"
    root.mkdir()
    campaign._append_ledger(
        root,
        {
            "attempt_id": "profile-startup-cost",
            "status": "failed_or_stopped",
            "action": "profile",
            "charged_process_seconds": 5399.0,
            "charged_profile_seconds": 5399.0,
            "charged_aux_epochs": 0.0,
        },
    )
    admission = Lock()
    gpu_lock = Lock()
    monkeypatch.setattr(campaign, "_require_environment", lambda: None)
    monkeypatch.setattr(campaign, "_acquire_admission", lambda _root: admission)
    monkeypatch.setattr(campaign, "_check_no_active_process", lambda _root: None)
    monkeypatch.setattr(campaign, "_acquire_physical_gpu_lock", lambda _gpu: gpu_lock)
    monkeypatch.setattr(campaign, "_gpu_record", lambda _gpu, _preflight: {"physical_index": 2})

    with pytest.raises(campaign.CampaignError, match="90 minutes"):
        campaign._run_foreground(
            root,
            command=["profile"],
            task_id="G",
            arm="profile",
            action="profile",
            gpu=2,
            preflight={},
            max_seconds=2.0,
            is_profile=True,
        )

    assert admission.closed is True
    assert gpu_lock.closed is True


def test_runtime_environment_is_scoped_to_owned_campaign_caches(tmp_path, monkeypatch):
    root = tmp_path / "campaign"
    root.mkdir()
    monkeypatch.setenv("OMP_NUM_THREADS", "99")
    env = campaign._campaign_environment(root, 2)
    assert env["CUDA_VISIBLE_DEVICES"] == "2"
    assert env["OMP_NUM_THREADS"] == "4"
    assert env["MKL_NUM_THREADS"] == "4"
    assert env["PHYCOFLOW_TOPOLOGY_WORKERS"] == "4"
    assert env["COST_AWARE_CAMPAIGN_ADMITTED"] == str(root)
    assert env["COST_AWARE_CAMPAIGN_GPU"] == "2"
    assert env["PHYCOFLOW_COMMAND_STARTED_MONOTONIC"] == str(
        campaign.CLI_START_MONOTONIC
    )
    for name in ("PYTHONPYCACHEPREFIX", "KEOPS_CACHE_FOLDER", "MPLCONFIGDIR", "TMPDIR"):
        assert str(root) in env[name]
    assert (root / "caches/keops").is_dir()
    assert (root / "tmp").is_dir()


def test_runtime_environment_uses_distinct_namespace_accepted_by_full_evaluator(
    tmp_path, monkeypatch
):
    from scripts.posttraining import evaluate_full_abt

    root = tmp_path / "campaign"
    root.mkdir()
    for name in evaluate_full_abt.CAMPAIGN_ENVIRONMENT:
        monkeypatch.setenv(name, "inherited")
    env = campaign._campaign_environment(root, 2)

    assert not set(evaluate_full_abt.CAMPAIGN_ENVIRONMENT).intersection(env)
    assert env["COST_AWARE_CAMPAIGN_ADMITTED"] == str(root)
    assert env["COST_AWARE_CAMPAIGN_GPU"] == "2"
    assert env["COST_AWARE_CAMPAIGN_ROOT"] == str(root)
    assert env["COST_AWARE_DATASET_IDENTITIES"] == str(
        root / campaign.IDENTITY_RECEIPT_PATH
    )
    with pytest.raises(evaluate_full_abt.PortableEvaluationError):
        evaluate_full_abt._reject_campaign_environment()


def test_campaign_evaluate_cli_accepts_disjoint_panel_without_changing_default():
    parser = campaign.build_parser()
    base = ["evaluate", "G", "dense", "--run", "/tmp/run"]
    assert parser.parse_args(base).panel == "uniform"
    assert parser.parse_args([*base, "--panel", "disjoint"]).panel == "disjoint"


def test_inspection_binding_rejects_config_bytes_changed_after_pin(tmp_path):
    root = tmp_path / "campaign"
    tasks = root / "tasks"
    tasks.mkdir(parents=True)
    config = root / "configs" / "G_dense.yaml"
    config.parent.mkdir()
    config.write_text("epochs: 100\n", encoding="utf-8")
    binding = {
        "config_path": str(config.resolve()),
        "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
    }
    campaign._write_immutable_json(
        tasks / "G_test_config_bindings.json",
        {"configs": {"dense": {"path": binding["config_path"], "sha256": binding["config_sha256"]}}},
    )
    campaign._verify_frozen_config_binding(root, "G", "dense", binding)
    config.write_text("epochs: 99\n", encoding="utf-8")
    with pytest.raises(campaign.CampaignError, match="immutable inspection binding"):
        campaign._verify_frozen_config_binding(root, "G", "dense", binding)


def test_inspect_and_profile_default_to_physical_gpu_two_without_execution():
    inspect = campaign.build_parser().parse_args(["inspect", "G"])
    profile = campaign.build_parser().parse_args(["profile", "G"])
    train = campaign.build_parser().parse_args(["train", "G", "dense"])
    assert inspect.task == "G"
    assert profile.gpu == campaign.DEFAULT_GPU == 2
    assert profile.execute is False
    assert train.gpu == 2
    assert train.execute is False


def test_public_child_bootstrap_forwards_child_arguments_once(tmp_path, monkeypatch):
    root = tmp_path / "campaign"
    identities = root / campaign.IDENTITY_RECEIPT_PATH
    identities.parent.mkdir(parents=True)
    identities.write_text("{}\n", encoding="utf-8")
    script = campaign.PROFILE_SCRIPT.resolve(strict=True)
    child_args = ["--root", str(root), "--task", "G", "--config", "/campaign/config.yaml"]
    command = campaign._child_script_command(root, script, child_args)
    observed: dict[str, object] = {}

    monkeypatch.setattr(
        campaign,
        "_verify_root",
        lambda path, *, create_payload_dirs: (root, {}),
    )
    monkeypatch.setattr(campaign, "_register_child_identities", lambda _root: None)

    def capture_run_path(path: str, *, run_name: str) -> dict[str, str]:
        observed["path"] = path
        observed["run_name"] = run_name
        observed["argv"] = list(campaign.sys.argv)
        return {}

    monkeypatch.setattr(campaign.runpy, "run_path", capture_run_path)

    assert campaign.main(command[2:]) == 0
    assert observed == {
        "path": str(script),
        "run_name": "__main__",
        "argv": [str(script), *child_args],
    }


def test_controller_and_child_share_cli_start_clock(tmp_path, monkeypatch):
    class Lock:
        closed = False

        def close(self):
            self.closed = True

    class CompletedChild:
        pid = 2_000_000_000
        returncode = 0

        def poll(self):
            return self.returncode

    root = tmp_path / "campaign"
    root.mkdir()
    (root / "tasks").mkdir()
    started = 100.0
    observed: dict[str, object] = {}
    monkeypatch.setattr(campaign, "CLI_START_MONOTONIC", started)
    monkeypatch.setattr(campaign.time, "monotonic", lambda: 101.0)
    monkeypatch.setattr(campaign, "_require_environment", lambda: None)
    monkeypatch.setattr(campaign, "_acquire_admission", lambda _root: Lock())
    monkeypatch.setattr(campaign, "_acquire_physical_gpu_lock", lambda _gpu: Lock())
    monkeypatch.setattr(
        campaign,
        "_resource_totals",
        lambda _root: {
            "process_seconds": 0.0,
            "profile_seconds": 0.0,
            "closeout_seconds": 0.0,
            "science_epochs": 0.0,
            "aux_epochs": 0.0,
        },
    )
    monkeypatch.setattr(campaign, "_check_no_active_process", lambda _root: None)
    monkeypatch.setattr(campaign, "_gpu_record", lambda _gpu, _preflight: {"physical_index": 2})
    ledger_rows = []
    monkeypatch.setattr(campaign, "_append_ledger", lambda _root, row: ledger_rows.append(row))

    def capture_popen(_command, *, env, **_kwargs):
        observed["child_start"] = env["PHYCOFLOW_COMMAND_STARTED_MONOTONIC"]
        return CompletedChild()

    monkeypatch.setattr(campaign.subprocess, "Popen", capture_popen)
    result = campaign._run_foreground(
        root,
        command=["profile"],
        task_id="G",
        arm="profile",
        action="profile",
        gpu=2,
        preflight={},
        max_seconds=10,
        is_profile=True,
    )

    assert observed["child_start"] == str(started)
    assert result["elapsed_seconds"] == pytest.approx(1.0)
    assert ledger_rows[-1]["reserved_profile_seconds"] == 10
    assert ledger_rows[-1]["charged_profile_seconds"] == pytest.approx(1.0)


@pytest.mark.parametrize("returncode", [0, 130])
def test_train_reservation_charges_accepted_epochs_and_stop_loss_only_when_incomplete(
    tmp_path, monkeypatch, returncode
):
    class Lock:
        closed = False

        def close(self):
            self.closed = True

    class CompletedChild:
        pid = 2_000_000_001

        def __init__(self, value):
            self.returncode = value

        def poll(self):
            return self.returncode

    root = tmp_path / "campaign"
    (root / "tasks").mkdir(parents=True)
    admission = Lock()
    gpu_lock = Lock()
    monkeypatch.setattr(campaign, "_require_environment", lambda: None)
    monkeypatch.setattr(campaign, "_acquire_admission", lambda _root: admission)
    monkeypatch.setattr(campaign, "_check_no_active_process", lambda _root: None)
    monkeypatch.setattr(campaign, "_acquire_physical_gpu_lock", lambda _gpu: gpu_lock)
    monkeypatch.setattr(campaign, "_gpu_record", lambda _gpu, _preflight: {"physical_index": 2})
    monkeypatch.setattr(campaign, "_process_group_members", lambda _pgid: [])
    child_run = root / "runs/G/dense/run-001"
    (child_run / "artifacts").mkdir(parents=True)
    (child_run / "artifacts/fidelity_budget_reference.json").write_text(
        json.dumps({"calibration_seconds": 13.25}), encoding="utf-8"
    )
    (child_run / "artifacts/coherence_calibration.json").write_text(
        json.dumps({"mode": "none"}), encoding="utf-8"
    )
    monkeypatch.setattr(
        campaign.subprocess,
        "Popen",
        lambda *_args, **_kwargs: CompletedChild(returncode),
    )
    observation = {
        "run_dir": str(child_run),
        "epoch_exposure": 2.0,
        "accepted_end_epoch": 2.0,
        "observed_history_end_epoch": 2.5,
        "uncheckpointed_epoch_exposure_lower_bound": 0.5,
        "training_complete": returncode == 0,
    }

    def run_child():
        return campaign._run_foreground(
            root,
            command=["train"],
            task_id="G",
            arm="dense",
            action="launch",
            gpu=2,
            preflight={},
            max_seconds=10,
            science_epochs=100,
            stop_loss_reserve_epochs=1,
            epoch_observer=lambda: observation,
        )

    if returncode == 0:
        result = run_child()
        row = campaign._latest_ledger(root)[result["attempt_id"]]
        assert row["charged_science_epochs"] == 2
        assert row["charged_stop_loss_epochs"] == 0
        assert row["charged_aux_epochs"] == 0
        assert row["charged_calibration_seconds"] == pytest.approx(13.25)
        assert row["reserved_calibration_seconds"] == 120
    else:
        with pytest.raises(campaign.CampaignError, match="exited with code 130"):
            run_child()
        row = next(iter(campaign._latest_ledger(root).values()))
        assert row["charged_science_epochs"] == 2
        assert row["charged_stop_loss_epochs"] == 1
        assert row["charged_aux_epochs"] == 1
        assert row["actual_science_epoch_observation"][
            "uncheckpointed_epoch_exposure_lower_bound"
        ] == 0.5


def test_resume_after_epoch_two_keeps_frozen_horizon_and_reuses_last_checkpoint(tmp_path):
    root = tmp_path / "campaign"
    run_dir = root / "runs" / "G" / "strategy" / "run-001"
    run_dir.mkdir(parents=True)
    source_run = tmp_path / "source"
    case_dir = tmp_path / "case"
    config_path = root / "configs" / "G_strategy.yaml"
    config = {
        "case": "turbulent_combustion",
        "stage": "post_training",
        "optimization": {"epochs": 100},
        "output": {
            "experiment_name": f"_CostAwareLPC/{root.name}/G/strategy"
        },
    }
    (run_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "case": config["case"],
                "stage": config["stage"],
                "parent_run": str(source_run),
                "config_sha256": campaign._config_digest(config),
                "last_checkpoint_step": 20,
                "last_checkpoint_epoch": 2,
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "resolved_config.yaml").write_text(
        campaign.yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    (run_dir / "checkpoints").mkdir()
    _write_cursor_checkpoint(
        run_dir,
        global_step=20,
        config_sha256=campaign._config_digest(config),
        source_run=source_run,
    )
    (run_dir / "status.json").write_text(
        json.dumps({"global_step": 20, "steps_per_epoch": 10}), encoding="utf-8"
    )
    (run_dir / "progress.json").write_text(
        json.dumps({"completed_updates": 20, "steps_per_epoch": 10}), encoding="utf-8"
    )

    resumed_run, remaining_epochs = campaign._verify_resume_run(
        root,
        "G",
        "strategy",
        case_dir,
        config_path,
        config,
        {
            "source_run": str(source_run),
            "source_checkpoint_sha256": "source-checkpoint-hash",
        },
        run_dir,
    )

    assert resumed_run == run_dir
    assert remaining_epochs == 98
    assert config["optimization"]["epochs"] == 100

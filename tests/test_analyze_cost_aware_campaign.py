from __future__ import annotations

import importlib
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

# Pytest may be invoked through the environment's console script, whose initial
# import path omits the repository root. These script modules intentionally do
# not form an installed package.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

ANALYZER = importlib.import_module("scripts.posttraining.analyze_cost_aware_campaign")


FIELDS = ["u", "v"]
START = "2026-01-01T00:00:00Z"
FINISH = "2026-01-01T02:00:00Z"
SOURCE_RUN = "/source/case/run"
DATA_PATH = "/metadata-only/validation.h5"
SOURCE_SHA = "a" * 64
DATA_SHA = "b" * 64
CONFIG_SHA = "c" * 64
SENSOR_SHA = "d" * 64


def _metric_group(index: int, *, source: bool, child_factor: float = 0.9) -> dict:
    factor = 1.0 if source else child_factor
    return {
        "sample_ids": [f"sample-{i}" for i in range(index * 32, (index + 1) * 32)],
        "query_ids": [[0] * 4096 for _ in range(32)],
        "generation_seed": 2027 + index * 100_003,
        "generation_steps": 2.0,
        "evaluation_weight_source": "configured_evaluation_weights",
        "coherence": {
            "total": factor,
            "families": {
                "global_distribution": {"weighted_total": factor, "components": {"a.component": factor}},
                "cross_spectrum": {"weighted_total": factor, "components": {"b.component": factor}},
                "topology": {
                    "weighted_total": factor,
                    "topology_scalar_filtrations": 6,
                    "components": {"topology.self.h0": factor, "topology.self.persistence": factor},
                },
            },
            "components": {"topology.self.h0": factor, "topology.self.persistence": factor},
        },
        "endpoint_mse_model_units": {
            "u": (2.0 if source else 1.6) * factor,
            "v": (1.0 if source else 0.9) * factor,
        },
    }


def _paired_baseline() -> dict:
    groups = [_metric_group(index, source=True) for index in range(4)]
    return {
        "generation_steps": 2.0,
        "evaluation_weight_source": "configured_evaluation_weights",
        "sensor_manifest_sha256": SENSOR_SHA,
        "fixed_estimator_scope": {
            "sample_count": 128,
            "group_size": 32,
            "groups": 4,
            "pooling": "query_weighted_mse_and_equal_sample_groups",
        },
        "sample_ids": [sample for group in groups for sample in group["sample_ids"]],
        "query_ids": [query for group in groups for query in group["query_ids"]],
        "fixed_estimator_groups": groups,
        "coherence": {
            "total": 1.0,
            "families": {
                "global_distribution": {"weighted_total": 1.0},
                "cross_spectrum": {"weighted_total": 1.0},
                "topology": {"weighted_total": 1.0, "topology_scalar_filtrations": 6},
            },
        },
        "endpoint_mse_model_units": {"u": 2.0, "v": 1.0},
    }


def _child_metrics(*, child_factor: float = 0.9) -> dict:
    groups = [_metric_group(index, source=False, child_factor=child_factor) for index in range(4)]
    return {
        "generation_steps": 2.0,
        "evaluation_weight_source": "configured_evaluation_weights",
        "sensor_manifest_sha256": SENSOR_SHA,
        "fixed_estimator_scope": {
            "sample_count": 128,
            "group_size": 32,
            "groups": 4,
            "pooling": "query_weighted_mse_and_equal_sample_groups",
        },
        "sample_ids": [sample for group in groups for sample in group["sample_ids"]],
        "query_ids": [query for group in groups for query in group["query_ids"]],
        "fixed_estimator_groups": groups,
        "evaluation_completed_utc": START,
        "command_elapsed_seconds": 1.0,
    }


def _disjoint_portable_evaluation(run_dir: Path, *, child_factor: float = 0.7) -> dict:
    validation_indices = list(range(128, 256))
    sample_ids = [f"disjoint-sample-{index}" for index in validation_indices]
    groups = []
    for group_index in range(4):
        start = group_index * 32
        group_indices = validation_indices[start : start + 32]
        group_ids = sample_ids[start : start + 32]
        source = _metric_group(group_index, source=True)
        child = _metric_group(group_index, source=False, child_factor=child_factor)
        source["sample_ids"] = group_ids
        child["sample_ids"] = group_ids
        groups.append(
            {
                "group": group_index,
                "sample_ids": group_ids,
                "validation_indices": group_indices,
                "source": source,
                "child": child,
            }
        )
    return {
        "schema": "phycoflow_full_abt_validation.v1",
        "run": str(run_dir),
        "config_sha256": CONFIG_SHA,
        "source": {"run": SOURCE_RUN, "hashes": {"checkpoint": SOURCE_SHA}},
        "dataset": {
            "path": DATA_PATH,
            "split": "validation",
            "fingerprint": DATA_SHA,
            "field_names": FIELDS,
        },
        "checkpoint": "last",
        "checkpoint_path": str(run_dir / "checkpoints" / "last.pt"),
        "checkpoint_sha256": "e" * 64,
        "checkpoint_training_epoch": 100.0,
        "evaluated_at_utc": FINISH,
        "inference_contract": {
            "weight_identity": "raw_live_model_weights",
            "model_ema_eval": False,
            "rollout_steps": 2,
            "rollout_solver": "euler",
        },
        "selection": {
            "split": "validation",
            "panel": {
                "kind": "disjoint",
                "algorithm": "chronological_linspace_after_uniform_exclusion_v1",
                "reference_count": 128,
                "excluded_validation_indices": list(range(128)),
            },
            "validation_indices": validation_indices,
            "sample_ids": sample_ids,
            "group_size": 32,
            "group_count": 4,
            "comparison_query_count": 4096,
            "query_indices_shape": [128, 4096],
            "groups": [group["validation_indices"] for group in groups],
        },
        "evaluation": {
            "panel_kind": "disjoint",
            "samples": 128,
            "topology_scalar_filtrations": 6,
        },
        "groups": groups,
    }


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _fixture_root(tmp_path: Path, *, duplicate_factor: float = 0.9) -> Path:
    root = tmp_path / "campaign"
    task = {
        "case": "synthetic_case",
        "arms": ["dense", "heuristic"],
        "source_run": SOURCE_RUN,
        "source_checkpoint_sha256": SOURCE_SHA,
        "dataset_identity": {"path": DATA_PATH, "fingerprint": DATA_SHA},
        "horizon": 100,
        "frozen_task_config": {
            "dataset": {"field_names": FIELDS, "split": "train"},
            "evaluation": {
                "split": "validation",
                "max_samples": 128,
                "group_size": 32,
                "query_points": 4096,
                "generation_steps": 2,
                "sample_selection": "uniform",
            },
        },
    }
    inventory = {
        "tasks": {"G": task},
        "quality_contract": {
            "additive_ratio_max": 0.9,
            "family_ratio_max": 1.05,
            "aggregate_endpoint_ratio_max": 1.05,
            "each_field_ratio_max": 1.05,
            "monitor_epochs": list(range(0, 101, 10)),
            "window": "e>=60 and (e-40,e]",
            "TEST": "forbidden",
        },
    }
    _write_json(root / "tasks" / "task_inventory.json", inventory)
    run_dir = root / "runs" / "G" / "dense" / "child-001"
    _write_json(
        run_dir / "run_manifest.json",
        {
            "parent_run": SOURCE_RUN,
            "source_hashes": {"checkpoint": SOURCE_SHA},
            "dataset_path": DATA_PATH,
            "dataset_fingerprint": DATA_SHA,
            "config_sha256": CONFIG_SHA,
            "checkpoint_hashes": {"last": "e" * 64},
            "last_checkpoint_epoch": 100,
            "steps_per_epoch": 38,
        },
    )
    _write_json(
        run_dir / "status.json",
        {"status": "completed", "updated_utc": FINISH, "checkpoint_epoch": 100},
    )
    _write_json(run_dir / "environment.json", {"python": "test only", "cuda_visible_devices": "2"})
    _write_json(run_dir / "evaluation" / "before.json", _paired_baseline())
    history = run_dir / "metrics" / "topology_validation.jsonl"
    history.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for epoch in range(0, 101, 10):
        rows.append(
            {
                "step": epoch * 38,
                "metrics": {
                    **_child_metrics(),
                    "evaluation_completed_utc": (
                        datetime(2026, 1, 1, tzinfo=timezone.utc)
                        + timedelta(seconds=epoch * 60)
                    ).isoformat().replace("+00:00", "Z"),
                    "command_elapsed_seconds": float(epoch * 60),
                },
            }
        )
    # The forced final save repeats epoch 100. The first native row remains primary.
    rows.append(
        {
            "step": 100 * 38,
            "metrics": {
                **_child_metrics(child_factor=duplicate_factor),
                "evaluation_completed_utc": "2026-01-01T01:41:00Z",
                "command_elapsed_seconds": 6060.0,
            },
        }
    )
    history.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    ledger = root / "manifests" / "campaign_runner_ledger.jsonl"
    ledger.parent.mkdir(parents=True, exist_ok=True)
    entries = [
        {
            "attempt_id": "profile-g",
            "task_id": "G",
            "action": "profile",
            "status": "running",
            "started_utc": "2025-12-31T23:57:00Z",
            "command": ["python", "cost_aware_profile.py", "--task", "G"],
        },
        {
            "attempt_id": "profile-g",
            "task_id": "G",
            "action": "profile",
            "status": "completed",
            "finished_utc": "2025-12-31T23:59:00Z",
            "charged_process_seconds": 120.0,
        },
        {
            "attempt_id": "launch-g",
            "task_id": "G",
            "arm": "dense",
            "action": "launch",
            "status": "running",
            "started_utc": START,
            "command": ["python", "cost_aware_campaign.py", "launch", "G", "dense"],
            "actual_science_epoch_observation": {"run_dir": str(run_dir), "accepted_end_epoch": 100},
        },
        {
            "attempt_id": "launch-g",
            "task_id": "G",
            "arm": "dense",
            "action": "launch",
            "status": "completed",
            "finished_utc": FINISH,
            "charged_process_seconds": 7000.0,
            "charged_science_epochs": 100.0,
            "charged_aux_epochs": 0.0,
            "actual_science_epoch_observation": {
                "run_dir": str(run_dir),
                "accepted_end_epoch": 100,
                "training_complete": True,
            },
        },
    ]
    ledger.write_text("".join(json.dumps(row) + "\n" for row in entries), encoding="utf-8")
    return root


def test_native_history_pairs_fixed_groups_pools_counts_and_assesses_windows(tmp_path, monkeypatch):
    root = _fixture_root(tmp_path)

    original_open = Path.open

    def guarded_open(path, *args, **kwargs):
        if path.suffix.lower() in {".h5", ".hdf5"}:
            raise AssertionError("analyzer attempted to open a dataset file")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    report, rows = ANALYZER.build_campaign_quality_history(root)
    run = report["tasks"]["G"]["arms"]["dense"]["runs"][0]
    assert run["native_history"]["baseline"]["status"] == "baseline_only"
    assert run["accepted_end_epoch"] == 100
    assert run["quality_assessment_status"] == "assessed"
    assessment = run["quality_assessment"]
    assert assessment["confirmation"]["status"] == "confirmed"
    assert assessment["confirmation"]["confirmation_epoch"] == 70
    assert assessment["retention"]["status"] == "retained"

    checks = run["uniform_evaluations"]
    assert [check["epoch"] for check in checks] == list(range(10, 101, 10))
    assert checks[-1]["duplicate_disagreement"] is False
    assert checks[-1]["duplicate_native_rows"][0]["metric_match_within_1e-6_relative_1e-12_absolute"] is True
    check60 = next(check for check in checks if check["epoch"] == 60)
    raw = check60["raw"]
    assert raw["coherence_sum"]["count"] == 128
    assert raw["coherence_sum"]["numerator"] == pytest.approx(0.9 * 128)
    assert len(raw["b_groups"]) == 4
    assert [group["snapshot_count"] for group in raw["b_groups"]] == [32] * 4
    assert all(group["measurement"]["count"] == 1 for group in raw["b_groups"])
    assert raw["endpoint_fields"]["u"]["count"] == 128 * 4096
    assert raw["endpoint_aggregate"]["count"] == 2 * 128 * 4096
    assert check60["active_process_seconds_from_first_training_command"] == 3600
    assert check60["task_profile_acquisition_seconds"] == 120
    assert check60["task_profile_acquisition_seconds_measured"] == 120
    assert check60["task_profile_acquisition_seconds_estimated"] == 0
    assert check60["allocated_profile_acquisition_seconds"] == 0
    assert check60["cold_process_seconds_including_profile_acquisition"] == 3600
    assert check60["wall_seconds_from_first_training_command"] == 3600

    assert run["training_attempt_segments"][0]["command"] == [
        "python",
        "cost_aware_campaign.py",
        "launch",
        "G",
        "dense",
    ]
    assert run["training_attempt_segments"][0]["charged_process_seconds"] == 7000
    components = ANALYZER._component_rows_from_report(report)
    assert any(
        row["metric_path"].endswith("topology.self.persistence")
        and row["source_value"] == pytest.approx(1.0)
        and row["child_value"] == pytest.approx(0.9)
        for row in components
    )
    assert len(rows) >= len(checks)


@pytest.mark.parametrize(
    ("duplicate_factor", "append_malformed_row", "expected_integrity"),
    [
        (0.95, False, "duplicate_metric_disagreement_first_row_retained"),
        (0.9, True, "invalid_history_rows"),
    ],
)
def test_native_integrity_issues_preserve_rows_but_suppress_assessment(
    tmp_path, duplicate_factor, append_malformed_row, expected_integrity
):
    root = _fixture_root(tmp_path, duplicate_factor=duplicate_factor)
    run_dir = root / "runs" / "G" / "dense" / "child-001"
    history = run_dir / "metrics" / "topology_validation.jsonl"
    if append_malformed_row:
        with history.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"step": 3801, "metrics": {}}) + "\n")

    report, _ = ANALYZER.build_campaign_quality_history(root)
    run = report["tasks"]["G"]["arms"]["dense"]["runs"][0]

    assert run["native_history"]["integrity_status"] == expected_integrity
    assert len(run["uniform_evaluations"]) == 10
    assert run["uniform_evaluations"][-1]["raw"]["coherence_sum"]["count"] == 128
    assert run["quality_assessment"] is None
    assert run["quality_assessment_status"] == (
        f"ambiguous_native_history_integrity:{expected_integrity}"
    )
    if append_malformed_row:
        assert run["native_history"]["invalid_rows"][0]["status"] == "rejected"
    else:
        assert run["uniform_evaluations"][-1]["duplicate_disagreement"] is True
        assert run["uniform_evaluations"][-1]["duplicate_native_rows"]


def test_native_history_requires_exact_four_groups_and_paired_queries():
    baseline = _paired_baseline()
    metrics = _child_metrics()
    malformed = json.loads(json.dumps(metrics))
    malformed["fixed_estimator_groups"].pop()
    with pytest.raises(ANALYZER.CampaignAnalysisError, match="exact selected sample/query panel"):
        ANALYZER._native_metric_identity(malformed, fields=FIELDS, expected=baseline)

    changed_query = json.loads(json.dumps(metrics))
    changed_query["fixed_estimator_groups"][2]["query_ids"][0][0] = 1
    with pytest.raises(ANALYZER.CampaignAnalysisError, match="SOURCE pairing"):
        ANALYZER._native_metric_identity(changed_query, fields=FIELDS, expected=baseline)


def test_pending_arms_are_preserved_without_promoting_portable_panels(tmp_path):
    root = _fixture_root(tmp_path)
    report, _ = ANALYZER.build_campaign_quality_history(root)
    assert report["tasks"]["G"]["arms"]["heuristic"]["status"] == "pending_run"


def test_disjoint_portable_metrics_remain_supplementary_and_keep_four_groups(tmp_path):
    root = _fixture_root(tmp_path)
    run_dir = root / "runs" / "G" / "dense" / "child-001"
    artifact_path = root / "analysis" / "G" / "dense" / "child-001" / "portable" / "disjoint.json"
    _write_json(artifact_path, _disjoint_portable_evaluation(run_dir))

    report, _ = ANALYZER.build_campaign_quality_history(root)
    run = report["tasks"]["G"]["arms"]["dense"]["runs"][0]

    assert len(run["uniform_evaluations"]) == 10
    assert {row["panel_kind"] for row in run["uniform_evaluations"]} == {"uniform"}
    assert [row["epoch"] for row in run["uniform_evaluations"]] == list(range(10, 101, 10))
    assert run["quality_assessment"]["confirmation"]["confirmation_epoch"] == 70

    assert len(run["supplementary_evaluations"]) == 1
    supplementary = run["supplementary_evaluations"][0]
    assert supplementary["panel_kind"] == "disjoint"
    assert supplementary["status"] == "supplementary_disjoint_panel"
    assert supplementary["epoch"] == 100
    assert supplementary["raw"]["coherence_sum"]["count"] == 128
    assert len(supplementary["raw"]["b_groups"]) == 4
    assert [group["snapshot_count"] for group in supplementary["raw"]["b_groups"]] == [32] * 4
    assert supplementary["raw"]["endpoint_fields"]["u"]["count"] == 128 * 4096
    assert supplementary["raw"]["endpoint_aggregate"]["count"] == 2 * 128 * 4096
    assert supplementary["ratios"]["coherence_sum"] == pytest.approx(0.7)
    assert set(supplementary["ratios"]["endpoint_fields"]) == set(FIELDS)
    assert run["uniform_evaluations"][-1]["ratios"]["coherence_sum"] == pytest.approx(0.9)


def test_live_launch_binds_pathless_ledger_to_its_new_run_manifest(tmp_path):
    root = _fixture_root(tmp_path)
    run_dir = root / "runs" / "G" / "dense" / "child-001"
    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["created_utc"] = "20260101T000004Z"
    manifest["experiment_name"] = f"_CostAwareLPC/{root.name}/G/dense"
    _write_json(manifest_path, manifest)

    ledger = root / "manifests" / "campaign_runner_ledger.jsonl"
    entries = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines()]
    profile_entries = [entry for entry in entries if entry["action"] == "profile"]
    active_launch = {
        "attempt_id": "launch-active",
        "task_id": "G",
        "arm": "dense",
        "action": "launch",
        "status": "running",
        "started_utc": START,
        "command": [
            "python",
            "cost_aware_campaign.py",
            "--root",
            str(root),
            "_child",
            "--",
            "_train",
            "--config",
            str(root / "configs" / "G_dense.yaml"),
        ],
    }
    ledger.write_text(
        "".join(json.dumps(entry) + "\n" for entry in [*profile_entries, active_launch]),
        encoding="utf-8",
    )

    report, _ = ANALYZER.build_campaign_quality_history(root)
    run = report["tasks"]["G"]["arms"]["dense"]["runs"][0]
    check = run["uniform_evaluations"][0]
    assert run["training_attempt_segments"][0]["attempt_id"] == "launch-active"
    assert check["status"] == "native_primary"
    assert check["active_process_seconds_from_first_training_command"] == 600
    assert check["task_profile_acquisition_seconds"] == 120
    assert check["allocated_profile_acquisition_seconds"] == 0
    assert check["cold_process_seconds_including_profile_acquisition"] == 600


def test_component_long_csv_pairs_source_and_child_values(tmp_path):
    root = _fixture_root(tmp_path)
    report, _ = ANALYZER.build_campaign_quality_history(root)
    rows = ANALYZER._component_rows_from_report(report)
    row = next(row for row in rows if row["metric_path"] == "coherence.total")
    assert set(row) == {
        "task",
        "arm",
        "run_id",
        "epoch",
        "panel",
        "group",
        "metric_path",
        "source_value",
        "child_value",
    }
    assert row["source_value"] == pytest.approx(1.0)
    assert row["child_value"] == pytest.approx(0.9)


def test_profile_acquisition_is_allocated_only_to_strategy_arm():
    assert ANALYZER._allocated_profile_seconds("dense", 120.0) == 0
    assert ANALYZER._allocated_profile_seconds("heuristic", 120.0) == 0
    assert ANALYZER._allocated_profile_seconds("strategy", 120.0) == 120
    assert ANALYZER._allocated_profile_seconds("strategy", None) is None


def test_pre_admission_profile_bound_is_separate_from_measured_time(tmp_path):
    root = tmp_path / "campaign"
    attempts_path = root / "analysis" / "readiness_attempts.jsonl"
    attempts_path.parent.mkdir(parents=True)
    attempts_path.write_text(
        json.dumps(
            {
                "phase": "D profile preflight",
                "outcome": "rejected before GPU admission",
                "gpu_process_seconds": 0,
                "optimizer_exposures": 0,
                "conservative_command_seconds": 1,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    estimate, records = ANALYZER._profile_readiness_estimate(root, "D")
    measured = 322.05613095900117
    assert estimate == 1
    assert records[0]["conservative_command_seconds"] == 1
    assert records[0]["basis"] == (
        "recorded conservative pre-admission command estimate; zero GPU time and optimizer exposures; not a proven timing bound"
    )
    assert ANALYZER._allocated_profile_seconds("strategy", measured + estimate) == pytest.approx(
        323.05613095900117
    )
    assert ANALYZER._allocated_profile_seconds("dense", measured + estimate) == 0

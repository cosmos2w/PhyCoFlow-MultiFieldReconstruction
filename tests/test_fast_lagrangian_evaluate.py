from __future__ import annotations

import hashlib
import json

import pytest
import torch
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


def test_cuda_evaluation_requires_one_authorized_physical_gpu_and_campaign_token(tmp_path, monkeypatch):
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
    monkeypatch.setattr(EVALUATOR, "_validate_device_admission", lambda *_args: None)
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

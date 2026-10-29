from __future__ import annotations

import json
from functools import partial
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from scripts.posttraining import evaluate_full_abt as PORTABLE


def _supported_config():
    return (
        {
            "stage": "post_training",
            "source": {"kind": "native_run"},
            "dataset": {"include_temporal_derivative": False},
            "model": {"model_ema_eval": False},
            "rollout": {"steps": 2, "solver": "euler"},
            "optimization": {"multitask": {"external_policy": {"method": "lagrangian"}}},
            "evaluation": {"split": "validation", "generation_steps": 2},
            "coherence": {
                "compute_budget": {"point_count": 16, "query_policy": "fixed_shared"},
                "families": {name: {} for name in PORTABLE.EXPECTED_FAMILIES},
            },
            "output": {"experiment_name": "fresh_eval"},
        },
        {"stage": "post_training", "coherence_families": list(PORTABLE.EXPECTED_FAMILIES)},
    )


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("evaluation", "split"), "test", "TEST"),
        (("dataset", "include_temporal_derivative"), True, "temporal_derivative"),
        (("model", "model_ema_eval"), True, "raw live weights"),
        (("rollout", "steps"), 4, "two Euler"),
        (("evaluation", "generation_steps"), 4, "two generation steps"),
        (("evaluation", "generation_steps"), None, "two generation steps"),
        (("output", "experiment_name"), "_LPCAccel/study/G1-C", "historical"),
    ],
)
def test_supported_contract_rejects_unapproved_inference_changes(path, value, message):
    config, manifest = _supported_config()
    config[path[0]][path[1]] = value
    with pytest.raises(PORTABLE.PortableEvaluationError, match=message):
        PORTABLE._validate_supported_config(config, manifest)


def test_native_deeponet_keeps_its_no_ema_model_contract():
    config, manifest = _supported_config()
    config["model"] = {"name": "deeponet", "width": 256, "basis_dim": 128}
    PORTABLE._validate_supported_config(config, manifest)
    config["model"] = {"name": "gl_rbf_cq"}
    with pytest.raises(PORTABLE.PortableEvaluationError, match="raw live weights"):
        PORTABLE._validate_supported_config(config, manifest)


@pytest.mark.parametrize(
    ("update", "message"),
    [
        ({"query_policy": "random_per_sample"}, "fixed_shared"),
        ({"point_count": 0}, "positive integer"),
    ],
)
def test_supported_contract_requires_fixed_shared_query_manifest(update, message):
    config, manifest = _supported_config()
    config["coherence"]["compute_budget"].update(update)
    with pytest.raises(PORTABLE.PortableEvaluationError, match=message):
        PORTABLE._validate_supported_config(config, manifest)


def test_uniform_validation_panel_remains_the_existing_frozen_selection():
    shared = PORTABLE._helpers()
    expected = shared._uniform_validation_indices(512, 128)
    selected, excluded = PORTABLE._select_validation_panel(shared, 512, 128, "uniform")
    assert selected == expected
    assert excluded == []
    assert PORTABLE.PANEL_ALGORITHMS["uniform"] == "chronological_linspace_v1"


def test_disjoint_validation_panel_is_chronological_and_excludes_uniform_panel():
    shared = PORTABLE._helpers()
    uniform = shared._uniform_validation_indices(512, 128)
    selected, excluded = PORTABLE._select_validation_panel(shared, 512, 128, "disjoint")
    candidates = [index for index in range(512) if index not in set(uniform)]
    expected_positions = np.linspace(0, len(candidates) - 1, 128).round().astype(np.int64)
    assert selected == [candidates[index] for index in expected_positions]
    assert excluded == uniform
    assert len(selected) == 128
    assert selected == sorted(selected)
    assert set(selected).isdisjoint(excluded)
    assert len(PORTABLE._helpers()._group_slices(len(selected))) == 4
    assert PORTABLE.PANEL_ALGORITHMS["disjoint"] == (
        "chronological_linspace_after_uniform_exclusion_v1"
    )


def test_disjoint_panel_requires_a_full_second_panel_and_valid_panel_name():
    shared = PORTABLE._helpers()
    with pytest.raises(PORTABLE.PortableEvaluationError, match="2 \\* max_samples"):
        PORTABLE._select_validation_panel(shared, 127, 64, "disjoint")
    with pytest.raises(PORTABLE.PortableEvaluationError, match="panel must be"):
        PORTABLE._select_validation_panel(shared, 256, 64, "test")


def test_panel_filename_is_distinct_while_uniform_filename_stays_compatible():
    from datetime import datetime, timezone

    evaluated_at = datetime(2026, 10, 23, tzinfo=timezone.utc)
    uniform = PORTABLE._evaluation_filename("best", 128, "a" * 64, evaluated_at, "uniform")
    disjoint = PORTABLE._evaluation_filename("best", 128, "a" * 64, evaluated_at, "disjoint")
    assert uniform == "full_abt_best_n128_aaaaaaaaaaaa_20261023T000000000000Z_validation.json"
    assert disjoint == "full_abt_best_n128_aaaaaaaaaaaa_20261023T000000000000Z_disjoint_validation.json"
    assert uniform != disjoint


def test_panel_cli_defaults_to_uniform_and_accepts_disjoint():
    parser = PORTABLE._parser()
    args = parser.parse_args(["--run", "/tmp/run"])
    assert args.panel == "uniform"
    args = parser.parse_args(["--run", "/tmp/run", "--panel", "disjoint"])
    assert args.panel == "disjoint"


def test_run_and_output_locations_are_confined_to_data_root(tmp_path):
    run = tmp_path / "runs" / "fresh" / "run-1"
    run.mkdir(parents=True)
    assert PORTABLE._validate_run_location(run, tmp_path) == run.resolve()
    with pytest.raises(PORTABLE.PortableEvaluationError, match="historical _LPCAccel"):
        PORTABLE._validate_run_location(tmp_path / "_LPCAccel" / "run", tmp_path)
    historical_campaign_run = (
        tmp_path / "experiments" / "LPCAccel_historical" / "runs" / "G1-C" / "run-1"
    )
    with pytest.raises(PORTABLE.PortableEvaluationError, match="historical _LPCAccel"):
        PORTABLE._validate_run_location(historical_campaign_run, tmp_path)

    output = PORTABLE._output_directory(
        run,
        None,
        data_root=tmp_path,
        require_mount=False,
    )
    assert output == (run / "evaluation").resolve()
    outside = tmp_path.parent / "outside-portable-evaluation"
    with pytest.raises(PORTABLE.PortableEvaluationError, match="below the mounted data root"):
        PORTABLE._output_directory(run, outside, data_root=tmp_path, require_mount=False)


def test_campaign_source_staging_environment_is_refused_before_helper_import(monkeypatch, tmp_path):
    monkeypatch.setenv("PHYCOFLOW_CAMPAIGN_SOURCE_STAGE", "goal1")
    with pytest.raises(PORTABLE.PortableEvaluationError, match="staging/admission"):
        PORTABLE.evaluate_run(tmp_path / "missing", _data_root=tmp_path, _require_mount=False)


def test_cli_keeps_the_production_mount_requirement(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(PORTABLE, "DATA_ROOT", tmp_path)
    assert PORTABLE.main(["--run", str(tmp_path / "missing")]) == 2
    assert "mounted /data2 filesystem" in capsys.readouterr().err


def test_child_lineage_verifier_rejects_checkpoint_binding_mismatches(tmp_path):
    from types import SimpleNamespace

    from phycoflow_reconstruction.training.run_store import file_sha256
    from phycoflow_reconstruction.training.source import source_checkpoint_path

    source_run = tmp_path / "source"
    source_checkpoint = source_run / "checkpoints" / "last.pt"
    source_checkpoint.parent.mkdir(parents=True)
    source_checkpoint.write_bytes(b"source checkpoint")
    config = {
        "source_run": str(source_run),
        "source_checkpoint": "last.pt",
        "optimization": {},
    }
    source_hashes = {"checkpoint": file_sha256(source_checkpoint)}
    run_dir = tmp_path / "child"
    artifact_dir = run_dir / "artifacts"
    artifact_dir.mkdir(parents=True)
    calibration = artifact_dir / "coherence_calibration.json"
    calibration.write_text('{"resolved_scales":{"A":1.0}}', encoding="utf-8")
    calibration_hash = file_sha256(calibration)
    checkpoint_path = run_dir / "checkpoints" / "last.pt"
    checkpoint_path.parent.mkdir()
    checkpoint_path.write_bytes(b"child checkpoint")
    config_hash = "c" * 64
    names = list(PORTABLE.EXPECTED_FAMILIES)
    normalizer = SimpleNamespace(
        offset=torch.zeros(2), scale=torch.ones(2), method="mean_std"
    )
    dataset = SimpleNamespace(field_names=("u", "v"), normalizer=normalizer)
    families = {name: object() for name in names}
    family_scales = {name: 1.0 for name in names}
    checkpoint = {
        "config_sha256": config_hash,
        "source_hashes": source_hashes,
        "source_run": str(source_run),
        "source_checkpoint": str(source_checkpoint_path(config)),
        "data_spec": {"field_names": ["u", "v"]},
        "normalization": {
            "offset": torch.zeros(2), "scale": torch.ones(2), "method": "mean_std"
        },
        "coherence_calibration_sha256": calibration_hash,
        "family_scales": family_scales,
        "family_states": {name: {} for name in names},
    }
    manifest = {
        "config_sha256": config_hash,
        "coherence_calibration_sha256": calibration_hash,
    }
    common = {
        "config": config,
        "manifest": manifest,
        "checkpoint": checkpoint,
        "checkpoint_path": checkpoint_path,
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "source_hashes": source_hashes,
        "train_dataset": dataset,
        "families": families,
        "family_scales": family_scales,
        "run_dir": run_dir,
        "shared": object(),
    }
    PORTABLE._validate_child_checkpoint(**common)

    invalid_cases = [
        ({"source_hashes": {"checkpoint": "0" * 64}}, "SOURCE hashes"),
        ({"data_spec": {"field_names": ["v", "u"]}}, "field order"),
        (
            {
                "normalization": {
                    "offset": torch.ones(2), "scale": torch.ones(2), "method": "mean_std"
                }
            },
            "normalization",
        ),
        ({"family_scales": {name: 2.0 for name in names}}, "family scales"),
    ]
    for updates, message in invalid_cases:
        candidate = dict(checkpoint)
        candidate.update(updates)
        with pytest.raises(PORTABLE.PortableEvaluationError, match=message):
            PORTABLE._validate_child_checkpoint(**{**common, "checkpoint": candidate})


def _write_full_validation_fixture(path):
    y, x = np.meshgrid(np.arange(8), np.arange(8), indexing="ij")
    coordinates = np.stack((x, y, np.zeros_like(x)), axis=-1).reshape(64, 1, 1, 3)
    fields = np.empty((34, 1, 64, 1, 1, 2), dtype=np.float32)
    for sample in range(fields.shape[0]):
        fields[sample, 0, :, 0, 0, 0] = x.reshape(-1) / 7.0 + sample * 0.01
        fields[sample, 0, :, 0, 0, 1] = y.reshape(-1) / 7.0 + sample * 0.02
    train_values = fields[:2, 0].reshape(-1, 2)
    mean = train_values.mean(axis=0)
    std = train_values.std(axis=0)
    std[std == 0] = 1.0
    with h5py.File(path, "w") as handle:
        handle.create_dataset("fields", data=fields)
        handle.create_dataset("coordinates", data=coordinates.astype(np.float32))
        handle.create_dataset("time", data=np.asarray([0.0], dtype=np.float32))
        handle.create_dataset("conditions", data=np.empty((34, 0), dtype=np.float32))
        handle.create_dataset(
            "trajectory_id",
            data=np.asarray([f"sample-{i:03d}" for i in range(34)], dtype=h5py.string_dtype()),
        )
        splits = handle.create_group("splits")
        splits.create_dataset("train", data=np.asarray([0, 1], dtype=np.int64))
        splits.create_dataset("validation", data=np.arange(2, 34, dtype=np.int64))
        splits.create_dataset("test", data=np.asarray([1], dtype=np.int64))
        statistics = handle.create_group("statistics")
        statistics.create_dataset("train_mean", data=mean)
        statistics.create_dataset("train_std", data=std)
        handle.attrs["field_names"] = '["u", "v"]'
        handle.attrs["field_units"] = '["1", "1"]'
        handle.attrs["grid_shape"] = "[8, 8]"
        handle.attrs["schema_version"] = "1.0"


def _small_gl_model(*, ema_eval):
    return {
        "name": "gl_rbf_cq",
        "backbone": "GL_rbf_ENH_CQ",
        "prior": "iid",
        "hidden_dim": 16,
        "cond_dim": 8,
        "field_embed_dim": 4,
        "latent_dim": 16,
        "num_latents": 4,
        "num_heads": 2,
        "num_latent_blocks": 1,
        "ff_mult": 2,
        "rff_features": 16,
        "fourier_pe_num_bands": 4,
        "gather_mode": "topk_rbf_glres",
        "gather_topk": 4,
        "neighbor_backend": "torch",
        "sensor_local_topk": 4,
        "cq_query_dim": 8,
        "cq_readout_rank": 4,
        "cq_readout_heads": 2,
        "cq_time_embed_dim": 8,
        "model_ema_enabled": True,
        "model_ema_decay": 0.9,
        "model_ema_eval": ema_eval,
        "query_points": 16,
    }


def _small_full_families():
    from helpers.multifamily import _cross_config, _global_config

    cross = _cross_config()
    cross["components"]["cross_frequency"]["enabled"] = False
    topology = {
        "enabled": True,
        "strategy": "cubical_persistence",
        "target_use": "paired_supervised",
        "units": "model_units",
        "fields": ["u", "v"],
        "geometry": {"grid_shape": [4, 4], "neighbors": 1, "periodic": False},
        "filtration": {
            "dimensions": [0, 1],
            "directions": ["sublevel", "superlevel"],
            "smoothing_sigma": 0.0,
        },
        "persistence": {"distance": "sliced_wasserstein", "projections": 4},
        "components": {
            "self": {"enabled": True, "weight": 1.0},
            "mutual": {"enabled": True, "weight": 1.0, "groups": [["u", "v"]], "lines": 1},
        },
    }
    return {
        "global_distribution": _global_config(),
        "cross_spectrum": cross,
        "topology": topology,
    }


def test_portable_cli_full_abt_end_to_end_cpu_on_synthetic_run(
    tmp_path, capsys, monkeypatch
):
    pytest.importorskip("gudhi")
    pytest.importorskip("cooper")
    from helpers.coherence import _base_config, _post_config

    # Exercise the real compact HDF5 source and collator without forking a
    # persistent DataLoader worker from a torch-initialized pytest process.
    from phycoflow_reconstruction.data import training_batches

    real_data_loader = torch.utils.data.DataLoader

    def synchronous_data_loader(*args, **kwargs):
        kwargs["num_workers"] = 0
        kwargs["persistent_workers"] = False
        kwargs.pop("prefetch_factor", None)
        return real_data_loader(*args, **kwargs)

    monkeypatch.setattr(training_batches, "DataLoader", synchronous_data_loader)

    from phycoflow_reconstruction.training.base_training import run_base_training
    from phycoflow_reconstruction.training.post_training import run_post_training

    dataset_path = tmp_path / "full-panel.h5"
    _write_full_validation_fixture(dataset_path)
    case_dir = tmp_path / "experiment"
    base = _base_config(dataset_path)
    base["dataset"]["grid_shape"] = [8, 8]
    base["dataset"]["allow_nonbenchmark"] = True
    base["model"] = _small_gl_model(ema_eval=True)
    base["optimization"].update(epochs=1, batch_size=2)
    base["evaluation"].update(generation_steps=4, preview={"enabled": False})
    source = run_base_training(base, case_dir=case_dir)

    post = _post_config(dataset_path, source)
    post["dataset"].update(grid_shape=[8, 8], allow_nonbenchmark=True)
    post["model"] = _small_gl_model(ema_eval=False)
    post["optimization"].update(
        epochs=1,
        steps_per_epoch=1,
        sampling="full_pass",
        batch_size=2,
        multitask={
            "family_scalarization": {"method": "fixed_sum"},
            "outer_aggregation": None,
            "external_policy": {"method": "lagrangian", "vector_reduction": "sum"},
            "fidelity_budget": {"relative_increase": 0.05, "calibration_batches": 2, "calibration_draws": 1},
        },
    )
    post["objectives"]["data_retention"] = {"enabled": False, "weight": 0.0}
    post["coherence"].update(
        compute_budget={"batch_size": 2, "point_count": 16, "query_policy": "fixed_shared"},
        families=_small_full_families(),
        family_balance={"mode": "none"},
    )
    post["rollout"] = {"steps": 2, "solver": "euler"}
    post["evaluation"].update(
        split="validation", max_samples=2, query_points=16, generation_steps=2,
        preview={"enabled": False},
    )
    post["checkpointing"] = {"every_epochs": 1, "selection_metric": "reconstruction_mse"}
    post["runtime"].update(device="cpu", progress=False, plot_every_steps=1000)
    child = run_post_training(post, case_dir=case_dir, max_steps=1)

    # Keep the real CLI and numerical evaluator; only replace the production
    # storage policy with the existing private seam for this temporary fixture.
    monkeypatch.setattr(
        PORTABLE,
        "evaluate_run",
        partial(PORTABLE.evaluate_run, _data_root=tmp_path, _require_mount=False),
    )
    assert PORTABLE.main(
        ["--run", str(child), "--checkpoint", "last", "--max-samples", "32"]
    ) == 0
    result_path = Path(capsys.readouterr().out.strip().splitlines()[-1])
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["schema"] == "phycoflow_full_abt_validation.v1"
    assert result["checkpoint_global_step"] == 1
    assert result["dataset"]["field_names"] == ["u", "v"]
    assert result["dataset"]["include_temporal_derivative"] is False
    assert result["dataset"]["logical_shape"] == [8, 8]
    assert result["inference_contract"] == {
        "model_ema_eval": False,
        "weight_identity": "raw_live_model_weights",
        "rollout_steps": 2,
        "rollout_solver": "euler",
        "include_temporal_derivative": False,
    }
    assert result["selection"]["split"] == "validation"
    assert result["selection"]["group_size"] == 32
    assert result["selection"]["group_count"] == 1
    assert result["selection"]["query_policy"] == "fixed_shared"
    assert result["selection"]["comparison_query_count"] == 16
    assert result["selection"]["query_indices_shape"][-1] == 16
    assert len(set(result["selection"]["sample_ids"])) == 32
    assert result["evaluation"]["mode"] == "full_existing_A_B_T_family_forward"
    assert result["evaluation"]["topology_scalar_filtrations"] == 6
    assert set(result["components"]["source"]) == set(PORTABLE.EXPECTED_FAMILIES)
    assert set(result["components"]["child"]) == set(PORTABLE.EXPECTED_FAMILIES)
    assert len(result["groups"]) == 1
    assert result["groups"][0]["source"]["coherence"]["families"]["topology"][
        "topology_scalar_filtrations"
    ] == 6
    assert result["groups"][0]["child"]["coherence"]["families"]["topology"][
        "topology_scalar_filtrations"
    ] == 6
    assert set(result["evaluation"]["full_metric_ratios_child_over_source"]["coherence"]["families"]) == set(
        PORTABLE.EXPECTED_FAMILIES
    )

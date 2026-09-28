"""Small canonical datasets and training configurations."""

from pathlib import Path

import h5py
import numpy as np


def _write_fixture(path: Path) -> None:
    generator = np.random.default_rng(7)
    with h5py.File(path, "w") as handle:
        handle.create_dataset(
            "fields", data=generator.normal(size=(3, 2, 16, 1, 1, 2)).astype("float32")
        )
        y, x = np.meshgrid(np.arange(4), np.arange(4), indexing="ij")
        coordinates = np.stack((x, y, np.zeros_like(x)), axis=-1).reshape(16, 1, 1, 3)
        handle.create_dataset("coordinates", data=coordinates.astype("float32"))
        handle.create_dataset("time", data=np.asarray([0.0, 1.0]))
        handle.create_dataset("conditions", data=np.empty((3, 0), dtype="float32"))
        handle.create_dataset(
            "trajectory_id",
            data=np.asarray(["train", "validation", "test"], dtype=h5py.string_dtype()),
        )
        splits = handle.create_group("splits")
        splits.create_dataset("train", data=np.asarray([0]))
        splits.create_dataset("validation", data=np.asarray([1]))
        splits.create_dataset("test", data=np.asarray([2]))
        statistics = handle.create_group("statistics")
        statistics.create_dataset("train_mean", data=np.zeros(2))
        statistics.create_dataset("train_std", data=np.ones(2))
        handle.attrs["field_names"] = '["u", "v"]'
        handle.attrs["field_units"] = '["1", "1"]'
        handle.attrs["grid_shape"] = "[4, 4]"
        handle.attrs["schema_version"] = "1.0"


def _family_config(target_use: str = "training_reference") -> dict:
    return {
        "enabled": True,
        "target_use": target_use,
        "units": "model_units",
        "fields": ["u", "v"],
        "reference_bank": {
            "enabled": target_use == "training_reference",
            "max_samples": 2,
            "points_per_sample": 8,
            "seed": 13,
        },
        "components": {
            "self": {"enabled": True, "weight": 1.0},
            "mutual": {
                "enabled": True,
                "weight": 1.0,
                "pairs": [["u", "v"]],
                "directions": 4,
                "seed": 3,
            },
            "cross": {
                "enabled": True,
                "weight": 1.0,
                "directions": 6,
                "top_fraction": 0.5,
                "seed": 5,
                "include_axes": True,
                "qmc": True,
            },
        },
    }


def _base_config(dataset_path: Path) -> dict:
    return {
        "stage": "base_training",
        "case": "fixture",
        "dataset": {
            "path": str(dataset_path),
            "split": "train",
            "field_names": ["u", "v"],
            "field_units": ["1", "1"],
            "normalization": "mean_std",
        },
        "model": {
            "name": "pointcloud_ffm",
            "backbone": "gl_rbf_enh",
            "gather_mode": "topk_rbf",
            "prior": "iid",
            "hidden_dim": 16,
            "latent_dim": 16,
            "num_latents": 4,
            "heads": 2,
            "latent_blocks": 1,
            "gather_topk": 2,
            "query_chunk_size": 16,
            "query_points": 8,
        },
        "observations": {"protocol": "random_uniform", "seed": 4, "fields": {"u": {"count": 4}}},
        "optimization": {"epochs": 1, "batch_size": 2, "lr": 1e-3, "grad_clip": 1.0},
        "runtime": {"seed": 9, "device": "cpu", "deterministic": True, "num_workers": 0},
        "evaluation": {"generation_steps": 1},
        "output": {"experiment_name": "source"},
    }


def _post_config(dataset_path: Path, source_run: Path) -> dict:
    config = _base_config(dataset_path)
    config.update(
        stage="post_training",
        source_run=str(source_run),
        source_checkpoint="last.pt",
        inherit_base_config=True,
        source={
            "kind": "native_run",
            "allow_integration_source": False,
            "inherited_base_keys": ["dataset", "model", "observations"],
            "config_origins": {"post_training": "test"},
        },
        objectives={
            "data_retention": {"enabled": True, "weight": 0.1},
            "coherence": {"enabled": True, "weight": 1.0},
        },
        coherence={
            "schedule": {
                "start_epoch": 1,
                "every_n_steps": 1,
                "weight_warmup_epochs": 0,
                "interval_rescale": False,
            },
            "compute_budget": {"batch_size": 1, "point_count": 8},
            "families": {"global_distribution": _family_config()},
        },
        rollout={"steps": 1, "solver": "euler"},
        observation_consistency={
            "mode": "endpoint_smooth",
            "strength": 1.0,
            "sigma": 0.2,
            "schedule_power": 2.0,
            "final_clamp": True,
        },
        trainable={"scope": "full_model"},
        evaluation={
            "split": "validation",
            "max_samples": 1,
            "query_points": 8,
            "generation_steps": 1,
            "seed": 77,
        },
        output={"experiment_name": "child"},
    )
    config["optimization"].update(
        train_fraction=1.0,
        weight_decay=0.0,
        gradient_balance="weighted_sum",
        config_missing_behavior="error",
    )
    return config

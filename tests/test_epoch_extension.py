"""RunStore epoch-horizon migration preserves the saved training state."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest
import torch
import yaml

from phycoflow_reconstruction.training import run_store as run_store_module
from phycoflow_reconstruction.training.run_store import (
    RunStore,
    config_digest,
    load_project_checkpoint,
)


def _config(epochs: int = 2) -> dict[str, Any]:
    return {
        "stage": "post_training",
        "case": "epoch-extension-fixture",
        "model": {"name": "synthetic-fixture"},
        "optimization": {"epochs": epochs, "batch_size": 2, "lr": 0.01},
        "output": {},
    }


def _checkpoint(step: int = 17) -> dict[str, Any]:
    """Build tensor-backed state without constructing or training a model."""
    return {
        "epoch": 2,
        "global_step": step,
        "model": {
            "weight": torch.tensor([1.25, -0.5]),
            "bias": torch.tensor(0.125),
        },
        "optimizer": {
            "state": {
                0: {
                    "step": torch.tensor(float(step)),
                    "exp_avg": torch.tensor([0.1, 0.2]),
                    "exp_avg_sq": torch.tensor([0.01, 0.04]),
                }
            },
            "param_groups": [{"lr": 0.01, "step": step}],
        },
        "component_controller": {
            "attempted": step,
            "accepted": step - 2,
            "rejected": 2,
            "dual": torch.tensor([0.3, 0.7]),
        },
        "model_ema": {"weight": torch.tensor([1.2, -0.45]), "num_updates": step},
        "rng_state": {"torch": torch.arange(16, dtype=torch.uint8)},
    }


def _new_run(tmp_path: Path) -> tuple[RunStore, dict[str, Any]]:
    config = _config()
    store = RunStore.create(tmp_path, "extension", config)
    store.save_checkpoint("last", _checkpoint())
    store.save_checkpoint("best", _checkpoint(step=12))
    store.append_history({"epoch": 2, "step": 17, "loss": 0.25})
    store.set_status("interrupted", epoch=2, global_step=17)
    return store, config


def _tree_snapshot(root: Path) -> dict[str, tuple[Any, ...]]:
    """Capture path membership, contents, and inode identity for no-mutation checks."""
    snapshot: dict[str, tuple[Any, ...]] = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            snapshot[relative] = ("symlink", os.readlink(path), path.lstat().st_ino)
        elif path.is_dir():
            snapshot[relative] = ("directory", path.stat().st_ino)
        else:
            stat = path.stat()
            snapshot[relative] = ("file", path.read_bytes(), stat.st_ino)
    return snapshot


def _assert_same_value(left: Any, right: Any) -> None:
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert isinstance(right, dict)
        assert left.keys() == right.keys()
        for key in left:
            _assert_same_value(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert isinstance(right, type(left))
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right, strict=True):
            _assert_same_value(left_item, right_item)
    else:
        assert left == right


def test_epoch_change_requires_explicit_extension_and_preserves_archived_state(tmp_path):
    store, old_config = _new_run(tmp_path)
    new_config = _config(epochs=3)
    with pytest.raises(ValueError, match="config hash"):
        RunStore.resume(store.run_dir, new_config)

    config_path = store.run_dir / "resolved_config.yaml"
    manifest_path = store.run_dir / "run_manifest.json"
    checkpoint_paths = sorted((store.run_dir / "checkpoints").glob("*.pt"))
    original_files = {
        path.relative_to(store.run_dir).as_posix(): path.read_bytes()
        for path in [config_path, manifest_path, *checkpoint_paths]
    }
    original_inodes = {
        path.relative_to(store.run_dir).as_posix(): path.stat().st_ino
        for path in [config_path, manifest_path, *checkpoint_paths]
    }
    original_last = store.load_checkpoint("last")
    original_history = (store.run_dir / "metrics" / "history.jsonl").read_bytes()
    run_id = json.loads(manifest_path.read_text(encoding="utf-8"))["run_id"]

    extended = RunStore.extend_epochs(store.run_dir, 3)

    archive = store.run_dir / "artifacts" / "epoch_extensions" / "2_to_3"
    assert archive.is_dir()
    for relative, original_bytes in original_files.items():
        archived = archive / relative
        assert archived.read_bytes() == original_bytes
        if relative.startswith("checkpoints/"):
            assert archived.stat().st_ino == original_inodes[relative]

    saved_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    expected_config = {**old_config, "optimization": {**old_config["optimization"], "epochs": 3}}
    assert saved_config == expected_config
    active_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert active_manifest["run_id"] == run_id == store.run_dir.name
    assert active_manifest["config_sha256"] == config_digest(expected_config)
    receipts = active_manifest["epoch_extensions"]
    assert isinstance(receipts, list) and receipts
    assert isinstance(receipts[-1], str)
    receipt_path = Path(receipts[-1])
    assert not receipt_path.is_absolute()
    assert receipt_path == Path("artifacts/epoch_extensions/2_to_3/receipt.json")
    receipt = json.loads((store.run_dir / receipt_path).read_text(encoding="utf-8"))
    assert receipt["old_epochs"] == 2
    assert receipt["new_epochs"] == 3

    resumed = RunStore.resume(store.run_dir, expected_config)
    assert resumed.config == expected_config
    with pytest.raises(ValueError, match="config hash"):
        RunStore.resume(store.run_dir, old_config)
    assert extended.config_hash == config_digest(expected_config)

    # Extending the configured horizon is metadata-only: the checkpoint and
    # training history retain their exact pre-extension state.
    last_path = store.run_dir / "checkpoints" / "last.pt"
    assert last_path.read_bytes() == original_files["checkpoints/last.pt"]
    assert store.load_checkpoint("last")["global_step"] == 17
    _assert_same_value(original_last, store.load_checkpoint("last"))
    assert (store.run_dir / "metrics" / "history.jsonl").read_bytes() == original_history

    # RunStore writes checkpoints through atomic replacement. The hard-linked
    # archive must remain the old checkpoint after the live last.pt is replaced.
    replacement = _checkpoint(step=18)
    replacement["model"]["weight"] = torch.tensor([8.0, 9.0])
    resumed.save_checkpoint("last", replacement)
    archived_last = archive / "checkpoints" / "last.pt"
    assert archived_last.read_bytes() == original_files["checkpoints/last.pt"]
    assert archived_last.stat().st_ino == original_inodes["checkpoints/last.pt"]
    assert last_path.stat().st_ino != archived_last.stat().st_ino
    _assert_same_value(original_last, load_project_checkpoint(archived_last))
    assert torch.equal(resumed.load_checkpoint()["model"]["weight"], torch.tensor([8.0, 9.0]))


@pytest.mark.parametrize("target", [0, -1, 1, 2, True, 2.5, None, "3"])
def test_epoch_extension_rejects_invalid_or_nonincreasing_target_without_mutation(tmp_path, target):
    store, _ = _new_run(tmp_path)
    before = _tree_snapshot(store.run_dir)

    with pytest.raises((TypeError, ValueError)):
        RunStore.extend_epochs(store.run_dir, target)

    assert _tree_snapshot(store.run_dir) == before


@pytest.mark.parametrize("corruption", ["saved_config", "manifest_digest"])
def test_epoch_extension_refuses_config_manifest_mismatch_without_mutation(tmp_path, corruption):
    store, _ = _new_run(tmp_path)
    config_path = store.run_dir / "resolved_config.yaml"
    manifest_path = store.run_dir / "run_manifest.json"
    if corruption == "saved_config":
        saved_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        saved_config["case"] = "tampered"
        config_path.write_text(yaml.safe_dump(saved_config), encoding="utf-8")
    else:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["config_sha256"] = "0" * 64
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    before = _tree_snapshot(store.run_dir)

    with pytest.raises(ValueError):
        RunStore.extend_epochs(store.run_dir, 3)

    assert _tree_snapshot(store.run_dir) == before
    assert not (store.run_dir / "artifacts" / "epoch_extensions").exists()


def test_epoch_extension_refuses_checkpoint_bound_to_different_config_without_mutation(tmp_path):
    store, _ = _new_run(tmp_path)
    checkpoint = store.load_checkpoint("last")
    checkpoint["config_sha256"] = config_digest(_config(epochs=4))
    store.save_checkpoint("last", checkpoint)
    before = _tree_snapshot(store.run_dir)

    with pytest.raises(ValueError, match="checkpoint config digest"):
        RunStore.extend_epochs(store.run_dir, 3)

    assert _tree_snapshot(store.run_dir) == before
    assert not (store.run_dir / "artifacts" / "epoch_extensions").exists()


def test_epoch_extension_rolls_back_config_and_manifest_when_manifest_write_fails(
    tmp_path, monkeypatch
):
    store, old_config = _new_run(tmp_path)
    config_path = store.run_dir / "resolved_config.yaml"
    manifest_path = store.run_dir / "run_manifest.json"
    checkpoint_paths = sorted((store.run_dir / "checkpoints").glob("*.pt"))
    original_files = {
        path: path.read_bytes() for path in [config_path, manifest_path, *checkpoint_paths]
    }
    archive = store.run_dir / "artifacts" / "epoch_extensions" / "2_to_3"
    atomic_text = run_store_module._atomic_text
    injected = False

    def fail_first_manifest_update(path, content):
        nonlocal injected
        path = Path(path)
        if path == manifest_path and not injected:
            injected = True
            assert (
                yaml.safe_load(config_path.read_text(encoding="utf-8"))["optimization"]["epochs"]
                == 3
            )
            raise OSError("simulated manifest replacement failure")
        return atomic_text(path, content)

    monkeypatch.setattr(run_store_module, "_atomic_text", fail_first_manifest_update)
    with pytest.raises(OSError, match="simulated manifest replacement failure"):
        RunStore.extend_epochs(store.run_dir, 3)

    assert injected
    for path, original_bytes in original_files.items():
        assert path.read_bytes() == original_bytes
    assert not archive.exists()
    assert RunStore.resume(store.run_dir, old_config).config == old_config

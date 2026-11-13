"""Load and verify immutable source models for post-training stages."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from ..data.factory import FieldDataset, open_field_dataset
from ..data.normalization import FieldNormalizer
from ..models import build_model
from ..models.compatibility import load_legacy_demo50
from .model_lifecycle import load_training_aux_state
from .run_store import file_sha256, load_model_state_strict, load_project_checkpoint


def source_checkpoint_path(config: Mapping[str, Any]) -> Path:
    source_run = Path(config["source_run"])
    checkpoint = Path(str(config["source_checkpoint"]))
    if checkpoint.is_absolute():
        return checkpoint
    name = checkpoint.name if checkpoint.suffix == ".pt" else f"{checkpoint.name}.pt"
    if config.get("source", {}).get("kind", "native_run") == "legacy_demo50":
        return source_run / name
    return source_run / "checkpoints" / name


def source_hashes(config: Mapping[str, Any]) -> dict[str, str]:
    """Hash all source artifacts that can influence the loaded child model."""
    source_run = Path(config["source_run"])
    candidates = {
        "checkpoint": source_checkpoint_path(config),
        "resolved_config": source_run / "resolved_config.yaml",
        "run_manifest": source_run / "run_manifest.json",
        "legacy_args": source_run / "args.json",
        "legacy_config": source_run / "run_config.yaml",
        "legacy_normalization": source_run / "dataset_stats.pt",
    }
    existing = {name: path for name, path in candidates.items() if path.is_file()}
    if "checkpoint" not in existing:
        raise FileNotFoundError(f"source checkpoint does not exist: {candidates['checkpoint']}")
    return {name: file_sha256(path) for name, path in existing.items()}


def _validate_native_source_status(config: Mapping[str, Any]) -> None:
    source = config.get("source", {})
    if source.get("admission") == "frozen_snapshot":
        receipt_path = Path(source["snapshot_receipt"])
        source_run = Path(config["source_run"])
        if receipt_path.resolve().parent != source_run.resolve():
            raise ValueError("snapshot receipt must belong to the admitted source directory")
        if source_checkpoint_path(config).resolve().parent != (source_run / "checkpoints").resolve():
            raise ValueError("admitted checkpoint must reside in the snapshot checkpoint directory")
        receipt = json.loads(receipt_path.read_text())
        if receipt.get("schema") != "frozen_source_snapshot.v1" or receipt.get("checkpoint_sha256") != file_sha256(source_checkpoint_path(config)):
            raise ValueError("frozen source snapshot receipt or checkpoint hash mismatch")
        metadata = receipt.get("metadata_sha256")
        if not isinstance(metadata, Mapping) or set(metadata) != {"resolved_config.yaml", "run_manifest.json", "status.json"}:
            raise ValueError("frozen source receipt lacks complete metadata hashes")
        if any(file_sha256(source_run / name) != digest for name, digest in metadata.items()):
            raise ValueError("frozen source snapshot metadata hash mismatch")
        return
    status_path = Path(config["source_run"]) / "status.json"
    if not status_path.is_file():
        raise FileNotFoundError(f"native source run has no status.json: {status_path}")
    status = json.loads(status_path.read_text())
    if status.get("status") != "completed" and not bool(
        config.get("source", {}).get("allow_integration_source", False)
    ):
        raise ValueError(
            "source run is not complete; set source.allow_integration_source=true "
            "only for a declared integration check"
        )


def materialize_ema_view(model: torch.nn.Module) -> str:
    """Strictly materialize an already restored EMA, then disable child EMA."""
    ema = getattr(model, "_ema", None)
    if ema is None or ema.num_updates < 1:
        raise ValueError("materialized EMA requires a populated strict-loaded EMA")
    ema.copy_to(model)
    model._ema = None
    model._ema_eval = False
    if hasattr(model, "model_ema_enabled"):
        model.model_ema_enabled = False
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def load_source_model(
    config: Mapping[str, Any], device: torch.device
) -> tuple[torch.nn.Module, FieldDataset, dict[str, Any]]:
    """Load a native or Demo50 source with strict identity and normalization checks."""
    source_kind = config.get("source", {}).get("kind", "native_run")
    checkpoint_path = source_checkpoint_path(config)
    if source_kind == "legacy_demo50":
        source = config["source"]
        model, compatibility = load_legacy_demo50(
            config["source_run"],
            config["dataset"]["path"],
            source["channel_mapping"],
            checkpoint=checkpoint_path.name,
            map_location="cpu",
        )
        normalizer = FieldNormalizer(
            torch.tensor(compatibility.normalization_mean),
            torch.tensor(compatibility.normalization_std),
            "legacy_checkpoint_mean_std",
        )
        dataset = open_field_dataset(
            config["dataset"],
            split=config["dataset"].get("split", "train"),
            normalizer=normalizer,
        )
        return (
            model.to(device),
            dataset,
            {
                "kind": source_kind,
                "compatibility": asdict(compatibility),
            },
        )

    _validate_native_source_status(config)
    dataset = open_field_dataset(config["dataset"], split=config["dataset"].get("split", "train"))
    checkpoint = load_project_checkpoint(checkpoint_path)
    if checkpoint.get("model_name") != config["model"]["name"]:
        raise ValueError("source checkpoint model identity disagrees with inherited model config")
    if tuple(checkpoint.get("data_spec", {}).get("field_names", ())) != tuple(
        dataset.data_spec.field_names
    ):
        raise ValueError("source checkpoint field order disagrees with the selected dataset")
    model = build_model(config["model"], dataset.data_spec)
    load_model_state_strict(model, checkpoint["model"])
    load_training_aux_state(model, checkpoint)
    weight_view = config.get("source", {}).get("weight_view", "live")
    view_hash = None
    if weight_view == "ema_materialized":
        view_hash = materialize_ema_view(model)
    elif weight_view != "live":
        raise ValueError("unknown source weight view")
    model.source_weight_view_sha256 = view_hash
    checkpoint_normalizer = FieldNormalizer(
        checkpoint["normalization"]["offset"],
        checkpoint["normalization"]["scale"],
        checkpoint["normalization"]["method"],
    )
    if not torch.equal(checkpoint_normalizer.offset, dataset.normalizer.offset) or not torch.equal(
        checkpoint_normalizer.scale, dataset.normalizer.scale
    ):
        raise ValueError("source checkpoint normalization disagrees with dataset statistics")
    return (
        model.to(device),
        dataset,
        {
            "kind": source_kind,
            "checkpoint_model_name": checkpoint["model_name"],
            "checkpoint_config_sha256": checkpoint["config_sha256"],
            "weight_view": weight_view,
            "materialized_view_sha256": view_hash,
        },
    )


def set_trainable_scope(model: torch.nn.Module, settings: Mapping[str, Any]) -> tuple[str, ...]:
    """Enable exactly the parameters declared by a child-run trainable scope."""
    scope = settings.get("scope", "full_model")
    prefixes = tuple(str(value) for value in settings.get("modules", ()))
    selected = []
    for name, parameter in model.named_parameters():
        parameter.requires_grad = scope == "full_model" or any(
            name == prefix or name.startswith(prefix + ".") for prefix in prefixes
        )
        if parameter.requires_grad:
            selected.append(name)
    if not selected:
        raise ValueError("trainable scope selected no model parameters")
    return tuple(selected)

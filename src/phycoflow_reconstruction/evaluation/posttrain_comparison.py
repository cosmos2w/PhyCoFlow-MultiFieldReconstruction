"""Validate and summarize matched base/post-training pairs for Phase 6."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import yaml

from ..data.manifest import SensorManifest

SUPPORTED_RUN_STATUSES = {"completed", "integration_truncated"}
UNSUPPORTED_REASON_CODES = {
    "no_plain_base_run",
    "not_inference_model",
    "non_differentiable_inference",
    "missing_required_geometry",
    "missing_dependency",
}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected a JSON mapping: {path}")
    return payload


def _read_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected a YAML mapping: {path}")
    return payload


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(path: str | Path, root: Path) -> Path:
    candidate = Path(path).expanduser()
    return candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()


def _query_digest(path: Path) -> str:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - PyTorch before weights_only support
        payload = torch.load(path, map_location="cpu")
    indices = payload.get("query_indices")
    if not isinstance(indices, torch.Tensor):
        raise TypeError(f"{path} does not contain tensor query_indices")
    tensor = indices.detach().cpu().contiguous()
    digest = hashlib.sha256(str(tuple(tensor.shape)).encode("utf-8"))
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _last_history(path: Path) -> dict[str, Any]:
    rows = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
    return json.loads(rows[-1]) if rows else {}


def _gain(before: float, after: float) -> dict[str, float | None]:
    absolute = before - after
    return {
        "absolute_reduction": absolute,
        "relative_reduction": None if before == 0 else absolute / before,
    }


def _pair_record(entry: Mapping[str, Any], matrix_root: Path) -> dict[str, Any]:
    source_run = _resolve(entry["source_run"], matrix_root)
    post_run = _resolve(entry["post_run"], matrix_root)
    manifest = _read_json(post_run / "run_manifest.json")
    status = _read_json(post_run / "status.json")
    config = _read_yaml(post_run / "resolved_config.yaml")
    environment = _read_json(post_run / "environment.json")
    before = _read_json(post_run / "evaluation" / "before.json")
    after = _read_json(post_run / "evaluation" / "after.json")
    history = _last_history(post_run / "metrics" / "history.jsonl")

    if Path(manifest["parent_run"]).resolve() != source_run:
        raise ValueError(f"{entry['id']}: child parent_run does not match source_run")
    if not bool(manifest.get("source_immutable_verified")):
        raise ValueError(f"{entry['id']}: source immutability was not verified")
    if manifest.get("source_hashes") != manifest.get("source_hashes_after"):
        raise ValueError(f"{entry['id']}: source hashes changed during post-training")
    if status.get("status") not in SUPPORTED_RUN_STATUSES:
        raise ValueError(f"{entry['id']}: unsupported child status {status.get('status')!r}")
    if before.get("sensor_manifest_sha256") != after.get("sensor_manifest_sha256"):
        raise ValueError(f"{entry['id']}: before/after sensor manifests differ")
    source_checkpoint = Path(manifest["source_checkpoint"]).resolve()
    if _file_sha256(source_checkpoint) != manifest["source_hashes"]["checkpoint"]:
        raise ValueError(
            f"{entry['id']}: current source checkpoint hash disagrees with child lineage"
        )
    for checkpoint_name, expected_hash in manifest["checkpoint_hashes"].items():
        checkpoint_path = post_run / "checkpoints" / f"{checkpoint_name}.pt"
        if _file_sha256(checkpoint_path) != expected_hash:
            raise ValueError(f"{entry['id']}: child {checkpoint_name} checkpoint hash mismatch")
    sensor_manifest = SensorManifest.load(
        post_run / "artifacts" / "evaluation_sensor_manifest.json"
    )
    if sensor_manifest.digest() != manifest["evaluation_sensor_manifest_sha256"]:
        raise ValueError(f"{entry['id']}: evaluation sensor artifact hash mismatch")

    before_mse = float(before["mse_normalized"])
    after_mse = float(after["mse_normalized"])
    before_coherence = float(before["coherence"]["total"])
    after_coherence = float(after["coherence"]["total"])
    data_gradient_norm = history.get("data_grad_norm")
    coherence_gradient_norm = history.get("coherence_grad_norm")
    for label, value in (
        ("data", data_gradient_norm),
        ("coherence", coherence_gradient_norm),
    ):
        if value is None or not math.isfinite(float(value)) or float(value) <= 0:
            raise ValueError(f"{entry['id']}: {label} gradient must be finite and positive")
    query_digest = _query_digest(post_run / "artifacts" / "evaluation_query_indices.pt")
    return {
        "id": str(entry["id"]),
        "family": str(entry["family"]),
        "model_name": config["model"]["name"],
        "model_variant": entry.get("model_variant"),
        "source_run": str(entry["source_run"]),
        "post_run": str(entry["post_run"]),
        "run_status": status["status"],
        "cuda_visible_devices": environment.get("cuda_visible_devices"),
        "cuda_device": (
            environment.get("cuda_devices", [None])[0] if environment.get("cuda_devices") else None
        ),
        "dataset_fingerprint": manifest["dataset_fingerprint"],
        "reference_bank_sha256": manifest.get("reference_bank_sha256"),
        "sensor_manifest_sha256": manifest["evaluation_sensor_manifest_sha256"],
        "query_indices_sha256": query_digest,
        "training_seed": int(config["runtime"]["seed"]),
        "evaluation_seed": int(config["evaluation"]["seed"]),
        "evaluation_split": config["evaluation"]["split"],
        "target_use": before["coherence"]["target_use"],
        "parameter_count": int(manifest["parameter_count"]),
        "trainable_parameter_count": int(manifest["trainable_parameter_count"]),
        "differentiable_adapter": manifest["differentiable_adapter"],
        "mse_normalized": {
            "before": before_mse,
            "after": after_mse,
            **_gain(before_mse, after_mse),
        },
        "coherence_total": {
            "before": before_coherence,
            "after": after_coherence,
            **_gain(before_coherence, after_coherence),
        },
        "post_training": {
            "seconds": float(status.get("post_training_seconds", 0.0)),
            "seconds_per_step": float(status.get("seconds_per_step", 0.0)),
            "peak_cuda_memory_bytes": int(status.get("peak_cuda_memory_bytes", 0)),
            "data_gradient_norm": data_gradient_norm,
            "coherence_gradient_norm": coherence_gradient_norm,
            "update_mode": history.get("update_mode"),
        },
        "inference": {
            "before_seconds": float(before["inference"]["seconds"]),
            "after_seconds": float(after["inference"]["seconds"]),
            "samples": int(after["inference"]["samples"]),
            "points_per_sample": int(after["inference"]["points_per_sample"]),
            "generation_steps": int(after["inference"]["generation_steps"]),
        },
    }


def _require_shared(records: list[dict[str, Any]], key: str) -> Any:
    values = {record[key] for record in records}
    if len(values) != 1:
        raise ValueError(f"Phase-6 pairs do not share {key}: {sorted(values, key=str)}")
    return next(iter(values))


def build_phase6_comparison(matrix_path: str | Path) -> dict[str, Any]:
    """Build a strict comparison payload and fail on any unmatched protocol."""
    matrix_path = Path(matrix_path).resolve()
    matrix = _read_yaml(matrix_path)
    entries = matrix.get("entries", [])
    unsupported = matrix.get("unsupported", [])
    if not entries:
        raise ValueError("Phase-6 matrix must contain at least one matched pair")
    identifiers = [str(entry.get("id", "")) for entry in entries]
    if not all(identifiers) or len(identifiers) != len(set(identifiers)):
        raise ValueError("Phase-6 entry IDs must be non-empty and unique")
    for item in unsupported:
        if item.get("reason_code") not in UNSUPPORTED_REASON_CODES:
            raise ValueError(f"unsupported entry {item.get('id')!r} needs a recognized reason_code")
        if not str(item.get("reason", "")).strip():
            raise ValueError(f"unsupported entry {item.get('id')!r} needs an explanation")

    records = [_pair_record(entry, matrix_path.parent) for entry in entries]
    required_environment = matrix.get("required_environment", {})
    for record in records:
        for key in ("cuda_visible_devices", "cuda_device"):
            expected = required_environment.get(key)
            if expected is not None and record[key] != str(expected):
                raise ValueError(
                    f"{record['id']}: required {key}={expected!r}, got {record[key]!r}"
                )
    shared = {
        key: _require_shared(records, key)
        for key in (
            "dataset_fingerprint",
            "reference_bank_sha256",
            "sensor_manifest_sha256",
            "query_indices_sha256",
            "training_seed",
            "evaluation_seed",
            "evaluation_split",
            "target_use",
            "cuda_visible_devices",
            "cuda_device",
        )
    }
    if shared["target_use"] != "training_reference":
        raise ValueError("formal Phase-6 comparison requires target_use=training_reference")
    return {
        "version": "1",
        "case": matrix["case"],
        "comparison_id": matrix["comparison_id"],
        "scope": matrix.get("scope", "integration"),
        "shared_protocol": shared,
        "matched_pairs": records,
        "unsupported": unsupported,
        "claims": {
            "scientific_performance": bool(matrix.get("scientific_performance", False)),
            "note": matrix.get(
                "claim_note",
                "Integration comparisons validate the pipeline and are not tuned scientific results.",
            ),
        },
    }


def comparison_markdown(payload: Mapping[str, Any]) -> str:
    """Render a compact human-auditable table from the validated payload."""
    lines = [
        f"# Phase 6 comparison: {payload['comparison_id']}",
        "",
        str(payload["claims"]["note"]),
        "",
        "| Model | Family | Params (M) | MSE before | MSE after | MSE reduction | Coherence reduction | Time/step (s) | Inference (ms) | Peak GPU (MiB) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["matched_pairs"]:
        mse = row["mse_normalized"]
        coherence = row["coherence_total"]
        runtime = row["post_training"]
        lines.append(
            "| {id} | {family} | {parameters:.3f} | {before:.6g} | {after:.6g} | "
            "{mse_gain:.6g} | {coherence_gain:.6g} | {seconds:.4g} | "
            "{inference:.3f} | {memory:.1f} |".format(
                id=row["id"],
                family=row["family"],
                parameters=row["parameter_count"] / 1_000_000,
                before=mse["before"],
                after=mse["after"],
                mse_gain=mse["absolute_reduction"],
                coherence_gain=coherence["absolute_reduction"],
                seconds=runtime["seconds_per_step"],
                inference=row["inference"]["after_seconds"] * 1000,
                memory=runtime["peak_cuda_memory_bytes"] / (1024**2),
            )
        )
    lines.extend(["", "## Unsupported combinations", ""])
    for item in payload["unsupported"]:
        lines.append(f"- `{item['id']}` — `{item['reason_code']}`: {item['reason']}")
    lines.extend(["", "## Shared protocol", "", "```json"])
    lines.append(json.dumps(payload["shared_protocol"], indent=2, sort_keys=True))
    lines.extend(["```", ""])
    return "\n".join(lines)


def write_phase6_comparison(
    matrix_path: str | Path,
    json_path: str | Path,
    markdown_path: str | Path,
) -> dict[str, Any]:
    payload = build_phase6_comparison(matrix_path)
    json_path, markdown_path = Path(json_path), Path(markdown_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(comparison_markdown(payload), encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()
    payload = write_phase6_comparison(args.matrix, args.json, args.markdown)
    print(
        json.dumps(
            {"matched_pairs": len(payload["matched_pairs"]), **payload["shared_protocol"]}, indent=2
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

"""Matched full-native inference and physical-field risk for coherence studies.

Inference streams one condition at a time and never passes targets to a model.
Each member is reconstructed over the complete ordered native support in one
model call; a model may internally chunk queries while retaining its global
latent realization. Saved members can be rescored by separately frozen family
artifacts, without repeating inference.
"""

from __future__ import annotations

import hashlib
import json
import resource
import time
from pathlib import Path

import numpy as np
import torch

from ..data.manifest import SensorManifest, dataset_fingerprint
from ..data.sensor_protocols import SensorProtocol, build_observation_batch
from ..training.common import sensor_protocol_from_config
from ..training.model_lifecycle import selected_evaluation_weight_context
from ..training.post_training import _without_target
from ..training.rollout import differentiable_reconstruction
from ..training.run_store import file_sha256


def _digest_arrays(*arrays):
    digest = hashlib.sha256()
    for value in arrays:
        array = np.ascontiguousarray(np.asarray(value))
        digest.update(f"{array.dtype}:{array.shape}:".encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _member_seed(base: int, sample_id: str, draw: int) -> int:
    encoded = f"native-coherence-v1:{int(base)}:{sample_id}:{int(draw)}".encode()
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big") % (2**63 - 1)


def _physical_weights(sample):
    """Return case-supplied weights or rectilinear trapezoidal nodal areas."""
    supplied = sample.metadata.get("auxiliary", {}).get("physical_weights")
    n = sample.values.shape[0]
    mask = (torch.ones(n, dtype=torch.bool) if sample.valid_points is None
            else sample.valid_points.detach().cpu())
    if supplied is not None:
        weights = torch.as_tensor(supplied).detach().double().cpu().numpy()
        receipt = sample.metadata.get("measure_metadata", {})
        provider = receipt.get("metadata", {}).get("provider", "case_supplied_physical_weights")
        measure = str(sample.metadata.get("physical_measure", provider))
    else:
        coords = sample.coordinates_raw.detach().double().cpu().numpy()
        if coords.ndim != 2 or coords.shape[1] != 2:
            raise ValueError("physical weights are required for non-2D or nonrectilinear native support")
        xs, ys = np.unique(coords[:, 0]), np.unique(coords[:, 1])
        if min(len(xs), len(ys)) < 2 or len(xs) * len(ys) != n:
            raise ValueError("physical weights are required for incomplete or nonrectilinear native support")
        order = np.lexsort((coords[:, 0], coords[:, 1]))
        grid = np.stack(np.meshgrid(xs, ys, indexing="xy"), -1).reshape(-1, 2)
        if not np.array_equal(coords[order], grid):
            raise ValueError("physical weights are required for nonrectilinear native support")
        def widths(axis):
            return np.diff(np.concatenate(([axis[0]], (axis[:-1] + axis[1:]) / 2, [axis[-1]])))
        sorted_weights = (widths(ys)[:, None] * widths(xs)[None, :]).ravel()
        weights = np.empty(n, dtype=np.float64)
        weights[order] = sorted_weights
        measure = "native_rectilinear_trapezoidal_area"
    valid = mask.numpy()
    if (weights.shape != (n,) or not valid.any() or not np.isfinite(weights[valid]).all()
            or np.any(weights[valid] <= 0)):
        raise ValueError("native physical weights must be positive and finite on the valid support")
    weights = np.where(valid, weights, 0.)
    weights = weights / weights.sum()
    return weights, valid, measure


def native_field_risk(draws, target, weights, mask):
    """Per-member/mean physical MSE, RMSE and relative L2 under fixed weights."""
    draws = np.asarray(draws, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    weights, mask = np.asarray(weights, dtype=np.float64), np.asarray(mask, dtype=bool)
    if (draws.ndim != 3 or target.ndim != 2 or draws.shape[1:] != target.shape
            or weights.shape != target.shape[:1] or mask.shape != weights.shape or not mask.any()):
        raise ValueError("native risk requires [J,N,C] draws, [N,C] target and aligned weights/mask")
    if (not np.isfinite(draws[:, mask]).all() or not np.isfinite(target[mask]).all()
            or not np.isfinite(weights[mask]).all() or (weights[mask] <= 0).any()):
        raise ValueError("native risk requires finite fields and positive finite support weights")
    mass = weights[mask] / weights[mask].sum()
    truth = target[mask]
    members = draws[:, mask]
    reference_power = np.sum(mass[:, None] * truth**2, axis=0)
    member_mse = np.sum(mass[None, :, None] * (members - truth[None])**2, axis=1)
    mean_mse = np.sum(mass[:, None] * (members.mean(axis=0) - truth)**2, axis=0)
    denominator = np.where(reference_power > 0, reference_power, np.nan)
    return {
        "member_mse_physical": member_mse,
        "member_rmse_physical": np.sqrt(member_mse),
        "member_relative_l2_physical": np.sqrt(member_mse / denominator[None]),
        "ensemble_mean_mse_physical": mean_mse,
        "ensemble_mean_rmse_physical": np.sqrt(mean_mse),
        "ensemble_mean_relative_l2_physical": np.sqrt(mean_mse / denominator),
        "reference_power_physical": reference_power,
    }


def _effective_weights(model, selection):
    if selection == "live":
        return "live"
    core = getattr(model, "core", model)
    if hasattr(core, "_ema_eval"):
        return "ema" if getattr(core, "_ema", None) is not None and core._ema_eval else "live"
    return "configured_model_hook" if callable(getattr(model, "evaluation_weight_context", None)) else "live"


def _manifest(endpoint_config, dataset):
    value = endpoint_config.get("sensor_manifest")
    if value is None:
        return None
    manifest = value if isinstance(value, SensorManifest) else SensorManifest.load(value)
    if dataset_fingerprint(dataset.path) != manifest.dataset_fingerprint:
        raise ValueError("sensor manifest dataset fingerprint differs from the native dataset")
    if manifest.split != dataset.split_name:
        raise ValueError("sensor manifest split differs from the native dataset")
    return manifest


def infer_native_panel(
    runtime, dataset, indices, output_dir, endpoint_config, *, panel_seed_base, draws=4,
    retain_payloads=True, consumer=None, retain_payload_indices=(),
):
    """Infer a matched panel and save fieldwise risks plus optional member payloads.

    ``runtime`` supplies ``model``, ``device``, ``config`` and ``checkpoint_path``.
    ``indices`` are dataset positions, not sparse spatial IDs. Endpoint settings
    require explicit ``steps``, ``solver``, ``observation_config`` and
    ``weight_selection`` (``live`` or ``configured``). A SensorManifest may be
    passed as ``sensor_manifest``; its observation IDs are honored and any
    recorded sparse query IDs are intentionally ignored. Without one the
    configured protocol is deterministically applied per condition. Relative L2
    is NaN where reference power is zero; MSE and RMSE remain available.
    Optional consumer(arrays, row) receives each complete condition in memory;
    retain_payload_indices selects additional dataset positions for saved examples.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists() or (output_dir / "field_risk.npz").exists():
        raise FileExistsError("native inference output already exists; choose a new output directory")
    required = {"steps", "solver", "observation_config", "weight_selection"}
    if required.difference(endpoint_config):
        raise ValueError(f"endpoint_config requires {sorted(required)}")
    steps = int(endpoint_config["steps"])
    solver = str(endpoint_config["solver"])
    obs_config = dict(endpoint_config["observation_config"])
    selection = str(endpoint_config["weight_selection"])
    if steps < 1 or solver not in {"euler", "heun"} or selection not in {"live", "configured"}:
        raise ValueError("native endpoint requires positive steps, euler/heun and live/configured weights")
    draws = int(draws)
    if draws < 1:
        raise ValueError("native inference requires at least one member")
    positions = [int(i) for i in indices]
    if not positions or len(set(positions)) != len(positions) or any(i < 0 or i >= len(dataset) for i in positions):
        raise ValueError("native panel indices must be nonempty, unique and within the dataset")
    retained_indices = {int(i) for i in retain_payload_indices}
    if not retained_indices.issubset(positions):
        raise ValueError("retained native positions must belong to the panel")
    if consumer is not None and not callable(consumer):
        raise TypeError("native consumer must be callable")
    sensor_manifest = _manifest(endpoint_config, dataset)
    protocol = (SensorProtocol(**sensor_manifest.protocol) if sensor_manifest is not None
                else sensor_protocol_from_config(runtime.config))
    model, device = runtime.model, torch.device(runtime.device)
    model.eval()
    model_attrs = {}
    for label, obj in (("model", model), ("core", getattr(model, "core", None))):
        if obj is not None:
            keys = ("reconstruction_execution_mode", "reconstruction_query_chunk_size",
                    "query_chunk_size", "rollout_context_cache", "rollout_checkpointing")
            model_attrs[label] = {key: getattr(obj, key) for key in keys if hasattr(obj, key)
                                  and isinstance(getattr(obj, key), (str, int, float, bool, type(None)))}
    sampler = {"path": "full_query_differentiable_reconstruction", "steps": steps, "solver": solver,
               "observation_config": obs_config, "weight_selection": selection,
               "effective_weights": _effective_weights(model, selection), "execution": model_attrs,
               "seed_scheme": "sha256_native_coherence_v1", "panel_seed_base": int(panel_seed_base),
               "draws": draws}
    rows, risks, ids = [], [], []
    callback_seconds = 0.0
    io_seconds = 0.0
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    with selected_evaluation_weight_context(model, selection), torch.no_grad():
        for ordinal, index in enumerate(positions):
            sample = dataset[index]
            sample.validate()
            sample_id = f"{sample.trajectory_id}:{sample.time_index if sample.time_index is not None else 'all'}"
            if sample_id in ids:
                raise ValueError("native panel sample identities must be unique")
            if sensor_manifest is not None and sample_id not in sensor_manifest.indices:
                raise KeyError(f"sensor manifest has no observations for {sample_id!r}")
            batch = build_observation_batch(
                [sample], protocol, query_points=None,
                manifest_indices=None if sensor_manifest is None else sensor_manifest.indices,
            ).to(device)
            native_ids = batch.metadata["query_indices"][0].detach().cpu().numpy()
            if not np.array_equal(native_ids, np.arange(sample.values.shape[0])):
                raise ValueError("native inference must use every original native point in order")
            weights, mask, measure = _physical_weights(sample)
            if not np.array_equal(batch.query_valid_mask[0].cpu().numpy(), mask):
                raise ValueError("inference and physical support masks differ")
            if tuple(sample.field_names) != tuple(dataset.field_names):
                raise ValueError("native sample field order differs from the dataset")
            if not mask[batch.obs_indices[0, batch.obs_valid_mask[0]].detach().cpu().numpy()].all():
                raise ValueError("native observations include points outside the valid physical mask")
            inference = _without_target(batch)
            seeds = [_member_seed(panel_seed_base, sample_id, j) for j in range(draws)]
            values = []
            for seed in seeds:
                generator = torch.Generator(device=device).manual_seed(seed)
                prediction = differentiable_reconstruction(
                    model, inference, steps=steps, solver=solver, generator=generator,
                    observation_config=obs_config,
                )
                if prediction.shape != batch.target_fields.shape:
                    raise ValueError("native prediction does not cover every native point and field")
                values.append(prediction[0].detach().cpu())
            member_model = torch.stack(values)
            member_physical = dataset.normalizer.decode(member_model).double().numpy()
            target_model = sample.values.detach().cpu()
            target_physical = dataset.normalizer.decode(target_model).double().numpy()
            risk = native_field_risk(member_physical, target_physical, weights, mask)
            risks.append(risk)
            arrays = {
                "draws_model": member_model.numpy(), "target_model": target_model.numpy(),
                "draws_physical": member_physical, "target_physical": target_physical,
                "model_coordinates": sample.coordinates.detach().cpu().numpy(),
                "raw_coordinates": sample.coordinates_raw.detach().cpu().numpy(),
                "native_ids": native_ids, "physical_mask": mask, "physical_weights": weights,
                "observation_coords": batch.obs_coords[0].cpu().numpy(),
                "observation_values": batch.obs_values[0].cpu().numpy(),
                "observation_field_ids": batch.obs_field_ids[0].cpu().numpy(),
                "observation_indices": batch.obs_indices[0].cpu().numpy(),
                "observation_mask": batch.obs_valid_mask[0].cpu().numpy(),
                "draw_seeds": np.asarray(seeds, dtype=np.int64),
                "field_names": np.asarray(sample.field_names), "sample_id": np.asarray(sample_id),
                "field_units": np.asarray(dataset.data_spec.field_units),
                "logical_shape": np.asarray(sample.logical_shape, dtype=np.int64),
                "normalizer_sha256": np.asarray(dataset.normalizer.digest()),
            }
            contract_hash = _digest_arrays(
                arrays["native_ids"], arrays["model_coordinates"], arrays["raw_coordinates"],
                arrays["physical_mask"], arrays["physical_weights"], arrays["target_model"],
                arrays["observation_coords"], arrays["observation_values"], arrays["observation_indices"],
                arrays["observation_field_ids"], arrays["observation_mask"], arrays["draw_seeds"],
            )
            name = f"condition_{ordinal:05d}.npz" if retain_payloads or index in retained_indices else None
            row = {"sample_id": sample_id, "dataset_index": index, "payload": name,
                   "contract_sha256": contract_hash, "point_count": len(native_ids),
                   "valid_count": int(mask.sum()), "physical_measure": measure,
                   "draw_seeds": seeds}
            if consumer is not None:
                callback_started = time.perf_counter()
                consumer(arrays, dict(row))
                callback_seconds += time.perf_counter() - callback_started
            if name is not None:
                path = output_dir / name
                if path.exists():
                    raise FileExistsError(f"native member payload already exists: {path}")
                io_started = time.perf_counter()
                np.savez_compressed(path, **arrays)
                io_seconds += time.perf_counter() - io_started
            rows.append(row)
            ids.append(sample_id)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    risk_arrays = {key: np.stack([row[key] for row in risks]) for key in risks[0]}
    risk_arrays.update({"sample_ids": np.asarray(ids), "field_names": np.asarray(sample.field_names)})
    io_started = time.perf_counter()
    np.savez_compressed(output_dir / "field_risk.npz", **risk_arrays)
    io_seconds += time.perf_counter() - io_started
    meta = {
        "schema": "native_coherence_inference_v1", "query_policy": "full_native",
        "sampler": sampler, "sensor_protocol": protocol.to_dict(),
        "sensor_manifest_sha256": None if sensor_manifest is None else sensor_manifest.digest(),
        "field_names": list(sample.field_names),
        "field_units": list(getattr(dataset.data_spec, "field_units", ("unknown",) * len(sample.field_names))),
        "normalizer_sha256": dataset.normalizer.digest(),
        "dataset_split": dataset.split_name,
        "conditions": rows,
        "checkpoint_path": str(runtime.checkpoint_path),
        "checkpoint_sha256": file_sha256(runtime.checkpoint_path),
        "wall_seconds": time.perf_counter() - started,
        "peak_cpu_maxrss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
    }
    if consumer is not None:
        meta["stream_timing"] = {"consumer_seconds": callback_seconds, "payload_and_risk_write_seconds": io_seconds,
                                 "wall_includes_consumer": True}
    manifest_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    return manifest_path


def require_matching_native_panels(left, right):
    """Reject comparisons whose support, targets, observations or sampler differ."""
    def read(value):
        return json.loads(Path(value).read_text(encoding="utf-8")) if not isinstance(value, dict) else value
    left, right = read(left), read(right)
    for key in ("schema", "query_policy", "sampler", "sensor_protocol", "sensor_manifest_sha256",
                "field_names", "field_units", "normalizer_sha256", "dataset_split"):
        if left.get(key) != right.get(key):
            raise ValueError(f"native comparison contract differs: {key}")
    if len(left["conditions"]) != len(right["conditions"]):
        raise ValueError("native comparison condition count differs")
    for first, second in zip(left["conditions"], right["conditions"]):
        for key in ("sample_id", "dataset_index", "contract_sha256", "point_count", "valid_count", "physical_measure"):
            if first[key] != second[key]:
                raise ValueError(f"native comparison condition contract differs: {key}")

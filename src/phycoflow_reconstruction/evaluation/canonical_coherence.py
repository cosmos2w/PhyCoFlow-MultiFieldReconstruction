"""Score saved draws against one already-frozen, common coherence yardstick.

This module never restores checkpoints, draws samples, or fits artifacts.  A caller
constructs the TRAIN-fitted family bundle and the 4096-query panel once, then
uses this same evaluator for every arm. Model/checkpoint identity is recorded but
is deliberately excluded from the common comparison fingerprint.
"""

from __future__ import annotations

import hashlib
import json
import math
import resource
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _hash_value(digest, value: Any) -> None:
    """Hash typed, nested artifact values, including tensor contents and shape."""
    def token(data: bytes) -> None:
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)

    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().contiguous()
        token(b"tensor")
        token(str(array.dtype).encode())
        token(str(tuple(array.shape)).encode())
        token(array.reshape(-1).view(torch.uint8).numpy().tobytes())
    elif isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError("object arrays are not allowed in frozen identities")
        array = np.ascontiguousarray(value)
        token(b"ndarray")
        token(array.dtype.str.encode())
        token(str(array.shape).encode())
        token(array.tobytes())
    elif isinstance(value, Mapping):
        token(b"mapping")
        for key in sorted(value, key=str):
            if not isinstance(key, str):
                raise TypeError("frozen identity mapping keys must be strings")
            token(key.encode())
            _hash_value(digest, value[key])
        token(b"end")
    elif isinstance(value, (list, tuple)):
        token(b"sequence")
        for item in value:
            _hash_value(digest, item)
        token(b"end")
    elif isinstance(value, np.generic):
        _hash_value(digest, value.item())
    elif value is None or isinstance(value, (bool, int, float, str)):
        token(json.dumps(value, allow_nan=False, ensure_ascii=False).encode())
    elif isinstance(value, Path):
        _hash_value(digest, str(value))
    else:
        raise TypeError(f"unsupported frozen identity value: {type(value).__name__}")


def content_sha256(value: Any) -> str:
    digest = hashlib.sha256()
    _hash_value(digest, value)
    return digest.hexdigest()


def _batch_identity(batch: Any) -> dict[str, Any]:
    names = ("sample_ids", "query_coords", "query_valid_mask", "target_fields", "obs_coords",
             "obs_values", "obs_field_ids", "obs_valid_mask", "obs_indices", "logical_shapes", "metadata")
    return {name: getattr(batch, name, None) for name in names}


def _artifact(family: Any) -> dict[str, Any]:
    method = getattr(family, "state_artifact", None)
    if not callable(method):
        raise TypeError("canonical families must expose state_artifact()")
    artifact = method()
    if not isinstance(artifact, dict):
        raise TypeError("state_artifact() must return a dict")
    if getattr(family, "target_use", None) != "paired_supervised":
        raise ValueError("canonical evaluator requires paired supervised family references")
    name = getattr(family, "family_name", "")
    if name == "cross_spectrum" and (not family.eigenvectors.numel() or not family.geometry_sha256):
        raise ValueError("canonical cross-spectrum graph must be frozen before evaluation")
    if name == "multiscale_spectrum" and (
        not family.fitted or family.calibration_record.get("split") != "train"
    ):
        raise ValueError("canonical spectral geometry and scales require explicit TRAIN provenance")
    if name == "topology":
        if not family.neighbor_indices.numel() or not family.geometry_sha256:
            raise ValueError("canonical topology raster must be frozen before evaluation")
        if str(family.version) == "4" and not family.spatial_objective.calibrated:
            raise ValueError("canonical topology leaf scales must be fitted on TRAIN")
    return artifact


def _tensor(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().to(torch.float64).numpy()


def _json_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return _json_value(value.detach().cpu().numpy())
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, Mapping):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    return str(value)


def _require_sampler(sampler: Mapping[str, Any], draws: int) -> None:
    required = {"path", "steps", "solver", "observation_config", "execution", "effective_weights", "draw_seeds"}
    if not required.issubset(sampler):
        raise ValueError(f"canonical sampler is missing {sorted(required - set(sampler))}")
    if sampler["effective_weights"] != "live":
        raise ValueError("canonical comparison requires declared live weights")
    seeds = sampler["draw_seeds"]
    if not isinstance(seeds, (tuple, list)) or len(seeds) != draws or len(set(seeds)) != draws:
        raise ValueError("canonical sampler requires one distinct recorded seed per draw")
    if not sampler["solver"] or int(sampler["steps"]) <= 0:
        raise ValueError("canonical sampler requires effective solver and positive steps")


class FrozenCanonicalEvaluator:
    """One immutable collection of fitted families and a declared evaluation policy.

    ``families`` is keyed by unique names such as legacy_state/next_state.
    ``modes`` maps each key to ``paired``, ``ensemble_group``, or ``conditional``.
    Paired scores are retained per draw/condition; ensemble-group scores use
    every fixed 32-condition group separately per draw; conditional scores use
    all four draws jointly. No family or grouping is silently dropped.
    """

    def __init__(self, families: Mapping[str, Any], modes: Mapping[str, str], *,
                 field_names: tuple[str, ...], query_seed: int = 100047,
                 query_points: int = 4096, draws: int = 4, group_size: int = 32):
        if not families or set(families) != set(modes):
            raise ValueError("families and modes must have the same nonempty keys")
        if not field_names or len(set(field_names)) != len(field_names):
            raise ValueError("canonical field names must be nonempty and unique")
        if any(mode not in {"paired", "ensemble_group", "conditional"} for mode in modes.values()):
            raise ValueError("unsupported canonical family aggregation")
        if query_points < 2 or draws < 2 or group_size < 1:
            raise ValueError("invalid canonical query, draw or group count")
        self.families, self.modes = dict(families), dict(modes)
        self.field_names = tuple(field_names)
        self.query_seed, self.query_points, self.draws, self.group_size = (
            int(query_seed), int(query_points), int(draws), int(group_size))
        self.artifact_hashes = {name: content_sha256(_artifact(family))
                                for name, family in self.families.items()}
        self.contract_hash: str | None = None

    def _verify_artifacts(self) -> None:
        for name, family in self.families.items():
            if content_sha256(_artifact(family)) != self.artifact_hashes[name]:
                raise ValueError(f"canonical family artifact changed: {name}")

    @torch.no_grad()
    def evaluate(self, members: torch.Tensor, batch: Any, *, normalizer: Any,
                 sampler: Mapping[str, Any], model_identity: Mapping[str, Any],
                 reference_ids: tuple[str, ...] | list[str] | None = None,
                 inference_batch: Any | None = None, query_seed: int = 100047):
        """Return numeric arrays and JSON-ready provenance for a single saved ensemble."""
        device = members.device if isinstance(members, torch.Tensor) else torch.device("cpu")
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        wall, cpu = time.perf_counter(), time.process_time()
        self._verify_artifacts()
        if not isinstance(members, torch.Tensor) or members.ndim != 4:
            raise ValueError("members must have shape [J,B,N,C]")
        target = batch.target_fields
        if target is None or members.shape[1:] != target.shape:
            raise ValueError("members and paired reference must align as [J,B,N,C] and [B,N,C]")
        count, conditions, points, channels = members.shape
        if (count, points, channels) != (self.draws, self.query_points, len(self.field_names)):
            raise ValueError("canonical draw/query/field shape differs from the frozen contract")
        if not torch.isfinite(members).all() or not torch.isfinite(target).all():
            raise ValueError("canonical members and references must be finite")
        if len(batch.sample_ids) != conditions or len(set(batch.sample_ids)) != conditions:
            raise ValueError("canonical sample IDs must be unique and aligned")
        if batch.query_coords.shape[:2] != (conditions, points) or batch.query_valid_mask.shape != (conditions, points):
            raise ValueError("canonical coordinates and masks must align with the panel")
        if not bool(batch.query_valid_mask.all()):
            raise ValueError("canonical family scoring requires the panel to contain only valid query points")
        if int(query_seed) != self.query_seed:
            raise ValueError("canonical query seed differs from the frozen contract")
        reference_ids = tuple(batch.sample_ids if reference_ids is None else reference_ids)
        if reference_ids != tuple(batch.sample_ids):
            raise ValueError("paired reference IDs/order must match the panel")
        if not model_identity:
            raise ValueError("model/checkpoint identity is required for provenance")
        _require_sampler(sampler, count)
        if "ensemble_group" in self.modes.values() and conditions % self.group_size:
            raise ValueError("legacy ensemble metric requires complete fixed-size condition groups")
        normalizer_identity = {"method": getattr(normalizer, "method", None),
                               "offset": normalizer.offset, "scale": normalizer.scale}
        contract = {"schema_version": 1, "field_names": self.field_names,
                    "query_seed": self.query_seed, "query_points": self.query_points,
                    "draws": self.draws, "group_size": self.group_size, "modes": self.modes,
                    "families": self.artifact_hashes, "panel": _batch_identity(batch),
                    "inference_panel": _batch_identity(inference_batch or batch),
                    "reference_ids": reference_ids, "normalizer": normalizer_identity,
                    "sampler": dict(sampler)}
        fingerprint = content_sha256(contract)
        if self.contract_hash is None:
            self.contract_hash = fingerprint
        elif self.contract_hash != fingerprint:
            raise ValueError("canonical panel, sampler, normalizer, or frozen artifacts differ")

        mask = batch.query_valid_mask.to(members.dtype)
        denominator = mask.sum(dim=1, keepdim=True)
        per_member_mse = ((members - target[None]).square() * mask[None, :, :, None]).sum(dim=2) / denominator[None]
        mean_mse = ((members.mean(dim=0) - target).square() * mask[:, :, None]).sum(dim=1) / denominator
        arrays = {"field_mse_per_member": _tensor(per_member_mse),
                  "field_mse_ensemble_mean": _tensor(mean_mse)}
        family_report = {}
        coordinates = batch.query_coords
        for name, family in self.families.items():
            mode = self.modes[name]
            results = []
            if mode == "conditional":
                results.append(family(members[0], target, coordinates=coordinates,
                    context={"sample_ids": tuple(batch.sample_ids), "reference_ids": reference_ids,
                             "conditional_draws": tuple(members[1:]), "score_all_conditional_draws": True}))
            elif mode == "paired":
                for member in members:
                    results.append(family(member, target, coordinates=coordinates,
                        context={"sample_ids": tuple(batch.sample_ids), "reference_ids": reference_ids}))
            else:
                for member in members:
                    for start in range(0, conditions, self.group_size):
                        stop = start + self.group_size
                        results.append(family(member[start:stop], target[start:stop],
                            coordinates=coordinates[start:stop],
                            context={"sample_ids": tuple(batch.sample_ids[start:stop]),
                                     "reference_ids": reference_ids[start:stop]}))
            if not results:
                raise RuntimeError(f"canonical family {name} produced no results")
            keys = tuple(results[0].component_results)
            if any(tuple(result.component_results) != keys for result in results):
                raise ValueError(f"canonical family {name} changed its component leaves")
            prefix = f"family/{name}"
            if mode == "ensemble_group":
                shape = (count, conditions // self.group_size)
                arrays[f"{prefix}/total_by_draw_group"] = np.asarray([float(r.scalar_loss.detach().cpu()) for r in results]).reshape(shape)
                for key in keys:
                    arrays[f"{prefix}/leaf/{key}"] = np.asarray([
                        float(r.component_results[key].scalar_loss.detach().cpu()) for r in results]).reshape(shape)
            else:
                if any(result.per_sample_cost is None for result in results):
                    raise ValueError(f"canonical {mode} family {name} must expose per-condition costs")
                totals = np.stack([_tensor(r.per_sample_cost) for r in results])
                arrays[f"{prefix}/total_by_draw_condition" if mode == "paired" else f"{prefix}/total_by_condition"] = totals if mode == "paired" else totals[0]
                for key in keys:
                    leaves = [r.component_results[key].per_sample_cost for r in results]
                    if any(leaf is None for leaf in leaves):
                        raise ValueError(f"canonical family {name} leaf {key} lacks per-condition costs")
                    values = np.stack([_tensor(leaf) for leaf in leaves])
                    arrays[f"{prefix}/leaf/{key}"] = values if mode == "paired" else values[0]
            family_report[name] = {"mode": mode, "artifact_sha256": self.artifact_hashes[name],
                                   "total_mean": float(np.mean([float(r.scalar_loss.detach().cpu()) for r in results])),
                                   "result_diagnostics": [_json_value(r.diagnostics) for r in results],
                                   "leaf_diagnostics": [{key: _json_value(r.component_results[key].diagnostics) for key in keys} for r in results]}
        self._verify_artifacts()
        for key, value in arrays.items():
            if not np.isfinite(value).all():
                raise FloatingPointError(f"canonical score contains non-finite values: {key}")
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        report = {"schema_version": 1, "comparison_fingerprint": fingerprint,
                  "family_artifact_sha256": self.artifact_hashes, "families": family_report,
                  "sample_ids": list(batch.sample_ids), "reference_ids": list(reference_ids),
                  "field_names": list(self.field_names), "query_seed": self.query_seed,
                  "query_points": self.query_points, "draws": count, "group_size": self.group_size,
                  "condition_groups": [list(batch.sample_ids[i:i+self.group_size])
                                       for i in range(0, conditions, self.group_size)] if "ensemble_group" in self.modes.values() else [],
                  "sampler": _json_value(sampler), "model_identity": _json_value(model_identity),
                  "members_sha256": content_sha256(members), "panel_sha256": content_sha256(_batch_identity(batch)),
                  "field_mse_measure": "equal valid query nodes in model units",
                  "timing": {"evaluation_wall_seconds": time.perf_counter() - wall,
                             "evaluation_process_cpu_seconds": time.process_time() - cpu},
                  "resources": {"process_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                                "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
                                "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None}}
        return arrays, report


def assert_comparable(*reports: Mapping[str, Any]) -> None:
    if not reports:
        raise ValueError("at least one canonical report is required")
    fingerprint = reports[0].get("comparison_fingerprint")
    if not fingerprint or any(report.get("comparison_fingerprint") != fingerprint for report in reports[1:]):
        raise ValueError("canonical reports have incompatible frozen comparison contracts")


def save_canonical_score(output: str | Path, arrays: Mapping[str, np.ndarray], report: Mapping[str, Any]) -> tuple[Path, Path]:
    """Persist reusable numerical leaves and a JSON manifest without model inference."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    payload = output / "canonical_scores.npz"
    np.savez_compressed(payload, **arrays)
    result = dict(report)
    result["payload"] = payload.name
    result["payload_sha256"] = hashlib.sha256(payload.read_bytes()).hexdigest()
    result["payload_persistence_wall_seconds"] = time.perf_counter() - started
    manifest = output / "canonical_scores.json"
    manifest.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return payload, manifest

"""Matched, loss-resolution Round 6 evaluation with resumable condition shards.

The caller owns model and frozen family loading. No calibration or training is done here.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import torch

from ..data.physical_measure import batch_measure_context
from ..data.training_batches import fixed_query_indices
from ..training.adaptive_coverage import AdaptiveCoverageSampler
from ..training.coherence_sampling import (
    _conditional_draw_count,
    _conditional_draw_seed,
    _family_context,
)
from ..training.fidelity_budget import field_error_sums
from ..training.post_training import _without_target
from ..training.rollout import differentiable_reconstruction

_COUNTS = {"train": 800, "validation": 100, "test": 100}
_PROVENANCE = ("observer_sha256", "normalizer_sha256", "artifacts_sha256", "support_sha256")


def _digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def model_state_digest(model):
    """SHA-256 of named model tensors, distinct from any checkpoint file hash."""
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        if not isinstance(value, torch.Tensor):
            raise TypeError("model state identity requires tensor-only state_dict")
        tensor = value.detach().cpu().contiguous()
        digest.update(
            json.dumps(
                [name, str(tensor.dtype), list(tensor.shape)], separators=(",", ":")
            ).encode()
        )
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _tensor_digest(value):
    if value is None:
        return None
    value = value.detach().cpu().contiguous()
    digest = hashlib.sha256(f"{value.dtype}:{tuple(value.shape)}:".encode())
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _json_value(value):
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def _batch_identity(batch):
    if len(batch.sample_ids) != 1:
        raise ValueError("evaluation requires one condition per batch")
    if batch.target_fields is None or batch.query_coords.shape[1] != 4096:
        raise ValueError("evaluation requires paired targets on Q4096 support")
    ids = batch.metadata.get("query_indices")
    if not isinstance(ids, torch.Tensor) or ids.shape != (1, 4096):
        raise ValueError("Q4096 original query indices are required")
    if not bool(batch.query_valid_mask.all()):
        raise ValueError("Q4096 support must be completely valid")
    return {
        "sample_id": str(batch.sample_ids[0]),
        "query_sha256": _tensor_digest(ids),
        "coordinates_sha256": _tensor_digest(batch.query_coords),
        "target_sha256": _tensor_digest(batch.target_fields),
        "observation_sha256": _digest(
            [
                _tensor_digest(x)
                for x in (
                    batch.obs_coords,
                    batch.obs_values,
                    batch.obs_field_ids,
                    batch.obs_valid_mask,
                    batch.obs_indices,
                )
            ]
        ),
        "measure_sha256": _tensor_digest(batch_measure_context(batch).get("generated_weights")),
    }


def _selection(bank, sample_id, seed):
    token = _digest([int(seed), sample_id])
    return AdaptiveCoverageSampler(bank, policy="uniform", seed=int(token[:16], 16)).select(4, 12)


def freeze_manifest(
    split_batches, query_bank, *, provenance, draw_seed, expected_sample_ids, shape_seed=100045
):
    """Freeze condition, observation, support, draw and sampled-query identities.

    Input batches are already at the fixed shared support. Verify its index set against
    Q4096 from seed 100045 before freezing, including when data loaders pre-subset it.
    """
    if set(split_batches) != set(_COUNTS) or set(expected_sample_ids) != set(_COUNTS):
        raise ValueError("expected train, validation and test cohorts")
    if any(
        not isinstance(provenance.get(k), str)
        or len(provenance[k]) != 64
        or any(c not in "0123456789abcdef" for c in provenance[k])
        for k in _PROVENANCE
    ):
        raise ValueError("all frozen provenance hashes must be lowercase SHA-256")
    splits = {}
    for split, count in _COUNTS.items():
        rows = []
        for batch in split_batches[split]:
            identity = _batch_identity(batch)
            native_count = math.prod(batch.logical_shapes[0])
            expected = fixed_query_indices(native_count, 4096, seed=100045)
            if not torch.equal(batch.metadata["query_indices"][0].cpu(), expected):
                raise ValueError("query support differs from fixed Q4096 seed 100045")
            identity["draw_seed"] = int(
                _digest([int(draw_seed), split, identity["sample_id"]])[:15], 16
            )
            identity["query_selection"] = _json_value(
                asdict(_selection(query_bank, identity["sample_id"], shape_seed))
            )
            rows.append(identity)
        if len(rows) != count or len({r["sample_id"] for r in rows}) != count:
            raise ValueError(f"{split} needs exactly {count} distinct conditions")
        if {r["sample_id"] for r in rows} != set(expected_sample_ids[split]) or len(
            expected_sample_ids[split]
        ) != count:
            raise ValueError(f"{split} condition IDs differ from supplied immutable cohort")
        splits[split] = rows
    manifest = {
        "version": "round6_loss_aligned_v1",
        "provenance": dict(provenance),
        "query_bank": query_bank.manifest(),
        "query_seed": 100045,
        "shape_seed": int(shape_seed),
        "draw_count": 2,
        "splits": splits,
    }
    manifest["sha256"] = _digest(manifest)
    return manifest


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
    os.replace(temp, path)


def _number(value):
    value = float(value.detach().cpu()) if isinstance(value, torch.Tensor) else float(value)
    if not math.isfinite(value):
        raise ValueError("nonfinite loss-aligned result")
    return value


def _collect_family_result(name, result, losses, component_diagnostics, query_leaf_diagnostics):
    """Keep selected persistence leaves separate from stable training losses."""
    losses[f"{name}.total"] = _number(result.scalar_loss)
    for key, term in result.component_results.items():
        if not term.available:
            continue
        details = {
            k: _number(v)
            for k, v in term.diagnostics.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v))
        }
        if name == "topology" and key not in {
            "topology.self.persistence",
            "topology.mutual.persistence",
        }:
            query_leaf_diagnostics[key] = {"value": _number(term.scalar_loss), **details}
        else:
            losses[key] = _number(term.scalar_loss)
            component_diagnostics[key] = details


def _evaluate_split(
    *,
    model,
    batches,
    families,
    config,
    manifest,
    split,
    checkpoint_sha256,
    output_dir,
    query_bank,
    family_scales=None,
    weight_identity_kind="checkpoint_file_sha256",
    scale_only=False,
):
    """Evaluate and persist one matched split; reuse only identity-matched shards.

    Batches must yield each frozen condition exactly once. This function creates the
    two endpoint draws once per condition and uses them for every family and fidelity.
    """
    expected_hash = manifest.get("sha256")
    if expected_hash != _digest({k: v for k, v in manifest.items() if k != "sha256"}):
        raise ValueError("manifest hash mismatch")
    if split not in _COUNTS or len(manifest["splits"][split]) != _COUNTS[split]:
        raise ValueError("invalid split coverage")
    if query_bank.manifest() != manifest["query_bank"]:
        raise ValueError("query bank mismatch")
    if weight_identity_kind not in {"checkpoint_file_sha256", "model_state_sha256"}:
        raise ValueError("unknown evaluated weight identity kind")
    if len(checkpoint_sha256) != 64 or any(c not in "0123456789abcdef" for c in checkpoint_sha256):
        raise ValueError("checkpoint SHA-256 required")
    expected_families = ({"multiscale_spectrum"} if scale_only else
                         {"global_distribution", "multiscale_spectrum", "topology"})
    if _conditional_draw_count(families) != 2 or set(families) != expected_families:
        raise ValueError("expected three retained families and J2")
    if any(f.target_use != "paired_supervised" for f in families.values()):
        raise ValueError("loss-aligned family references must be paired")
    records = {item["sample_id"]: item for item in manifest["splits"][split]}
    if len(records) != _COUNTS[split]:
        raise ValueError("duplicate manifest condition")
    signature_inputs = {
            "manifest": expected_hash,
            "checkpoint": checkpoint_sha256,
            "config": config,
            "family_scales": family_scales or {},
            "split": split,
            "weight_identity_kind": weight_identity_kind,
            "reduction_schema": 2,
        }
    if scale_only:
        signature_inputs["active_families"] = ["multiscale_spectrum"]
    signature = _digest(signature_inputs)
    root = Path(output_dir)
    seen, rows = set(), []
    started = perf_counter()
    model.eval()
    with torch.no_grad():
        for batch in batches:
            identity = _batch_identity(batch)
            sample_id = identity["sample_id"]
            if sample_id not in records or sample_id in seen:
                raise ValueError("unexpected or repeated condition")
            seen.add(sample_id)
            frozen = records[sample_id]
            if any(frozen.get(k) != v for k, v in identity.items()):
                raise ValueError(f"condition identity mismatch: {sample_id}")
            selection = _selection(query_bank, sample_id, manifest["shape_seed"])
            if _json_value(asdict(selection)) != frozen["query_selection"]:
                raise ValueError("query selection identity mismatch")
            path = root / "shards" / f"{_digest(sample_id)}.json"
            if path.exists():
                row = json.loads(path.read_text())
                if row.get("signature") != signature or row.get("sample_id") != sample_id:
                    raise ValueError("incompatible cached condition shard")
                rows.append(row)
                continue
            tick = perf_counter()
            inference_batch = _without_target(batch)
            predictions = []
            for draw in range(2):
                seed = (
                    frozen["draw_seed"]
                    if draw == 0
                    else _conditional_draw_seed(frozen["draw_seed"], draw)
                )
                generator = torch.Generator(device=batch.query_coords.device).manual_seed(seed)
                predictions.append(
                    differentiable_reconstruction(
                        model,
                        inference_batch,
                        steps=int(config["rollout"]["steps"]),
                        solver=config["rollout"]["solver"],
                        generator=generator,
                        observation_config=dict(config["observation_consistency"]),
                    )
                )
            if model.capabilities.structured_grid_required:
                # Structured models must be supplied complete input and a separate
                # support batch; this API intentionally refuses ambiguous inputs.
                raise ValueError("structured-grid model requires a complete input adapter")
            losses, diagnostics, component_diagnostics, query_leaf_diagnostics = {}, {}, {}, {}
            base = {
                "sample_ids": batch.sample_ids,
                "reference_ids": batch.sample_ids,
                **batch_measure_context(batch),
            }
            for name, family in families.items():
                context = _family_context(family, base, predictions[1:])
                if name == "topology":
                    context["query_selection"] = selection
                result = family(
                    predictions[0],
                    batch.target_fields,
                    coordinates=batch.query_coords,
                    context=context,
                )
                _collect_family_result(
                    name, result, losses, component_diagnostics, query_leaf_diagnostics
                )
                diagnostics[name] = {
                    "weight": float(getattr(family, "family_weight", 1.0)),
                    "calibration_scale": float((family_scales or {}).get(name, 1.0)),
                }
            sums, counts = field_error_sums(
                predictions[0], batch.target_fields, batch.query_valid_mask
            )
            if bool((counts <= 0).any()):
                raise ValueError("empty fidelity support")
            row = {
                "signature": signature,
                "sample_id": sample_id,
                "losses": losses,
                "field_error_sums": sums.cpu().tolist(),
                "field_error_counts": counts.cpu().tolist(),
                "family_weights": diagnostics,
                "component_diagnostics": component_diagnostics,
                "query_leaf_diagnostics": query_leaf_diagnostics,
                "query_selection": frozen["query_selection"],
                "elapsed_seconds": perf_counter() - tick,
            }
            _write_json(path, row)
            rows.append(row)
    if seen != set(records):
        raise ValueError(f"incomplete split: missing {len(set(records) - seen)} conditions")
    ordered = sorted(rows, key=lambda r: r["sample_id"])
    _write_json(
        root / "reductions.json",
        {
            "version": "round6_loss_aligned_v1",
            "split": split,
            "manifest_sha256": expected_hash,
            "weight_sha256": checkpoint_sha256,
            "weight_identity_kind": weight_identity_kind,
            "signature": signature,
            "provenance": manifest["provenance"],
            "rows": ordered,
            "timing": {
                "invocation_seconds": perf_counter() - started,
                "condition_seconds": sum(x["elapsed_seconds"] for x in ordered),
            },
        },
    )
    return root / "reductions.json"


def evaluate_split(**kwargs):
    """Evaluate a frozen split while restoring the caller's RNG and model mode."""
    from ..training.execution_strategy.audit import preserve_rng_and_modes

    with preserve_rng_and_modes(kwargs["model"]):
        return _evaluate_split(**kwargs)


def legacy_metrics(reductions_path, field_names):
    """Translate saved reductions to the trainer callback's reporting contract."""
    payload = json.loads(Path(reductions_path).read_text())
    rows = payload["rows"]
    if not rows or len(field_names) != len(rows[0]["field_error_sums"]):
        raise ValueError("field names do not match reductions")
    risks = {
        name: sum(r["field_error_sums"][i] for r in rows)
        / sum(r["field_error_counts"][i] for r in rows)
        for i, name in enumerate(field_names)
    }
    keys = set.intersection(*(set(r["losses"]) for r in rows))
    means = {key: sum(r["losses"][key] for r in rows) / len(rows) for key in keys}
    families = {}
    for name in ("global_distribution", "multiscale_spectrum", "topology"):
        if f"{name}.total" not in means:
            raise ValueError("missing family total")
        weights = rows[0]["family_weights"][name]
        components = {
            key: value
            for key, value in means.items()
            if key.startswith(name + ".") and key != f"{name}.total"
        }
        families[name] = {
            "total": means[f"{name}.total"],
            "component_scalars": components,
            "components": components,
            **weights,
        }
    return {
        "mse_normalized": sum(risks.values()) / len(risks),
        "per_field_mse_normalized": risks,
        "endpoint_mse_model_units": risks,
        "coherence": {"families": families},
        "loss_aligned_reductions": str(reductions_path),
    }


def make_batches(dataset, positions, config, seed_base, device):
    """Yield one fixed-observation, Q4096 batch per selected dataset position.

    Uses the ordinary observation assembler directly, with no loader workers.
    Observation seeds depend on physical sample identity, not iteration order.
    """
    from dataclasses import replace

    from ..data.sensor_protocols import build_observation_batch
    from ..training.common import sensor_protocol_from_config

    protocol = sensor_protocol_from_config(config)
    for position in positions:
        sample = dataset[int(position)]
        if sample.valid_points is not None and not bool(sample.valid_points.all()):
            raise ValueError("Q4096 evaluator requires compact all-valid nodes")
        sample_id = f"{sample.trajectory_id}:{sample.time_index if sample.time_index is not None else 'all'}"
        seed = int(_digest([int(seed_base), sample_id])[:15], 16)
        query = fixed_query_indices(sample.values.shape[0], 4096, seed=100045)
        if query.numel() != 4096:
            raise ValueError("dataset has fewer than 4096 support points")
        yield build_observation_batch(
            [sample], replace(protocol, seed=seed), query_indices=query
        ).to(device)

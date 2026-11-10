"""Frozen TRAIN calibration and conditional-draw contracts for coherence families."""

from __future__ import annotations

import hashlib
import math
import random
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import replace
from typing import Any

import numpy as np
import torch

from ..data.factory import FieldDataset
from ..data.sensor_protocols import build_observation_batch
from ..data.training_batches import fixed_query_indices
from .common import sensor_protocol_from_config
from .rollout import differentiable_reconstruction


def _coherence_query_seed(config: Mapping[str, Any]) -> int:
    """Seed of the fixed shared coherence queries, as used by every batch builder."""
    compute = config["coherence"]["compute_budget"]
    default = config["observations"].get("seed", config["runtime"].get("seed", 42)) + 100_003
    return int(compute.get("query_seed", default))


def _conditional_draw_count(families: Mapping[str, Any]) -> int:
    return max(
        (int(getattr(family, "conditional_draws", 1)) for family in families.values()), default=1
    )


def _conditional_draw_seed(base_seed: int, draw: int) -> int:
    # A golden-ratio stride keeps extra-draw seeds clear of the per-step rollout seeds.
    return (int(base_seed) + int(draw) * 0x9E3779B97F4A7C15) % (1 << 63)


def _family_context(
    family,
    context: Mapping[str, Any],
    conditional_draws: list[torch.Tensor],
    *,
    report_detail: bool = False,
) -> dict[str, Any]:
    """Add the extra conditional draws and report requests a family declares it uses."""
    payload = dict(context)
    needed = int(getattr(family, "conditional_draws", 1))
    if needed > 1:
        payload["conditional_draws"] = tuple(conditional_draws[: needed - 1])
    if report_detail and getattr(family, "supports_report_detail", False):
        payload["report_detail"] = True
        if getattr(family, "supports_ensemble_report", False):
            # Every available independent draw, for per-band ensemble diagnostics only.
            payload["ensemble_draws"] = tuple(conditional_draws)
    return payload


def _fit_training_references(
    families: Mapping[str, Any],
    train_dataset: FieldDataset,
    config: Mapping[str, Any],
    *,
    device: torch.device,
    structured_grid_required: bool,
) -> None:
    """Freeze training-only geometry and reference scales before any model evaluation."""
    total = math.prod(train_dataset.data_spec.logical_shape)
    point_count = min(int(config["coherence"]["compute_budget"]["point_count"]), total)
    frozen = config["coherence"].get("frozen_family_artifacts", {})
    if frozen and (getattr(train_dataset, "split_name", None) != "train"
                   or getattr(getattr(train_dataset, "selection", None), "split", None) != "train"):
        raise ValueError("frozen coherence artifacts require the live TRAIN split")
    query_indices = fixed_query_indices(total, point_count, seed=_coherence_query_seed(config))
    if any(callable(getattr(family, "fit_training_reference", None)) for family in families.values()):
        first = train_dataset[0]
        if first.valid_points is not None and not bool(first.valid_points.all()):
            raise ValueError("coherence calibration requires compact all-valid nodes; full-grid invalid-point masks are unsupported")
    for name, family in families.items():
        if name in frozen:
            from ..data.manifest import dataset_fingerprint
            from .run_store import file_sha256
            declaration = frozen[name]
            if file_sha256(declaration["path"]) != declaration["sha256"]:
                raise ValueError(f"frozen TRAIN family artifact hash mismatch: {name}")
            artifact = torch.load(declaration["path"], map_location=device, weights_only=True)
            binding = artifact.get("frozen_training_context")
            expected = {
                "dataset_fingerprint": dataset_fingerprint(train_dataset.path),
                "query_indices_sha256": hashlib.sha256(query_indices.numpy().tobytes()).hexdigest(),
                "normalizer_digest": train_dataset.normalizer.digest(),
                "field_names": list(train_dataset.data_spec.field_names),
                "split": "train",
            }
            if not isinstance(binding, Mapping) or any(binding.get(k) != v for k, v in expected.items()):
                raise ValueError(f"frozen TRAIN family data/query/normalizer binding mismatch: {name}")
            native = artifact.get("definition") == "self_rho_native_quadrature_v1"
            base_artifact = artifact["native_reference"] if native else artifact
            for key in ("normalization_offset", "normalization_scale"):
                if not torch.equal(base_artifact["state_dict"][key].cpu(), getattr(family, key).cpu()):
                    raise ValueError(f"frozen TRAIN family normalization mismatch: {name}")
            if (getattr(family, "score_definition", None) == "conditional_quantile_fair_v1"
                    and artifact.get("score_definition") == "marginal_mixed_weighted_v3"):
                family.load_base_artifact(artifact)
            else:
                family.load_state_artifact(artifact)
            if getattr(family, "weighted_measure", False):
                family.bind_measure_artifact(train_dataset.physical_measure_receipt,
                                             weights=train_dataset.physical_weights)
            record = getattr(family, "calibration_record", None)
            if isinstance(record, Mapping):
                if record.get("split") != "train":
                    raise ValueError(f"frozen family lacks TRAIN provenance: {name}")
                digest = hashlib.sha256(query_indices.numpy().tobytes()).hexdigest()
                if not native and record.get("query_indices_sha256") != digest:
                    raise ValueError(f"frozen TRAIN family query identity mismatch: {name}")
                if record.get("dataset_fingerprint") != dataset_fingerprint(train_dataset.path):
                    raise ValueError(f"frozen TRAIN family dataset identity mismatch: {name}")
                family._require_ready(first.values[query_indices][None].to(device),
                                      first.coordinates[query_indices][None].to(device))
            # The family's strict loader owns geometry, measure, units, backend,
            # config and normalization checks. No reference is fitted again.
            continue
        bind_measure = getattr(family, "bind_measure_artifact", None)
        receipt = getattr(train_dataset, "physical_measure_receipt", None)
        if callable(bind_measure) and getattr(family, "weighted_measure", False) and receipt is not None:
            bind_measure(receipt, weights=train_dataset.physical_weights)
        fit = getattr(family, "fit_training_reference", None)
        if fit is None:
            continue
        if (
            structured_grid_required
            and getattr(family, "requires_full_grid_calibration", False)
            and point_count < total
        ):
            raise ValueError(
                f"{name} with a structured-grid source needs coherence.compute_budget.point_count "
                f"to cover all {total} grid points"
            )
        query_indices = fixed_query_indices(total, point_count, seed=_coherence_query_seed(config))
        with _preserved_rng(device):
            fit(train_dataset, query_indices=query_indices, device=device)


def _fit_source_calibrations(
    families, model, train_dataset, config, *, device, source_checkpoint_sha256
) -> None:
    """Freeze new descriptor scales on 32 spaced TRAIN/SOURCE endpoints.

    This is distinct from family gradient balancing and the endpoint-MSE budget.
    Reference fitting and this source calibration never advance training streams.
    """
    frozen = config["coherence"].get("frozen_family_artifacts", {})
    selected = [f for name, f in families.items()
                if getattr(f, "needs_source_calibration", False) and name not in frozen]
    for name in frozen:
        family = families[name]
        if getattr(family, "needs_source_calibration", False):
            record = family.spatial_objective.calibration_record
            if record.get("provenance", {}).get("source_checkpoint_sha256") != source_checkpoint_sha256:
                raise ValueError(f"frozen family SOURCE calibration identity mismatch: {name}")
    if not selected:
        return
    if any(getattr(f, "version", None) == "5" and getattr(f, "family_name", None) == "topology" for f in selected):
        raise ValueError("fair Shape requires pre-frozen TRAIN800 transform and SOURCE calibration artifact")
    if train_dataset.split_name != "train":
        raise ValueError("SOURCE descriptor calibration requires TRAIN")
    total = math.prod(train_dataset.data_spec.logical_shape)
    count = min(int(config["coherence"]["compute_budget"]["point_count"]), total)
    query = fixed_query_indices(total, count, seed=_coherence_query_seed(config))
    positions = np.linspace(0, len(train_dataset) - 1, min(32, len(train_dataset))).round().astype(int)
    predictions, references, sample_ids = [], [], []
    was_training = model.training
    coordinates = None
    try:
        model.eval()
        with _preserved_rng(device), torch.no_grad():
            for position, index in enumerate(positions):
                sample = train_dataset[int(index)]
                if sample.valid_points is not None and not bool(sample.valid_points.all()):
                    raise ValueError("SOURCE calibration requires compact all-valid nodes")
                batch = build_observation_batch(
                    [sample], replace(sensor_protocol_from_config(config), seed=700171 + position),
                    query_indices=None if model.capabilities.structured_grid_required else query,
                ).to(device)
                target = batch.target_fields
                batch.target_fields = None
                prediction = differentiable_reconstruction(
                    model, batch, steps=int(config["rollout"]["steps"]),
                    solver=config["rollout"]["solver"],
                    generator=torch.Generator(device=device).manual_seed(700271 + position),
                    observation_config=dict(config["observation_consistency"]),
                )
                if model.capabilities.structured_grid_required:
                    prediction, target = prediction[:, query], target[:, query]
                    coordinate = batch.query_coords[:, query]
                else:
                    coordinate = batch.query_coords
                if coordinates is not None and not torch.equal(coordinates, coordinate):
                    raise ValueError("SOURCE descriptor calibration geometry changed")
                coordinates = coordinate
                predictions.append(prediction.detach())
                references.append(target.detach())
                sample_ids.extend(batch.sample_ids)
            for family in selected:
                family.fit_calibration(
                    torch.cat(predictions), torch.cat(references),
                    coordinates=coordinates.expand(len(sample_ids), -1, -1),
                    provenance={"split": "train", "sample_ids": sample_ids,
                                "source_checkpoint_sha256": source_checkpoint_sha256,
                                "positions": positions.tolist(),
                                "weight_identity": "live_training_weights"},
                )
    finally:
        model.train(was_training)


def _requires_frozen_checkpoint_state(family) -> bool:
    """New score definitions require checkpoint buffers to match their frozen artifact."""
    return (
        (family.family_name == "topology" and family.version == "4")
        or getattr(family, "mean_augmented", False)
        or getattr(family, "score_definition", None) in {
            "marginal_mixed_mean_v2", "marginal_mixed_weighted_v3", "conditional_quantile_fair_v1"
        }
    )


def _load_checkpoint_family_state(family, state, *, source_checkpoint_sha256=None):
    """Never let checkpoint buffers override a validated, frozen new descriptor."""
    finite_primary = family.family_name == "topology" and family.version == "4"
    if _requires_frozen_checkpoint_state(family):
        frozen = family.state_dict()
        if frozen.keys() != state.keys() or any(
            not torch.equal(value.detach().cpu(), state[key].detach().cpu())
            for key, value in frozen.items()
        ):
            raise ValueError("checkpoint family state differs from the frozen family artifact")
        if finite_primary and source_checkpoint_sha256 is not None:
            recorded = family.spatial_objective.calibration_record["provenance"][
                "source_checkpoint_sha256"
            ]
            if recorded != source_checkpoint_sha256:
                raise ValueError("Shape calibration SOURCE checkpoint identity mismatch")
    family.load_state_dict(state, strict=True)


@contextmanager
def _preserved_rng(device: torch.device) -> Iterator[None]:
    """Keep calibration draws out of the training random stream."""
    torch_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if device.type == "cuda" else []
    python_rng = random.getstate()
    numpy_rng = np.random.get_state()
    try:
        yield
    finally:
        torch.set_rng_state(torch_rng)
        if cuda_rng:
            torch.cuda.set_rng_state_all(cuda_rng)
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)

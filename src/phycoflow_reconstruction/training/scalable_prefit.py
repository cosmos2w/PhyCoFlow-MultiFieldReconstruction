"""Build or verify bounded TRAIN/SOURCE scalable-coherence artifacts."""

from __future__ import annotations

import math
from dataclasses import replace
from time import perf_counter

import torch

from ..data.manifest import dataset_fingerprint
from ..data.physical_measure import attach_batch_measure, batch_measure_context
from ..data.sensor_protocols import build_observation_batch
from ..data.training_batches import fixed_query_indices
from .coherence_sampling import _coherence_query_seed, _conditional_draw_seed, _preserved_rng
from .common import sensor_protocol_from_config
from .rollout import differentiable_reconstruction
from .scalable_fitting import _digest, fit_scalable_artifacts, validate_scalable_artifacts


def prepare_scalable_artifacts(
    model, dataset, families, config, *, source_sha256, device, artifact_path
):
    """Fit once from <=32 reproducibly selected ~10% TRAIN conditions; resume verifies identity."""
    if dataset.split_name != "train" or dataset.selection.split != "train":
        raise ValueError("scalable fit requires TRAIN dataset")
    settings = config["coherence"].get("scalable_anchored_coherence", {})
    n = min(int(settings.get("max_samples", 32)), max(1, math.ceil(len(dataset) * 0.1)))
    seed = int(config["runtime"].get("seed", 42)) + 810001
    indexes = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(seed))[
        :n
    ].tolist()
    samples = [dataset[i] for i in indexes]
    query = fixed_query_indices(
        math.prod(dataset.data_spec.logical_shape),
        min(
            int(config["coherence"]["compute_budget"]["point_count"]),
            math.prod(dataset.data_spec.logical_shape),
        ),
        seed=_coherence_query_seed(config),
    )
    protocol = replace(sensor_protocol_from_config(config), seed=seed)
    batch = build_observation_batch(samples, protocol, query_indices=query)
    batch = attach_batch_measure(dataset, batch).to(device)
    if not bool(batch.query_valid_mask.all()):
        raise ValueError("scalable fit requires compact valid query support")
    context = batch_measure_context(batch)
    if not context:
        raise ValueError("scalable fit requires an explicit physical measure")
    coords = batch.query_coords
    weights = context["generated_weights"]
    provenance = {
        "split": "train",
        "field_order": list(dataset.data_spec.field_names),
        "sample_ids": list(batch.sample_ids),
        "source_checkpoint_sha256": source_sha256,
        "normalizer_sha256": dataset.normalizer.digest(),
        "support_sha256": _digest(
            {
                "coordinates": coords,
                "weights": weights,
                "valid_mask": batch.query_valid_mask,
                "query_indices": batch.metadata["query_indices"],
                "measure": context["measure_metadata"],
                "train_targets": batch.target_fields,
                "dataset_fingerprint": dataset_fingerprint(dataset.path),
            }
        ),
        "config_sha256": _digest(
            {
                "coherence": config["coherence"],
                "seed": seed,
                "sample_indices": indexes,
                "rollout": config["rollout"],
                "observations": config["observations"],
                "observation_consistency": config["observation_consistency"],
            }
        ),
        "definition_version": "scalable_anchored_coherence_v1",
    }
    if artifact_path.exists():
        artifacts = torch.load(artifact_path, map_location="cpu", weights_only=True)
        validate_scalable_artifacts(artifacts, provenance)
        return artifacts, provenance, None
    if model.capabilities.structured_grid_required:
        raise ValueError("scalable SOURCE prefit currently requires unstructured query rollout")
    reference = batch.target_fields
    batch.target_fields = None
    began = perf_counter()
    was_training = model.training
    try:
        model.eval()
        with _preserved_rng(device), torch.no_grad():
            draws = [
                differentiable_reconstruction(
                    model,
                    batch,
                    steps=int(config["rollout"]["steps"]),
                    solver=config["rollout"]["solver"],
                    generator=torch.Generator(device=device).manual_seed(
                        _conditional_draw_seed(seed, j)
                    ),
                    observation_config=dict(config["observation_consistency"]),
                ).detach()
                for j in range(2 if model.capabilities.generative else 1)
            ]
        artifacts = fit_scalable_artifacts(
            families=families,
            source=torch.stack(draws, dim=1),
            reference=reference,
            coordinates=coords,
            provenance=provenance,
            source_weights=weights,
            reference_weights=weights,
            seed=seed,
            max_samples=int(settings.get("max_samples", 32)),
            candidate_count=settings.get("candidate_count"),
            shape_batches=int(settings.get("shape_batches", 8)),
            field_floor=float(settings.get("field_floor", 1e-6)),
        )
    finally:
        model.train(was_training)
    return (
        artifacts,
        provenance,
        {"sample_count": n, "source_forward_count": len(draws), "seconds": perf_counter() - began},
    )

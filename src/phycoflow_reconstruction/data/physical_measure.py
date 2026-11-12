"""Case-selected physical measures, carried independently of model inputs."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping

import torch

from ..coherence.measures import measure_fingerprint, rectilinear_nodal_weights


def configure_physical_measure(dataset, policy):
    if policy is None:
        return dataset
    sample = dataset[0]
    if sample.reconstruction_unit != "snapshot":
        raise ValueError("physical measure requires snapshot data")
    if policy == "native_rectilinear_area":
        if sample.valid_points is not None and not bool(sample.valid_points.all()):
            raise ValueError("masked physical measures require explicit native weights")
        weights = rectilinear_nodal_weights(sample.coordinates_raw.double(), sample.logical_shape)
        metadata = {
            "provider": policy,
            "boundary": "outermost_coordinate_rectangle",
            "coordinate_units": "dataset_raw_units_unspecified",
        }
    elif isinstance(policy, Mapping) and policy.get("provider") == "explicit_weights":
        if (
            set(policy) != {"provider", "weights", "receipt"}
            or not isinstance(policy["receipt"], Mapping)
            or not policy["receipt"]
        ):
            raise ValueError("explicit_weights requires weights and a nonempty provider receipt")
        weights = torch.as_tensor(policy["weights"], dtype=torch.float64).detach().cpu().clone()
        if (
            weights.shape != (sample.coordinates_raw.shape[0],)
            or not torch.isfinite(weights).all()
            or (weights < 0).any()
            or not bool(weights.sum() > 0)
        ):
            raise ValueError(
                "explicit physical weights must match support and be finite nonnegative with positive mass"
            )
        if sample.valid_points is not None and bool(
            (weights[~sample.valid_points.cpu()] != 0).any()
        ):
            raise ValueError("explicit physical weights must be zero outside the valid domain")
        metadata = {"provider": "explicit_weights", "provider_receipt": dict(policy["receipt"])}
    else:
        raise ValueError("unsupported dataset.physical_measure")
    metadata.update(
        {
            "coordinates_sha256": hashlib.sha256(
                sample.coordinates_raw.double().contiguous().numpy().tobytes()
            ).hexdigest(),
            "logical_shape": list(sample.logical_shape),
        }
    )
    receipt = measure_fingerprint(weights, mask=sample.valid_points, metadata=metadata)
    dataset.physical_weights = weights
    dataset.physical_measure_receipt = receipt
    return dataset


def attach_sample_measure(dataset, sample):
    weights = getattr(dataset, "physical_weights", None)
    if weights is not None:
        sample.metadata.setdefault("auxiliary", {})["physical_weights"] = weights
        sample.metadata["measure_metadata"] = dataset.physical_measure_receipt
        sample.metadata["physical_measure"] = dataset.physical_measure_receipt["metadata"][
            "provider"
        ]
    return sample


def attach_batch_measure(dataset, batch):
    weights = getattr(dataset, "physical_weights", None)
    if weights is not None:
        ids = batch.metadata["query_indices"]
        batch.metadata.setdefault("sample_context", {}).setdefault("auxiliary", {})[
            "physical_weights"
        ] = weights.to(ids.device)[ids]
        batch.metadata["measure_metadata"] = {
            "native_measure_sha256": dataset.physical_measure_receipt["sha256"],
            "query_policy": "native_nodal_mass_self_normalized",
        }
    return batch


def batch_measure_context(batch):
    weights = batch.metadata.get("sample_context", {}).get("auxiliary", {}).get("physical_weights")
    if weights is None:
        return {}
    return {
        "generated_weights": weights,
        "reference_weights": weights,
        "generated_mask": batch.query_valid_mask,
        "reference_mask": batch.query_valid_mask,
        "measure_metadata": batch.metadata.get("measure_metadata", {}),
    }

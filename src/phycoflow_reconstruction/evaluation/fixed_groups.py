"""Fixed-estimator panel evaluation without changing covariance membership."""

import math
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import fields, replace
from numbers import Real

import torch

_QUERY_WEIGHTED_METRICS = {
    "mse_normalized",
    "per_field_mse_normalized",
    "mse_physical",
    "per_field_mse_physical",
    "endpoint_mse_model_units",
}
_GROUP_ONLY_METRICS = {
    "observed_entry_mse_normalized",
    "unobserved_entry_mse_normalized",
}


def slice_observation_batch(batch, start, stop):
    count = len(batch.sample_ids)
    if not 0 <= start < stop <= count:
        raise ValueError("invalid fixed panel group")

    def sliced(value):
        if isinstance(value, torch.Tensor) and value.ndim and value.shape[0] == count:
            return value[start:stop]
        if isinstance(value, Mapping):
            return {key: sliced(item) for key, item in value.items()}
        return value

    updates = {field.name: sliced(getattr(batch, field.name)) for field in fields(batch)}
    updates["sample_ids"] = batch.sample_ids[start:stop]
    updates["logical_shapes"] = batch.logical_shapes[start:stop] if batch.logical_shapes else ()
    result = replace(batch, **updates)
    result.validate()
    return result


def evaluate_fixed_groups(
    evaluate_one,
    model,
    complete,
    comparison,
    family,
    bank,
    field_names,
    config,
    *,
    normalizer=None,
    family_scales=None,
):
    """Pool equal-size full evaluations and retain every group's raw metrics.

    It never computes one covariance on the concatenated panel. Query-wise MSE
    metrics are pooled by valid-query counts; per-sample metrics are averaged
    across equal-size groups. Group-only observed/unobserved MSEs remain in the
    raw records. Non-numeric identities come from the first group in the legacy
    summary, and all identities remain in ``fixed_estimator_groups``.
    """
    size = config["evaluation"]["group_size"]
    count = len(complete.sample_ids)
    if tuple(comparison.sample_ids) != tuple(complete.sample_ids):
        raise ValueError("complete and comparison batches must contain the same aligned samples")
    if isinstance(size, bool) or not isinstance(size, int) or size < 1 or count % size:
        raise ValueError("fixed panel requires a positive exact group size")
    valid_query_mask = getattr(comparison, "query_valid_mask", None)
    if (
        not isinstance(valid_query_mask, torch.Tensor)
        or valid_query_mask.ndim != 2
        or valid_query_mask.shape[0] != count
        or valid_query_mask.dtype != torch.bool
    ):
        raise ValueError("comparison batch must provide a boolean [sample, query] validity mask")
    valid_query_counts = [
        int(valid_query_mask[start : start + size].sum().item())
        for start in range(0, count, size)
    ]
    if any(value < 1 for value in valid_query_counts):
        raise ValueError("every fixed panel group must contain valid queries")
    groups = []
    for index, start in enumerate(range(0, count, size)):
        group_config = deepcopy(config)
        group_config["evaluation"].pop("group_size")
        group_config["evaluation"]["seed"] = (
            int(config["evaluation"].get("seed", 2027)) + index * 100_003
        )
        groups.append(
            evaluate_one(
                model,
                slice_observation_batch(complete, start, start + size),
                slice_observation_batch(comparison, start, start + size),
                family,
                bank,
                field_names,
                group_config,
                normalizer=normalizer,
                family_scales=family_scales,
            )
        )

    def pooled(values, key=None):
        first = values[0]
        if isinstance(first, Mapping):
            if any(set(value) != set(first) for value in values):
                raise ValueError("fixed group metric schemas differ")
            if key in _QUERY_WEIGHTED_METRICS:
                return {
                    metric: weighted_mean([value[metric] for value in values])
                    for metric in first
                }
            return {
                metric: pooled([value[metric] for value in values], metric)
                for metric in first
                if key is not None or metric not in _GROUP_ONLY_METRICS
            }
        if key in _QUERY_WEIGHTED_METRICS:
            return weighted_mean(values)
        if isinstance(first, Real) and not isinstance(first, bool):
            return sum(values) / len(values)
        return first

    def weighted_mean(values):
        if any(not isinstance(value, Real) or isinstance(value, bool) for value in values):
            raise ValueError("query-weighted group metrics must be finite numeric values")
        if any(not math.isfinite(float(value)) for value in values):
            raise ValueError("query-weighted group metrics must be finite")
        total_weight = sum(valid_query_counts)
        return sum(float(value) * weight for value, weight in zip(values, valid_query_counts)) / total_weight

    result = pooled(groups)
    if "worst_field_relative_l2" in result:
        per_field = result.get("per_field_relative_l2")
        if not isinstance(per_field, Mapping) or not per_field:
            raise ValueError("worst_field_relative_l2 requires pooled per-field relative errors")
        result["worst_field_relative_l2"] = max(float(value) for value in per_field.values())
    if "worst_field_relative_l2_physical" in result:
        per_field = result.get("per_field_relative_l2_physical")
        if not isinstance(per_field, Mapping) or not per_field:
            raise ValueError(
                "worst_field_relative_l2_physical requires pooled per-field relative errors"
            )
        result["worst_field_relative_l2_physical"] = max(
            float(value) for value in per_field.values()
        )
    result["sample_ids"] = list(complete.sample_ids)
    query_indices = comparison.metadata.get("query_indices")
    result["query_ids"] = (
        query_indices.detach().cpu().tolist() if isinstance(query_indices, torch.Tensor) else None
    )
    result["fixed_estimator_groups"] = groups
    result["fixed_estimator_scope"] = {
        "sample_count": count,
        "group_size": size,
        "groups": len(groups),
        "covariance": "mean_of_exact_group_scores",
        "pooling": "query_weighted_mse_and_equal_sample_groups",
    }
    result["inference"]["seconds"] = sum(group["inference"]["seconds"] for group in groups)
    result["inference"]["samples"] = count
    return result

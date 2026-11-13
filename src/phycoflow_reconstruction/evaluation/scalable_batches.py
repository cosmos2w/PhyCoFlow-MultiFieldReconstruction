"""Memory-bounded evaluation of independent per-condition coherence scores."""

from __future__ import annotations

from copy import deepcopy

from .fixed_groups import slice_observation_batch


def evaluate_scalable_batches(
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
    scalable_runtime=None,
):
    """Keep the full selected panel, query pack and sample weighting intact.

    This batches independent scores, not ensemble/covariance groups. A fixed
    batch partition and explicit draw seeds are part of the matched inference
    contract. Uneven final batches retain their actual sample/entry weights.
    """
    count = len(complete.sample_ids)
    size = config["evaluation"]["batch_size"]
    if not count or tuple(comparison.sample_ids) != tuple(complete.sample_ids):
        raise ValueError("evaluation batches require aligned nonempty panels")
    if not bool(comparison.query_valid_mask.all()):
        raise ValueError("scalable evaluation requires valid fixed query support")
    if isinstance(size, bool) or not isinstance(size, int) or size < 1:
        raise ValueError("evaluation.batch_size must be a positive integer")
    child_config = deepcopy(config)
    child_config["evaluation"].pop("batch_size")
    base_seed = int(config["evaluation"].get("seed", 2027))
    groups, sizes, seeds = [], [], []
    for index, start in enumerate(range(0, count, size)):
        stop = min(start + size, count)
        seed = base_seed + index * 100_003
        seeds.append(seed)
        sizes.append(stop - start)
        groups.append(
            evaluate_one(
                model,
                slice_observation_batch(complete, start, stop),
                slice_observation_batch(comparison, start, stop),
                family,
                bank,
                field_names,
                child_config,
                normalizer=normalizer,
                family_scales=family_scales,
                scalable_runtime=scalable_runtime,
                _generation_seed=seed,
            )
        )

    def mean(values, weights=sizes):
        if isinstance(values[0], dict):
            if any(set(v) != set(values[0]) for v in values):
                raise ValueError("evaluation batch metric schemas differ")
            return {k: mean([v[k] for v in values], weights) for k in values[0]}
        total = sum(weights)
        return None if not total else sum(v * w for v, w in zip(values, weights) if w) / total

    result = deepcopy(groups[0])
    for key in groups[0]:
        if key in {
            "coherence",
            "inference",
            "generation_seed",
            "generation_steps",
            "sample_ids",
            "query_ids",
            "native_topology",
            "evaluation_weight_source",
            "evaluation_completed_utc",
            "command_elapsed_seconds",
        }:
            continue
        if key in {"observed_entry_count", "unobserved_entry_count"}:
            result[key] = sum(g[key] for g in groups)
        elif key in {"observed_entry_mse_normalized", "unobserved_entry_mse_normalized"}:
            count_key = key.replace("mse_normalized", "count")
            result[key] = mean([g[key] for g in groups], [g[count_key] for g in groups])
        else:
            result[key] = mean([g[key] for g in groups])
    for suffix in ("", "_physical"):
        values = result.get("per_field_relative_l2" + suffix)
        if values:
            result["worst_field_relative_l2" + suffix] = max(values.values())
    result["coherence"]["total"] = mean([g["coherence"]["total"] for g in groups])
    result["coherence"]["reference_ids"] = list(
        dict.fromkeys(ref for g in groups for ref in g["coherence"]["reference_ids"])
    )
    for name, payload in result["coherence"]["families"].items():
        rows = [g["coherence"]["families"][name] for g in groups]
        for key in ("total", "weighted_total"):
            payload[key] = mean([row[key] for row in rows])
        payload["reference_ids"] = [ref for row in rows for ref in row["reference_ids"]]
        payload["diagnostics"].pop("observer_sha256", None)
        payload["diagnostics"]["observer_sha256s"] = [
            row["diagnostics"]["observer_sha256"] for row in rows
        ]
    for key in ("seconds", "conditional_extra_seconds", "all_member_seconds"):
        result["inference"][key] = sum(g["inference"][key] for g in groups)
    result["inference"]["samples"] = count
    result["sample_ids"] = list(complete.sample_ids)
    result["query_ids"] = [row for g in groups for row in g["query_ids"]]
    result["generation_seed"] = base_seed
    result["evaluation_batching"] = {
        "sample_count": count,
        "batch_size": size,
        "batch_sizes": sizes,
        "generation_seeds": seeds,
        "observer_query_seed": base_seed,
        "pooling": "sample_weighted_independent_scores_and_entry_weighted_observed_errors",
    }
    return result

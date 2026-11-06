"""Compare frozen spectral scores across post-training checkpoints.

Run this module with ``arms --help``. Run-owned recovery scores retain each
run's training geometry; use canonical_coherence for a separately frozen common
cross-arm yardstick. Artifacts are written only to the selected output path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import resource
import time
from pathlib import Path

import numpy as np
import torch

from phycoflow_reconstruction.coherence.families.multiscale_spectrum.features import raw_statistics
from phycoflow_reconstruction.coherence.families.multiscale_spectrum.losses import (
    SCORE_DEFINITION_VERSION,
    almost_fair_repulsion_factor,
    band_coherence,
    conditional_energy_score,
    field_projections,
    joint_spectral_energy_score,
    signed_band_energy_score,
    vector_energy_score,
)
from phycoflow_reconstruction.coherence.families.multiscale_spectrum.reporting import (
    finite_values,
    positive_ratio,
)
from phycoflow_reconstruction.evaluation.ensemble import ensemble_crps
from phycoflow_reconstruction.training.model_lifecycle import selected_evaluation_weight_context
from phycoflow_reconstruction.training.post_training import (
    _conditional_draw_seed,
    _gather_prediction,
    _without_target,
)
from phycoflow_reconstruction.training.posttrain_rescore import restore_evaluation_panel
from phycoflow_reconstruction.training.rollout import differentiable_reconstruction

EVALUATION_SCHEMA_VERSION = 3


def effective_sampler(panel, *, steps=None, seed=None, noise_scale=1.0, weight_selection="configured", draws=None):
    """Describe the endpoint call and resolved lifecycle settings used by draw_members."""
    config, model = panel.config, panel.model
    evaluation = config["evaluation"]
    if weight_selection not in {"configured", "live"}:
        raise ValueError("weight_selection must be configured or live")
    core = getattr(model, "core", model)
    external = bool(config.get("optimization", {}).get("multitask", {}).get("external_policy"))
    flow_hooks = callable(getattr(model, "sample_source", None)) and callable(getattr(model, "velocity", None))
    seed = int(evaluation.get("seed", 2027) if seed is None else seed)
    actual_steps = int(evaluation.get("generation_steps", 2) if steps is None else steps)
    # Native reconstruction receives no override: these are the actual model attributes.
    attributes = (
        "ode_solver", "obs_consistency_mode", "obs_consistency_strength",
        "obs_consistency_sigma", "obs_consistency_schedule_power",
        "obs_consistency_final_clamp", "obs_consistency_chunk_size",
        "reconstruction_execution_mode", "reconstruction_query_chunk_size",
        "reconstruction_cache_level", "gather_query_chunk_size", "query_chunk_size",
        "rollout_context_cache", "rollout_checkpointing",
    )
    execution = {}
    for prefix, obj in (("adapter", model), ("core", core), ("backbone", getattr(core, "model", None))):
        if obj is not None:
            execution[prefix] = {key: getattr(obj, key) for key in attributes
                                 if hasattr(obj, key) and isinstance(getattr(obj, key), (str, int, float, bool, type(None)))}
    ema = getattr(core, "_ema", None)
    ema_eval = bool(getattr(core, "_ema_eval", False))
    if weight_selection == "live":
        weights = "live"
    elif hasattr(core, "_ema_eval"):
        weights = "ema" if ema is not None and ema_eval else "live"
    elif callable(getattr(model, "evaluation_weight_context", None)):
        weights = "configured_model_hook"
    else:
        weights = "live"
    return {
        "path": "external_differentiable_reconstruction" if external else "native_reconstruct",
        "dispatch": ("flow_hooks" if flow_hooks else "native_differentiable_hook") if external else "native_reconstruct",
        "model_type": f"{type(model).__module__}.{type(model).__qualname__}",
        "steps": actual_steps,
        "solver": config.get("rollout", {}).get("solver") if external and flow_hooks else getattr(core, "ode_solver", None),
        "requested_rollout_solver": config.get("rollout", {}).get("solver") if external else None,
        "observation_config": dict(config.get("observation_consistency", {})) if external else None,
        "observation_effective": (
            {**dict(config.get("observation_consistency", {})),
             "mode": str(config.get("observation_consistency", {}).get("mode", "endpoint_smooth")),
             "final_clamp": bool(config.get("observation_consistency", {}).get("final_clamp", True))
                 and str(config.get("observation_consistency", {}).get("mode", "endpoint_smooth")) != "none"}
            if external else None
        ),
        "model_configuration": dict(config.get("model", {})),
        "execution": execution,
        "weight_selection": weight_selection,
        "effective_weights": weights,
        "seed": seed,
        "draw_seeds": None if draws is None else [seed if i == 0 else _conditional_draw_seed(seed, i) for i in range(draws)],
        "noise_scale": float(noise_scale),
        "structured_grid": bool(getattr(getattr(model, "capabilities", None), "structured_grid_required", False)),
    }


def comparison_fingerprint(panel, *, steps=None, seed=None, noise_scale=1.0, weight_selection="configured", draws=None):
    """Bind states, sensors, targets, normalization, geometry and frozen score scales."""
    family = panel.families["multiscale_spectrum"]
    digest = hashlib.sha256()
    descriptor = {
        "sample_ids": list(panel.comparison_batch.sample_ids),
        "evaluation_sample_ids": list(getattr(panel.evaluation_batch, "sample_ids", ())),
        "comparison_logical_shapes": getattr(panel.comparison_batch, "logical_shapes", ()),
        "evaluation_logical_shapes": getattr(panel.evaluation_batch, "logical_shapes", ()),
        "fields": family.field_names,
        "field_ids": family.field_ids,
        "units": family.units,
        "bands": family.settings.bands.names,
        "layout": family.layout.names,
        "descriptor_weights": family.layout.weights,
        "standardize_descriptors": family.standardize_descriptors,
        "sampler": effective_sampler(panel, steps=steps, seed=seed, noise_scale=noise_scale, weight_selection=weight_selection, draws=draws),
        "score_definition_version": SCORE_DEFINITION_VERSION,
        "family_configuration": family.config,
    }
    digest.update(json.dumps(descriptor, sort_keys=True, default=str).encode())
    tensors = dict(family.named_buffers())
    for batch_name in ("comparison_batch", "evaluation_batch"):
        batch = getattr(panel, batch_name)
        for name in (
            "query_coords",
            "query_valid_mask",
            "target_fields",
            "obs_coords",
            "obs_values",
            "obs_field_ids",
            "obs_valid_mask",
            "obs_indices",
            "metadata",
        ):
            value = getattr(batch, name, None)
            if isinstance(value, torch.Tensor):
                tensors[f"{batch_name}.{name}"] = value
            elif isinstance(value, dict):
                def hash_metadata(item, path):
                    if isinstance(item, torch.Tensor):
                        tensors[path] = item
                    elif isinstance(item, np.ndarray):
                        digest.update(f"{path}:{item.dtype}:{item.shape}".encode())
                        if item.dtype.kind in "OUS":
                            digest.update(json.dumps(item.tolist(), default=str).encode())
                        else:
                            digest.update(np.ascontiguousarray(item).tobytes())
                    elif isinstance(item, dict):
                        for key in sorted(item):
                            hash_metadata(item[key], f"{path}.{key}")
                    elif isinstance(item, (list, tuple)):
                        for index, part in enumerate(item):
                            hash_metadata(part, f"{path}[{index}]")
                    else:
                        digest.update(f"{path}:{json.dumps(item, sort_keys=True, default=str)}".encode())
                hash_metadata(value, f"{batch_name}.{name}")
    for name, value in sorted(tensors.items()):
        data = value.detach().cpu().contiguous()
        digest.update(f"{name}:{data.dtype}:{tuple(data.shape)}".encode())
        digest.update(data.numpy().tobytes())
    return digest.hexdigest()


def fmt_scalar(value, precision=3):
    """Render undefined calibration ratios without inventing a finite value."""
    return "undefined" if value is None or not math.isfinite(value) else f"{value:.{precision}f}"


# ----------------------------------------------------------------------------- sampling
def draw_members(panel, count, *, steps=None, seed=None, noise_scale=1.0, states=None, weight_selection="configured"):
    """K independent draws of the panel states; [K,B,N,C] on the panel device (float32)."""
    model = panel.model
    evaluation = panel.config["evaluation"]
    seed = int(evaluation.get("seed", 2027)) if seed is None else int(seed)
    steps = int(evaluation.get("generation_steps", 2)) if steps is None else int(steps)
    structured = model.capabilities.structured_grid_required
    batch = panel.evaluation_batch if structured else panel.comparison_batch
    inference = _without_target(batch)
    original = getattr(model, "_sample_prior", None)
    if noise_scale != 1.0:
        if original is None:
            raise ValueError("noise-scale diagnostics require a model with _sample_prior")

        def scaled(coords, generator):
            return original(coords, generator) * float(noise_scale)

        model._sample_prior = scaled
    values = []
    model.eval()
    try:
        with selected_evaluation_weight_context(model, weight_selection), torch.no_grad():
            for draw in range(count):
                generator = torch.Generator(device=batch.query_coords.device).manual_seed(
                    seed if draw == 0 else _conditional_draw_seed(seed, draw)
                )
                if panel.config.get("optimization", {}).get("multitask", {}).get("external_policy"):
                    value = differentiable_reconstruction(
                        model, inference, steps=steps, solver=panel.config["rollout"]["solver"],
                        generator=generator,
                        observation_config=dict(panel.config["observation_consistency"]),
                    )
                else:
                    value = model.reconstruct(inference, steps=steps, generator=generator).prediction
                if structured:
                    value = _gather_prediction(
                        value, panel.evaluation_batch, panel.comparison_batch
                    )
                if states is not None:
                    value = value[:states]
                values.append(value.to(torch.float32))
    finally:
        if noise_scale != 1.0:
            model._sample_prior = original
    return torch.stack(values)


# ----------------------------------------------------------------------------- analysis
class BandAnalyzer:
    def __init__(self, family):
        self.family = family
        self.dtype = torch.float64
        self.geometry = family._geometry_tensors(self.dtype)
        self.mass = self.geometry["mass"] / self.geometry["mass"].sum()
        self.scales = family.band_scale.to(self.dtype)
        self.floor = family.power_floor.to(self.dtype)
        self.fields = list(family.field_names)
        self.bands = list(family.settings.bands.names)
        count = len(self.fields)
        self.projections = field_projections(
            count, "axes_and_pairs", dtype=self.dtype, device=self.mass.device
        )
        pairs = torch.triu_indices(count, count, offset=1)
        self.pairs = [
            f"{self.fields[int(i)]}*{self.fields[int(j)]}" for i, j in zip(pairs[0], pairs[1])
        ]
        self.weights = family.layout.weight_vector(dtype=self.dtype, device=self.mass.device)
        if getattr(family, "standardize_descriptors", False):
            self.weights = self.weights / family.descriptor_variance.to(self.weights)

    def statistics(self, fields):
        selected = self.family._select(fields).to(self.dtype)
        return raw_statistics(
            selected, manifest=self.family.manifest, with_band_fields=True, with_piece_means=True, **self.geometry
        )

    def descriptors(self, statistics):
        # The family's frozen buffers are in its compute dtype (float32 on CoastalBench).
        dtype = self.family.compute_dtype
        blocks = self.family._descriptors(
            {key: value.to(dtype) for key, value in statistics.items()}
        )
        return torch.cat([blocks[key] for key in self.family.layout.keys], dim=1).to(self.dtype)

    def power(self, left, right):
        return torch.einsum("n,bsnc->bsc", self.mass, left * right)

    def signed_band_scores(self, draws, target, *, alpha=1.0):
        """Per-band score for one state using the same kernel as training."""
        return tuple(
            value[0]
            for value in signed_band_energy_score(
                draws[:, None],
                target[None],
                mass=self.mass,
                scales=self.scales,
                projections=self.projections,
                alpha=alpha,
            )
        )


def analyse_ensemble(panel, members, *, label, score_alpha=1.0, score_draws=None, steps=None, seed=None, noise_scale=1.0, weight_selection="configured"):
    """Per-state arrays (numpy) and their state means for one checkpoint's K-draw ensemble."""
    family = panel.families["multiscale_spectrum"]
    analyzer = BandAnalyzer(family)
    target = panel.comparison_batch.target_fields
    mask = panel.comparison_batch.query_valid_mask
    count, states = members.shape[0], members.shape[1]
    score_draws = count if score_draws is None else int(score_draws)
    if not 2 <= score_draws <= count:
        raise ValueError("score_draws must be between 2 and the ensemble size")
    almost_fair_repulsion_factor(score_draws, score_alpha)
    started = time.perf_counter()
    # Field-level fidelity per state.
    mean = members.mean(dim=0)
    variance = members.var(dim=0, unbiased=True)

    def masked(values):  # [B,N,C] -> [B,C]
        weights = mask.to(values.dtype)
        return (values * weights[..., None]).sum(dim=1) / weights.sum(dim=1, keepdim=True)

    ens_mse = masked((mean - target).square())
    single_mse = masked((members[0] - target).square())
    spread = masked(variance)
    debiased = ens_mse - spread / count
    crps = torch.stack(
        [
            ensemble_crps([m[b : b + 1, mask[b]] for m in members], target[b : b + 1, mask[b]])
            for b in range(states)
        ]
    )
    # Band statistics: target once, members one by one (band fields kept in float32).
    target_stats = analyzer.statistics(target)
    truth_fields = target_stats["band_fields"]
    truth_power = analyzer.power(truth_fields, truth_fields)
    member_power = torch.zeros_like(truth_power)
    running = torch.zeros_like(truth_fields)
    band_fields = []
    descriptors = []
    coefficients = []
    piece_means = []
    rho = []
    for k in range(count):
        stats = analyzer.statistics(members[k])
        fields = stats["band_fields"]
        member_power += analyzer.power(fields, fields)
        running += fields
        band_fields.append(fields.to(torch.float32))
        descriptors.append(analyzer.descriptors(stats))
        coefficients.append(stats["coefficients"])
        piece_means.append(stats["piece_means"])
        rho.append(band_coherence(stats["gram"], analyzer.floor))
        del stats
    member_power /= count
    mean_fields = running / count
    mean_power = analyzer.power(mean_fields, mean_fields)
    varying = (count * (member_power - mean_power) / (count - 1)).clamp_min(0)
    shared = mean_power - varying / count
    mean_error = (
        analyzer.power(mean_fields - truth_fields, mean_fields - truth_fields) - varying / count
    )
    correlation = positive_ratio(
        analyzer.power(mean_fields, truth_fields), (mean_power * truth_power).sqrt()
    )
    # Canonical verification is fair and uses every draw. Legacy alpha/J are explicit options.
    es, es_attraction, es_repulsion = [], [], []
    for b in range(states):
        draws = torch.stack([band_fields[j][b].to(analyzer.dtype) for j in range(score_draws)])
        s, a, r = analyzer.signed_band_scores(draws, truth_fields[b], alpha=score_alpha)
        es.append(s), es_attraction.append(a), es_repulsion.append(r)
    es, es_attraction, es_repulsion = (
        torch.stack(es),
        torch.stack(es_attraction),
        torch.stack(es_repulsion),
    )
    rho_draws = torch.stack(rho)  # [K,B,S,P2]
    rho_target = band_coherence(target_stats["gram"], analyzer.floor)
    coherence_crps, coherence_attraction, coherence_repulsion = vector_energy_score(
        rho_draws[..., None], rho_target[..., None], alpha=score_alpha
    )
    joint_es, joint_attraction, joint_repulsion = joint_spectral_energy_score(
        torch.stack(coefficients),
        target_stats["coefficients"],
        filters=analyzer.geometry["filters"],
        scales=analyzer.scales,
        mass=analyzer.geometry["mass"],
    )
    mean_augmented = None
    if getattr(family, "mean_augmented", False):
        mean_augmented = joint_spectral_energy_score(
            torch.stack(coefficients), target_stats["coefficients"],
            filters=analyzer.geometry["filters"], scales=analyzer.scales,
            mass=analyzer.geometry["mass"], draw_piece_means=torch.stack(piece_means),
            target_piece_means=target_stats["piece_means"],
            piece_weights=family.piece_measures.sum(dim=1),
            mean_scales=family.piece_mean_scale, mean_weight=family.settings.mean_weight,
        )
    stacked = torch.stack(descriptors, dim=1)  # [B,K,D]
    v1_es, _v1_attraction, _v1_repulsion = conditional_energy_score(
        stacked, analyzer.descriptors(target_stats), analyzer.weights
    )
    seconds = time.perf_counter() - started

    def npy(value):
        return value.detach().cpu().numpy().astype(np.float64)

    arrays = {
        "ens_mse": npy(ens_mse),
        "single_mse": npy(single_mse),
        "spread": npy(spread),
        "debiased_mse": npy(debiased),
        "crps": npy(crps),
        "truth_power": npy(truth_power),
        "member_power": npy(member_power),
        "shared_power": npy(shared),
        "varying_power": npy(varying),
        "mean_error": npy(mean_error),
        "correlation": npy(correlation),
        "signed_band_es": npy(es),
        "signed_band_attraction": npy(es_attraction),
        "signed_band_repulsion": npy(es_repulsion),
        "coherence_crps": npy(coherence_crps),
        "rho_target": npy(rho_target),
        "rho_mean": npy(rho_draws.mean(dim=0)),
        "coherence_attraction": npy(coherence_attraction),
        "coherence_repulsion": npy(coherence_repulsion),
        "joint_spectral_es": npy(joint_es),
        "joint_attraction": npy(joint_attraction),
        "joint_repulsion": npy(joint_repulsion),
        "v1_descriptor_es": npy(v1_es),
    }
    if mean_augmented is not None:
        arrays.update({
            "joint_spectral_mean_es": npy(mean_augmented[0]),
            "joint_spectral_mean_attraction": npy(mean_augmented[1]),
            "joint_spectral_mean_repulsion": npy(mean_augmented[2]),
            "target_piece_means": npy(target_stats["piece_means"]),
            "member_piece_means": npy(torch.stack(piece_means)),
        })
    summary = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "score_definition_version": SCORE_DEFINITION_VERSION,
        "comparison_fingerprint": comparison_fingerprint(panel, steps=steps, seed=seed, noise_scale=noise_scale, weight_selection=weight_selection, draws=count),
        "metric_settings": {
            "signed_band_es": {
                "alpha": score_alpha,
                "draws": score_draws,
                "projections": "axes_and_pairs",
            },
            "coherence_crps": {"alpha": score_alpha, "draws": count},
            "joint_spectral_es": {"alpha": 1.0, "draws": count},
            "field_crps": {"alpha": 1.0, "draws": count},
            "v1_descriptor_es": {"alpha": 1.0, "draws": count},
            "spectral_weighting": "normalized_analysis_mass",
            "field_weighting": "uniform_query_nodes",
            "spectral_units": family.units,
            "field_units": "model_units",
        },
        "sampler": effective_sampler(panel, steps=steps, seed=seed, noise_scale=noise_scale, weight_selection=weight_selection, draws=count),
        "label": label,
        "draws": count,
        "score_draws": score_draws,
        "states": states,
        "seconds": seconds,
        "fields": analyzer.fields,
        "bands": analyzer.bands,
        "pairs": analyzer.pairs,
        "sample_ids": list(panel.comparison_batch.sample_ids),
        "per_field": {
            "ens_mean_mse": arrays["ens_mse"].mean(0).tolist(),
            "debiased_mse": arrays["debiased_mse"].mean(0).tolist(),
            "single_draw_mse": arrays["single_mse"].mean(0).tolist(),
            "spread": arrays["spread"].mean(0).tolist(),
            "fair_crps": arrays["crps"].mean(0).tolist(),
        },
        "per_band_field": {
            "mean_correlation": arrays["correlation"].mean(0).tolist(),
        },
        "per_band": {
            "signed_band_es": arrays["signed_band_es"].mean(0).tolist(),
            "signed_band_attraction": arrays["signed_band_attraction"].mean(0).tolist(),
            "signed_band_repulsion": arrays["signed_band_repulsion"].mean(0).tolist(),
            "coherence_crps": arrays["coherence_crps"].mean(axis=(0, 2)).tolist()
            if analyzer.pairs
            else [None] * len(analyzer.bands),
        },
        "per_pair_coherence_crps": arrays["coherence_crps"].mean(axis=(0, 1)).tolist(),
        "signed_band_es_mean": float(arrays["signed_band_es"].mean()),
        "coherence_crps_mean": float(arrays["coherence_crps"].mean()) if analyzer.pairs else None,
        "v1_descriptor_es": float(arrays["v1_descriptor_es"].mean()),
        "joint_spectral_es": float(arrays["joint_spectral_es"].mean()),
    }
    if mean_augmented is not None:
        summary["metric_settings"]["joint_spectral_mean_es"] = {
            "definition": family.settings.score_definition_version, "alpha": 1.0, "draws": count,
            "mean_weight": family.settings.mean_weight,
            "piece_mean_scales": "frozen_train_calibration",
        }
        summary["joint_spectral_mean_es"] = float(arrays["joint_spectral_mean_es"].mean())
    field_ratio = positive_ratio(spread.mean(0), debiased.mean(0))
    band_ratio = positive_ratio(varying.mean(0), mean_error.mean(0))
    summary["per_field"]["spread_over_debiased"] = finite_values(field_ratio)
    summary["per_field"]["rms_spread_over_rmse"] = finite_values(field_ratio.sqrt())
    summary["per_band_field"].update(
        spread_skill=finite_values(band_ratio),  # retained legacy alias, variance / error
        variance_over_debiased_error=finite_values(band_ratio),
        rms_spread_over_rmse=finite_values(band_ratio.sqrt()),
        variance_ratio_valid_states=finite_values(
            torch.isfinite(positive_ratio(varying, mean_error)).sum(0)
        ),
        member_power_over_truth=finite_values(
            positive_ratio(member_power.mean(0), truth_power.mean(0))
        ),
        shared_power_over_truth=finite_values(positive_ratio(shared.mean(0), truth_power.mean(0))),
        varying_power_over_truth=finite_values(
            positive_ratio(varying.mean(0), truth_power.mean(0))
        ),
        shared_over_rho2_truth=finite_values(
            positive_ratio(shared.mean(0), (correlation.square() * truth_power).mean(0))
        ),
        median_member_power_ratio=finite_values(
            positive_ratio(member_power, truth_power).nanmedian(dim=0).values
        ),
    )
    summary["spread_skill_definition"] = "variance_over_debiased_mean_error (legacy alias)"
    return arrays, finite_values(summary)


def effective_dimension(panel, *, draws, states, steps=None, chunk=32, weight_selection="configured"):
    """Effective ensemble dimension (sum sqrt(lambda))^2 / sum lambda of the band-passed
    covariance per band and field, from ``draws`` members of the first ``states`` states."""
    family = panel.families["multiscale_spectrum"]
    analyzer = BandAnalyzer(family)
    weights = analyzer.mass.sqrt()
    count = 0
    collected = []
    for start in range(0, draws, chunk):
        members = draw_members(
            panel, min(chunk, draws - start), steps=steps, seed=4242 + start, states=states, weight_selection=weight_selection
        )
        for k in range(members.shape[0]):
            fields = analyzer.statistics(members[k])["band_fields"] * weights[None, None, :, None]
            collected.append(fields.to(torch.float32))
        count += members.shape[0]
    stacked = torch.stack(collected)  # [K,B,S,N,C]
    stacked = stacked - stacked.mean(dim=0, keepdim=True)
    result = np.zeros((states, len(analyzer.bands), len(analyzer.fields)))
    ninety = np.zeros_like(result)
    for b in range(states):
        for s in range(len(analyzer.bands)):
            for c in range(len(analyzer.fields)):
                matrix = stacked[:, b, s, :, c].to(torch.float64)  # [K,N]
                eigenvalues = torch.linalg.eigvalsh(matrix @ matrix.T).clamp_min(0.0)
                total = float(eigenvalues.sum())
                if total <= 0:
                    continue
                result[b, s, c] = float(eigenvalues.sqrt().sum()) ** 2 / total
                sorted_values = torch.sort(eigenvalues, descending=True).values.cumsum(0) / total
                ninety[b, s, c] = int((sorted_values < 0.9).sum()) + 1
    return {
        "draws": count,
        "states": states,
        "bands": analyzer.bands,
        "fields": analyzer.fields,
        "effective_dimension_mean": result.mean(0).tolist(),
        "modes_for_90pct_variance_mean": ninety.mean(0).tolist(),
        "note": "sample covariance of K draws has rank <= K-1; values near K-1 mean the ensemble is richer than K",
    }


# ----------------------------------------------------------------------------- bootstrap
def _group_indices(clusters, count):
    if clusters is None:
        return [], []
    labels = np.asarray(clusters, dtype=str)
    if labels.ndim != 1 or len(labels) != count or any(not label for label in labels):
        raise ValueError("explicit grouping must contain one nonempty label per condition")
    unique = np.unique(labels)
    return unique.tolist(), [np.flatnonzero(labels == label) for label in unique]


def clustered_bootstrap(values, clusters, *, reps=2000, seed=0, statistic="mean"):
    """Cluster percentile interval when at least two independent clusters are declared."""
    if statistic != "mean":
        raise ValueError("only the state mean is supported")
    values = np.asarray(values)
    _, groups = _group_indices(clusters, len(values))
    result = {"mean": values.mean(axis=0).tolist(), "low": None, "high": None,
              "clusters": len(groups), "interval_available": len(groups) >= 2}
    if len(groups) >= 2:
        rng = np.random.default_rng(seed)
        estimates = []
        for _ in range(reps):
            index = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
            estimates.append(values[index].mean(axis=0))
        result.update(low=np.percentile(estimates, 2.5, axis=0).tolist(),
                      high=np.percentile(estimates, 97.5, axis=0).tolist())
    return finite_values(result)


def ratio_bootstrap(arm, control, clusters=None, *, reps=2000, seed=0, grouping="independent_cluster"):
    """Paired condition-mean ratio; only declared independent clusters get a CI.

    Declared time blocks produce descriptive block ratios, not inferential intervals.
    """
    arm, control = np.asarray(arm), np.asarray(control)
    if arm.shape != control.shape or arm.ndim < 1 or not len(arm):
        raise ValueError("paired arm and control arrays must have the same nonempty shape")
    if grouping not in {"independent_cluster", "time_block", "none"}:
        raise ValueError("grouping must be independent_cluster, time_block, or none")
    names, groups = _group_indices(clusters, len(arm))
    if grouping == "none" and groups:
        raise ValueError("grouping none cannot contain labels")
    def ratio(numerator, denominator):
        return np.divide(numerator, denominator, out=np.full_like(numerator, np.nan, dtype=float),
                         where=np.isfinite(denominator) & (denominator > 0))
    point = ratio(arm.mean(axis=0), control.mean(axis=0))
    result = {"ratio": point.tolist(), "low": None, "high": None,
              "valid_bootstrap_fraction": None, "grouping": grouping,
              "group_count": len(groups), "condition_count": len(arm),
              "group_sizes": {name: len(group) for name, group in zip(names, groups)},
              "interval_available": grouping == "independent_cluster" and len(groups) >= 2}
    if result["interval_available"]:
        rng = np.random.default_rng(seed)
        ratios = []
        for _ in range(reps):
            index = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
            ratios.append(ratio(arm[index].mean(axis=0), control[index].mean(axis=0)))
        ratios = np.stack(ratios)
        result.update(low=np.percentile(ratios, 2.5, axis=0).tolist(),
                      high=np.percentile(ratios, 97.5, axis=0).tolist(),
                      valid_bootstrap_fraction=np.isfinite(ratios).mean(axis=0).tolist())
    elif grouping == "time_block":
        result["block_ratios"] = {name: ratio(arm[group].mean(axis=0), control[group].mean(axis=0)).tolist()
                                  for name, group in zip(names, groups)}
    return finite_values(result)


def load_grouping(path, sample_ids):
    """Read case-supplied group labels, never infer independence from sample IDs."""
    if path is None:
        return None, "none"
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    unit, labels = data.get("unit"), data.get("labels")
    if unit not in {"independent_cluster", "time_block"} or not isinstance(labels, dict):
        raise ValueError("group file requires unit independent_cluster or time_block and an ID-to-label mapping")
    if set(labels) != set(sample_ids) or len(set(sample_ids)) != len(sample_ids):
        raise ValueError("group labels must exactly match unique panel sample IDs")
    clusters = [labels[sample_id] for sample_id in sample_ids]
    if any(not isinstance(label, str) or not label.strip() for label in clusters):
        raise ValueError("group labels must be nonempty strings")
    _group_indices(clusters, len(sample_ids))
    return clusters, unit


# ----------------------------------------------------------------------------- commands
def _timing_start(device=None):
    if device is not None and device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter(), time.process_time()


def _timing_stop(start, device=None):
    if device is not None and device.type == "cuda":
        torch.cuda.synchronize(device)
    return {"wall_seconds": time.perf_counter() - start[0],
            "process_cpu_seconds": time.process_time() - start[1]}


def parse_spec(spec):
    name = None
    if "=" in spec.split(":")[0]:
        name, spec = spec.split("=", 1)
    run_dir, checkpoints = spec.rsplit(":", 1)
    return name, run_dir, checkpoints.split(",")


def open_panel(run_dir, draws, device, split=None):
    del draws  # Members are drawn explicitly after restoring the frozen panel.
    overrides = {}
    if split:
        overrides["split"] = str(
            split
        )  # a split other than the run's own builds a fresh panel with the run's protocol
    return restore_evaluation_panel(run_dir, evaluation_overrides=overrides, device_name=device)


def command_capacity(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report = {
        "draws": args.draws,
        "steps": args.steps,
        "noise_scales": args.noise_scales,
        "split": args.split or "validation",
        "conditions": {},
    }
    for spec in args.specs:
        _, run_dir, checkpoints = parse_spec(spec)
        phase = _timing_start()
        panel = open_panel(run_dir, args.draws, args.device, args.split)
        device = panel.comparison_batch.query_coords.device
        initialization = _timing_stop(phase, device)
        default_steps = int(panel.config["evaluation"].get("generation_steps", 2))
        run_label = Path(run_dir).parent.name
        for checkpoint in sorted(checkpoints, key=lambda name: name != "source"):
            phase = _timing_start(device)
            if checkpoint != "source":
                panel.load_checkpoint(checkpoint)
            checkpoint_loading = _timing_stop(phase, device)
            conditions = [(steps, 1.0) for steps in args.steps] + [
                (default_steps, scale) for scale in args.noise_scales if scale != 1.0
            ]
            for steps, scale in conditions:
                label = f"{run_label}:{checkpoint}|steps={steps}|noise={scale}"
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                total_phase = _timing_start(device)
                phase = _timing_start(device)
                members = draw_members(panel, args.draws, steps=steps, noise_scale=scale, weight_selection=args.weight_selection)
                sampling = _timing_stop(phase, device)
                phase = _timing_start(device)
                arrays, summary = analyse_ensemble(
                    panel,
                    members,
                    label=label,
                    score_alpha=args.score_alpha,
                    score_draws=args.score_draws,
                    steps=steps, noise_scale=scale, weight_selection=args.weight_selection,
                )
                scoring_and_transfer = _timing_stop(phase, device)
                phase = _timing_start(device)
                np.savez_compressed(
                    out / (label.replace("|", "_").replace(":", "_").replace("=", "") + ".npz"),
                    **arrays,
                )
                persistence = _timing_stop(phase, device)
                summary["timing"] = {"panel_initialization": initialization, "checkpoint_load": checkpoint_loading,
                    "all_draws": sampling, "scoring_and_device_to_cpu": scoring_and_transfer,
                    "payload_persistence": persistence, "condition_total": _timing_stop(total_phase, device),
                    "note": "panel and checkpoint setup are shared; scoring includes transfers; training/calibration costs are separate"}
                summary["resources"] = {"process_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                    "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
                    "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None}
                report["conditions"][label] = summary
                print(
                    label,
                    f"condition_total={summary['timing']['condition_total']['wall_seconds']:.0f}s",
                    "debiased mse",
                    [round(v, 6) for v in summary["per_field"]["debiased_mse"]],
                    flush=True,
                )
                del members
                torch.cuda.empty_cache()
            if args.effective_dimension_draws > 0:
                label = f"{run_label}:{checkpoint}|effective_dimension"
                phase = _timing_start(device)
                report["conditions"][label] = effective_dimension(
                    panel,
                    draws=args.effective_dimension_draws,
                    states=args.effective_dimension_states,
                    steps=default_steps, weight_selection=args.weight_selection,
                )
                report["conditions"][label]["timing"] = _timing_stop(phase, device)
                print(label, report["conditions"][label]["effective_dimension_mean"], flush=True)
                torch.cuda.empty_cache()
        del panel
        torch.cuda.empty_cache()
    (out / "capacity.json").write_text(json.dumps(finite_values(report), indent=1, allow_nan=False))
    (out / "capacity.md").write_text(render_capacity(report))
    print("wrote", out / "capacity.md")


def render_capacity(report):
    lines = ["# Capacity diagnostic", "", f"draws per condition: {report['draws']}", ""]
    for label, summary in report["conditions"].items():
        lines.append(f"## {label}")
        lines.append("")
        if "effective_dimension_mean" in summary:
            lines.append(
                f"effective ensemble dimension from {summary['draws']} draws on {summary['states']} states (rows bands, columns fields {summary['fields']}):"
            )
            lines.append("")
            for band, row in zip(summary["bands"], summary["effective_dimension_mean"]):
                lines.append(f"- {band}: " + ", ".join(f"{v:.1f}" for v in row))
            lines.append("")
            continue
        fields, bands = summary["fields"], summary["bands"]
        lines.append(f"Score settings: `{json.dumps(summary['metric_settings'], sort_keys=True)}`")
        lines.append(
            "Spread-skill is variance / debiased error; undefined denominators are not clamped."
        )
        lines.append("")
        pf = summary["per_field"]
        lines.append(
            "| field | debiased mean MSE | ens-mean MSE | single-draw MSE | spread | spread/debiased | fair CRPS |"
        )
        lines.append("|---|---|---|---|---|---|---|")
        for i, f in enumerate(fields):
            lines.append(
                f"| {f} | {pf['debiased_mse'][i]:.3e} | {pf['ens_mean_mse'][i]:.3e} | {pf['single_draw_mse'][i]:.3e} | {pf['spread'][i]:.3e} | {fmt_scalar(pf['spread_over_debiased'][i])} | {pf['fair_crps'][i]:.4e} |"
            )
        lines.append("")
        pb = summary["per_band_field"]
        lines.append(
            "| band | field | member/truth | shared/truth | varying/truth | shared/(rho^2 truth) | spread-skill | corr(mean,truth) | median member/truth |"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for s, band in enumerate(bands):
            for c, f in enumerate(fields):
                lines.append(
                    f"| {band} | {f} | "
                    + " | ".join(
                        fmt_scalar(pb[key][s][c])
                        for key in (
                            "member_power_over_truth",
                            "shared_power_over_truth",
                            "varying_power_over_truth",
                            "shared_over_rho2_truth",
                            "spread_skill",
                            "mean_correlation",
                            "median_member_power_ratio",
                        )
                    )
                    + " |"
                )
        lines.append("")
        lines.append(
            "per band: signed-band ES "
            + ", ".join(
                f"{b} {v:.4f}" for b, v in zip(bands, summary["per_band"]["signed_band_es"])
            )
            + "; coherence CRPS "
            + ", ".join(
                f"{b} {fmt_scalar(v, 4)}"
                for b, v in zip(bands, summary["per_band"]["coherence_crps"])
            )
            + f"; v1 descriptor ES {summary['v1_descriptor_es']:.4f}"
        )
        lines.append("")
    return "\n".join(lines) + "\n"


def command_arms(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results = {}
    sample_ids = None
    fingerprint = None
    for spec in args.specs:
        name, run_dir, checkpoints = parse_spec(spec)
        name = name or Path(run_dir).parent.name
        phase = _timing_start()
        panel = open_panel(run_dir, args.draws, args.device, args.split)
        device = panel.comparison_batch.query_coords.device
        initialization = _timing_stop(phase, device)
        current_fingerprint = comparison_fingerprint(panel, steps=args.steps, draws=args.draws, weight_selection=args.weight_selection)
        if fingerprint is not None and current_fingerprint != fingerprint:
            raise ValueError(
                f"{name}: comparison states, sensors, normalization or spectral calibration differ"
            )
        fingerprint = current_fingerprint
        for checkpoint in sorted(checkpoints, key=lambda name: name != "source"):
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            total_phase = _timing_start(device)
            phase = _timing_start(device)
            if checkpoint != "source":
                panel.load_checkpoint(checkpoint)
            checkpoint_loading = _timing_stop(phase, device)
            # load_checkpoint restores family states too; verify after loading.
            if comparison_fingerprint(panel, steps=args.steps, draws=args.draws, weight_selection=args.weight_selection) != fingerprint:
                raise ValueError(f"{name}:{checkpoint}: frozen evaluator or effective sampler differs after checkpoint load")
            label = f"{name}:{checkpoint}"
            phase = _timing_start(device)
            members = draw_members(panel, args.draws, steps=args.steps, weight_selection=args.weight_selection)
            sampling = _timing_stop(phase, device)
            phase = _timing_start(device)
            arrays, summary = analyse_ensemble(
                panel,
                members,
                label=label,
                score_alpha=args.score_alpha,
                score_draws=args.score_draws,
                steps=args.steps, weight_selection=args.weight_selection,
            )
            scoring_and_transfer = _timing_stop(phase, device)
            if sample_ids is None:
                sample_ids = summary["sample_ids"]
            elif summary["sample_ids"] != sample_ids:
                raise ValueError(f"{label}: validation panel states differ from the first arm")
            phase = _timing_start(device)
            np.savez_compressed(out / f"{name}_{checkpoint}.npz", **arrays)
            persistence = _timing_stop(phase, device)
            summary["timing"] = {
                "panel_initialization": initialization,
                "checkpoint_load": checkpoint_loading,
                "all_draws": sampling,
                "scoring_and_device_to_cpu": scoring_and_transfer,
                "payload_persistence": persistence,
                "checkpoint_evaluation_total": _timing_stop(total_phase, device),
                "note": "panel initialization is shared across checkpoints; scoring includes transfers; training/calibration costs are separate",
            }
            summary["resources"] = {
                "process_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
                "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None,
            }
            results[label] = {"arrays": arrays, "summary": summary}
            print(
                label,
                f"checkpoint_total={summary['timing']['checkpoint_evaluation_total']['wall_seconds']:.0f}s",
                "debiased mse",
                [round(v, 6) for v in summary["per_field"]["debiased_mse"]],
                "ES",
                round(summary["signed_band_es_mean"], 4),
                "cohCRPS",
                summary["coherence_crps_mean"],
                flush=True,
            )
            del members
            torch.cuda.empty_cache()
        del panel
        torch.cuda.empty_cache()
    clusters, grouping = load_grouping(args.groups_json, sample_ids)
    comparison = compare_arms(results, clusters, control=args.control, reference=args.reference, grouping=grouping)
    split = args.split or "validation"
    (out / "comparison.json").write_text(
        json.dumps(
            finite_values(
                {
                    "split": split,
                    "draws": args.draws,
                    "steps": args.steps,
                    "summaries": {k: v["summary"] for k, v in results.items()},
                    "comparison": comparison,
                }
            ),
            indent=1,
            allow_nan=False,
        )
    )
    (out / "comparison.md").write_text(
        render_comparison(
            results,
            comparison,
            control=args.control,
            reference=args.reference,
            draws=args.draws,
            steps=args.steps,
            split=split,
        )
    )
    print("wrote", out / "comparison.md")


def compare_arms(results, clusters=None, *, control, reference, grouping="independent_cluster"):
    if control not in results or (reference is not None and reference not in results):
        raise ValueError("control and reference labels must identify evaluated checkpoints")
    baseline = results[control]["summary"]
    for label, item in results.items():
        summary = item["summary"]
        for key in ("comparison_fingerprint", "metric_settings", "score_definition_version"):
            if summary[key] != baseline[key]:
                raise ValueError(f"{label}: incompatible comparison {key}")
        if summary["sampler"] != baseline["sampler"]:
            raise ValueError(f"{label}: incompatible evaluation sampler")
    comparison = {}
    bases = {"control": control, "reference": reference}
    for label, item in results.items():
        comparison[label] = {}
        for kind, base in bases.items():
            if base is None or base not in results or base == label:
                continue
            a, b = item["arrays"], results[base]["arrays"]
            comparison[label][f"vs_{kind}"] = {
                "debiased_mse": ratio_bootstrap(a["debiased_mse"], b["debiased_mse"], clusters, grouping=grouping),
                "ens_mean_mse": ratio_bootstrap(a["ens_mse"], b["ens_mse"], clusters, grouping=grouping),
                "fair_crps": ratio_bootstrap(a["crps"], b["crps"], clusters, grouping=grouping),
                "spread": ratio_bootstrap(a["spread"], b["spread"], clusters, grouping=grouping),
                "signed_band_es": ratio_bootstrap(
                    a["signed_band_es"], b["signed_band_es"], clusters, grouping=grouping
                ),
                "joint_spectral_es": ratio_bootstrap(
                    a["joint_spectral_es"], b["joint_spectral_es"], clusters, grouping=grouping
                ),
                "coherence_crps": ratio_bootstrap(
                    a["coherence_crps"].mean(axis=2), b["coherence_crps"].mean(axis=2), clusters, grouping=grouping
                ),
                "v1_descriptor_es": ratio_bootstrap(
                    a["v1_descriptor_es"], b["v1_descriptor_es"], clusters, grouping=grouping
                ),
                "varying_power": ratio_bootstrap(
                    a["varying_power"].mean(axis=2), b["varying_power"].mean(axis=2), clusters, grouping=grouping
                ),
            }
            if "joint_spectral_mean_es" in a:
                if "joint_spectral_mean_es" not in b:
                    raise ValueError(f"{label}: mean-augmented score unavailable in baseline")
                comparison[label][f"vs_{kind}"]["joint_spectral_mean_es"] = ratio_bootstrap(
                    a["joint_spectral_mean_es"], b["joint_spectral_mean_es"], clusters, grouping=grouping
                )
    return comparison


def fmt_ratio(entry, index=None, *, neutral=False):
    """Percent change with its interval; lower-is-better metrics get **better** / !!worse marks,
    neutral quantities (spread, varying power) only a significance mark (†)."""
    if index is None:
        r, lo, hi = entry["ratio"], entry["low"], entry["high"]
    else:
        r = entry["ratio"][index]
        lo = None if entry["low"] is None else entry["low"][index]
        hi = None if entry["high"] is None else entry["high"][index]
    if r is None or not math.isfinite(r):
        return "undefined"
    if lo is None or hi is None or not math.isfinite(lo) or not math.isfinite(hi):
        return f"{100 * (r - 1):+.1f}% (interval unavailable)"
    significant = not (lo <= 1.0 <= hi)
    text = f"{100 * (r - 1):+.1f}% [{100 * (lo - 1):+.1f}, {100 * (hi - 1):+.1f}]"
    if not significant:
        return text
    if neutral:
        return text + "†"
    return f"**{text}**" if r < 1 else f"!!{text}"


def render_comparison(results, comparison, *, control, reference, draws, steps, split="validation"):
    any_summary = next(iter(results.values()))["summary"]
    fields, bands = any_summary["fields"], any_summary["bands"]
    lines = [
        "# Arm comparison (offline yardstick)",
        "",
        (
            f"{draws} draws per state, sampler `{json.dumps(any_summary['sampler'], sort_keys=True)}`, {any_summary['states']} {split} states; "
            f"percent change of the state mean; paired intervals require explicitly declared independent clusters; time blocks are descriptive; "
            f"**bold** = interval excludes 0 and the arm is better, !! = interval excludes 0 and the arm is worse, "
            f"† = interval excludes 0 for a neutral quantity (spread, varying power). "
            f"Control = {control}; reference = {reference}."
        ),
        "",
    ]
    settings = any_summary["metric_settings"]
    lines.extend(
        [
            (
                f"Metric definition `{any_summary['score_definition_version']}`; "
                f"signed ES alpha={settings['signed_band_es']['alpha']}, J={settings['signed_band_es']['draws']}; "
                f"coherence CRPS alpha={settings['coherence_crps']['alpha']}, J={settings['coherence_crps']['draws']}; "
                f"joint ES is fair with J={settings['joint_spectral_es']['draws']}. "
                "Spectral distances use normalized analysis mass; field scores use uniform valid query nodes. "
                "Spread-skill is variance / debiased error; the RMS ratio is its square root. "
                "Nonpositive debiased denominators are reported as undefined."
            ),
            "",
        ]
    )
    if "joint_spectral_mean_es" in settings:
        lines.extend([f"Mean-augmented joint score: `{json.dumps(settings['joint_spectral_mean_es'], sort_keys=True)}`; the separately named legacy joint score remains mean-free.", ""])
    for kind in ("vs_control", "vs_reference"):
        base = control if kind == "vs_control" else reference
        rows = [(label, c[kind]) for label, c in comparison.items() if kind in c]
        if not rows:
            continue
        lines.append(f"## Versus {base}")
        lines.append("")
        for label, entry in rows:
            grouping_info = entry["joint_spectral_es"]
            lines.append(f"{label}: {grouping_info['condition_count']} conditions; {grouping_info['group_count']} {grouping_info['grouping']} groups, sizes `{json.dumps(grouping_info['group_sizes'], sort_keys=True)}`.")
            if grouping_info.get("block_ratios") is not None:
                lines.append(f"Descriptive joint-score ratios by time block: `{json.dumps(grouping_info['block_ratios'], sort_keys=True)}`.")
        lines.append("")
        if "joint_spectral_mean_es" in rows[0][1]:
            lines.append("Mean-augmented joint score change: " + "; ".join(f"{label}: {fmt_ratio(entry['joint_spectral_mean_es'])}" for label, entry in rows))
            lines.append("")
        lines.append(
            "| arm | "
            + " | ".join(f"debiased MSE {f}" for f in fields)
            + " | "
            + " | ".join(f"CRPS {f}" for f in fields)
            + " |"
        )
        lines.append("|---|" + "---|" * (2 * len(fields)))
        for label, entry in rows:
            lines.append(
                f"| {label} | "
                + " | ".join(fmt_ratio(entry["debiased_mse"], i) for i in range(len(fields)))
                + " | "
                + " | ".join(fmt_ratio(entry["fair_crps"], i) for i in range(len(fields)))
                + " |"
            )
        lines.append("")
        lines.append(
            "| arm | "
            + " | ".join(f"signed-band ES {b}" for b in bands)
            + " | "
            + " | ".join(f"coherence CRPS {b}" for b in bands)
            + " | v1 descriptor ES | joint ES |"
        )
        lines.append("|---|" + "---|" * (2 * len(bands) + 2))
        for label, entry in rows:
            lines.append(
                f"| {label} | "
                + " | ".join(fmt_ratio(entry["signed_band_es"], i) for i in range(len(bands)))
                + " | "
                + " | ".join(fmt_ratio(entry["coherence_crps"], i) for i in range(len(bands)))
                + " | "
                + fmt_ratio(entry["v1_descriptor_es"])
                + " | "
                + fmt_ratio(entry["joint_spectral_es"])
                + " |"
            )
        lines.append("")
        lines.append(
            "| arm | "
            + " | ".join(f"spread {f}" for f in fields)
            + " | "
            + " | ".join(f"varying band power {b}" for b in bands)
            + " |"
        )
        lines.append("|---|" + "---|" * (len(fields) + len(bands)))
        for label, entry in rows:
            lines.append(
                f"| {label} | "
                + " | ".join(
                    fmt_ratio(entry["spread"], i, neutral=True) for i in range(len(fields))
                )
                + " | "
                + " | ".join(
                    fmt_ratio(entry["varying_power"], i, neutral=True) for i in range(len(bands))
                )
                + " |"
            )
        lines.append("")
    lines.append("## Absolute values")
    lines.append("")
    lines.append(
        "| arm | "
        + " | ".join(f"debiased MSE {f}" for f in fields)
        + " | "
        + " | ".join(f"spread/debiased {f}" for f in fields)
        + " | ES mean | coherence CRPS mean | v1 ES |"
    )
    lines.append("|---|" + "---|" * (2 * len(fields) + 3))
    for label, item in results.items():
        s = item["summary"]
        lines.append(
            f"| {label} | "
            + " | ".join(f"{v:.3e}" for v in s["per_field"]["debiased_mse"])
            + " | "
            + " | ".join(fmt_scalar(v, 2) for v in s["per_field"]["spread_over_debiased"])
            + f" | {s['signed_band_es_mean']:.4f} | {fmt_scalar(s['coherence_crps_mean'], 4)} | {s['v1_descriptor_es']:.4f} |"
        )
    lines.append("")
    lines.append(
        "| arm | band | "
        + " | ".join(f"member/truth {f}" for f in fields)
        + " | "
        + " | ".join(f"spread-skill {f}" for f in fields)
        + " |"
    )
    lines.append("|---|---|" + "---|" * (2 * len(fields)))
    for label, item in results.items():
        pb = item["summary"]["per_band_field"]
        for s, band in enumerate(bands):
            lines.append(
                f"| {label} | {band} | "
                + " | ".join(
                    fmt_scalar(pb["member_power_over_truth"][s][c], 2) for c in range(len(fields))
                )
                + " | "
                + " | ".join(fmt_scalar(pb["spread_skill"][s][c], 2) for c in range(len(fields)))
                + " |"
            )
    lines.append("")
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    cap = sub.add_parser("capacity")
    cap.add_argument("out")
    cap.add_argument("specs", nargs="+", help="RUN_DIR:ckpt[,ckpt]")
    cap.add_argument("--draws", type=int, default=32)
    cap.add_argument("--steps", type=int, nargs="+", default=[8, 16])
    cap.add_argument("--noise-scales", type=float, nargs="+", default=[1.0, 1.15, 1.3])
    cap.add_argument("--effective-dimension-draws", type=int, default=192)
    cap.add_argument("--effective-dimension-states", type=int, default=4)
    cap.add_argument("--device", default="cuda:0")
    cap.add_argument(
        "--split", default=None, help="evaluation split (default: the run's own validation panel)"
    )
    cap.set_defaults(func=command_capacity)
    arms = sub.add_parser("arms")
    arms.add_argument("out")
    arms.add_argument("specs", nargs="+", help="NAME=RUN_DIR:ckpt[,ckpt]")
    arms.add_argument("--control", required=True, help="label NAME:ckpt of the control arm")
    arms.add_argument(
        "--reference", default=None, help="label NAME:ckpt of the shared starting checkpoint"
    )
    arms.add_argument("--draws", type=int, default=32)
    arms.add_argument("--steps", type=int, default=16)
    arms.add_argument("--device", default="cuda:0")
    arms.add_argument(
        "--split", default=None, help="evaluation split (default: the run's own validation panel)"
    )
    arms.add_argument("--groups-json", default=None, help="case-declared independent_cluster or time_block labels keyed by sample ID")
    arms.set_defaults(func=command_arms)
    for command in (cap, arms):
        command.add_argument("--weight-selection", choices=("configured", "live"), default="configured")
        command.add_argument(
            "--score-alpha",
            type=float,
            default=1.0,
            help="signed/coherence score alpha; 1 is fair, 0.95 reproduces the legacy surrogate",
        )
        command.add_argument(
            "--score-draws",
            type=int,
            default=None,
            help="draws for signed ES (default: all); legacy evaluation used 16",
        )
    args = parser.parse_args(argv)
    count = args.draws if args.score_draws is None else args.score_draws
    if not 2 <= count <= args.draws:
        parser.error("--score-draws must be between 2 and --draws")
    if not 0 < args.score_alpha <= 1:
        parser.error("--score-alpha must lie in (0, 1]")
    args.func(args)


if __name__ == "__main__":
    main()

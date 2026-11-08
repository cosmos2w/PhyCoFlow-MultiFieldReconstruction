"""Versioned common native observers for Round 5.

All outputs retain the condition axis. Endpoint order in State matrices is the
reference followed by every supplied member. Nothing in this module changes the
historical native bundle or its summary definitions.
"""
from __future__ import annotations

import copy
from time import perf_counter
import torch

from ..coherence.base import projection_bank
from ..coherence.families.multiscale_spectrum.losses import vector_energy_score, band_coherence
from ..coherence.families.multiscale_spectrum.objectives import signed_band_terms
from .relation_metrics import relation_coordinates

OBSERVER_VERSION = "round5_common_native_v1"
HELDOUT_SEED = 700543


def _validate_members(members, target):
    if members.ndim != 4 or target.ndim != 3 or members.shape[1:] != target.shape or members.shape[0] < 2:
        raise ValueError("members [J,B,N,C] must match target [B,N,C], with J >= 2")
    if not torch.isfinite(members).all() or not torch.isfinite(target).all():
        raise ValueError("native endpoints must be finite")


def squared_distance_accounting(matrix):
    """Fair ES and Hilbert bias/variation from exact squared distances [J+1,J+1,B]."""
    if matrix.ndim != 3 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] < 3:
        raise ValueError("squared distance matrix must be [J+1,J+1,B], J >= 2")
    if not torch.isfinite(matrix).all() or (matrix < 0).any():
        raise ValueError("squared distances must be finite and nonnegative")
    if not torch.allclose(matrix, matrix.transpose(0, 1)) or not torch.equal(matrix.diagonal(dim1=0, dim2=1), torch.zeros_like(matrix.diagonal(dim1=0, dim2=1))):
        raise ValueError("squared distance matrix must be symmetric with zero diagonal")
    j = matrix.shape[0] - 1
    i, k = torch.triu_indices(j, j, 1, device=matrix.device)
    member_sq = matrix[0, 1:]
    pair_sq = matrix[1:, 1:][i, k]
    member = member_sq.mean(0)
    variation = pair_sq.sum(0) / (j * j)
    bank = member - variation
    attraction = member_sq.sqrt().mean(0)
    separation = pair_sq.sqrt().sum(0) / (j * (j - 1))
    return {"paired_primary": member_sq[0], "m_member": member, "m_bank": bank,
            "v_bank": variation, "u_mean": bank - variation / (j - 1),
            "attraction": attraction, "separation": separation, "fair_score": attraction - separation}


class CommonStateObserver:
    """Score a frozen v3 paired family against all endpoints, with two direction banks.

    The independent held-out copy is frozen at construction and exposed through
    ``heldout_directions`` for inclusion in the campaign's metric binding.
    """
    def __init__(self, family, *, cache=False):
        if getattr(family, "score_definition", None) != "marginal_mixed_weighted_v3":
            raise ValueError("common State observer requires the historical weighted v3 family")
        self.cache = bool(cache)
        self.main = copy.deepcopy(family)
        self.heldout = copy.deepcopy(family)
        cross = (self.heldout.components_by_key["cross_joint_topk_swd"]
                 if "cross_joint_topk_swd" in self.heldout.components_by_key else None)
        if cross is not None:
            if not cross.mean_reduction or cross.top_fraction != 1.0:
                raise ValueError("State observer needs the complete fixed direction bank")
            cross.directions = projection_bank(len(cross.field_ids), 32, seed=HELDOUT_SEED, include_axes=False, qmc=True).to(cross.directions)
        self.heldout_directions = None if cross is None else cross.directions.detach().clone()

    @torch.no_grad()
    def score(self, members, target, *, weights, mask=None, measure_metadata=None, timings=None):
        _validate_members(members, target)
        context = {"generated_weights": weights, "reference_weights": weights,
                   "generated_mask": mask, "reference_mask": mask,
                   "measure_metadata": measure_metadata or {}}
        if timings is not None and members.device.type == "cuda":
            torch.cuda.synchronize(members.device)
        started = perf_counter()
        main = self._bank(self.main, members, target, context, cache=self.cache)
        if timings is not None and members.device.type == "cuda":
            torch.cuda.synchronize(members.device)
        main_seconds = perf_counter() - started
        started = perf_counter()
        heldout = self._bank(self.heldout, members, target, context, cache=self.cache)
        if timings is not None and members.device.type == "cuda":
            torch.cuda.synchronize(members.device)
        heldout_seconds = perf_counter() - started
        if timings is not None:
            timings["state_main_seconds"] = timings.get("state_main_seconds", 0.0) + main_seconds
            timings["state_heldout_seconds"] = timings.get("state_heldout_seconds", 0.0) + heldout_seconds
        return {"version": OBSERVER_VERSION, "main": main, "heldout": heldout}

    @staticmethod
    def _bank(family, members, target, context, *, cache=False):
        if cache:
            if members.device.type != 'cpu':
                raise ValueError('cached native State is CPU-only')
            from .quantile_state_cache import cached_state_matrices
            claimed = context.get('measure_metadata', {}).get('native_measure_sha256')
            if family._measure_artifact is not None and claimed is not None and claimed != family._measure_artifact['sha256']:
                raise ValueError('runtime State measure differs from the frozen native measure')
            result = cached_state_matrices(family, members, target, weights=context['generated_weights'],
                                           mask=context.get('generated_mask'))
            return {**result, **squared_distance_accounting(result['squared_distances'])}
        endpoints = [target, *members.unbind(0)]
        n, b = len(endpoints), target.shape[0]
        matrix = target.new_zeros((n, n, b))
        leaves = {}
        for i in range(n):
            for k in range(i + 1, n):
                result = family(endpoints[i], endpoints[k], context=context)
                matrix[i, k] = matrix[k, i] = result.per_sample_cost
                for name, term in result.component_results.items():
                    for leaf_name in ("per_field_w2", "per_direction_w2"):
                        value = term.diagnostics.get(leaf_name)
                        if value is not None:
                            key = name + "." + leaf_name
                            if key not in leaves:
                                leaves[key] = value.new_zeros((n, n, *value.shape))
                            leaves[key][i, k] = leaves[key][k, i] = value
        return {"squared_distances": matrix, "leaf_squared_distances": leaves,
                **squared_distance_accounting(matrix)}


def query_accounting(draws, target):
    """Mean-over-coordinate squared risk and fair scalar-coordinate CRPS per condition."""
    if draws.ndim < 3 or draws.shape[1:] != target.shape or draws.shape[0] < 2:
        raise ValueError("query draws [J,B,...] must match target [B,...], J >= 2")
    if target[0].numel() == 0:
        return None
    if not torch.isfinite(draws).all() or not torch.isfinite(target).all():
        raise ValueError("queries must be finite")
    j = draws.shape[0]
    flat, truth = draws.flatten(start_dim=2), target.flatten(start_dim=1)
    member = (flat - truth.unsqueeze(0)).square().mean((0, 2))
    bank = (flat.mean(0) - truth).square().mean(-1)
    variation = (flat - flat.mean(0, keepdim=True)).square().mean((0, 2))
    score, attraction, separation = vector_energy_score(flat[..., None], truth[..., None])
    return {"m_member": member, "m_bank": bank, "v_bank": variation,
            "u_mean": bank - variation / (j - 1), "fair_score": score.mean(-1),
            "attraction": attraction.mean(-1), "separation": separation.mean(-1)}


def physical_relation_observer(draw_gram, target_gram, scales, power_floor):
    """Retain normalized self, signed mutual and floor-regularized rho coordinates."""
    if draw_gram.ndim != 5 or draw_gram.shape[1:] != target_gram.shape or draw_gram.shape[0] < 2:
        raise ValueError("Gram draws [J,B,S,C,C] must match target and have J >= 2")
    target = relation_coordinates(target_gram, scales, power_floor)
    draws = [relation_coordinates(item, scales, power_floor) for item in draw_gram]
    return {name: {"target": value, "draws": torch.stack([d[name] for d in draws]),
                   "accounting": query_accounting(torch.stack([d[name] for d in draws]), value)}
            for name, value in target.items()}


@torch.no_grad()
def scale_companion(family, members, target, *, coordinates, gram_scales, gram_power_floor):
    """Extract native statistics once and reproduce signed-v4 and R proper scores.

    Coefficients, piece means and Gram tensors are retained for independent
    re-scoring; input and outputs use the frozen family's declared units.
    """
    _validate_members(members, target)
    family._require_ready(target, coordinates)
    stats = [family._statistics(x) for x in (target, *members.unbind(0))]
    reference, draws = stats[0], stats[1:]
    signed = signed_band_terms(draws[0], draws[1:], reference, settings=family.settings,
        mass=family.mass, filters=family.filters, band_scale=family.band_scale,
        power_floor=family.power_floor, projections=family._projections(family.compute_dtype, target.device),
        band_weights=family._band_weights(family.compute_dtype, target.device),
        piece_measures=family.piece_measures if family.mean_augmented else None,
        mean_scale=family.piece_mean_scale if family.mean_augmented else None)
    blocks = [family._descriptors(s) for s in stats]
    self_target = blocks[0]["log_band_power"]
    self_draws = torch.stack([v["log_band_power"] for v in blocks[1:]])
    denominator = family.relation_self_scale.to(self_draws) * self_target.shape[-1] ** 0.5
    self_score = vector_energy_score(self_draws / denominator, self_target / denominator)
    rho_target = band_coherence(reference["gram"], family.power_floor)
    rho_draws = torch.stack([band_coherence(s["gram"], family.power_floor) for s in draws])
    rho_score = None if rho_target.numel() == 0 else tuple(x.mean((1, 2)) for x in vector_energy_score(rho_draws[..., None], rho_target[..., None]))
    proper = lambda x: {"score": x[0], "attraction": x[1], "separation": x[2]}
    retained = {key: {"target": reference[key], "draws": torch.stack([s[key] for s in draws])}
                for key in ("gram", "coefficients", "piece_means") if key in reference}
    return {"retained": retained, "physical": physical_relation_observer(retained["gram"]["draws"], retained["gram"]["target"], gram_scales, gram_power_floor),
            "signed_score": signed.per_sample_cost, "signed_report": signed.report,
            "self_proper": proper(self_score), "rho_proper": None if rho_score is None else proper(rho_score)}

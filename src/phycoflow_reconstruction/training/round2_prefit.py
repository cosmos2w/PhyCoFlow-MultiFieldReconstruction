"""Full-target SOURCE parameter graphs for common Round 2 calibration."""

import time
from dataclasses import replace

import torch

from ..coherence.query_sampling import full_target_query_batch
from ..data.physical_measure import attach_batch_measure, batch_measure_context
from ..data.sensor_protocols import build_observation_batch
from .coherence_sampling import _conditional_draw_seed
from .common import sensor_protocol_from_config
from .rollout import differentiable_reconstruction


def matched_panel_batches(runtime, dataset, config, panel, *, device, seed=1129001, deadline_unix=None):
    """Reconstruct the frozen B64 TRAIN panel and verify its saved support."""
    if panel.get("schema") != "car_source_panel_v1" or panel.get("provenance") != runtime.artifacts["provenance"]:
        raise ValueError("SOURCE panel and full-target runtime provenance differ")
    if dataset.split_name != "train":
        raise ValueError("Round 2 full-target calibration requires TRAIN and B64")
    ids = panel["sample_ids"]
    indices = panel["sample_indices"]
    if ids != panel["provenance"]["sample_ids"] or len(indices) != len(ids) or not ids:
        raise ValueError("SOURCE calibration sample manifest differs")
    if not isinstance(panel.get("observation_manifest"), dict) or set(panel["observation_manifest"]) != set(ids):
        raise ValueError("calibration requires explicit saved observation IDs for every sample")
    protocol = sensor_protocol_from_config(config)
    for start in range(0, len(indices), 64):
        if deadline_unix is not None and time.time() >= deadline_unix:
            raise TimeoutError("common prefit deadline reached at the batch boundary")
        stop = min(start + 64, len(indices))
        samples = [dataset[i] for i in indices[start:stop]]
        query_ids = panel["query_indices"][start:stop]
        if not torch.equal(query_ids, query_ids[0].expand_as(query_ids)):
            raise ValueError("calibration SOURCE panel requires its declared shared query support")
        complete = build_observation_batch(samples, replace(protocol, seed=seed + start),
                                           query_indices=query_ids[0],
                                           manifest_indices=panel["observation_manifest"])
        complete = attach_batch_measure(dataset, complete)
        if list(complete.sample_ids) != ids[start:stop]:
            raise ValueError("calibration sample ordering changed")
        for key, actual in (("reference", complete.target_fields), ("coordinates", complete.query_coords),
                            ("masses", batch_measure_context(complete)["generated_weights"]),
                            ("query_indices", complete.metadata["query_indices"])):
            if not torch.equal(panel[key][start:stop], actual):
                raise ValueError(f"calibration SOURCE panel {key} changed")
        yield start, complete.to(device)


def full_target_source_blocks(runtime, model, dataset, config, panel, *, device, batch_size=64, seed=1129001, deadline_unix=None):
    """Yield complete semantic block graphs on the frozen matched TRAIN panel.

    A fresh J1/J2 SOURCE rollout uses recorded deterministic batch/member seeds;
    no cached endpoints or endpoint-only Jacobians are used.
    """
    if batch_size != 64 or model.capabilities.structured_grid_required:
        raise ValueError("Round 2 SOURCE calibration requires point models and B64")
    target_queries = full_target_query_batch(len(runtime.fields))
    model.eval()
    batches = matched_panel_batches(runtime, dataset, config, panel, device=device, seed=seed,
                                    deadline_unix=deadline_unix)
    for start, complete in batches:
        reference = complete.target_fields
        source_batch = replace(complete, target_fields=None)
        members = []
        member_seeds = [_conditional_draw_seed(seed + start, j)
                        for j in range(2 if model.capabilities.generative else 1)]
        for member_seed in member_seeds:
            members.append(differentiable_reconstruction(
                model, source_batch, steps=int(config["rollout"]["steps"]),
                solver=config["rollout"]["solver"],
                generator=torch.Generator(device=device).manual_seed(member_seed),
                observation_config=dict(config["observation_consistency"])))
        context = {"sample_ids": complete.sample_ids, "reference_ids": complete.sample_ids,
                   **batch_measure_context(complete), "conditional_draws": tuple(members[1:]),
                   "exhaustive_shape_queries": True}
        blocks = {}
        for name, family in runtime.families.items():
            if family.target_use != "paired_supervised":
                raise ValueError("Round 2 common calibration requires paired target families")
            _, values, _ = runtime._score(name, family, members[0], reference,
                                          complete.query_coords, context, target_queries)
            blocks.update(values)
        expected = {key for key, active in runtime.calibration.active.items() if active}
        if not expected.issubset(blocks):
            raise ValueError("full target did not produce every active semantic block")
        yield list(complete.sample_ids), {key: blocks[key] for key in expected}


def fit_round2_common_prefit(runtime, model, dataset, config, panel, *, device, seed=1129001, deadline_unix=None):
    """Fit a common vector against real full-target SOURCE parameter gradients."""
    from ..coherence.adaptive_fitting import bank_target_digest
    from .internal_calibration import fit_common_prefit

    if not all(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("common prefit requires the complete trainable SOURCE parameterization")
    if runtime.calibration.coefficients is not None:
        raise ValueError("common prefit requires an unfrozen calibration")
    provenance = runtime.artifacts["provenance"]
    binding = {key: provenance[key] for key in
               ("source_checkpoint_sha256", "normalizer_sha256", "support_sha256", "field_order", "sample_ids")}
    binding.update({"panel_schema": panel["schema"], "member_seed_base": seed,
                    "batch_size": 64, "query_target": "full_outer_and_exhaustive_shape_inner_v1",
                    "bank_target_sha256": bank_target_digest(runtime.bank)})
    graphs = full_target_source_blocks(runtime, model, dataset, config, panel, device=device,
                                       seed=seed, deadline_unix=deadline_unix)
    return fit_common_prefit(runtime.calibration, graphs, tuple(model.parameters()), binding=binding)

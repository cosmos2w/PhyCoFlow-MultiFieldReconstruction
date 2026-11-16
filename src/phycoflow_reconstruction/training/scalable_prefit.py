"""Build or verify bounded TRAIN/SOURCE scalable-coherence artifacts."""

from __future__ import annotations

import json
import math
import os
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import torch

from ..data.manifest import dataset_fingerprint
from ..data.physical_measure import attach_batch_measure, batch_measure_context
from ..data.sensor_protocols import build_observation_batch
from ..data.training_batches import fixed_query_indices
from .coherence_sampling import _coherence_query_seed, _conditional_draw_seed, _preserved_rng
from .common import sensor_protocol_from_config
from .rollout import differentiable_reconstruction
from .run_store import file_sha256
from .scalable_fitting import _digest, fit_scalable_artifacts, validate_scalable_artifacts


def _load_common_core(declaration, *, provenance, model, config, batch, indexes, samples, query, protocol, manifest_indices, seed, members, sensor_count, microbatch):
    if members != 4:
        raise ValueError("common SOURCE panel reuse requires the frozen J4 member bank")
    core_path, panel_path = Path(declaration["path"]), Path(declaration["source_panel_path"])
    if file_sha256(core_path) != declaration["sha256"] or file_sha256(panel_path) != declaration["source_panel_sha256"]:
        raise ValueError("common core or SOURCE panel hash mismatch")
    core = torch.load(core_path, map_location="cpu", weights_only=True)
    panel = torch.load(panel_path, map_location="cpu", weights_only=True)
    validate_scalable_artifacts(core, core["provenance"])
    if panel.get("schema") != "car_source_panel_v1" or panel.get("provenance") != core["provenance"]:
        raise ValueError("common SOURCE panel and core identity mismatch")
    if any(core["provenance"][k] != provenance[k] for k in provenance if k != "config_sha256"):
        raise ValueError("common core SOURCE, TRAIN population or support mismatch")
    if panel.get("parameter_buffer_sha256") != _digest(model.state_dict()) or panel.get("source_weight_view_sha256") != getattr(model, "source_weight_view_sha256", None):
        raise ValueError("common SOURCE panel materialized parameter/buffer view mismatch")
    if panel.get("source_weight_view") != config.get("source", {}).get("weight_view", "live") or panel.get("members_count") != members or panel.get("sensor_count") != sensor_count:
        raise ValueError("common SOURCE weight view, member or sensor contract mismatch")
    if panel.get("sample_indices") != indexes or panel.get("sample_ids") != list(batch.sample_ids) or panel.get("observation_manifest") != manifest_indices:
        raise ValueError("common SOURCE sample or observation manifest mismatch")
    for key, expected in (("reference", batch.target_fields), ("coordinates", batch.query_coords), ("masses", batch_measure_context(batch)["generated_weights"]), ("query_indices", batch.metadata["query_indices"])):
        if not torch.equal(panel[key], expected):
            raise ValueError(f"common SOURCE panel {key} mismatch")
    expected_coupling = []
    for start in range(0, len(indexes), microbatch):
        model_batch = build_observation_batch(samples[start:start + microbatch], replace(protocol, seed=seed + start), query_indices=None if model.capabilities.structured_grid_required else query, manifest_indices=manifest_indices)
        expected_coupling.append({"sample_ids": list(model_batch.sample_ids), "observation_sha256": _digest([model_batch.obs_indices, model_batch.obs_values, model_batch.obs_field_ids, model_batch.obs_valid_mask]), "member_seeds": [_conditional_draw_seed(seed + start, j) for j in range(members)]})
    if panel.get("coupling") != expected_coupling or panel["members"].shape != (len(indexes), members, *batch.target_fields.shape[1:]) or not torch.isfinite(panel["members"]).all():
        raise ValueError("common SOURCE observation or member stream mismatch")
    parent_run = core_path.parent.parent
    import yaml

    parent_config = yaml.safe_load((parent_run / "resolved_config.yaml").read_text())
    for key in ("rollout", "observations", "observation_consistency"):
        if parent_config[key] != config[key]:
            raise ValueError(f"common SOURCE {key} contract differs")
    if parent_config["coherence"]["families"] != config["coherence"]["families"] or parent_config["coherence"]["compute_budget"] != config["coherence"]["compute_budget"] or parent_config["runtime"].get("seed", 42) != config["runtime"].get("seed", 42):
        raise ValueError("common core family, support or seed configuration differs")
    parent_settings = parent_config["coherence"].get("scalable_anchored_coherence", {})
    current_settings = config["coherence"].get("scalable_anchored_coherence", {})
    for key, default in (("field_floor", 1e-6), ("candidate_count", None), ("shape_batches", 8)):
        if parent_settings.get(key, default) != current_settings.get(key, default):
            raise ValueError(f"common core fitting setting differs: {key}")
    parent_manifest = json.loads((parent_run / "run_manifest.json").read_text())
    frozen = config["coherence"].get("frozen_family_artifacts", {})
    family_hashes = {}
    for name in config["coherence"]["families"]:
        if not config["coherence"]["families"][name].get("enabled", True):
            continue
        path = core_path.parent / f"{name}_family.pt"
        digest = file_sha256(path)
        if parent_manifest.get("coherence_family_state_sha256s", {}).get(name) != digest or frozen.get(name, {}).get("sha256") != digest or file_sha256(Path(frozen[name]["path"])) != digest:
            raise ValueError(f"common core requires the exact frozen parent family: {name}")
        family_hashes[name] = digest
    return core, panel, family_hashes


def _core_body_digest(artifacts):
    body = {k: deepcopy(artifacts[k]) for k in ("state_bank", "shape_artifact", "priorities", "units", "active")}
    body["shape_artifact"]["provenance"].pop("config_sha256", None)
    return _digest(body)


def prepare_scalable_artifacts(
    model, dataset, families, config, *, source_sha256, device, artifact_path
):
    """Fit once from reproducibly selected ~10% TRAIN conditions; resume verifies identity."""
    if dataset.split_name != "train" or dataset.selection.split != "train":
        raise ValueError("scalable fit requires TRAIN dataset")
    settings = config["coherence"].get("scalable_anchored_coherence", {})
    n = max(1, math.ceil(len(dataset) * 0.1))
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
    sensor_count = settings.get("source_sensor_count")
    manifest_indices = None
    if sensor_count is not None:
        if "T" not in dataset.data_spec.field_names:
            raise ValueError("fixed source sensor panel requires field T")
        field = dataset.data_spec.field_names.index("T")
        manifest_indices = {}
        for index, sample in zip(indexes, samples):
            condition_seed = 2047 + int(index)
            points = torch.randperm(math.prod(dataset.data_spec.logical_shape),
                                   generator=torch.Generator().manual_seed(condition_seed))[:int(sensor_count)]
            sample_id = f"{sample.trajectory_id}:{sample.time_index}"
            manifest_indices[sample_id] = [(int(point), field) for point in points]
    batch = build_observation_batch(samples, protocol, query_indices=query,
                                    manifest_indices=manifest_indices)
    batch = attach_batch_measure(dataset, batch)
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
        "definition_version": "scalable_anchored_coherence_streamed_v2",
    }
    if artifact_path.exists():
        artifacts = torch.load(artifact_path, map_location="cpu", weights_only=True)
        validate_scalable_artifacts(artifacts, provenance)
        if settings.get("common_core") is not None:
            receipt_path = artifact_path.with_name("common_core_receipt.json")
            receipt = json.loads(receipt_path.read_text())
            if receipt.get("new_provenance") != provenance or receipt.get("core_body_sha256") != _core_body_digest(artifacts) or receipt.get("parent_artifact_sha256") != settings["common_core"]["sha256"] or receipt.get("parent_source_panel_sha256") != settings["common_core"]["source_panel_sha256"]:
                raise ValueError("common-core resume receipt mismatch")
        return artifacts, provenance, None
    reference = batch.target_fields
    began = perf_counter()
    was_training = model.training
    try:
        model.eval()
        with _preserved_rng(device), torch.no_grad():
            members = int(settings.get("source_members", 2)) if model.capabilities.generative else 1
            microbatch = 1 if model.capabilities.structured_grid_required else min(8, n)
            common = settings.get("common_core")
            reuse_core = None
            family_hashes = {}
            if common is None:
                draws = [[] for _ in range(members)]
                # Full-grid models must not materialize the whole reference panel
                # as a native-grid batch. Each chunk has a recorded deterministic
                # sensor and member stream, independent of the training stream.
                coupling = []
                for start in range(0, n, microbatch):
                    chunk_protocol = replace(protocol, seed=seed + start)
                    model_batch = build_observation_batch(
                        samples[start : start + microbatch],
                        chunk_protocol,
                        query_indices=None if model.capabilities.structured_grid_required else query,
                        manifest_indices=manifest_indices,
                    ).to(device)
                    coupling.append({
                        "sample_ids": list(model_batch.sample_ids),
                        "observation_sha256": _digest([model_batch.obs_indices, model_batch.obs_values,
                                                        model_batch.obs_field_ids, model_batch.obs_valid_mask]),
                        "member_seeds": [_conditional_draw_seed(seed + start, j) for j in range(members)],
                    })
                    model_batch.target_fields = None
                    for j in range(members):
                        prediction = differentiable_reconstruction(
                            model,
                            model_batch,
                            steps=int(config["rollout"]["steps"]),
                            solver=config["rollout"]["solver"],
                            generator=torch.Generator(device=device).manual_seed(
                                _conditional_draw_seed(seed + start, j)
                            ),
                            observation_config=dict(config["observation_consistency"]),
                        ).detach()
                        if model.capabilities.structured_grid_required:
                            prediction = prediction[:, query.to(device)]
                        draws[j].append(prediction.cpu())
                draws = [torch.cat(member, dim=0) for member in draws]
            else:
                reuse_core, panel, family_hashes = _load_common_core(
                    common, provenance=provenance, model=model, config=config, batch=batch,
                    indexes=indexes, samples=samples, query=query, protocol=protocol,
                    manifest_indices=manifest_indices, seed=seed, members=members,
                    sensor_count=sensor_count, microbatch=microbatch,
                )
                draws = [panel["members"][:, j] for j in range(members)]
                coupling = panel["coupling"]
        fit_families = deepcopy(families)
        for family in fit_families.values():
            if hasattr(family, "cpu"):
                family.cpu()
        if settings.get("save_source_panel", False) or reuse_core is not None:
            torch.save({
                "schema": "car_source_panel_v1", "provenance": provenance,
                "members": torch.stack(draws, dim=1), "reference": reference,
                "coordinates": coords, "masses": weights,
                "sample_indices": indexes, "sample_ids": list(batch.sample_ids),
                "query_indices": batch.metadata["query_indices"],
                "observation_manifest": manifest_indices, "coupling": coupling,
                "source_weight_view": config.get("source", {}).get("weight_view", "live"),
                "source_weight_view_sha256": getattr(model, "source_weight_view_sha256", None),
                "parameter_buffer_sha256": _digest(model.state_dict()),
                "members_count": members, "sensor_count": sensor_count,
                "prior_buffers_sha256": _digest({k: v for k, v in model.state_dict().items()
                                                 if k in ("prior.omega", "prior.phase")}),
            }, artifact_path.with_name("source_panel.pt"))
        artifacts = fit_scalable_artifacts(
            families=fit_families,
            source=torch.stack(draws, dim=1),
            reference=reference,
            coordinates=coords,
            provenance=provenance,
            source_weights=weights,
            reference_weights=weights,
            seed=seed,
            max_samples=n,
            candidate_count=settings.get("candidate_count"),
            shape_batches=int(settings.get("shape_batches", 8)),
            field_floor=float(settings.get("field_floor", 1e-6)),
            reuse_core=reuse_core,
            adaptive_settings=settings.get("adaptive")
            if settings.get("adaptive", {}).get("enabled", False)
            else None,
        )
        if reuse_core is not None:
            body_hash = _core_body_digest(reuse_core)
            if _core_body_digest(artifacts) != body_hash:
                raise ValueError("rebased common core changed its fitted body")
            receipt = {"schema": "car_common_core_reuse_v1", "parent_artifact_sha256": common["sha256"], "parent_source_panel_sha256": common["source_panel_sha256"], "parent_provenance": reuse_core["provenance"], "new_provenance": provenance, "core_body_sha256": body_hash, "parent_family_sha256": family_hashes, "parameter_buffer_sha256": _digest(model.state_dict()), "source_weight_view_sha256": getattr(model, "source_weight_view_sha256", None)}
            receipt_path = artifact_path.with_name("common_core_receipt.json")
            if receipt_path.exists():
                if json.loads(receipt_path.read_text()) != receipt:
                    raise ValueError("existing common-core receipt differs")
            else:
                with receipt_path.open("x", encoding="utf-8") as stream:
                    json.dump(receipt, stream, sort_keys=True, allow_nan=False)
                    stream.write("\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                receipt_path.chmod(0o444)
    finally:
        model.train(was_training)
    return (
        artifacts,
        provenance,
        {
            "sample_count": n,
            "source_forward_count": 0 if reuse_core is not None else members * math.ceil(n / microbatch),
            "microbatch_size": microbatch,
            "stream_seed": seed,
            "seconds": perf_counter() - began,
        },
    )

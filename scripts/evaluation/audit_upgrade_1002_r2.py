"""Audit R2 LIVE checkpoints; final test access requires a frozen recipe and one-use receipt."""

from __future__ import annotations

import argparse
import json
import os
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from phycoflow_reconstruction.coherence import ReferenceBank, build_enabled_families
from phycoflow_reconstruction.config import load_config
from phycoflow_reconstruction.data.factory import open_field_dataset
from phycoflow_reconstruction.data.manifest import dataset_fingerprint, manifest_from_batch
from phycoflow_reconstruction.evaluation.checkpoint import _checkpoint_path, load_evaluation_runtime
from phycoflow_reconstruction.evaluation.coherence_set import _restore_evaluation_family
from phycoflow_reconstruction.evaluation.upgrade_1002_r2_protocol import (
    build_panel_batch,
    declare_panels,
    grouped_evaluate,
    grouped_native_monitor,
    render_pressure_comparison,
    save_declaration,
    selection_indices,
)
from phycoflow_reconstruction.training.common import sensor_protocol_from_config
from phycoflow_reconstruction.training.fidelity_controller import coherence_selection_report
from phycoflow_reconstruction.training.native_topology_audit import _preserve_audit_state
from phycoflow_reconstruction.training.post_training import _build_comparison_batch, _without_target
from phycoflow_reconstruction.training.run_store import (
    config_digest,
    file_sha256,
    load_project_checkpoint,
)
from phycoflow_reconstruction.training.source import source_checkpoint_path


def validate_recipe_lock(lock_path, run, checkpoint, declaration, test_root):
    """CPU identities only: never construct a model or open any dataset."""
    if lock_path is None:
        raise ValueError("test access requires --recipe-lock")
    lock_path.resolve().relative_to(test_root.resolve())
    lock = json.loads(lock_path.read_text())
    if (
        lock.get("schema") != "phycoflow.upgrade_1002_r2_final_recipe_lock.v1"
        or lock.get("locked_final_recipe") is not True
    ):
        raise ValueError("test access requires a locked final recipe")
    if (
        lock.get("use_test_for_selection") is not False
        or lock.get("use_test_for_adaptation") is not False
    ):
        raise ValueError("test cannot be used for selection or adaptation")
    run = run.resolve()
    run.relative_to(test_root.resolve())
    child = _checkpoint_path(run, checkpoint).resolve()
    if not child.is_relative_to(run / "checkpoints"):
        raise ValueError("locked child checkpoint must belong to this run/checkpoints")
    config = load_config(run / "resolved_config.yaml")
    source = source_checkpoint_path(config).resolve()
    if (
        lock.get("run") != str(run)
        or lock.get("checkpoint") != str(child)
        or lock.get("source_checkpoint") != str(source)
    ):
        raise ValueError("recipe lock paths do not match the resolved run/checkpoint/source")
    panel_path = test_root / "_audit/R2_campaign/final_test_panel_declaration.json"
    identities = {
        "checkpoint_sha256": child,
        "config_sha256": run / "resolved_config.yaml",
        "run_manifest_sha256": run / "run_manifest.json",
        "source_checkpoint_sha256": source,
        "declaration_sha256": declaration,
        "test_panel_declaration_sha256": panel_path,
    }
    if any(lock.get(key) != file_sha256(path) for key, path in identities.items()):
        raise ValueError("recipe lock artifact hash mismatch")
    manifest = json.loads((run / "run_manifest.json").read_text())
    if manifest["source_hashes"]["checkpoint"] != lock["source_checkpoint_sha256"]:
        raise ValueError("recipe lock source differs from the child's immutable source")
    for key, filename in (
        ("resolved_config", "resolved_config.yaml"),
        ("run_manifest", "run_manifest.json"),
    ):
        expected_source_hash = manifest["source_hashes"].get(key)
        if (
            expected_source_hash is not None
            and file_sha256(Path(config["source_run"]) / filename) != expected_source_hash
        ):
            raise ValueError("immutable source metadata hash mismatch: " + key)
    payload = load_project_checkpoint(child)
    semantic_digest = config_digest(config)
    if (
        manifest.get("config_sha256") != semantic_digest
        or payload.get("config_sha256") != semantic_digest
    ):
        raise ValueError("checkpoint/manifest semantic config digest association mismatch")
    if payload.get("source_hashes") != manifest["source_hashes"]:
        raise ValueError("checkpoint source hashes differ from the hashed run manifest")
    divisor = int(manifest["steps_per_epoch"])
    step = int(payload["global_step"])
    if divisor < 1 or step % divisor or step // divisor < 100:
        raise ValueError("locked final test requires a mature exact epoch boundary")
    panel = json.loads(panel_path.read_text())
    indices = selection_indices(1000, 128, 16)
    expected = {
        "schema": "phycoflow.upgrade_1002_r2_final_test_panel.v1",
        "split": "test",
        "data_role": "locked_final_test",
        "access_allowed": False,
        "dataset_size": 1000,
        "sample_count": 128,
        "strata": 16,
        "dataset_indices": indices,
        "groups": [indices[start : start + 32] for start in range(0, 128, 32)],
        "equal_sample_weights": True,
        "sensor_seed_offset": 2000,
        "generation_seed": 6027,
        "native_seeds": [6027, 7027],
        "weight_selection": "live",
        "query_count": 4096,
        "topology_bank": "full_16",
        "dataset_opened_to_declare": False,
        "model_or_recipe_chosen": False,
    }
    if any(panel.get(key) != value for key, value in expected.items()):
        raise ValueError(
            "final test panel differs from the immutable 128/16-strata/groups32 declaration"
        )
    query_count = config.get("evaluation", {}).get(
        "query_points", config.get("coherence", {}).get("compute_budget", {}).get("point_count")
    )
    if query_count != panel["query_count"]:
        raise ValueError("frozen recipe query count differs from final test declaration")
    topology = config.get("coherence", {}).get("families", {}).get("topology", {})
    if topology.get("components", {}).get("mutual", {}).get("line_bank_size") != 16:
        raise ValueError("final test requires the frozen full 16-line C bank")
    return {
        "lock": lock,
        "lock_path": str(lock_path.resolve()),
        "lock_sha256": file_sha256(lock_path),
        "panel": panel,
        "epoch": step // divisor,
    }


def begin_final_test_access(validated, test_root, output):
    """Campaign-wide exclusive receipt: failures also consume the one access attempt."""
    if output.exists():
        raise ValueError("final test output must be a fresh path")
    path = test_root / "_audit/R2_campaign/final_test_access.json"
    if path.resolve() == output.resolve() or path.resolve() in output.resolve().parents:
        raise ValueError("final test output must not collide with the campaign access receipt")
    payload = {
        "schema": "phycoflow.upgrade_1002_r2_final_test_access.v1",
        "recipe_lock": validated["lock_path"],
        "recipe_lock_sha256": validated["lock_sha256"],
        "recipe": validated["lock"],
        "output": str(output.resolve()),
        "state": "execution_begins_before_any_test_dataset_open",
        "second_access_for_any_recipe_forbidden": True,
        "failure_consumes_access_attempt": True,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, indent=2, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    return {"path": str(path.resolve()), "sha256": file_sha256(path)}


def validate_native_field_indices(indices, dataset_size, epoch):
    if not indices:
        return
    if epoch < 100:
        raise ValueError("native field review requires a mature checkpoint at epoch >=100")
    if not 2 <= len(indices) <= 4 or len(set(indices)) != len(indices):
        raise ValueError("native field review requires 2–4 unique validation indices")
    if any(index < 0 or index >= dataset_size for index in indices):
        raise ValueError("native field index lies outside validation")


def save_native_fields(runtime, source_model, indices, config, protocol, output, *, epoch):
    """Only matched full-lattice reconstruction; no family/PH evaluation or fitting."""
    batch = build_panel_batch(runtime.dataset, protocol, indices, runtime.device)
    query_count = batch.query_coords.shape[1]
    if (
        query_count != int(np.prod(runtime.dataset.data_spec.logical_shape))
        or not bool(batch.query_valid_mask.all())
        or not torch.equal(
            batch.metadata["query_indices"],
            torch.arange(query_count, device=runtime.device).expand(len(indices), -1),
        )
    ):
        raise ValueError("native field review requires each complete lattice point exactly once")
    inference_batch = _without_target(batch)
    predictions = {}
    seed = int(config.get("evaluation", {}).get("seed", 2027))
    with _preserve_audit_state((runtime.model, source_model)), torch.no_grad():
        for role, model in (("source_live", source_model), ("candidate_live", runtime.model)):
            model.eval()
            generator = torch.Generator(device=runtime.device).manual_seed(seed)
            prediction = model.reconstruct(
                inference_batch, steps=runtime.generation_steps, generator=generator
            ).prediction
            if prediction.shape != batch.target_fields.shape:
                raise ValueError(
                    "native reconstruction did not return all fields on the full lattice"
                )
            predictions[role] = prediction.detach().cpu()
    raw_coordinates = torch.stack([runtime.dataset[index].coordinates_raw for index in indices])
    observation_indices = batch.obs_indices.detach().cpu()
    observation_coordinates = torch.stack(
        [raw_coordinates[row][observation_indices[row].clamp_min(0)] for row in range(len(indices))]
    )
    target = batch.target_fields.detach().cpu()
    payload = {
        "coordinates_raw": raw_coordinates.numpy(),
        "ground_truth_model": target.numpy(),
        "ground_truth": runtime.dataset.normalizer.decode(target).numpy(),
        "field_names": np.asarray(runtime.dataset.field_names),
        "field_units": np.asarray(runtime.dataset.data_spec.field_units),
        "sample_ids": np.asarray(batch.sample_ids),
        "epoch": np.asarray(epoch),
        "full_native_grid": np.asarray(True),
        "query_indices": batch.metadata["query_indices"].detach().cpu().numpy(),
        "observation_coordinates_raw": observation_coordinates.numpy(),
        "observation_indices": observation_indices.numpy(),
        "observation_field_ids": batch.obs_field_ids.detach().cpu().numpy(),
        "observation_valid_mask": batch.obs_valid_mask.detach().cpu().numpy(),
        "generation_seed": np.asarray(seed),
        "generation_steps": np.asarray(runtime.generation_steps),
        "dataset_indices": np.asarray(indices),
        "split": np.asarray("validation"),
        "data_role": np.asarray("development_visual_review_not_fresh_audit"),
    }
    for role, prediction in predictions.items():
        payload[role + "_model"] = prediction.numpy()
        payload[role] = runtime.dataset.normalizer.decode(prediction).numpy()
        payload[role + "_signed_error_model"] = (prediction - target).numpy()
        payload[role + "_signed_error"] = payload[role] - payload["ground_truth"]
    path = output / f"fields_epoch_{epoch:03d}_native.npz"
    np.savez_compressed(path, **payload)
    return {
        "path": path.name,
        "sha256": file_sha256(path),
        "sample_count": len(indices),
        "dataset_indices": indices,
        "query_points": target.shape[1],
        "full_native_grid": True,
        "data_role": "development_visual_review_not_fresh_audit",
        "generation_seed": seed,
        "source_and_candidate_weight_selection": "live",
        "runs_extra_ABC_or_PH": False,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--declaration", required=True, type=Path)
    parser.add_argument("--declare-only", action="store_true")
    parser.add_argument("--dataset-size", type=int, default=1000)
    parser.add_argument(
        "--previously-used-until",
        type=int,
        default=96,
        help="Exclusive split index for already inspected R1 development blocks",
    )
    parser.add_argument(
        "--previously-used-index",
        action="append",
        type=int,
        default=[643],
        help="Additional previously inspected split index; R1 native audit includes 643",
    )
    parser.add_argument("--run", type=Path)
    parser.add_argument("--checkpoint", default="last")
    parser.add_argument(
        "--recipe-lock", type=Path, help="Frozen final recipe JSON; test access only"
    )
    parser.add_argument(
        "--panel",
        choices=("training", "selection", "extended", "audit", "test"),
        default="extended",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--native-noise-seed",
        action="append",
        type=int,
        default=[],
        help="Additional predeclared native draw/replay seeds at epoch >=100; separate evidence",
    )
    parser.add_argument(
        "--native-field-index",
        action="append",
        type=int,
        default=[],
        help="2–4 validation indices for full-native field review at epoch >=100",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--pressure-samples",
        type=int,
        default=0,
        help="PDF raw-coordinate p review on this many already evaluated query fields",
    )
    args = parser.parse_args(argv)
    repository = Path(__file__).resolve().parents[2]
    test_root = repository / "cases/turbulent_combustion/runs/Test_1002"
    args.declaration.resolve().relative_to(test_root.resolve())
    if args.declare_only:
        declaration = declare_panels(
            args.dataset_size,
            previously_used_indices=list(range(args.previously_used_until))
            + args.previously_used_index,
        )
        save_declaration(args.declaration, declaration)
        print(
            json.dumps(
                {
                    "declaration": str(args.declaration.resolve()),
                    "sha256": file_sha256(args.declaration),
                    "test_locked": True,
                }
            )
        )
        return
    if args.run is None or args.output is None:
        parser.error("audits require --run and --output")
    args.output.resolve().relative_to(test_root.resolve())
    if args.device.startswith("cuda") and (
        args.device != "cuda:0" or os.environ.get("CUDA_VISIBLE_DEVICES") != "0"
    ):
        raise ValueError("R2 audits require physical GPU0 via CUDA_VISIBLE_DEVICES=0 and cuda:0")
    declaration = json.loads(args.declaration.read_text())
    if not declaration.get("test_locked") or declaration.get("split") != "validation":
        raise ValueError("R2 declaration must preserve validation roles and locked test")
    final_test = None
    if args.panel == "test":
        if args.native_field_index or args.native_noise_seed or args.pressure_samples:
            raise ValueError("test forbids exploratory field probes and additional noise draws")
        final_test = validate_recipe_lock(
            args.recipe_lock, args.run, args.checkpoint, args.declaration, test_root
        )
        final_test["access_receipt"] = begin_final_test_access(final_test, test_root, args.output)
    elif args.recipe_lock is not None:
        raise ValueError("--recipe-lock is reserved for the final test panel")
    runtime_split = "test" if final_test is not None else "validation"
    runtime = load_evaluation_runtime(
        args.run,
        split=runtime_split,
        checkpoint=args.checkpoint,
        sensor_config=None,
        generation_steps=None,
        device_name=args.device,
        include_temporal_derivative=False,
    )
    config = deepcopy(runtime.config)
    checkpoint = load_project_checkpoint(runtime.checkpoint_path)
    manifest = json.loads((args.run / "run_manifest.json").read_text())
    steps_per_epoch = int(manifest["steps_per_epoch"])
    committed_updates = int(checkpoint["global_step"])
    if steps_per_epoch < 1 or committed_updates % steps_per_epoch:
        runtime.dataset.close()
        raise ValueError("R2 broad audits require an exact saved historical epoch boundary")
    epoch = committed_updates // steps_per_epoch
    if args.native_noise_seed and epoch < 100:
        runtime.dataset.close()
        raise ValueError("native noise checks require a mature checkpoint at epoch >=100")
    validate_native_field_indices(args.native_field_index, len(runtime.dataset), epoch)
    if args.panel in {"extended", "audit"} and epoch < 100:
        runtime.dataset.close()
        raise ValueError("broader R2 audit requires a mature checkpoint at epoch >=100")
    source_run = Path(config["source_run"]).resolve()
    source_checkpoint = source_checkpoint_path(config).resolve()
    source = load_evaluation_runtime(
        source_run,
        split=runtime_split,
        checkpoint=str(source_checkpoint),
        sensor_config=None,
        generation_steps=runtime.generation_steps,
        device_name=args.device,
        include_temporal_derivative=False,
    )
    audited_dataset = runtime.dataset
    if args.panel == "training":
        audited_dataset = open_field_dataset(config["dataset"], split="train")
    try:
        if len(runtime.dataset) != (
            1000 if final_test is not None else declaration["dataset_size"]
        ):
            raise ValueError("declared split size differs from actual dataset")
        if runtime.dataset.normalizer.digest() != source.dataset.normalizer.digest():
            raise ValueError("source/candidate normalizers differ")
        if file_sha256(source_checkpoint) != manifest["source_hashes"]["checkpoint"]:
            raise ValueError("immutable source identity changed")
        split = "train" if args.panel == "training" else runtime_split
        if final_test is not None:
            panel = final_test["panel"]
        elif args.panel == "training":
            panel = {
                "dataset_indices": selection_indices(len(audited_dataset), 32, 16),
                "sensor_seed_offset": 0,
                "generation_seed": 2027,
                "native_seeds": [2027, 3027],
            }
        else:
            panel = declaration[args.panel]
        protocol = sensor_protocol_from_config(config)
        protocol = replace(protocol, seed=protocol.seed + panel["sensor_seed_offset"])
        complete = build_panel_batch(
            audited_dataset, protocol, panel["dataset_indices"], runtime.device
        )
        config["evaluation"]["seed"] = panel["generation_seed"]
        config["evaluation"]["native_topology"] = False
        comparison = _build_comparison_batch(complete, config)
        realization = manifest_from_batch(comparison, audited_dataset.path, split)
        families = build_enabled_families(
            config["coherence"], runtime.dataset.data_spec, runtime.dataset.normalizer
        )
        banks, artifacts = {}, {}
        for name, family in families.items():
            family.to(runtime.device)
            artifacts[name] = _restore_evaluation_family(runtime, family, name)
            banks[name] = None
            if family.target_use == "training_reference":
                bank_name = (
                    "coherence_reference.pt"
                    if name == "global_distribution"
                    else f"coherence_reference_{name}.pt"
                )
                bank = ReferenceBank.load(args.run / "artifacts" / bank_name)
                if bank.digest() != manifest["reference_bank_sha256s"][name]:
                    raise ValueError(f"reference bank identity changed: {name}")
                banks[name] = bank
        scales = manifest.get("coherence_family_scales", {})
        results = {}
        pressure_captures = {}
        field_positions = np.linspace(
            0,
            len(comparison.sample_ids) - 1,
            min(max(args.pressure_samples, 0), len(comparison.sample_ids)),
            dtype=int,
        ).tolist()
        field_probe_positions = {
            comparison.sample_ids[position]: position for position in field_positions
        }
        with _preserve_audit_state((runtime.model, source.model)):
            runtime.model.eval()
            source.model.eval()
            for model in (runtime.model, source.model):
                if any(module.__dict__.get("_ema_eval") is True for module in model.modules()):
                    raise RuntimeError("R2 source/candidate inference must bypass configured EMA")
            for role, model in (("source_live", source.model), ("candidate_live", runtime.model)):
                captured = []

                def capture_pressure(prediction, batch, captured=captured):
                    if (
                        len(captured) >= args.pressure_samples
                        or "p" not in runtime.dataset.field_names
                    ):
                        return
                    field_index = runtime.dataset.field_names.index("p")
                    decoded = runtime.dataset.normalizer.decode(prediction)
                    for index, sample_id in enumerate(batch.sample_ids):
                        if sample_id not in field_probe_positions:
                            continue
                        if len(captured) >= args.pressure_samples:
                            break
                        captured.append(
                            {
                                "sample_id": sample_id,
                                "panel_position": field_probe_positions[sample_id],
                                "p": decoded[index, :, field_index].cpu().numpy(),
                                "decoded_fields": decoded[index].cpu().numpy(),
                                "model_fields": prediction[index].cpu().numpy(),
                            }
                        )

                results[role] = grouped_evaluate(
                    model,
                    complete,
                    comparison,
                    families,
                    banks,
                    tuple(runtime.dataset.field_names),
                    config,
                    normalizer=runtime.dataset.normalizer,
                    family_scales=scales,
                    endpoint_callback=capture_pressure,
                )
                pressure_captures[role] = captured
            native = grouped_native_monitor(
                runtime.model, source.model, {args.panel: comparison}, seeds=panel["native_seeds"]
            )
            noise_check = (
                grouped_native_monitor(
                    runtime.model,
                    source.model,
                    {args.panel: comparison},
                    seeds=args.native_noise_seed,
                )
                if args.native_noise_seed
                else None
            )
        if "posttrain_fidelity" in config and final_test is None:
            selection = coherence_selection_report(
                results["source_live"], results["candidate_live"], config["posttrain_fidelity"]
            )
            selection.pop("metrics", None)
        else:
            selection = None
        output = {
            "schema": "phycoflow.upgrade_1002_r2_audit.v1",
            "epoch": epoch,
            "purpose": "non_scientific_forensic_probe"
            if epoch < 100
            else "mature_matched_panel_audit",
            "scientific_maturity": epoch >= 100,
            "run": str(args.run.resolve()),
            "checkpoint": str(runtime.checkpoint_path),
            "checkpoint_sha256": file_sha256(runtime.checkpoint_path),
            "source_checkpoint_sha256": file_sha256(source_checkpoint),
            "source_weight_selection": "live",
            "candidate_weight_selection": "live",
            "configured_source_ema_eval": bool(source.config["model"].get("model_ema_eval", True)),
            "ema_evaluation_bypassed_and_restored": True,
            "declaration_sha256": file_sha256(args.declaration),
            "panel": args.panel,
            "split": split,
            "dataset_indices": panel["dataset_indices"],
            "dataset_fingerprint": dataset_fingerprint(runtime.dataset.path),
            "sensor_manifest_sha256": realization.digest(),
            "family_artifacts": artifacts,
            "test_locked": True,
            "fits_descriptors": False,
            "updates_model": False,
            "results": results,
            "matched_native_monitor": native,
            "fidelity_coherence_audit": selection,
        }
        if final_test is not None:
            output.update(
                {
                    "purpose": "locked_final_test",
                    "data_role": "locked_final_test",
                    "frozen_final_recipe": final_test["lock"],
                    "recipe_lock_sha256": final_test["lock_sha256"],
                    "access_receipt": final_test["access_receipt"],
                    "test_used_for_selection": False,
                    "test_used_for_adaptation": False,
                    "test_for_final_frozen_evaluation_only": True,
                    "sample_count": 128,
                    "test_split_count": 1000,
                    "coverage_limitation": "128 predeclared snapshots of 1000; not the whole test split",
                }
            )
        args.output.mkdir(parents=True, exist_ok=False)
        realization.save(args.output / "sensor_manifest.json")
        if noise_check is not None:
            replay = []
            for draw in noise_check[args.panel]["draws"]:
                baseline = next(
                    (item for item in native[args.panel]["draws"] if item["seed"] == draw["seed"]),
                    None,
                )
                if baseline is not None:
                    replay.append(
                        {
                            "seed": draw["seed"],
                            "candidate_abs_difference": abs(
                                draw["candidate"] - baseline["candidate"]
                            ),
                            "source_abs_difference": abs(draw["source"] - baseline["source"]),
                        }
                    )
            noise_payload = {
                "schema": "phycoflow.upgrade_1002_r2_native_noise_check.v1",
                "epoch": epoch,
                "panel": args.panel,
                "dataset_indices": panel["dataset_indices"],
                "sensor_manifest_sha256": realization.digest(),
                "checkpoint_sha256": output["checkpoint_sha256"],
                "source_checkpoint_sha256": output["source_checkpoint_sha256"],
                "requested_seeds": args.native_noise_seed,
                "baseline_seeds": panel["native_seeds"],
                "baseline_replay_precision": replay,
                "matched_native_draws": noise_check,
                "role": "development_noise_check_only_no_selection_controller_or_fitting",
                "tolerance_interpretation": "2_percent_descriptive_only_no_claim_above_MC_variation_from_two_draws",
                "test_locked": True,
                "fits_descriptors": False,
                "updates_model": False,
            }
            noise_path = args.output / "native_noise_check.json"
            noise_path.write_text(json.dumps(noise_payload, indent=2, allow_nan=False) + "\n")
            output["native_noise_check"] = {
                "path": noise_path.name,
                "sha256": file_sha256(noise_path),
            }
        if args.native_field_index:
            output["native_fields"] = save_native_fields(
                runtime,
                source.model,
                args.native_field_index,
                config,
                protocol,
                args.output,
                epoch=epoch,
            )
        pressure_figures = []
        saved_fields = []
        for position, (base, child) in enumerate(
            zip(pressure_captures["source_live"], pressure_captures["candidate_live"])
        ):
            if base["sample_id"] != child["sample_id"]:
                raise ValueError("source/candidate pressure snapshots differ")
            panel_position = base["panel_position"]
            sample = audited_dataset[panel["dataset_indices"][panel_position]]
            query_ids = comparison.metadata["query_indices"][panel_position].cpu()
            coordinates_raw = sample.coordinates_raw[query_ids].numpy()
            field_index = runtime.dataset.field_names.index("p")
            target = (
                runtime.dataset.normalizer.decode(comparison.target_fields[panel_position])[
                    ..., field_index
                ]
                .cpu()
                .numpy()
            )
            path = render_pressure_comparison(
                coordinates_raw,
                target,
                base["p"],
                child["p"],
                args.output / f"pressure_{position:02d}_epoch_{epoch:03d}.pdf",
                sample_id=base["sample_id"],
                epoch=epoch,
            )
            pressure_figures.append(
                {
                    "path": path.name,
                    "sample_id": base["sample_id"],
                    "query_points": len(query_ids),
                    "raw_coordinates": True,
                    "full_native_grid": len(query_ids) == len(sample.coordinates_raw),
                }
            )
            saved_fields.append(
                {
                    "coordinates_raw": coordinates_raw,
                    "ground_truth": runtime.dataset.normalizer.decode(
                        comparison.target_fields[panel_position]
                    )
                    .cpu()
                    .numpy(),
                    "ground_truth_model": comparison.target_fields[panel_position].cpu().numpy(),
                    "source_live": base["decoded_fields"],
                    "candidate_live": child["decoded_fields"],
                    "source_live_model": base["model_fields"],
                    "candidate_live_model": child["model_fields"],
                    "query_indices": query_ids.numpy(),
                    "sample_ids": base["sample_id"],
                }
            )
        if saved_fields:
            field_path = args.output / f"fields_epoch_{epoch:03d}.npz"
            np.savez_compressed(
                field_path,
                **{
                    key: np.stack([record[key] for record in saved_fields])
                    for key in saved_fields[0]
                },
                field_names=np.asarray(runtime.dataset.field_names),
                epoch=np.asarray(epoch),
                full_native_grid=np.asarray(
                    all(item["full_native_grid"] for item in pressure_figures)
                ),
            )
            output["saved_fields"] = {
                "path": field_path.name,
                "sha256": file_sha256(field_path),
                "sample_count": len(saved_fields),
                "decoded_dataset_values": True,
                "panel_positions": field_positions,
                "field_probe_policy": "prospective_uniform_positions_across_declared_panel",
            }
        output["pressure_review"] = pressure_figures
        (args.output / "audit.json").write_text(
            json.dumps(output, indent=2, allow_nan=False) + "\n"
        )
        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "audit": str((args.output / "audit.json").resolve()),
                    "panel": args.panel,
                }
            )
        )
    finally:
        if audited_dataset is not runtime.dataset:
            audited_dataset.close()
        runtime.dataset.close()
        source.dataset.close()


if __name__ == "__main__":
    main()

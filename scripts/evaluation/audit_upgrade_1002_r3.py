"""Bounded matched R3 development audit with cached coefficients; TEST forbidden."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from audit_upgrade_1002_r2 import save_native_fields

from phycoflow_reconstruction.coherence import ReferenceBank, build_enabled_families
from phycoflow_reconstruction.data.factory import open_field_dataset
from phycoflow_reconstruction.data.manifest import SensorManifest, manifest_from_batch
from phycoflow_reconstruction.evaluation.checkpoint import load_evaluation_runtime
from phycoflow_reconstruction.evaluation.coherence_set import _restore_evaluation_family
from phycoflow_reconstruction.evaluation.upgrade_1002_r2_protocol import (
    build_panel_batch,
    grouped_evaluate,
    grouped_native_monitor,
)
from phycoflow_reconstruction.training.coherence_diagnostics import cached_B_regrouping
from phycoflow_reconstruction.training.common import sensor_protocol_from_config
from phycoflow_reconstruction.training.fidelity_controller import r3_coherence_selection_report
from phycoflow_reconstruction.training.native_topology_audit import (
    _preserve_audit_state,
    build_native_topology_family,
    initialize_native_geometry,
)
from phycoflow_reconstruction.training.post_training import _build_comparison_batch, _without_target
from phycoflow_reconstruction.training.run_store import file_sha256, load_project_checkpoint
from phycoflow_reconstruction.training.source import source_checkpoint_path

SOURCE_SHA = "03f2f45b966a00850cc45c011b4cf1cd644b9408de355324fec8dc434a84a810"


def write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


def component_values(result):
    return {
        name: float(value.scalar_loss.detach())
        for name, value in result.component_results.items()
        if name.endswith((".finite", ".essential", "_raw", "_normalized"))
        or name in {"topology.legacy", "topology.finite_primary", "topology.normalized_finite", "topology.normalized_essential"}
    }


def native_review(
    runtime,
    source_model,
    config,
    protocol,
    output,
    *,
    epoch,
    development_indices,
    train_indices=(0, 2671, 5342, 7999),
):
    """One <=4 TRAIN SOURCE calibration; <=4 DEV fields, separate native scales."""
    if not 1 <= len(train_indices) <= 4 or not 2 <= len(development_indices) <= 4:
        raise ValueError("bounded native review requires <=4 TRAIN and 2-4 development snapshots")
    native, provenance = build_native_topology_family(
        config, runtime.dataset, runtime.dataset.normalizer, runtime.run_dir, runtime.device
    )
    train = open_field_dataset(config["dataset"], split="train", normalizer=runtime.dataset.normalizer)
    try:
        if train.normalizer.digest() != runtime.dataset.normalizer.digest():
            raise ValueError("native TRAIN/source normalizers differ")
        batch = build_panel_batch(train, protocol, train_indices, runtime.device)
        initialize_native_geometry(
            native, batch, tuple(config["dataset"]["grid_shape"]), excluded_sample_ids=()
        )
        with _preserve_audit_state((source_model,)), torch.no_grad():
            source_model.eval()
            generator = torch.Generator(device=runtime.device).manual_seed(
                int(config["evaluation"]["seed"])
            )
            prediction = source_model.reconstruct(
                _without_target(batch), steps=runtime.generation_steps, generator=generator
            ).prediction
            result = native(
                prediction,
                batch.target_fields,
                coordinates=batch.query_coords,
                context={"phase": "calibration", "sample_ids": batch.sample_ids,
                         "reference_ids": batch.sample_ids},
            )
            records = [native.spatial_objective.collect_source_components(result)]
            calibration = native.spatial_objective.freeze_source_calibration(
                records,
                {
                    "split": "train",
                    "source_checkpoint_sha256": SOURCE_SHA,
                    "source_weight_selection": "live",
                    "sample_ids": list(batch.sample_ids),
                    "dataset_indices": list(train_indices),
                    "generation_seed": int(config["evaluation"]["seed"]),
                    "query_count": int(batch.query_coords.shape[1]),
                },
                representation="native_grid",
            )
        field_metadata = save_native_fields(
            runtime, source_model, development_indices, config, protocol, output, epoch=epoch
        )
        dev_batch = build_panel_batch(
            runtime.dataset, protocol, development_indices, runtime.device
        )
        with (
            np.load(output / field_metadata["path"], allow_pickle=False) as arrays,
            torch.no_grad(),
        ):
            values = {}
            for role in ("source_live", "candidate_live"):
                prediction = torch.as_tensor(arrays[role + "_model"], device=runtime.device)
                result = native(
                    prediction,
                    dev_batch.target_fields,
                    coordinates=dev_batch.query_coords,
                    context={"phase": "evaluation", "sample_ids": dev_batch.sample_ids,
                             "reference_ids": dev_batch.sample_ids},
                )
                values[role] = component_values(result)
        return {
            "representation": "native_grid",
            "grid_shape": list(native.grid_shape),
            "source_TRAIN_calibration": calibration,
            "field_arrays": field_metadata,
            "native_family_provenance": provenance,
            "components": values,
            "scope": "development only; independent TRAIN native scales; not training-raster scores",
        }
    finally:
        train.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--checkpoint", default="last")
    parser.add_argument("--declaration", type=Path, required=True)
    parser.add_argument("--panel", choices=("selection", "robustness"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-cache", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--native-field-index", action="append", type=int, default=[])
    parser.add_argument("--native-train-index", action="append", type=int, default=[])
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[2] / "cases/turbulent_combustion/runs/Test_1002"
    for path in (args.run, args.output, args.source_cache):
        path.resolve().relative_to(root.resolve())
    if not args.run.resolve().relative_to(root.resolve()).parts[0].startswith("R3_"):
        raise ValueError("R3 audits require an R3 lineage")
    if args.device.startswith("cuda") and (
        args.device != "cuda:0" or os.getenv("CUDA_VISIBLE_DEVICES") != "0"
    ):
        raise ValueError("R3 audits require physical GPU0 via CUDA_VISIBLE_DEVICES=0 and cuda:0")
    declaration = json.loads(args.declaration.read_text())
    if declaration.get("split") != "validation" or not declaration.get("test_locked"):
        raise ValueError("R3 development declaration must lock TEST and declare validation")
    panel = declaration[args.panel]
    runtime = load_evaluation_runtime(
        args.run,
        split="validation",
        checkpoint=args.checkpoint,
        sensor_config=None,
        generation_steps=None,
        device_name=args.device,
        include_temporal_derivative=False,
    )
    source = None
    try:
        config = deepcopy(runtime.config)
        manifest = json.loads((args.run / "run_manifest.json").read_text())
        checkpoint = load_project_checkpoint(runtime.checkpoint_path)
        divisor = int(manifest["steps_per_epoch"])
        if checkpoint["global_step"] % divisor:
            raise ValueError("R3 broad audit requires an exact historical epoch boundary")
        epoch = checkpoint["global_step"] // divisor
        if args.panel == "robustness" and epoch not in (100, 150, 200):
            raise ValueError("R3 robustness is restricted to epochs100/150/200")
        if args.native_field_index and epoch < 100:
            raise ValueError("new native review requires a mature checkpoint")
        source_checkpoint = source_checkpoint_path(config).resolve()
        if (
            file_sha256(source_checkpoint) != SOURCE_SHA
            or manifest["source_hashes"]["checkpoint"] != SOURCE_SHA
        ):
            raise ValueError("immutable SOURCE identity mismatch")
        source = load_evaluation_runtime(
            Path(config["source_run"]),
            split="validation",
            checkpoint=str(source_checkpoint),
            sensor_config=None,
            generation_steps=runtime.generation_steps,
            device_name=args.device,
            include_temporal_derivative=False,
        )
        if runtime.dataset.normalizer.digest() != source.dataset.normalizer.digest():
            raise ValueError("SOURCE/candidate normalizers differ")
        protocol = sensor_protocol_from_config(config)
        protocol = replace(protocol, seed=protocol.seed + int(panel["sensor_seed_offset"]))
        complete = build_panel_batch(
            runtime.dataset, protocol, panel["dataset_indices"], runtime.device
        )
        config["evaluation"]["seed"] = int(panel["generation_seed"])
        config["evaluation"]["native_topology"] = False
        comparison = _build_comparison_batch(complete, config)
        sensor_manifest = manifest_from_batch(comparison, runtime.dataset.path, "validation")
        frozen_path = Path(panel["frozen_sensor_manifest"])
        if file_sha256(frozen_path) != panel["frozen_sensor_manifest_file_sha256"]:
            raise ValueError("prospectively frozen sensor manifest file changed")
        if sensor_manifest.digest() != SensorManifest.load(frozen_path).digest():
            raise ValueError(
                "regenerated panel differs from prospectively frozen sensor/query manifest"
            )
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
                banks[name] = ReferenceBank.load(args.run / "artifacts" / bank_name)
                if banks[name].digest() != manifest["reference_bank_sha256s"][name]:
                    raise ValueError("reference bank identity mismatch")
        context = {
            "source_checkpoint_sha256": SOURCE_SHA,
            "source_weight_selection": "live",
            "sensor_manifest_sha256": sensor_manifest.digest(),
            "family_artifacts": artifacts,
            "family_scales": manifest.get("coherence_family_scales", {}),
            "generation_seed": panel["generation_seed"],
            "generation_steps": runtime.generation_steps,
            "normalizer_sha256": runtime.dataset.normalizer.digest(),
            "panel": args.panel,
        }
        cache_key = hashlib.sha256(json.dumps(context, sort_keys=True).encode()).hexdigest()
        args.source_cache.mkdir(parents=True, exist_ok=True)
        cache_json, cache_npz = (
            args.source_cache / f"source_{cache_key}.{suffix}" for suffix in ("json", "npz")
        )
        captured = {}

        def capture(role):
            def callback(generated, reference):
                captured[role] = (generated.cpu().numpy(), reference.cpu().numpy())

            return callback

        args.output.mkdir(parents=True, exist_ok=False)
        source_cache_reused = cache_json.is_file()
        with _preserve_audit_state((runtime.model, source.model)):
            runtime.model.eval()
            source.model.eval()
            if source_cache_reused:
                saved = json.loads(cache_json.read_text())
                if saved["context"] != context or saved["coefficients_sha256"] != file_sha256(
                    cache_npz
                ):
                    raise ValueError("cached SOURCE context/array identity mismatch")
                source_result = saved["result"]
                with np.load(cache_npz, allow_pickle=False) as arrays:
                    captured["source_live"] = (arrays["source"], arrays["reference"])
            else:
                source_result = grouped_evaluate(
                    source.model,
                    complete,
                    comparison,
                    families,
                    banks,
                    tuple(runtime.dataset.field_names),
                    config,
                    normalizer=runtime.dataset.normalizer,
                    family_scales=manifest.get("coherence_family_scales", {}),
                    coefficient_callback=capture("source_live"),
                )
                np.savez_compressed(
                    cache_npz,
                    source=captured["source_live"][0],
                    reference=captured["source_live"][1],
                )
                write_json(
                    cache_json,
                    {
                        "context": context,
                        "result": source_result,
                        "coefficients_sha256": file_sha256(cache_npz),
                    },
                )
            child_result = grouped_evaluate(
                runtime.model,
                complete,
                comparison,
                families,
                banks,
                tuple(runtime.dataset.field_names),
                config,
                normalizer=runtime.dataset.normalizer,
                family_scales=manifest.get("coherence_family_scales", {}),
                coefficient_callback=capture("candidate_live"),
            )
            native = grouped_native_monitor(
                runtime.model,
                source.model,
                {args.panel: comparison},
                seeds=panel.get("native_seeds", [2027, 3027]),
            )
            if not np.array_equal(captured["source_live"][1], captured["candidate_live"][1]):
                raise ValueError("SOURCE/child coefficient reference mismatch")
            sensitivity = cached_B_regrouping(
                captured["source_live"][0],
                captured["candidate_live"][0],
                captured["source_live"][1],
                families["cross_spectrum"],
            )
            np.savez_compressed(
                args.output / "graph_coefficients.npz",
                source=captured["source_live"][0],
                candidate=captured["candidate_live"][0],
                reference=captured["source_live"][1],
                sample_ids=np.asarray(comparison.sample_ids),
                split=np.asarray("validation"),
            )
            native_fields = (
                native_review(
                    runtime,
                    source.model,
                    config,
                    protocol,
                    args.output,
                    epoch=epoch,
                    development_indices=args.native_field_index,
                    train_indices=args.native_train_index or (0, 2671, 5342, 7999),
                )
                if args.native_field_index
                else None
            )
        output = {
            "schema": "phycoflow.upgrade_1002_r3_audit.v1",
            "epoch": epoch,
            "run": str(args.run.resolve()),
            "checkpoint": str(runtime.checkpoint_path),
            "checkpoint_sha256": file_sha256(runtime.checkpoint_path),
            "panel": args.panel,
            "split": "validation",
            "dataset_indices": panel["dataset_indices"],
            "context": context,
            "source_cache_key": cache_key,
            "source_cache_reused": source_cache_reused,
            "results": {"source_live": source_result, "candidate_live": child_result},
            "R3_selection_descriptive": r3_coherence_selection_report(
                source_result, child_result, config["posttrain_fidelity"]
            ),
            "matched_native_monitor": native,
            "B_regrouping": sensitivity,
            "native_fields": native_fields,
            "test_accessed": False,
            "fits_on_TEST": False,
            "updates_model": False,
            "source_weight_selection": "live",
            "candidate_weight_selection": "live",
        }
        sensor_manifest.save(args.output / "sensor_manifest.json")
        write_json(args.output / "audit.json", output)
        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "panel": args.panel,
                    "audit": str(args.output / "audit.json"),
                    "B_grouped_ratio_median": sensitivity["ratio_median"],
                    "test_accessed": False,
                }
            )
        )
    finally:
        runtime.dataset.close()
        if source is not None:
            source.dataset.close()


if __name__ == "__main__":
    main()

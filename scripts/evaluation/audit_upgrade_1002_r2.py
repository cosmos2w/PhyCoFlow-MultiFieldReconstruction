"""Freeze R2 panels or audit a mature LIVE checkpoint without accessing test data."""

from __future__ import annotations

import argparse
import json
import os
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np

from phycoflow_reconstruction.coherence import ReferenceBank, build_enabled_families
from phycoflow_reconstruction.data.factory import open_field_dataset
from phycoflow_reconstruction.data.manifest import dataset_fingerprint, manifest_from_batch
from phycoflow_reconstruction.evaluation.checkpoint import load_evaluation_runtime
from phycoflow_reconstruction.evaluation.coherence_set import _restore_evaluation_family
from phycoflow_reconstruction.evaluation.upgrade_1002_r2_protocol import (
    build_panel_batch,
    declare_panels,
    grouped_evaluate,
    matched_native_monitor,
    render_pressure_comparison,
    save_declaration,
    selection_indices,
)
from phycoflow_reconstruction.training.common import sensor_protocol_from_config
from phycoflow_reconstruction.training.fidelity_controller import coherence_selection_report
from phycoflow_reconstruction.training.native_topology_audit import _preserve_audit_state
from phycoflow_reconstruction.training.post_training import _build_comparison_batch
from phycoflow_reconstruction.training.run_store import file_sha256, load_project_checkpoint
from phycoflow_reconstruction.training.source import source_checkpoint_path


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
        "--panel", choices=("training", "selection", "extended", "audit"), default="extended"
    )
    parser.add_argument("--device", default="cuda:0")
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
    runtime = load_evaluation_runtime(
        args.run,
        split="validation",
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
    if args.panel in {"extended", "audit"} and epoch < 100:
        runtime.dataset.close()
        raise ValueError("broader R2 audit requires a mature checkpoint at epoch >=100")
    source_run = Path(config["source_run"]).resolve()
    source_checkpoint = source_checkpoint_path(config).resolve()
    source = load_evaluation_runtime(
        source_run,
        split="validation",
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
        if len(runtime.dataset) != declaration["dataset_size"]:
            raise ValueError("declared split size differs from actual dataset")
        if runtime.dataset.normalizer.digest() != source.dataset.normalizer.digest():
            raise ValueError("source/candidate normalizers differ")
        if file_sha256(source_checkpoint) != manifest["source_hashes"]["checkpoint"]:
            raise ValueError("immutable source identity changed")
        split = "train" if args.panel == "training" else "validation"
        panel = (
            declaration[args.panel]
            if args.panel != "training"
            else {
                "dataset_indices": selection_indices(len(audited_dataset), 32, 16),
                "sensor_seed_offset": 0,
                "generation_seed": 2027,
                "native_seeds": [2027, 3027],
            }
        )
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
            native = matched_native_monitor(
                runtime.model, source.model, {args.panel: comparison}, seeds=panel["native_seeds"]
            )
        if "posttrain_fidelity" in config:
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
        args.output.mkdir(parents=True, exist_ok=False)
        realization.save(args.output / "sensor_manifest.json")
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

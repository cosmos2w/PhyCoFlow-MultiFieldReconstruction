#!/usr/bin/env python3
"""Qualify an adaptive bank on one frozen TRAIN/SOURCE panel, on CPU only."""

from __future__ import annotations

import argparse
import hashlib
import json
import signal
import time
from pathlib import Path

import torch
import yaml

from phycoflow_reconstruction.coherence import build_enabled_families
from phycoflow_reconstruction.coherence.families.multiscale_spectrum.statistics import (
    piecewise_fluctuations,
    piecewise_means,
    spectral_coefficients,
)
from phycoflow_reconstruction.coherence.representation_contracts import retained_scale_constant
from phycoflow_reconstruction.data.factory import open_field_dataset
from phycoflow_reconstruction.training.run_store import file_sha256
from phycoflow_reconstruction.training.scalable_fitting import (
    fit_scalable_artifacts,
    validate_scalable_artifacts,
)

FAILURE_DIR = None
STARTED = time.monotonic()


def _deadline_handler(signum, frame):
    raise TimeoutError("CPU discovery exceeded its wall-clock limit")


def main():
    global FAILURE_DIR
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    run, output = args.run_dir.resolve(), args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    FAILURE_DIR = output
    started = time.perf_counter()
    config = yaml.safe_load((run / "resolved_config.yaml").read_text())
    dataset = open_field_dataset(config["dataset"], split="train")
    families = build_enabled_families(config["coherence"], dataset.data_spec, dataset.normalizer)
    manifest = json.loads((run / "run_manifest.json").read_text())
    for name, family in families.items():
        path = run / "artifacts" / f"{name}_family.pt"
        if file_sha256(path) != manifest["coherence_family_state_sha256s"][name]:
            raise ValueError(f"{name} family artifact hash mismatch")
        family.load_state_artifact(torch.load(path, map_location="cpu", weights_only=True))
    panel_path = run / "artifacts/source_panel.pt"
    core_path = run / "artifacts/scalable_coherence.pt"
    panel = torch.load(panel_path, map_location="cpu", weights_only=True)
    core = torch.load(core_path, map_location="cpu", weights_only=True)
    if panel["schema"] != "car_source_panel_v1" or panel["provenance"] != core["provenance"]:
        raise ValueError("pilot panel and C8 core provenance differ")
    validate_scalable_artifacts(core, panel["provenance"])
    if len(panel["sample_ids"]) != 800 or panel["sample_ids"] != panel["provenance"]["sample_ids"]:
        raise ValueError("expected full declared TRAIN800 panel")
    seed = int(config["runtime"].get("seed", 42)) + 810001
    artifact = fit_scalable_artifacts(
        families=families,
        source=panel["members"][:, :2],
        reference=panel["reference"],
        coordinates=panel["coordinates"],
        provenance=panel["provenance"],
        source_weights=panel["masses"],
        reference_weights=panel["masses"],
        seed=seed,
        adaptive_settings={
            "panel_batch_size": args.batch_size,
            "floor": 0.01,
            "sketch_size": 128,
            "redundancy_limit": 0.95,
            "units": "model",
        },
        reuse_core=core,
    )
    bank = artifact["adaptive_bank"]
    (output / "bank.json").write_text(json.dumps(bank, indent=2, allow_nan=False) + "\n")
    scale = families["multiscale_spectrum"]
    g = scale._geometry_tensors(scale.compute_dtype)
    retained = retained_scale_constant(
        g["filters"],
        scale.band_scale,
        g["piece_measures"].sum(1),
        scale.piece_mean_scale,
        scale.settings.mean_weight,
    )
    squared_sum = 0.0
    max_residual = 0.0
    observations = 0
    for start in range(0, len(panel["reference"]), args.batch_size):
        stop = min(start + args.batch_size, len(panel["reference"]))
        z = scale._select(panel["reference"][start:stop])
        for member in range(2):
            delta = scale._select(panel["members"][start:stop, member]) - z
            means = piecewise_means(delta, g["piece_measures"])
            fluct = piecewise_fluctuations(delta, g["piece_measures"], g["pieces"])
            coeff = spectral_coefficients(fluct, g["basis"] * g["mass"][:, None])
            reconstructed = (
                torch.einsum("nk,bkc->bnc", g["basis"], coeff) + means[:, g["pieces"], :]
            )
            residual = (
                ((delta - reconstructed).square() * g["mass"][None, :, None])
                .sum((1, 2))
                .div(g["mass"].sum())
                .sqrt()
            )
            squared_sum += float(residual.square().sum())
            max_residual = max(max_residual, float(residual.max()))
            observations += len(residual)
    record = {
        "schema": "car_preformal_train_discovery_v1",
        "purpose": "pilot selection evidence only",
        "source_panel_sha256": file_sha256(panel_path),
        "core_artifact_sha256": file_sha256(core_path),
        "source_identity": bank["source_identity"],
        "support_identity": bank["support_identity"],
        "sample_count": len(panel["sample_ids"]),
        "available_members": 4,
        "discovery_members": 2,
        "sample_ids_sha256": hashlib.sha256(json.dumps(panel["sample_ids"]).encode()).hexdigest(),
        "seed": seed,
        "batch_size": args.batch_size,
        "threads": args.threads,
        "elapsed_seconds": time.perf_counter() - started,
        "bank_sha256": bank["sha256"],
        "retained_scale": {
            "c_min": retained["c_min"],
            "inverse_available": retained["inverse_available"],
            "modal_weights": retained["modal_weights"].tolist(),
            "mean_weights": retained["mean_weights"].tolist(),
            "unresolved_rms": (squared_sum / observations) ** 0.5,
            "unresolved_max": max_residual,
            "source_truth_comparisons": observations,
            "units": f"{scale.units}; mass-normalized physical support",
        },
        "settings": {
            "scalable_anchored_coherence": {
                "enabled": True,
                "adaptive": {
                    "enabled": True,
                    "panel_batch_size": args.batch_size,
                    "floor": 0.01,
                    "sketch_size": 128,
                    "redundancy_limit": 0.95,
                    "units": "model",
                },
            }
        },
    }
    (output / "receipt.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    if hasattr(dataset, "close"):
        dataset.close()
    print(
        json.dumps(
            {
                "output": str(output),
                "elapsed_seconds": record["elapsed_seconds"],
                "selected": len(bank["selected"]),
            }
        )
    )


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, _deadline_handler)
    try:
        main()
    except Exception as exc:
        if FAILURE_DIR is not None:
            (FAILURE_DIR / "failure.json").write_text(
                json.dumps(
                    {
                        "status": "failed",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "elapsed_seconds": time.monotonic() - STARTED,
                    },
                    indent=2,
                )
                + "\n"
            )
        raise

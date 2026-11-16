#!/usr/bin/env python3
"""Freeze the TRAIN-truth-only Round 2 regional covariance definition."""

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import torch
import yaml

from phycoflow_reconstruction.data.factory import open_field_dataset
from phycoflow_reconstruction.data.manifest import dataset_fingerprint


def freeze(config, fraction=0.1):
    if not 0 < fraction <= 0.1:
        raise ValueError("task freeze must use at most the declared 10% TRAIN panel")
    settings = dict(config["dataset"])
    settings["physical_measure"] = "native_rectilinear_area"
    dataset = open_field_dataset(settings, split="train")
    try:
        count = max(1, math.ceil(len(dataset) * fraction))
        first = dataset[0]
        names = tuple(first.field_names)
        fields = [names.index("CO"), names.index("U_1")]
        x = first.coordinates_raw[:, 0].double()
        mass = dataset.physical_weights.double()
        support = mass > 0
        sorted_indices = torch.argsort(x[support], stable=True)
        sorted_x = x[support][sorted_indices]
        sorted_mass = mass[support][sorted_indices]
        median = sorted_x[torch.searchsorted(sorted_mass.cumsum(0), sorted_mass.sum() / 2)]
        region = support & (x >= median)
        weights = mass[region] / mass[region].sum()
        sums = torch.zeros(2, dtype=torch.float64)
        squares = torch.zeros(2, dtype=torch.float64)
        sample_ids = []
        for i in range(count):
            sample = dataset[i]
            if not torch.equal(sample.coordinates_raw, first.coordinates_raw):
                raise ValueError("TRAIN samples do not share the frozen native geometry")
            values = dataset.normalizer.decode(sample.values)[region][:, fields].double()
            if not torch.isfinite(values).all():
                raise ValueError("nonfinite TRAIN truth on task region")
            sums += (weights[:, None] * values).sum(0)
            squares += (weights[:, None] * values.square()).sum(0)
            sample_ids.append(f"{sample.trajectory_id}:{sample.time_index if sample.time_index is not None else 'all'}")
        variance = squares / count - (sums / count).square()
        if not torch.isfinite(variance).all() or (variance <= 0).any():
            raise ValueError("regional TRAIN-truth field scales are degenerate")
        indices = torch.nonzero(region, as_tuple=False).flatten().tolist()
        return {
            "schema": "car_round2_reserved_task_v1", "split": "train",
            "dataset_fingerprint": dataset_fingerprint(settings["path"]),
            "sample_selection": "first_fraction_in_dataset_order", "sample_fraction": fraction,
            "sample_count": count, "sample_ids": sample_ids,
            "field_names": ["CO", "U_1"],
            "field_units": [dataset.data_spec.field_units[i] for i in fields],
            "scale_definition": "population standard deviation over selected TRAIN samples and normalized regional native masses",
            "truth_scales": variance.sqrt().tolist(),
            "coordinate": "raw x, axis 0", "median_rule": "first sorted x at cumulative positive native mass >= half total mass",
            "median_x": float(median), "region_rule": "positive native mass and x >= median_x",
            "region_point_indices": indices,
            "region_indices_sha256": hashlib.sha256(torch.tensor(indices, dtype=torch.int64).numpy().tobytes()).hexdigest(),
            "native_measure": dataset.physical_measure_receipt,
            "regional_mass_fraction": float(mass[region].sum() / mass.sum()),
            "score": "sum(normalized regional mass * (CO - regional mean CO) * (U_1 - regional mean U_1)) / (truth_scale_CO * truth_scale_U_1)",
            "interpretation": "regional signed species-velocity covariance; dimensionless after TRAIN-truth scaling",
        }
    finally:
        dataset.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.time()
    manifest = freeze(yaml.safe_load(args.source_config.read_text()))
    manifest["source_config_sha256"] = hashlib.sha256(args.source_config.read_bytes()).hexdigest()
    manifest["timer"] = {"start_unix": started, "end_unix": time.time(), "gpu_lease": None}
    manifest["timer"]["wall_seconds"] = manifest["timer"]["end_unix"] - started
    with args.output.open("x") as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()

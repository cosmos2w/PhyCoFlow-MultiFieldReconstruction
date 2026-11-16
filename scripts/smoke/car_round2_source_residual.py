#!/usr/bin/env python3
"""One TRAIN-sample source/residual numerical microcase; no efficacy assessment."""

import argparse
import hashlib
import json
import time
from dataclasses import replace
from pathlib import Path

import torch
import yaml

from phycoflow_reconstruction.data.sensor_protocols import SensorProtocol, build_observation_batch
from phycoflow_reconstruction.data.training_batches import fixed_query_indices
from phycoflow_reconstruction.models.source_residual import FrozenSourceResidual
from phycoflow_reconstruction.training.rollout import differentiable_reconstruction
from phycoflow_reconstruction.training.source import load_source_model


def digest(state):
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--family", choices=("ffm", "senseiver"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    started = time.time()
    torch.set_num_threads(4)
    config = yaml.safe_load((args.snapshot / "resolved_config.yaml").read_text())
    config.update(source_run=str(args.snapshot), source_checkpoint="last.pt")
    config["source"] = {"kind": "native_run", "admission": "frozen_snapshot",
                        "snapshot_receipt": str(args.snapshot / "snapshot_receipt.json"),
                        "weight_view": "ema_materialized" if args.family == "ffm" else "live"}
    device = torch.device(args.device)
    model, dataset, identity = load_source_model(config, device)
    model.eval()
    sample = dataset[0]
    query = fixed_query_indices(sample.values.shape[0], 128, seed=100045)
    batch = build_observation_batch([sample], SensorProtocol(field_counts={"T": 256}, seed=2047),
                                    query_indices=query).to(device)
    batch.target_fields = None
    consistency = ({"mode": "endpoint_smooth", "strength": 1.0, "sigma": 0.05,
                    "schedule_power": 2.0, "final_clamp": True, "chunk_size": 4096}
                   if args.family == "ffm" else {"mode": "none", "final_clamp": False})
    wrapper = FrozenSourceResidual(model, coordinate_dim=sample.coordinates.shape[-1],
                                   field_scales=[1.0] * len(sample.field_names), width=16).to(device)
    state_before = digest(model.state_dict())

    def run(candidate, input_batch, seed):
        generator = torch.Generator(device=device).manual_seed(seed)
        result = differentiable_reconstruction(candidate, input_batch, steps=4, solver="euler",
                                                generator=generator, observation_config=consistency)
        return result, hashlib.sha256(generator.get_state().cpu().numpy().tobytes()).hexdigest()

    with torch.no_grad():
        original, state1 = run(model, batch, 31001)
        wrapped, state2 = run(wrapper, batch, 31001)
        second, _ = run(wrapper, batch, 31002)
        native = model.reconstruct(batch, steps=4, generator=torch.Generator(device=device).manual_seed(31001)).prediction
        pieces, corrections = [], []
        wrapper.head[-1].bias.fill_(0.2)
        full_correction = wrapper.correction(batch, original)
        for start in range(0, 128, 32):
            stop = start + 32
            sub = replace(batch, query_coords=batch.query_coords[:, start:stop],
                          query_valid_mask=batch.query_valid_mask[:, start:stop],
                          metadata={**batch.metadata, "query_indices": batch.metadata["query_indices"][:, start:stop]})
            piece, _ = run(model, sub, 31001)
            pieces.append(piece)
            corrections.append(wrapper.correction(sub, original[:, start:stop]))
        chunked = torch.cat(pieces, dim=1)
        head_chunked = torch.cat(corrections, dim=1)
        state_after = digest(model.state_dict())
    max_error = lambda x, y: float((x - y).abs().max().cpu())
    result = {"schema": "car_round2_source_residual_microcase_v1", "family": args.family,
              "scope": "one TRAIN sample; numerical microcase only", "identity": identity,
              "sample_id": batch.sample_ids[0], "query_count": 128, "chunk_size": 32,
              "source_wrapper_zero_max_abs": max_error(original, wrapped),
              "source_native_max_abs": max_error(original, native),
              "source_chunk_max_abs": max_error(original, chunked),
              "head_fixed_source_chunk_max_abs": max_error(full_correction, head_chunked),
              "independent_seed_max_abs": max_error(wrapped, second),
              "generator_states_equal": state1 == state2,
              "source_state_before": state_before, "source_state_after": state_after,
              "device": str(device), "consistency": consistency,
              "timer": {"start_unix": started, "end_unix": time.time(), "gpu_lease": None if device.type == "cpu" else str(device)}}
    result["timer"]["wall_seconds"] = result["timer"]["end_unix"] - started
    dataset.close()
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()

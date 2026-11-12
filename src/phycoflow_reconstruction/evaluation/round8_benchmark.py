"""Same-device batch-one J2 latency and resident-memory measurements."""
from __future__ import annotations

from dataclasses import replace
import gc
import json
import os
from pathlib import Path
import statistics
import subprocess
import time

import torch

from .round8_campaign import canonical_hash, file_hash
from .round8_inference import prior_identity, _capture_initial
from ..data.factory import open_field_dataset
from ..data.sensor_protocols import build_observation_batch
from ..training.common import sensor_protocol_from_config
from ..training.model_lifecycle import load_training_aux_state
from ..training.post_training import _without_target
from ..training.rollout import differentiable_reconstruction
from ..training.run_store import load_model_state_strict, load_project_checkpoint
from ..training.source import load_source_model


def benchmark(config, manifest_path, endpoints, *, output_path, device="cuda:0", requests=5):
    """Endpoints map fixed names to checkpoint/alpha; SOURCE is included once.

    All requests come from the existing frozen ASSESS panel. Loading is outside
    latency because both models remain resident throughout each measurement.
    """
    uuid = os.environ.get("ROUND8_PHYSICAL_GPU_UUID")
    if not uuid or os.environ.get("CUDA_VISIBLE_DEVICES") != uuid:
        raise ValueError("benchmark requires a verified physical device")
    manifest = json.loads(Path(manifest_path).read_text())
    selected = manifest["ASSESS"][:requests]
    if len(selected) != requests or requests < 1:
        raise ValueError("benchmark requests must belong to the frozen panel")
    request_hash = canonical_hash(selected)
    result = {"schema":"round8_same_device_inference_benchmark_v1", "endpoints":{}}
    for name, declaration in {"SOURCE":{"checkpoint":None, "alpha":0.0}, **endpoints}.items():
        gc.collect()
        torch.cuda.empty_cache()
        source, train, _ = load_source_model(config, torch.device(device))
        source.eval().requires_grad_(False)
        alpha = float(declaration["alpha"])
        if not 0 <= alpha <= 1:
            raise ValueError("benchmark coefficient must be global and in [0,1]")
        repair = None
        if alpha > 0:
            repair, duplicate, _ = load_source_model(config, torch.device(device))
            duplicate.close()
            checkpoint = load_project_checkpoint(declaration["checkpoint"])
            load_model_state_strict(repair, checkpoint["model"])
            load_training_aux_state(repair, checkpoint)
            repair.eval().requires_grad_(False)
            if prior_identity(source) != prior_identity(repair):
                raise ValueError("benchmark endpoints must share prior buffers")
        datasets = {"train":train}
        try:
            for split in {row["source"] for row in selected}:
                datasets[split] = open_field_dataset(config["dataset"], split=split, normalizer=train.normalizer)
            protocol = sensor_protocol_from_config(config)
            batches = []
            for row in selected:
                sample = datasets[row["source"]][int(row["dataset_index"])]
                batch = build_observation_batch(
                    [sample], replace(protocol, seed=row["observation_seed"]), query_points=None).to(device)
                batches.append(_without_target(batch))
            def request(batch, row, *, verify=False):
                outputs = []
                for seed in row["member_seeds"]:
                    def reconstruct(model):
                        generator = torch.Generator(device=device).manual_seed(int(seed))
                        def call():
                            return differentiable_reconstruction(
                                model, batch, steps=int(config["rollout"]["steps"]),
                                solver=config["rollout"]["solver"], generator=generator,
                                observation_config=dict(config["observation_consistency"]))
                        if verify:
                            with _capture_initial(model) as initial:
                                value = call()
                        else:
                            value, initial = call(), None
                        return value, initial
                    if alpha == 0:
                        value, _ = reconstruct(source)
                    elif alpha == 1:
                        value, _ = reconstruct(repair)
                    else:
                        left, a = reconstruct(source)
                        right, b = reconstruct(repair)
                        if verify and a != b:
                            raise ValueError("benchmark initial random fields differ")
                        value = left + alpha * (right - left)
                    outputs.append(value)
                return outputs
            with torch.no_grad():
                for warmup in range(2):
                    values = request(batches[0], selected[0], verify=warmup == 0)
                    del values
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
                times = []
                for batch, row in zip(batches, selected):
                    torch.cuda.synchronize(device)
                    start = time.perf_counter()
                    values = request(batch, row)
                    torch.cuda.synchronize(device)
                    times.append(time.perf_counter() - start)
                    del values
            # CUDA context/non-PyTorch overhead is included conservatively with
            # the allocator's peak reservation, rather than peak tensors alone.
            used_mib = float(subprocess.check_output([
                "nvidia-smi", f"--id={uuid}", "--query-gpu=memory.used",
                "--format=csv,noheader,nounits"], text=True).strip())
            reserved = torch.cuda.memory_reserved(device)
            overhead = max(0, int(used_mib * 1024**2) - reserved)
            props = torch.cuda.get_device_properties(device)
            result["endpoints"][name] = {
                "device_uuid":uuid, "device_name":props.name,
                "device_vram_bytes":props.total_memory, "batch_size":1, "member_count":2,
                "request_ids_sha256":request_hash, "warmup_policy":"two untimed identical J2 requests",
                "synchronized":True, "latency_seconds":statistics.median(times),
                "request_seconds":times, "alpha":alpha,
                "peak_allocated_bytes":torch.cuda.max_memory_allocated(device),
                "peak_resident_bytes":torch.cuda.max_memory_reserved(device) + overhead,
                "resident_measure":"peak PyTorch reservation plus measured context/non-PyTorch overhead",
                "both_models_resident":0 < alpha < 1,
                "checkpoint_sha256":declaration.get("checkpoint") and file_hash(declaration["checkpoint"]),
            }
        finally:
            for dataset in datasets.values():
                dataset.close()
            del repair, source
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return output

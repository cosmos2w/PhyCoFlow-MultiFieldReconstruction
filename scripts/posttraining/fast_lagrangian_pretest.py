"""TRAIN-only real-SOURCE estimator/oracle probe; performs no model updates."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from time import perf_counter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--anchor", type=Path, required=True)
    parser.add_argument("--estimator", choices=["full", "p3", "s2"], default="full")
    args = parser.parse_args()
    root = args.campaign.resolve()
    if not (root / "manifests/preflight.json").is_file() or not str(root).startswith("/data2/"):
        raise ValueError("verified campaign required")
    # Only Unix sockets use /tmp: Linux limits AF_UNIX paths to 108 bytes.
    import multiprocessing
    import tempfile

    multiprocessing.current_process()._config["tempdir"] = tempfile.mkdtemp(
        prefix="lpc-sock-", dir="/tmp"
    )
    import torch
    import yaml

    from phycoflow_reconstruction.coherence import build_enabled_families
    from phycoflow_reconstruction.coherence.families.topology.filtration_sampling import (
        TopologyFiltrationSampler,
        sampled_family_forward,
    )
    from phycoflow_reconstruction.data.training_batches import build_training_batch_source
    from phycoflow_reconstruction.training.common import iter_unique_batch_indices
    from phycoflow_reconstruction.training.gradient_balance import _flat_gradient
    from phycoflow_reconstruction.training.periodic_config import extract_config_coefficients
    from phycoflow_reconstruction.training.post_training import (
        _coherence_objective,
        _configure_persistence_workers,
    )
    from phycoflow_reconstruction.training.retention import post_training_mode
    from phycoflow_reconstruction.training.source import load_source_model
    from phycoflow_reconstruction.training.update_budget import post_training_steps_per_epoch
    from phycoflow_reconstruction.utils.reproducibility import seed_everything

    cfg = yaml.safe_load(args.anchor.read_text())
    cfg["runtime"]["device"] = "cuda:0"
    seed_everything(42, True)
    torch.set_num_threads(4)
    _configure_persistence_workers(cfg)
    device = torch.device("cuda:0")
    start = perf_counter()
    model, dataset, metadata = load_source_model(cfg, device)
    post_training_mode(model, cfg)
    families = {
        k: v.to(device)
        for k, v in build_enabled_families(
            cfg["coherence"], dataset.data_spec, dataset.normalizer
        ).items()
    }
    indices = iter_unique_batch_indices(
        len(dataset), 4, 32, generator=torch.Generator().manual_seed(700103)
    )
    batches = build_training_batch_source(
        dataset, indices, cfg, query_points=4096, device=device, start_step=0
    )
    report = {
        "estimator": args.estimator,
        "source": cfg["source_run"],
        "checkpoint": cfg["source_checkpoint"],
        "split": "train",
        "training_epochs": 0,
        "dataset_length": len(dataset),
        "steps_per_epoch": post_training_steps_per_epoch(cfg, len(dataset)),
        "device": torch.cuda.get_device_name(0),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "grid": [32, 128],
        "homology": [0, 1],
        "model_metadata": metadata,
        "sampling": {},
        "coefficient_replay": [],
    }
    alpha = None
    previous_alpha = None
    for cursor, batch in enumerate(batches):
        endpoint, losses = {}, {}
        execution = {
            "estimator": args.estimator,
            "topology_included": True,
            "family_probability": 1.0,
            "topology_seed": 3700228 + cursor,
        }
        result, _ = _coherence_objective(
            model,
            batch,
            families,
            None,
            cfg,
            step=cursor,
            generator=torch.Generator(device=device).manual_seed(1000045 + cursor),
            endpoint_context=endpoint,
            family_losses=losses,
            execution=execution,
        )
        if cursor == 0:
            # Reuse one real endpoint as an input leaf. Each outcome constructs
            # fresh generated pairings; no second model call is made.
            leaf = endpoint["prediction"].detach().requires_grad_()
            ref = endpoint["target"]
            coords = batch.query_coords
            topo = families["topology"]

            # _coherence_objective keeps all 4096 points; fixed_shared coords align.
            def compute(strategy, choice=None, topo=topo, leaf=leaf, ref=ref, coords=coords):
                if strategy == "oracle":
                    return topo(leaf, ref, coordinates=coords)
                sampler = TopologyFiltrationSampler(
                    topo.spatial_objective, strategy=strategy, seed=700211
                )
                return sampled_family_forward(
                    topo, leaf, ref, coordinates=coords, sampler=sampler, choice=choice
                )

            torch.cuda.synchronize()
            t = perf_counter()
            full = compute("oracle")
            fg = torch.autograd.grad(full.scalar_loss, leaf)[0]
            torch.cuda.synchronize()
            report["full_topology"] = {
                "loss": float(full.scalar_loss),
                "seconds": perf_counter() - t,
                "diagnostics": full.diagnostics,
            }
            for strategy in ["p3", "s2"]:
                sampler = TopologyFiltrationSampler(
                    topo.spatial_objective, strategy=strategy, seed=700211
                )
                values = []
                gradients = []
                durations = []
                for choice in sampler.enumerate_choices():
                    torch.cuda.synchronize()
                    t = perf_counter()
                    sample = compute(strategy, choice)
                    grad = torch.autograd.grad(sample.scalar_loss, leaf)[0]
                    torch.cuda.synchronize()
                    durations.append(perf_counter() - t)
                    values.append(float(sample.scalar_loss))
                    gradients.append(grad)
                gs = torch.stack(gradients)
                mean = gs.mean(0)
                report["sampling"][strategy] = {
                    "outcomes": len(values),
                    "losses": values,
                    "mean_loss": sum(values) / len(values),
                    "loss_relative_error": abs(sum(values) / len(values) - float(full.scalar_loss))
                    / max(abs(float(full.scalar_loss)), 1e-30),
                    "gradient_relative_error": float((mean - fg).norm() / fg.norm()),
                    "gradient_variance_relative": float(
                        ((gs - fg).square().flatten(1).sum(1)).mean() / fg.square().sum()
                    ),
                    "cosines": [
                        float(
                            torch.nn.functional.cosine_similarity(g.flatten(), fg.flatten(), dim=0)
                        )
                        for g in gradients
                    ],
                    "seconds": durations,
                }
                del gs, gradients
            del leaf, full, fg
        params = [p for p in model.parameters() if p.requires_grad]
        matrix = torch.stack(
            [
                _flat_gradient(losses[n], params)
                for n in ["global_distribution", "cross_spectrum", "topology"]
            ]
        ).double()
        recovered = extract_config_coefficients(matrix)
        fresh = recovered.oracle_direction
        if alpha is None:
            alpha = recovered.coefficients
        reused = matrix.T @ alpha
        report["coefficient_replay"].append(
            {
                "age": cursor,
                **recovered.diagnostics,
                "stale_cosine": float(torch.nn.functional.cosine_similarity(fresh, reused, dim=0)),
                "stale_relative_error": float((fresh - reused).norm() / fresh.norm()),
                "fresh_products": (matrix @ fresh).cpu().tolist(),
                "reuse_products": (matrix @ reused).cpu().tolist(),
            }
        )
        if previous_alpha is not None:
            previous_reuse = matrix.T @ previous_alpha
            report["coefficient_replay"][-1]["age_one_cosine"] = float(
                torch.nn.functional.cosine_similarity(fresh, previous_reuse, dim=0)
            )
        previous_alpha = recovered.coefficients
        (root / f"analysis/pretest_{args.estimator}.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        print("completed TRAIN probe", cursor, flush=True)
        del result, losses, endpoint, matrix, fresh, reused, recovered, batch
    batches.close()
    dataset.close()
    torch.cuda.synchronize()
    report["inclusive_gpu_reservation_seconds"] = perf_counter() - start
    report["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
    (root / f"analysis/pretest_{args.estimator}.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

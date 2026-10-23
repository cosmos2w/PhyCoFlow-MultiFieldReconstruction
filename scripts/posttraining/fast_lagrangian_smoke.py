"""One-epoch real deterministic smoke or two-epoch flow recovery smoke."""

from __future__ import annotations

import argparse
import multiprocessing
import tempfile
from copy import deepcopy
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", required=True, type=Path)
    parser.add_argument("--kind", required=True, choices=["deterministic", "recovery"])
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    root = args.campaign.resolve(strict=True)
    if not str(root).startswith("/data2/") or not (root / "manifests/preflight.json").is_file():
        raise ValueError("verified /data2 campaign required")
    multiprocessing.current_process()._config["tempdir"] = tempfile.mkdtemp(
        prefix="lpc-sock-", dir="/tmp"
    )
    import torch

    from phycoflow_reconstruction.training.post_training import run_post_training
    from phycoflow_reconstruction.training.update_budget import post_training_steps_per_epoch

    repo = Path(__file__).resolve().parents[2]
    case = repo / "cases/turbulent_combustion"
    anchor = (
        case
        / "runs/_MultiTask/11f_lagrangian_lr_warmup_dense_fast_100ep/20261017T170131Z_615167aa/resolved_config.yaml"
    )
    cfg = yaml.safe_load(anchor.read_text())
    cfg["optimization"]["epochs"] = 1 if args.kind == "deterministic" else 2
    cfg["optimization"]["fast_lagrangian"] = {
        "topology_estimator": "p3",
        "family_probability": 0.5,
        "coefficient_interval": 1,
        "seed": 700211,
    }
    cfg["optimization"]["multitask"].pop("coherence_aggregation")
    cfg["optimization"]["multitask"]["family_scalarization"] = {"method": "fixed_sum"}
    cfg["runtime"]["device"] = "cuda:0"
    cfg["runtime"]["progress"] = False
    cfg["output"]["experiment_name"] = f"_LPCAccel/{root.name}/smoke_{args.kind}"
    if args.kind == "deterministic":
        source = case / "runs/tc_deeponet_5000ep/20260829T125329Z_2ebc837b"
        base = yaml.safe_load((source / "resolved_config.yaml").read_text())
        cfg["source_run"] = str(source)
        for key in ["model", "dataset", "observations"]:
            cfg[key] = deepcopy(base[key])
    path = root / f"configs/smoke_{args.kind}.yaml"
    text = yaml.safe_dump(cfg, sort_keys=False)
    if path.exists() and path.read_text() != text:
        raise ValueError("smoke configuration differs")
    path.write_text(text)
    torch.set_num_threads(4)
    # Dataset metadata identifies historical epoch exposure; do not read TEST.
    from phycoflow_reconstruction.data.factory import open_field_dataset

    ds = open_field_dataset(cfg["dataset"], split="train")
    steps = post_training_steps_per_epoch(cfg, len(ds))
    ds.close()
    run = run_post_training(cfg, case_dir=case, max_steps=steps, resume=args.resume)
    print(run, flush=True)


if __name__ == "__main__":
    main()

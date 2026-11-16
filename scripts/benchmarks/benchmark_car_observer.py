#!/usr/bin/env python3
"""Compare frozen compiled and independent C8 observers on a saved TRAIN panel."""

from __future__ import annotations

import argparse
import io
import json
import signal
import time
from dataclasses import asdict
from pathlib import Path

import torch
import yaml

from phycoflow_reconstruction.coherence import build_enabled_families
from phycoflow_reconstruction.data.factory import open_field_dataset
from phycoflow_reconstruction.evaluation.frozen_coherence import FrozenCoherenceObserver
from phycoflow_reconstruction.training.run_store import file_sha256
from phycoflow_reconstruction.training.scalable_fitting import _digest, validate_scalable_artifacts


class BudgetExceeded(Exception):
    pass


def _deadline(_signum, _frame):
    raise BudgetExceeded("CPU benchmark reached wall-clock budget")


def tensor_bytes(value):
    """Count unique tensor storages in nested cache objects."""
    seen_objects, seen_storage = set(), set()

    def visit(x):
        if id(x) in seen_objects:
            return 0
        seen_objects.add(id(x))
        if isinstance(x, torch.Tensor):
            storage = x.untyped_storage()
            key = (str(x.device), storage.data_ptr())
            if key in seen_storage:
                return 0
            seen_storage.add(key)
            return storage.nbytes()
        if isinstance(x, dict):
            return sum(visit(k) + visit(v) for k, v in x.items())
        if isinstance(x, (tuple, list)):
            return sum(visit(v) for v in x)
        if hasattr(x, "__dict__"):
            return visit(vars(x))
        return 0

    return visit(value)


def save_json(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--query-seed", type=int, default=0)
    p.add_argument("--query-batches", type=int, default=8)
    p.add_argument("--max-seconds", type=int, default=3600)
    args = p.parse_args()
    if min(args.batch_size, args.threads, args.query_batches, args.max_seconds) < 1:
        raise ValueError("positive benchmark budgets required")
    torch.set_num_threads(args.threads)
    run, output = args.run_dir.resolve(), args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    started = time.perf_counter()
    signal.signal(signal.SIGALRM, _deadline)
    signal.alarm(args.max_seconds)
    receipt = {
        "schema": "car_observer_engine_benchmark_v1",
        "status": "running",
        "source_run": str(run),
        "population": "TRAIN source panel",
        "members": 4,
        "conditions_expected": 800,
        "conditions_completed": 0,
        "batch_size": args.batch_size,
        "threads": args.threads,
        "query_seed": args.query_seed,
        "query_batches": args.query_batches,
        "max_seconds": args.max_seconds,
        "device": "cpu",
        "scope": "same J4 target and fixed queries, no model inference",
    }
    save_json(output / "receipt.json", receipt)
    columns = {}
    try:
        config = yaml.safe_load((run / "resolved_config.yaml").read_text())
        dataset = open_field_dataset(config["dataset"], split="train")
        families = build_enabled_families(
            config["coherence"], dataset.data_spec, dataset.normalizer
        )
        manifest = json.loads((run / "run_manifest.json").read_text())
        for name, family in families.items():
            path = run / "artifacts" / f"{name}_family.pt"
            if file_sha256(path) != manifest["coherence_family_state_sha256s"][name]:
                raise ValueError(f"{name} family hash differs from manifest")
            family.load_state_artifact(torch.load(path, map_location="cpu", weights_only=True))
        panel_path, artifact_path = (
            run / "artifacts/source_panel.pt",
            run / "artifacts/scalable_coherence.pt",
        )
        panel = torch.load(panel_path, map_location="cpu", weights_only=True)
        artifact = torch.load(artifact_path, map_location="cpu", weights_only=True)
        if (
            panel["schema"] != "car_source_panel_v1"
            or panel["provenance"] != artifact["provenance"]
        ):
            raise ValueError("SOURCE panel and artifact identity mismatch")
        validate_scalable_artifacts(artifact, panel["provenance"])
        ids = list(panel["sample_ids"])
        if (
            len(ids) != 800
            or ids != artifact["provenance"]["sample_ids"]
            or panel["members"].shape[:2] != (800, 4)
        ):
            raise ValueError("exact TRAIN800 J4 panel required")
        import phycoflow_reconstruction

        package = Path(phycoflow_reconstruction.__file__).resolve().parent
        code_files = (
            "coherence/query_sampling.py",
            "coherence/query_runtime.py",
            "coherence/shape_compute.py",
            "coherence/state_compute.py",
            "coherence/scale_compute.py",
            "evaluation/frozen_coherence.py",
        )
        receipt["observer_package_path"] = str(package)
        receipt["observer_code_sha256s"] = {
            name: file_sha256(package / name) for name in code_files
        }
        receipt["benchmark_script_sha256"] = file_sha256(Path(__file__))
        receipt.update(
            {
                "source_panel_file_sha256": file_sha256(panel_path),
                "scalable_artifact_sha256": artifact["sha256"],
                "state_bank_sha256": artifact["state_bank"]["sha256"],
                "family_artifact_sha256s": manifest["coherence_family_state_sha256s"],
                "panel_ids_sha256": _digest(ids),
                "pilot_provenance": (
                    "preformal SOURCE panel; recorded parameter/buffer and weight-view binding present"
                    if panel.get("parameter_buffer_sha256") and panel.get("source_weight_view_sha256")
                    else "preformal pilot; predates parameter/buffer and weight-view binding"
                ),
                "source_view_binding": {
                    key: panel.get(key) for key in (
                        "parameter_buffer_sha256", "source_weight_view_sha256", "source_weight_view"
                    )
                },
            }
        )
        setup_total = compile_total = compiled_total = direct_total = pairing_total = 0.0
        cache_tensor_total = cache_serialized_total = 0
        rows, pack = [], None
        for start in range(0, len(ids), args.batch_size):
            stop = min(start + args.batch_size, len(ids))
            mark = time.perf_counter()
            reference, coordinates, masses = (
                panel[k][start:stop].contiguous() for k in ("reference", "coordinates", "masses")
            )
            members = panel["members"][start:stop].transpose(0, 1).contiguous()
            pairing_total += time.perf_counter() - mark
            mark = time.perf_counter()
            observer = FrozenCoherenceObserver(
                artifact,
                families,
                expected_provenance=artifact["provenance"],
                reference=reference,
                coordinates=coordinates,
                sample_ids=ids[start:stop],
                member_weights=masses,
                reference_weights=masses,
                mode="conditional",
                query_seed=args.query_seed,
                query_batches=args.query_batches,
                complete_scale=True,
            )
            setup_total += time.perf_counter() - mark
            compile_total += observer.compile_seconds
            pack_now = [asdict(q) for q in observer.batches]
            if pack is None:
                pack = pack_now
            elif _digest(pack_now) != _digest(pack):
                raise ValueError("query pack changed across panel")
            cache = {
                "state": observer.state_reference,
                "scale": observer.scale_reference,
                "shape": observer.shape_reference,
                "shape_queries": observer.shape_queries,
            }
            cache_tensor_total += tensor_bytes(cache)
            stream = io.BytesIO()
            torch.save(cache, stream)
            cache_serialized_total += stream.tell()
            with torch.no_grad():
                mark = time.perf_counter()
                fast = observer.score_tensors(members, compiled=True)
                compiled_total += time.perf_counter() - mark
                mark = time.perf_counter()
                slow = observer.score_tensors(members, compiled=False)
                direct_total += time.perf_counter() - mark
            if set(fast) != set(slow):
                raise ValueError("block identity differs")
            for key in fast:
                a, b = fast[key].detach().double(), slow[key].detach().double()
                if not torch.isfinite(a).all() or not torch.isfinite(b).all():
                    raise ValueError("nonfinite observer result")
                columns.setdefault(key, {"compiled": [], "direct": []})
                columns[key]["compiled"].append(a.clone())
                columns[key]["direct"].append(b.clone())
            rows.extend(ids[start:stop])
            receipt.update(
                {
                    "conditions_completed": len(rows),
                    "elapsed_seconds": time.perf_counter() - started,
                    "cumulative_seconds": {
                        "cold_observer_setup": setup_total,
                        "reference_compilation": compile_total,
                        "warm_compiled_callbacks": compiled_total,
                        "direct_callbacks": direct_total,
                        "panel_pairing_cpu": pairing_total,
                        "host_device_transfer": 0.0,
                    },
                    "cache": {
                        "total_batch_tensor_storage_bytes": cache_tensor_total,
                        "total_batch_serialized_estimate_bytes": cache_serialized_total,
                        "persistent_cache_disk_bytes": 0,
                    },
                    "query_pack": pack,
                    "query_pack_sha256": _digest(pack),
                }
            )
            save_json(output / "receipt.json", receipt)
        comparisons = {}
        for key, value in columns.items():
            a, b = torch.cat(value["compiled"]), torch.cat(value["direct"])
            absolute = (a - b).abs()
            relative = absolute / b.abs().clamp_min(torch.finfo(b.dtype).tiny)
            comparisons[key] = {
                "compiled": a,
                "direct": b,
                "absolute_difference": absolute,
                "relative_difference": relative,
            }
        data_path = output / "per_condition.pt"
        torch.save({"sample_ids": rows, "blocks": comparisons}, data_path)
        receipt["blocks"] = {
            key: {
                "max_absolute": float(v["absolute_difference"].max()),
                "max_relative": float(v["relative_difference"].max()),
                "max_absolute_sample_id": rows[int(v["absolute_difference"].argmax())],
                "max_relative_sample_id": rows[int(v["relative_difference"].argmax())],
            }
            for key, v in comparisons.items()
        }
        receipt.update(
            {
                "status": "completed",
                "per_condition_file_sha256": file_sha256(data_path),
                "result_disk_bytes": data_path.stat().st_size,
                "elapsed_seconds": time.perf_counter() - started,
                "interpretation": "one callback per path per panel batch after shared setup; no GPU transfer; zero cache persistence on disk",
            }
        )
        save_json(output / "receipt.json", receipt)
        if hasattr(dataset, "close"):
            dataset.close()
        print(
            json.dumps(
                {
                    "status": "completed",
                    "conditions": len(rows),
                    "receipt": str(output / "receipt.json"),
                }
            )
        )
    except Exception as exc:
        receipt.update(
            {
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
        save_json(output / "receipt.json", receipt)
        raise
    finally:
        signal.alarm(0)


if __name__ == "__main__":
    main()

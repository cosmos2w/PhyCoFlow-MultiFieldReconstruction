"""Rerender topology diagnostics from a pinned training-preview payload."""

from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import numpy as np
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
CASE_ROOT = REPOSITORY_ROOT / "cases" / "turbulent_combustion"
DIAGNOSTICS_ROOT = CASE_ROOT / "diagnostics"
COHERENCE_ROOT = DIAGNOSTICS_ROOT / "coherence"
SOURCE_ROOT = COHERENCE_ROOT / "source"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    started = perf_counter()
    # The example uses one immutable saved preview and CPU-only GUDHI pairing.
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    import sys

    import gudhi

    sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

    from phycoflow_reconstruction.config import load_config
    from phycoflow_reconstruction.contracts import DataSpec
    from phycoflow_reconstruction.data.normalization import FieldNormalizer
    from phycoflow_reconstruction.evaluation.topology_set import TopologySetAccumulator

    provenance_path = SOURCE_ROOT / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    payload_path = SOURCE_ROOT / "epoch_000800_latest_reconstruction.npz"
    if _sha256(payload_path) != provenance["payload_sha256"]:
        raise ValueError("pinned epoch-800 reconstruction payload checksum mismatch")

    config = load_config(SOURCE_ROOT / "resolved_config.yaml")
    preview = json.loads(
        (SOURCE_ROOT / "epoch_000800_preview_metrics.json").read_text(encoding="utf-8")
    )
    field_names = tuple(str(value) for value in config["dataset"]["field_names"])
    field_units = tuple(str(value) for value in config["dataset"]["field_units"])
    normalizer_path = SOURCE_ROOT / "normalization_train_mean_std.json"
    stats = json.loads(normalizer_path.read_text(encoding="utf-8"))
    normalizer = FieldNormalizer.from_artifact(
        normalizer_path,
        field_names=field_names,
        dataset_fingerprint=str(stats["dataset_fingerprint"]),
    )

    with np.load(payload_path, allow_pickle=False) as payload:
        payload_fields = tuple(str(value) for value in payload["field_names"])
        if payload_fields != field_names:
            raise ValueError("preview field order does not match resolved configuration")
        physical_prediction = torch.from_numpy(
            np.asarray(payload["prediction_physical"], dtype=np.float32)
        )
        physical_target = torch.from_numpy(
            np.asarray(payload["target_physical"], dtype=np.float32)
        )
        coordinates = torch.from_numpy(np.asarray(payload["query_coords"], dtype=np.float32))
        logical_shape = tuple(int(value) for value in payload["logical_shape"])

    data_spec = DataSpec(
        field_names=field_names,
        field_units=field_units,
        coordinate_dim=int(coordinates.shape[-1]),
        logical_shape=logical_shape,
        mesh_type="point",
    )
    data_spec.validate()
    dataset = SimpleNamespace(
        field_names=field_names,
        normalizer=normalizer,
        data_spec=data_spec,
    )
    runtime = SimpleNamespace(
        config=config,
        dataset=dataset,
        device=torch.device("cpu"),
    )

    accumulator = TopologySetAccumulator.build(runtime)
    accumulator.use_preselected_query_points(
        seed=int(preview["seed"]),
        description=(
            "saved training_preview query_coords; preview selection seed="
            f"{preview['seed']} (identity order in pinned payload)"
        ),
    )
    prediction = normalizer.encode(physical_prediction).unsqueeze(0)
    target = normalizer.encode(physical_target).unsqueeze(0)
    accumulator.update(
        prediction,
        target,
        coordinates.unsqueeze(0),
        sample_id=str(preview["sample_id"]),
    )
    output = accumulator.finalize(
        DIAGNOSTICS_ROOT,
        split=str(config["evaluation"].get("split", config["dataset"].get("split", "validation"))),
        checkpoint_label="epoch_000800",
        run_label=str(provenance["source_run_name"]),
        scale="linear",
    )
    render_manifest = {
        "source_run_id": provenance["source_run_id"],
        "sample_id": str(preview["sample_id"]),
        "preview_epoch": float(preview["training_epoch"]),
        "input_payload_sha256": provenance["payload_sha256"],
        "command": (
            "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 "
            "PYTHONPATH=src conda run -n phycoflow_env python "
            "cases/turbulent_combustion/diagnostics/coherence/render_epoch800_topology.py"
        ),
        "device": "cpu",
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "gudhi_version": str(gudhi.__version__),
        "script_elapsed_seconds": round(perf_counter() - started, 3),
    }
    (Path(output["directory"]) / "render_manifest.json").write_text(
        json.dumps(render_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

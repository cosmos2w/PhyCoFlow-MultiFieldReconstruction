"""Rebuild physical-coordinate reconstruction examples from the pinned payload."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from phycoflow_reconstruction.data.training_batches import fixed_query_indices
from phycoflow_reconstruction.evaluation.reconstruction_visualization import (
    render_reconstruction_payload,
)
from phycoflow_reconstruction.training.preview import render_preview_payload

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "source" / "fullgrid_validation_frame8000.npz"
OUTPUT = ROOT / "reconstruction"
PREVIEW_QUERY_SEED = 2027
PREVIEW_QUERY_COUNT = 4096


def render() -> dict[str, object]:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    provenance = json.loads((ROOT / "snapshot_provenance.json").read_text())
    epoch = int(provenance["checkpoint_epoch"])
    fullgrid_path = OUTPUT / "fullgrid_validation_frame8000.png"
    render_reconstruction_payload(
        SOURCE,
        fullgrid_path,
        title=f"A+B+C formal · validation frame 8000 · epoch {epoch}",
    )
    fullgrid_path.with_suffix(".svg").unlink()

    with np.load(SOURCE, allow_pickle=False) as payload:
        normalized_coordinates = np.asarray(payload["query_coords"])[0]
        physical_coordinates = np.asarray(payload["query_coords_physical"])[0]
        prediction = np.asarray(payload["prediction_physical"])[0]
        target = np.asarray(payload["target_physical"])[0]
        observation_indices = np.asarray(payload["obs_indices"])[0]
        selected = fixed_query_indices(
            len(normalized_coordinates), PREVIEW_QUERY_COUNT, seed=PREVIEW_QUERY_SEED
        )
        if selected is None:
            raise RuntimeError("preview query selection is unavailable")
        indices = selected.numpy()
        if not np.allclose(
            physical_coordinates.min(axis=0), (0.0003868074854835868, -0.09947119653224945),
            rtol=0.0, atol=1e-7,
        ):
            raise ValueError("full-grid physical coordinate minimum changed")
        preview_payload = {
            "prediction_physical": prediction[indices],
            "target_physical": target[indices],
            "query_coords": normalized_coordinates[indices],
            "query_coords_physical": physical_coordinates[indices],
            "obs_coords": np.asarray(payload["obs_coords"])[0],
            "obs_coords_physical": physical_coordinates[observation_indices],
            "obs_values_physical": np.asarray(payload["obs_values_physical"])[0],
            "obs_field_ids": np.asarray(payload["obs_field_ids"])[0],
            "obs_valid_mask": np.asarray(payload["obs_valid_mask"])[0],
            "logical_shape": np.asarray(payload["logical_shape"]),
            "field_names": np.asarray(payload["field_names"]),
            "field_units": np.asarray(["unknown"] * prediction.shape[1]),
            "sample_id": np.asarray(str(payload["sample_id"])),
            "source_fullgrid_query_indices": indices,
            "preview_query_seed": np.asarray(PREVIEW_QUERY_SEED),
        }
    sparse_payload_path = ROOT / "source" / "sparse_preview_from_fullgrid.npz"
    np.savez_compressed(sparse_payload_path, **preview_payload)
    sparse_stem = OUTPUT / "sparse_preview_from_fullgrid"
    render_preview_payload(sparse_payload_path, output_stem=sparse_stem, epoch=epoch)
    sparse_stem.with_suffix(".svg").unlink()
    return {
        "checkpoint_epoch": epoch,
        "source_payload": str(SOURCE.relative_to(ROOT)),
        "sparse_payload": str(sparse_payload_path.relative_to(ROOT)),
        "sparse_query_count": PREVIEW_QUERY_COUNT,
        "sparse_query_seed": PREVIEW_QUERY_SEED,
        "sparse_view_contract": (
            "fixed 4096-point subset of a full-grid evaluation, not the trainer's "
            "direct sparse-preview inference"
        ),
        "figures": [
            str(fullgrid_path.relative_to(ROOT)),
            str(sparse_stem.with_suffix(".png").relative_to(ROOT)),
        ],
    }


if __name__ == "__main__":
    report = render()
    (OUTPUT / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))

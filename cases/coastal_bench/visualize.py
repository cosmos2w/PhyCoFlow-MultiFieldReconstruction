#!/usr/bin/env python
"""Plot matched reference, ensemble-mean, member and deviation maps by spectral band."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from phycoflow_reconstruction.evaluation.spectral import (
    BandAnalyzer,
    draw_members,
    open_panel,
)
from phycoflow_reconstruction.training.post_training import _slice_batch


def plot_bands(
    coordinates,
    target,
    mean,
    member,
    *,
    band_names,
    field_names,
    output,
    length_unit="geometry units",
):
    """Use common signed colour limits per band/field for all four panels."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    panels = (target, mean, member, member - mean)
    titles = ("Reference", "Ensemble mean", "One draw", "Draw minus mean")
    paths = []
    for field, name in enumerate(field_names):
        fig, axes = plt.subplots(
            len(band_names), 4, figsize=(15, 3 * len(band_names)), squeeze=False
        )
        for band, label in enumerate(band_names):
            limit = max(float(np.max(np.abs(value[band, :, field]))) for value in panels)
            limit = max(limit, np.finfo(float).eps)
            for column, (values, title) in enumerate(zip(panels, titles)):
                axis = axes[band, column]
                artist = axis.scatter(
                    coordinates[:, 0],
                    coordinates[:, 1],
                    c=values[band, :, field],
                    s=3,
                    cmap="RdBu_r",
                    vmin=-limit,
                    vmax=limit,
                    rasterized=True,
                )
                axis.set_title(f"{label}: {title}")
                axis.set_aspect("equal")
                axis.set_xlabel(f"Coordinate 0 ({length_unit})")
                if column == 0:
                    axis.set_ylabel(f"Coordinate 1 ({length_unit})")
            fig.colorbar(artist, ax=axes[band].tolist(), shrink=0.8, label="Signed band field")
        fig.suptitle(name)
        path = output / f"field_{field:02d}.png"
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--checkpoint", default="last")
    parser.add_argument("--draws", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    if args.draws < 2:
        parser.error("--draws must be at least two")
    panel = open_panel(args.run_dir, args.draws, args.device)
    if args.checkpoint != "source":
        panel.load_checkpoint(args.checkpoint)
    # Keep the first recorded condition and its exact sensors/queries; bound map-generation memory.
    panel.evaluation_batch = _slice_batch(panel.evaluation_batch, 1)
    panel.comparison_batch = _slice_batch(panel.comparison_batch, 1)
    family = panel.families["multiscale_spectrum"]
    analyzer = BandAnalyzer(family)
    with torch.no_grad():
        members = draw_members(panel, args.draws)
        bands = torch.stack([analyzer.statistics(member)["band_fields"][0] for member in members])
        target = analyzer.statistics(panel.comparison_batch.target_fields)["band_fields"][0]
    coords = panel.comparison_batch.query_coords[0, family.analysis_index].cpu().numpy()
    geometry = family.geometry_record
    coords = coords[:, geometry["active_axes"]] * np.asarray(geometry["active_axis_lengths"])
    if coords.shape[1] != 2:
        raise ValueError("CoastalBench band maps require two active coordinate axes")
    coords = coords - coords.min(axis=0)

    paths = plot_bands(
        coords,
        target.cpu().numpy(),
        bands.mean(0).cpu().numpy(),
        bands[0].cpu().numpy(),
        band_names=family.settings.bands.names,
        field_names=family.field_names,
        output=args.output,
        length_unit=geometry["length_unit"],
    )
    metadata = {
        "run_dir": str(panel.run_dir),
        "checkpoint": args.checkpoint,
        "sample_id": panel.comparison_batch.sample_ids[0],
        "draws": args.draws,
        "seed": panel.config["evaluation"].get("seed", 2027),
        "steps": panel.config["evaluation"]["generation_steps"],
        "units": family.units,
        "fields": family.field_names,
        "bands": family.settings.bands.names,
        "files": [p.name for p in paths],
    }
    (args.output / "maps.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()

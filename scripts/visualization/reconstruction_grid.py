"""Render target/prediction/error grids from a portable evaluation payload."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["svg.fonttype"] = "none"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--payload", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--claim", required=True)
    parser.add_argument("--caveat", default="")
    args = parser.parse_args()

    payload = np.load(args.payload)
    prediction = payload["prediction_physical"][0]
    target = payload["target_physical"][0]
    logical_shape = tuple(int(value) for value in payload["logical_shape"])
    fields = [str(value) for value in payload["field_names"]]
    if len(logical_shape) != 2:
        raise ValueError("reconstruction_grid requires a two-dimensional logical shape")
    prediction = prediction.reshape(*logical_shape, len(fields))
    target = target.reshape(*logical_shape, len(fields))
    error = np.abs(prediction - target)

    fig, axes = plt.subplots(
        len(fields), 3, figsize=(8.0, max(2.1 * len(fields), 3.0)), squeeze=False
    )
    for row, field in enumerate(fields):
        shared_low, shared_high = np.nanpercentile(target[..., row], [1, 99])
        error_high = max(float(np.nanpercentile(error[..., row], 99)), 1e-12)
        panels = (
            (target[..., row], "Target", "viridis", shared_low, shared_high),
            (prediction[..., row], "Reconstruction", "viridis", shared_low, shared_high),
            (error[..., row], "Absolute error", "magma", 0.0, error_high),
        )
        for column, (values, title, cmap, low, high) in enumerate(panels):
            axis = axes[row, column]
            image = axis.imshow(
                values, origin="lower", aspect="auto", cmap=cmap, vmin=low, vmax=high
            )
            if row == 0:
                axis.set_title(title)
            axis.set_ylabel(field if column == 0 else "")
            axis.set_xticks([])
            axis.set_yticks([])
            fig.colorbar(image, ax=axis, fraction=0.046, pad=0.02)
    fig.tight_layout()
    args.output.mkdir(parents=True, exist_ok=True)
    for suffix, dpi in (("svg", None), ("pdf", None), ("png", 220)):
        fig.savefig(args.output / f"reconstruction.{suffix}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    contract = f"""# Figure contract

- Core scientific claim: {args.claim}
- Source file: `{args.payload.resolve()}`
- Panel map: rows are physical fields; columns are target, reconstruction, and absolute error.
- Metrics/statistics: color limits use target 1st--99th percentiles per field; error limits use the 99th percentile.
- Caveat: {args.caveat or 'This visualization reports the saved integration result without additional inference.'}
"""
    (args.output / "figure_contract.md").write_text(contract, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

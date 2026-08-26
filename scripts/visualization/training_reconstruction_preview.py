"""Re-render an automatic training preview from its portable NPZ payload."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _relative_l2_error(estimate: np.ndarray, truth: np.ndarray) -> float | None:
    """Return ||estimate - truth||_2 / ||truth||_2, or None for a zero reference."""
    estimate64 = np.asarray(estimate, dtype=np.float64)
    truth64 = np.asarray(truth, dtype=np.float64)
    denominator = float(np.linalg.norm(truth64.ravel()))
    if denominator == 0.0 or not np.isfinite(denominator):
        return None
    value = float(np.linalg.norm((estimate64 - truth64).ravel()) / denominator)
    return value if np.isfinite(value) else None


def _absolute_error_title(relative_l2: float | None) -> str:
    metric = "N/A" if relative_l2 is None else f"{relative_l2:.3e}"
    return f"Absolute error\nRelative $L_2$ = {metric}"


def render(payload_path: Path, output_stem: Path, epoch: float) -> tuple[Path, ...]:
    """Render target, checkpoint reconstruction, and error without PyTorch."""
    with np.load(payload_path, allow_pickle=False) as payload:
        prediction = payload["prediction_physical"]
        target = payload["target_physical"]
        query_coords = payload["query_coords"]
        obs_coords = payload["obs_coords"]
        obs_values = payload["obs_values_physical"]
        obs_fields = payload["obs_field_ids"]
        obs_valid = payload["obs_valid_mask"].astype(bool)
        field_names = tuple(str(value) for value in payload["field_names"])
        logical_shape = tuple(int(value) for value in payload["logical_shape"])

    plt.rcParams["svg.fonttype"] = "none"
    figure, axes = plt.subplots(
        len(field_names),
        3,
        figsize=(12.0, max(3.2, 2.8 * len(field_names))),
        squeeze=False,
        constrained_layout=True,
    )
    complete_grid = prediction.shape[0] == math.prod(logical_shape)
    for field_index, field_name in enumerate(field_names):
        truth = target[:, field_index]
        estimate = prediction[:, field_index]
        error = np.abs(estimate - truth)
        error_title = _absolute_error_title(_relative_l2_error(estimate, truth))
        sensor_mask = obs_valid & (obs_fields == field_index)
        if len(logical_shape) == 1 and complete_grid:
            x = query_coords[:, 0]
            order = np.argsort(x)
            panels = (
                (truth, "Target"),
                (estimate, "Reconstruction"),
                (error, error_title),
            )
            for column, (values, title) in enumerate(panels):
                axes[field_index, column].plot(x[order], values[order], linewidth=1.4)
                axes[field_index, column].set_title(title)
            axes[field_index, 1].scatter(
                obs_coords[sensor_mask, 0],
                obs_values[sensor_mask],
                s=15,
                facecolors="none",
                edgecolors="black",
                linewidths=0.8,
                label="sensors",
                zorder=3,
            )
        elif len(logical_shape) == 2 and complete_grid:
            low = float(min(truth.min(), estimate.min()))
            high = float(max(truth.max(), estimate.max()))
            panels = (
                (truth.reshape(logical_shape), "Target", "viridis", low, high),
                (estimate.reshape(logical_shape), "Reconstruction", "viridis", low, high),
                (error.reshape(logical_shape), error_title, "magma", 0.0, None),
            )
            for column, (values, title, cmap, vmin, vmax) in enumerate(panels):
                image = axes[field_index, column].imshow(
                    values,
                    origin="lower",
                    aspect="auto",
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                )
                axes[field_index, column].set_title(title)
                figure.colorbar(image, ax=axes[field_index, column], fraction=0.046, pad=0.03)
            if sensor_mask.any():
                rows, columns = logical_shape
                axes[field_index, 1].scatter(
                    obs_coords[sensor_mask, 0] * max(columns - 1, 1),
                    obs_coords[sensor_mask, 1] * max(rows - 1, 1),
                    s=8,
                    facecolors="none",
                    edgecolors="white",
                    linewidths=0.5,
                    label="sensors",
                )
        else:
            x = query_coords[:, 0]
            y = query_coords[:, 1] if query_coords.shape[1] > 1 else np.zeros_like(x)
            panels = (
                (truth, "Target", "viridis"),
                (estimate, "Reconstruction", "viridis"),
                (error, error_title, "magma"),
            )
            for column, (values, title, cmap) in enumerate(panels):
                points = axes[field_index, column].scatter(x, y, c=values, s=8, cmap=cmap)
                axes[field_index, column].set_title(title)
                figure.colorbar(points, ax=axes[field_index, column], fraction=0.046, pad=0.03)
        axes[field_index, 0].set_ylabel(field_name)
        if sensor_mask.any():
            axes[field_index, 1].legend(loc="best", frameon=False, fontsize=7)

    figure.suptitle(f"Sparse reconstruction preview — epoch {epoch:.3f}")
    outputs = tuple(output_stem.with_suffix(suffix) for suffix in (".png", ".svg", ".pdf"))
    for output in outputs:
        figure.savefig(output, dpi=160)
    plt.close(figure)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--payload", type=Path, required=True)
    parser.add_argument("--output-stem", type=Path)
    parser.add_argument("--epoch", type=float)
    args = parser.parse_args()
    epoch = args.epoch
    if epoch is None:
        report_path = args.payload.with_name("latest_metrics.json")
        if not report_path.is_file():
            raise FileNotFoundError(
                "--epoch is required when latest_metrics.json is absent beside the payload"
            )
        epoch = float(json.loads(report_path.read_text())["training_epoch"])
    output_stem = args.output_stem or args.payload.with_suffix("")
    print("\n".join(str(path) for path in render(args.payload, output_stem, epoch)))


if __name__ == "__main__":
    main()

"""Saved-data visualization for the actual finite-primary topology bank.

This path is separate from the legacy topology-set renderer. Persistence
diagrams and Betti curves use the objective's exact normalized filtrations;
native event maps use an independently declared physical measure.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import torch

from ..coherence.families.topology.betti_curves import gaussian_blur
from ..coherence.families.topology.exact import exact_betti_curves, reference_levels
from ..coherence.families.topology.geometry import rasterize_fields
from ..coherence.families.topology.persistence import cubical_diagrams


def _pack_diagrams(diagrams, dimensions):
    finite, essential = [], []
    finite_offsets = np.zeros((len(diagrams), len(dimensions), 2), dtype=np.int64)
    essential_offsets = np.zeros_like(finite_offsets)
    nf = ne = 0
    for i, diagram in enumerate(diagrams):
        for j, dim in enumerate(dimensions):
            f = diagram[dim].finite.detach().cpu().numpy().reshape(-1, 2)
            e = diagram[dim].essential.detach().cpu().numpy().reshape(-1)
            finite_offsets[i, j] = (nf, nf + len(f))
            essential_offsets[i, j] = (ne, ne + len(e))
            nf, ne = nf + len(f), ne + len(e)
            finite.append(f)
            essential.append(e)
    return (
        np.concatenate(finite, axis=0), finite_offsets,
        np.concatenate(essential, axis=0), essential_offsets,
    )


def _event_payload(event):
    if event is None:
        return {}
    required = {"reference", "prediction", "coordinates", "valid_mask", "weights",
                "field_names", "pair", "signs", "thresholds"}
    missing = required.difference(event)
    if missing:
        raise ValueError(f"native event is missing {sorted(missing)}")
    reference = np.asarray(event["reference"], dtype=np.float64)
    prediction = np.asarray(event["prediction"], dtype=np.float64)
    coordinates = np.asarray(event["coordinates"], dtype=np.float64)
    mask = np.asarray(event["valid_mask"], dtype=bool)
    weights = np.asarray(event["weights"], dtype=np.float64)
    if reference.ndim != 3 or prediction.shape != reference.shape:
        raise ValueError("event fields must be matching [field,H,W] arrays")
    shape = reference.shape[-2:]
    names = tuple(str(x) for x in event["field_names"])
    if len(names) != len(reference) or len(set(names)) != len(names):
        raise ValueError("event field names must uniquely identify every field")
    if coordinates.shape != (*shape, 2) or mask.shape != shape or weights.shape != shape:
        raise ValueError("event coordinates, mask and weights must match field support")
    if (not mask.any() or not np.isfinite(weights[mask]).all() or (weights[mask] <= 0).any()
            or not np.isfinite(coordinates[mask]).all()
            or not np.isfinite(reference[:, mask]).all()
            or not np.isfinite(prediction[:, mask]).all()):
        raise ValueError("native event requires finite fields and positive weights on the valid support")
    # A raster renderer must not connect physically separate irregular cells.
    x, y = coordinates[0, :, 0], coordinates[:, 0, 1]
    if (not np.isfinite(coordinates).all() or min(shape) < 2 or np.any(np.diff(x) <= 0)
            or np.any(np.diff(y) <= 0) or not np.allclose(coordinates[..., 0], x[None, :])
            or not np.allclose(coordinates[..., 1], y[:, None])):
        raise ValueError("native event renderer requires increasing rectilinear coordinates")
    pair = tuple(event["pair"])
    signs = tuple(event["signs"])
    thresholds = tuple(float(v) for v in event["thresholds"])
    if (len(pair) != 2 or len(signs) != 2 or len(thresholds) != 2
            or any(name not in names for name in pair) or any(s not in (-1, 1) for s in signs)
            or not np.isfinite(thresholds).all()):
        raise ValueError("native event requires a named pair, two signs and finite thresholds")
    ids = [names.index(name) for name in pair]
    ref_event = mask & np.logical_and.reduce([
        signs[i] * reference[ids[i]] >= thresholds[i] for i in range(2)
    ])
    pred_event = mask & np.logical_and.reduce([
        signs[i] * prediction[ids[i]] >= thresholds[i] for i in range(2)
    ])
    return {
        "event_reference_fields": reference,
        "event_prediction_fields": prediction,
        "event_coordinates": coordinates,
        "event_valid_mask": mask,
        "event_weights": weights,
        "event_reference": ref_event,
        "event_prediction": pred_event,
        "event_metadata": np.asarray(json.dumps({
            "field_names": names, "pair": pair, "signs": signs, "thresholds": thresholds,
            "units": str(event.get("units", "unknown")),
            "measure": str(event.get("measure", "declared physical weights")),
        })),
    }


@torch.no_grad()
def save_finite_primary_payload(
    family, prediction: torch.Tensor, reference: torch.Tensor, *, coordinates: torch.Tensor,
    output_path: str | Path, event: dict | None = None,
    quantiles: tuple[float, ...] = (0.1, 0.25, 0.5, 0.75, 0.9),
    sample_id: str = "unspecified", endpoint: str = "unspecified",
) -> Path:
    """Score one matched member and store the entire signed bank for rerendering.

    ``event`` optionally supplies native physical fields [C,H,W], coordinates,
    fixed support/weights and a named signed pair with fixed physical thresholds.
    Persistence scores and physical event diagnostics remain distinct queries.
    """
    if family.version != "4" or family.spatial_objective is None:
        raise ValueError("finite-primary payload requires a finite_primary_v1 topology family")
    if not bool(family.spatial_objective.calibrated):
        raise ValueError("finite-primary payload requires frozen TRAIN/SOURCE calibration")
    if prediction.ndim != 3 or prediction.shape != reference.shape or prediction.shape[0] != 1:
        raise ValueError("finite-primary payload requires one matched [1,N,C] member")
    if coordinates.ndim != 3 or coordinates.shape[:2] != prediction.shape[:2]:
        raise ValueError("finite-primary coordinates must align with the member")
    if not quantiles or any(not 0 < q < 1 for q in quantiles):
        raise ValueError("Betti quantiles must lie strictly between 0 and 1")
    family._raster_map(coordinates)
    pred, ref = family._units_and_fields(prediction, reference)
    grids = []
    for values in (pred, ref):
        grid = rasterize_fields(values, family.neighbor_indices, family.neighbor_weights.to(values), family.grid_shape)
        grids.append(gaussian_blur(grid, family.smoothing_sigma, family.periodic, radius=family.smoothing_radius).float())
    pred_grid, ref_grid = grids
    objective = family.spatial_objective
    valid = family.domain_mask if family.native_masked else None
    descriptors = objective._descriptor_fields(ref_grid)
    support = descriptors if valid is None else descriptors[..., valid].unsqueeze(-2)
    mean = support.mean((-2, -1), keepdim=True)
    scale = support.std((-2, -1), keepdim=True, unbiased=False).clamp_min(objective.scale_floor)
    pred_bank = objective._filtrations((objective._descriptor_fields(pred_grid) - mean) / scale)[:, 0]
    ref_bank = objective._filtrations((descriptors - mean) / scale)[:, 0]
    dimensions = tuple(objective.dimensions)
    pred_diagrams = cubical_diagrams(pred_bank, periodic=family.periodic, dimensions=dimensions, valid_mask=valid)
    ref_diagrams = cubical_diagrams(ref_bank, periodic=family.periodic, dimensions=dimensions, valid_mask=valid)
    raw, present, normalization = objective.raw_leaves(pred_grid, ref_grid, valid_mask=valid)
    levels = reference_levels(-ref_bank, quantiles, valid_mask=valid)
    pred_betti = exact_betti_curves(-pred_bank, levels, periodic=family.periodic, valid_mask=valid)
    ref_betti = exact_betti_curves(-ref_bank, levels, periodic=family.periodic, valid_mask=valid)
    pf, pfo, pe, peo = _pack_diagrams(pred_diagrams, dimensions)
    rf, rfo, re_, reo = _pack_diagrams(ref_diagrams, dimensions)
    bank = []
    for component, name, ids, signs, line in objective.units:
        bank.append({
            "component": component, "name": name,
            "fields": [objective.fields[i] for i in ids], "signs": list(signs), "line": line,
            "direction": None if line is None else objective.line_directions_0[line].tolist(),
            "offset": None if line is None else objective.line_offsets_0[line].tolist(),
            "line_weight": None if line is None else float(objective.line_weights[line]),
        })
    metadata = {
        "definition": objective.definition,
        "sample_id": str(sample_id),
        "endpoint": str(endpoint),
        "geometry": dict(family.geometry_diagnostics),
        "geometry_sha256": family.geometry_sha256,
        "grid_shape": list(family.grid_shape),
        "periodic": family.periodic,
        "units": family.units,
        "point_normalization": normalization,
        "calibration": objective.calibration_record,
        "bank": bank,
        "descriptor_names": list(objective.fields),
        "reference_descriptor_mean": mean[0, :, 0, 0].detach().cpu().tolist(),
        "reference_descriptor_scale": scale[0, :, 0, 0].detach().cpu().tolist(),
        "threshold_rule": "reference quantiles of negative lower-star filtration; inclusive superlevel",
    }
    output_path = Path(output_path)
    if output_path.suffix.lower() != ".npz":
        raise ValueError("finite-primary payload path must end in .npz")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
        unit_names=np.asarray([unit[1] for unit in objective.units]),
        dimensions=np.asarray(dimensions),
        raw_leaves=raw[:, 0].cpu().numpy(),
        leaf_scales=objective.leaf_scales.detach().cpu().numpy(),
        essential_present=present.cpu().numpy(),
        prediction_filtrations=pred_bank.cpu().numpy(),
        reference_filtrations=ref_bank.cpu().numpy(),
        valid_mask=(valid.cpu().numpy() if valid is not None else np.ones(family.grid_shape, dtype=bool)),
        raster_coordinates=family.grid_coordinates.detach().cpu().numpy(),
        quantiles=np.asarray(quantiles), betti_levels=levels.numpy(),
        prediction_betti=pred_betti.numpy(), reference_betti=ref_betti.numpy(),
        prediction_finite=pf, prediction_finite_offsets=pfo,
        prediction_essential=pe, prediction_essential_offsets=peo,
        reference_finite=rf, reference_finite_offsets=rfo,
        reference_essential=re_, reference_essential_offsets=reo,
        **_event_payload(event),
    )
    return output_path


def _event_metrics(reference, prediction, mask, weights):
    common = reference & prediction & mask
    missing = reference & ~prediction & mask
    excess = prediction & ~reference & mask
    area = float(weights[mask].sum())
    tp = float(weights[common].sum())
    miss = float(weights[missing].sum())
    extra = float(weights[excess].sum())
    return {
        "reference_support": (tp + miss) / area,
        "prediction_support": (tp + extra) / area,
        "missing": miss / area,
        "excess": extra / area,
        "precision": tp / (tp + extra) if tp + extra > 0 else None,
        "recall": tp / (tp + miss) if tp + miss > 0 else None,
        "iou": tp / (tp + miss + extra) if tp + miss + extra > 0 else None,
    }


def render_finite_primary_payload(
    payload_path: str | Path, output_dir: str | Path, *, units: tuple[str, ...] | None = None,
    dpi: int = 180,
) -> list[Path]:
    """Render selected exact bank units and an optional native event from saved arrays."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    with np.load(payload_path, allow_pickle=False) as data:
        meta = json.loads(str(data["metadata"].item()))
        if meta.get("definition") != "finite_primary_v1":
            raise ValueError("payload does not contain finite_primary_v1")
        names = list(map(str, data["unit_names"]))
        selected = names if units is None else list(units)
        if not selected or any(name not in names for name in selected):
            raise ValueError("requested filtration names must exist in the saved bank")
        dimensions = list(map(int, data["dimensions"]))
        for name in selected:
            index = names.index(name)
            fig, axes = plt.subplots(len(dimensions), 2, figsize=(9.4, 3.4 * len(dimensions)), squeeze=False)
            for j, dim in enumerate(dimensions):
                ax, curve = axes[j]
                all_points = []
                for prefix, color, marker in (("reference", "#355caa", "o"), ("prediction", "#d0563e", "x")):
                    f0, f1 = data[f"{prefix}_finite_offsets"][index, j]
                    e0, e1 = data[f"{prefix}_essential_offsets"][index, j]
                    finite = data[f"{prefix}_finite"][f0:f1]
                    essential = data[f"{prefix}_essential"][e0:e1]
                    all_points.extend(finite.reshape(-1).tolist())
                    all_points.extend(essential.tolist())
                    if len(finite):
                        ax.scatter(finite[:, 0], finite[:, 1], s=13, c=color, marker=marker, alpha=.7, label=prefix)
                    if len(essential):
                        ax.scatter(essential, np.ones_like(essential), transform=ax.get_xaxis_transform(),
                                   s=30, c=color, marker=marker, clip_on=False)
                low, high = (min(all_points), max(all_points)) if all_points else (0., 1.)
                if high <= low:
                    low, high = low - .5, high + .5
                margin = (high - low) * .08
                ax.plot([low - margin, high + margin], [low - margin, high + margin], c="#999999", lw=.7)
                ax.set(xlim=(low-margin, high+margin), ylim=(low-margin, high+margin),
                       xlabel="Birth", ylabel="Death", title=f"H{dim} persistence (top markers: essential)")
                leaf = data["raw_leaves"][index, j]
                scale = data["leaf_scales"][index, j]
                ax.text(.02, .98, f"finite {leaf[0]:.3g} / scale {scale[0]:.3g}\nessential {leaf[1]:.3g} / scale {scale[1]:.3g}",
                        transform=ax.transAxes, va="top", fontsize=8)
                for prefix, color, ls in (("reference", "#355caa", "-"), ("prediction", "#d0563e", "--")):
                    curve.plot(data["quantiles"], data[f"{prefix}_betti"][index, dim], ls, c=color, label=prefix)
                curve.set(xlabel="Reference quantile of negative filtration", ylabel=f"Betti H{dim}",
                          title="Exact fixed-domain excursion counts")
                curve.legend(fontsize=8)
            fig.suptitle(f"{name} · {meta['endpoint']} · {meta['grid_shape'][0]}×{meta['grid_shape'][1]} raster", fontsize=11)
            fig.tight_layout()
            stem = re.sub(r"[^a-zA-Z0-9_.-]+", "_", name)
            for suffix in ("pdf", "png"):
                path = output_dir / f"shape_{stem}.{suffix}"
                fig.savefig(path, dpi=dpi, bbox_inches="tight")
                outputs.append(path)
            plt.close(fig)
        if "event_reference" in data.files:
            event_meta = json.loads(str(data["event_metadata"].item()))
            mask, weights = data["event_valid_mask"], data["event_weights"]
            ref, pred = data["event_reference"], data["event_prediction"]
            metrics = _event_metrics(ref, pred, mask, weights)
            status = np.zeros(mask.shape, dtype=int)
            status[ref & pred] = 1
            status[ref & ~pred] = 2
            status[pred & ~ref] = 3
            xy = data["event_coordinates"]
            x, y = xy[0, :, 0], xy[:, 0, 1]
            x_edges = np.concatenate(([x[0]], (x[:-1] + x[1:]) / 2, [x[-1]]))
            y_edges = np.concatenate(([y[0]], (y[:-1] + y[1:]) / 2, [y[-1]]))
            fig, axes = plt.subplots(1, 3, figsize=(12, 3.2), constrained_layout=True)
            cmap = ListedColormap(["#eeeeee", "#516f9e", "#d45846", "#e4a539"])
            for ax, arr, title in zip(axes, (ref.astype(int), pred.astype(int), status), ("Reference event", "Predicted event", "Agreement / missing / excess")):
                ax.pcolormesh(x_edges, y_edges, np.ma.masked_where(~mask, arr),
                              shading="flat", cmap=cmap, vmin=0, vmax=3)
                ax.set(title=title, xlabel="x (raw coordinate units)", ylabel="y (raw coordinate units)")
            axes[-1].legend(handles=[Patch(color=cmap(i / 3), label=label) for i, label in
                                     ((1, "both"), (2, "missing"), (3, "excess"))], fontsize=8)
            def shown(value):
                return "unavailable" if value is None else f"{value:.3f}"
            fig.suptitle(
                f"Native event {event_meta['pair']} · {event_meta['measure']} · "
                f"missing={metrics['missing']:.3f}, excess={metrics['excess']:.3f}, IoU={shown(metrics['iou'])}\n"
                f"reference support={metrics['reference_support']:.3f}, prediction support={metrics['prediction_support']:.3f}, "
                f"precision={shown(metrics['precision'])}, recall={shown(metrics['recall'])}\n"
                f"Signs={event_meta['signs']}; signed thresholds={event_meta['thresholds']} ({event_meta['units']})", fontsize=10,
            )
            for suffix in ("pdf", "png"):
                path = output_dir / f"shape_native_event.{suffix}"
                fig.savefig(path, dpi=dpi, bbox_inches="tight")
                outputs.append(path)
            plt.close(fig)
    return outputs

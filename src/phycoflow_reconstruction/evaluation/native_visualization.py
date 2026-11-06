"""Full-native field and weighted-State atlas from matched saved members.

The preparation step verifies frozen native identities and saves all arrays
needed for rendering. Rendering the resulting atlas payload never runs a model,
loads a checkpoint, or fits a metric.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from collections.abc import Mapping
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..data.normalization import FieldNormalizer
from .canonical_coherence import content_sha256
from .native_coherence import _digest_arrays
from .native_scoring import load_native_metric_bundle

_CONTRACT_ARRAYS = (
    "native_ids", "model_coordinates", "raw_coordinates", "physical_mask", "physical_weights",
    "target_model", "observation_coords", "observation_values", "observation_indices",
    "observation_field_ids", "observation_mask", "draw_seeds",
)
_MATCHED_ARRAYS = _CONTRACT_ARRAYS + (
    "target_physical", "field_names", "field_units", "sample_id", "logical_shape",
    "normalizer_sha256",
)
_MATCHED_MANIFESTS = (
    "schema", "query_policy", "sampler", "sensor_protocol", "sensor_manifest_sha256",
    "field_names", "field_units", "normalizer_sha256", "dataset_split",
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_member_payload(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any], dict[str, Any]]:
    """Read the exact saved condition and its inference provenance."""
    manifest_path = path.parent / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"native atlas payload has no sibling manifest: {path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "native_coherence_inference_v1" or manifest.get("query_policy") != "full_native":
        raise ValueError("native atlas requires a full-native inference manifest")
    rows = [row for row in manifest.get("conditions", [])
            if row.get("payload") is not None and (manifest_path.parent / row["payload"]).resolve() == path]
    if len(rows) != 1:
        raise ValueError("native atlas payload must have exactly one manifest condition")
    with np.load(path, allow_pickle=False) as source:
        required = set(_MATCHED_ARRAYS) | {"draws_model", "draws_physical"}
        missing = required.difference(source.files)
        if missing:
            raise ValueError(f"native atlas payload is missing {sorted(missing)}")
        arrays = {key: np.array(source[key], copy=True) for key in required}
    row = rows[0]
    if _digest_arrays(*(arrays[key] for key in _CONTRACT_ARRAYS)) != row.get("contract_sha256"):
        raise ValueError("native atlas payload does not match its condition contract hash")
    if str(arrays["sample_id"].item()) != row.get("sample_id"):
        raise ValueError("native atlas payload sample ID differs from its manifest")
    if arrays["draw_seeds"].tolist() != row.get("draw_seeds"):
        raise ValueError("native atlas payload draw seeds differ from its manifest")
    if (row.get("point_count") != len(arrays["native_ids"])
            or row.get("valid_count") != int(arrays["physical_mask"].sum())):
        raise ValueError("native atlas payload support counts differ from its manifest")
    return arrays, manifest, row


def _pair_bank(bundle: Mapping[str, Any], state: Any) -> tuple[tuple[str, str], ...]:
    """Prefer declared Shape groups, then State pairs, then all State pairs."""
    configs = bundle["configs"]
    groups = configs.get("topology", {}).get("components", {}).get("mutual", {}).get("groups")
    if not groups:
        groups = state.config.get("components", {}).get("mutual", {}).get("pairs")
    pairs = list(groups or combinations(state.field_names, 2))
    selected = []
    seen = set()
    for pair in pairs:
        if (len(pair) != 2 or pair[0] == pair[1]
                or any(name not in state.field_names for name in pair)):
            raise ValueError("native atlas pair bank must contain named State-field pairs")
        names = (str(pair[0]), str(pair[1]))
        if frozenset(names) not in seen:
            selected.append(names)
            seen.add(frozenset(names))
    return tuple(selected)


def _bounds(values: list[np.ndarray]) -> tuple[float, float]:
    minimum = min(float(value.min()) for value in values)
    maximum = max(float(value.max()) for value in values)
    width = maximum - minimum
    padding = 0.02 * width if width else max(abs(minimum) * 0.02, 1e-6)
    return minimum - padding, maximum + padding


def _relative_l2_and_rmse(prediction, target, weights, mask):
    mass = weights[mask] / weights[mask].sum()
    reference = target[mask]
    error = prediction[mask] - reference
    mse = np.sum(mass * error**2, dtype=np.float64)
    power = np.sum(mass * reference**2, dtype=np.float64)
    return (float(np.sqrt(mse / power)) if power > 0 else None, float(np.sqrt(mse)))


def _check_bundle_and_payload(bundle, state, payload, manifest):
    if bundle.get("query_policy") != "full_native" or state.score_definition != "marginal_mixed_weighted_v3":
        raise ValueError("native atlas requires the frozen full-native weighted State definition")
    if state.units != "model_units":
        raise ValueError("native atlas State diagnostics require the frozen model-unit definition")
    if manifest.get("normalizer_sha256") != bundle["normalizer_digest"]:
        raise ValueError("native atlas manifest normalizer differs from the frozen bundle")
    for key in ("field_names", "field_units"):
        if manifest.get(key) != list(bundle["data_spec"][key]):
            raise ValueError(f"native atlas manifest {key} differs from the frozen bundle")
    checks = {
        "model_coordinates": np.asarray(bundle["coordinates"])[0],
        "raw_coordinates": np.asarray(bundle["raw_coordinates"]),
        "native_ids": np.asarray(bundle["native_ids"]),
        "physical_mask": np.asarray(bundle["native_mask"]),
        "field_names": np.asarray(bundle["data_spec"]["field_names"]),
        "field_units": np.asarray(bundle["data_spec"]["field_units"]),
        "logical_shape": np.asarray(bundle["data_spec"]["logical_shape"]),
    }
    for key, expected in checks.items():
        if not np.array_equal(payload[key], expected):
            raise ValueError(f"native atlas {key} differs from the frozen bundle")
    if str(payload["normalizer_sha256"].item()) != bundle["normalizer_digest"]:
        raise ValueError("native atlas payload normalizer differs from the frozen bundle")
    mask = payload["physical_mask"]
    if mask.dtype != np.bool_ or mask.ndim != 1 or not mask.any():
        raise ValueError("native atlas requires a nonempty Boolean physical mask")
    weights = np.asarray(payload["physical_weights"], dtype=np.float64)
    raw_weights = np.asarray(bundle["native_weights"], dtype=np.float64)
    expected = np.where(mask, raw_weights, 0.0)
    if expected.shape != mask.shape or not np.isfinite(expected).all() or np.any(expected < 0) or expected.sum() <= 0:
        raise ValueError("native atlas bundle has invalid physical weights")
    expected = expected / expected.sum()
    if (weights.shape != expected.shape or not np.isfinite(weights).all()
            or not np.allclose(weights, expected, rtol=1e-12, atol=1e-15)):
        raise ValueError("native atlas physical weights differ from the frozen bundle")
    target = payload["target_model"]
    draws = payload["draws_model"]
    if (target.ndim != 2 or target.shape[0] != mask.size or draws.ndim != 3
            or draws.shape[1:] != target.shape or draws.shape[0] != len(payload["draw_seeds"])):
        raise ValueError("native atlas members and target must align as [J,N,C] and [N,C]")
    if (payload["raw_coordinates"].ndim != 2 or payload["raw_coordinates"].shape[1] < 2
            or payload["raw_coordinates"].shape[0] != mask.size):
        raise ValueError("native atlas requires at least two raw coordinate columns")
    if (not np.isfinite(target).all() or not np.isfinite(draws).all()
            or not np.isfinite(payload["target_physical"]).all()
            or not np.isfinite(payload["draws_physical"]).all()):
        raise ValueError("native atlas fields must be finite")
    normalizer = FieldNormalizer(**bundle["normalization"])
    if normalizer.digest() != bundle["normalizer_digest"]:
        raise ValueError("native atlas frozen normalizer does not match its digest")
    for model_key, physical_key in (("target_model", "target_physical"), ("draws_model", "draws_physical")):
        decoded = normalizer.decode(torch.from_numpy(payload[model_key])).double().numpy()
        if not np.array_equal(decoded, payload[physical_key]):
            raise ValueError("native atlas decoded fields do not match the frozen normalizer")


def _occupation_arrays(state, bundle, inputs, member, bins=32, marginal_bins=80):
    names = tuple(state.field_names)
    field_ids = tuple(state.field_ids)
    labels = tuple(inputs)
    first = next(iter(inputs.values()))
    target = first["target_model"]
    mask = first["physical_mask"]
    weights = first["physical_weights"].astype(np.float64)
    weights = np.where(mask, weights, 0.0)
    weights /= weights.sum()
    generated = [inputs[label]["draws_model"][member] for label in labels]
    histograms = np.zeros((len(names), len(labels) + 1, marginal_bins), dtype=np.float64)
    marginal_edges = np.zeros((len(names), marginal_bins + 1), dtype=np.float64)
    for column, field_id in enumerate(field_ids):
        values = [target[:, field_id], *(x[:, field_id] for x in generated)]
        low, high = _bounds([value[mask] for value in values])
        edges = np.linspace(low, high, marginal_bins + 1)
        marginal_edges[column] = edges
        for k, value in enumerate(values):
            histograms[column, k] = np.histogram(value, bins=edges, weights=weights)[0]

    pairs = _pair_bank(bundle, state)
    raw_hist = np.zeros((len(pairs), len(labels) + 1, bins, bins), dtype=np.float64)
    rank_hist = np.zeros_like(raw_hist)
    pair_edges = np.zeros((len(pairs), 2, bins + 1), dtype=np.float64)
    diagnostic_raw = np.zeros((len(pairs), len(labels), 2, bins, bins), dtype=np.float64)
    diagnostic_bounds = np.zeros((len(pairs), len(labels), 4), dtype=np.float64)
    scores = {}
    context = {
        "generated_weights": torch.from_numpy(weights),
        "reference_weights": torch.from_numpy(weights),
        "generated_mask": torch.from_numpy(mask),
        "reference_mask": torch.from_numpy(mask),
        "measure_metadata": {"native_measure_sha256": bundle["measure"]["sha256"]},
    }
    target_tensor = torch.from_numpy(target)[None]
    initial = content_sha256(state.state_artifact())
    for endpoint_index, (label, values) in enumerate(zip(labels, generated)):
        generated_tensor = torch.from_numpy(values)[None]
        with torch.no_grad():
            score = state(generated_tensor, target_tensor, context=context)
            pair_diagnostics = state.pair_diagnostics(
                generated_tensor, target_tensor, context=context, pairs=pairs, bins=bins,
            )
        marginal = score.component_results.get("global_distribution.self.marginal_w2")
        joint = score.component_results.get("global_distribution.cross.joint_topk_swd")
        scores[label] = {
            "family_total": float(score.scalar_loss),
            "marginal_mean": None if marginal is None else float(marginal.scalar_loss),
            "marginal_per_field": None if marginal is None else marginal.diagnostics["per_field_w2"][0].tolist(),
            "mixed_projection_mean": None if joint is None else float(joint.scalar_loss),
            "mixed_projection_count": 0 if joint is None else int(joint.diagnostics["per_direction_w2"].shape[-1]),
            "component_weights": dict(state.component_weights),
        }
        for pair_index, pair in enumerate(pairs):
            item = pair_diagnostics[f"{pair[0]}::{pair[1]}"]
            diagnostic_raw[pair_index, endpoint_index, 0] = item["reference_occupation"][0].numpy()
            diagnostic_raw[pair_index, endpoint_index, 1] = item["generated_occupation"][0].numpy()
            diagnostic_bounds[pair_index, endpoint_index] = item["raw_value_bounds"][0].numpy()
            reference_rank = item["reference_own_rank_occupation"][0].numpy()
            if endpoint_index and not np.array_equal(rank_hist[pair_index, 0], reference_rank):
                raise ValueError("reference rank occupation changed across matched endpoints")
            rank_hist[pair_index, 0] = reference_rank
            rank_hist[pair_index, endpoint_index + 1] = item["generated_own_rank_occupation"][0].numpy()
    if content_sha256(state.state_artifact()) != initial:
        raise RuntimeError("native atlas changed its frozen State artifact")
    for pair_index, (left, right) in enumerate(pairs):
        i, j = names.index(left), names.index(right)
        x_id, y_id = field_ids[i], field_ids[j]
        values = [target, *generated]
        xlow, xhigh = _bounds([value[mask, x_id] for value in values])
        ylow, yhigh = _bounds([value[mask, y_id] for value in values])
        xedges, yedges = np.linspace(xlow, xhigh, bins + 1), np.linspace(ylow, yhigh, bins + 1)
        pair_edges[pair_index] = xedges, yedges
        for k, value in enumerate(values):
            raw_hist[pair_index, k] = np.histogram2d(
                value[:, x_id], value[:, y_id], bins=(xedges, yedges), weights=weights,
            )[0]
    return {
        "state_field_names": np.asarray(names), "state_pairs": np.asarray(pairs, dtype=str).reshape(-1, 2),
        "marginal_edges": marginal_edges, "marginal_occupation": histograms,
        "pair_edges": pair_edges, "pair_raw_occupation": raw_hist,
        "pair_own_rank_occupation": rank_hist, "pair_family_raw_occupation": diagnostic_raw,
        "pair_family_raw_bounds": diagnostic_bounds,
    }, scores


def render_native_atlas(
    bundle_path: str | Path,
    payload_paths: Mapping[str, str | Path],
    output_dir: str | Path,
    *,
    sample_label: str = "",
    member: int = 0,
    include_ensemble_mean: bool = True,
) -> dict[str, Any]:
    """Validate matched native payloads, save scientific data, and render an atlas.

    `payload_paths` maps a SOURCE label and optional checkpoint labels to
    condition NPZs emitted by `evaluate-native-coherence`. Every payload needs
    its sibling native inference manifest. The returned State scores use only
    the selected member. The ensemble mean is a separate fidelity visualization.
    """
    if not isinstance(payload_paths, Mapping) or not payload_paths:
        raise ValueError("native atlas requires a mapping of SOURCE and optional checkpoint payload paths")
    if any(not isinstance(label, str) for label in payload_paths):
        raise TypeError("native atlas endpoint labels must be strings")
    labels = tuple(str(label) for label in payload_paths)
    if (len(set(labels)) != len(labels) or any(not label.strip() for label in labels)
            or sum(label.casefold() == "source" for label in labels) != 1):
        raise ValueError("native atlas requires unique nonempty labels including exactly one SOURCE")
    source_label = next(label for label in labels if label.casefold() == "source")
    labels = (source_label, *(label for label in labels if label != source_label))
    paths = {label: Path(payload_paths[label]).resolve() for label in labels}
    bundle_path = Path(bundle_path).resolve()
    bundle, families = load_native_metric_bundle(bundle_path)
    state = families.get("global_distribution")
    if state is None:
        raise ValueError("native atlas bundle has no frozen State family")
    inputs, manifests, rows = {}, {}, {}
    for label, path in paths.items():
        inputs[label], manifests[label], rows[label] = _load_member_payload(path)
        _check_bundle_and_payload(bundle, state, inputs[label], manifests[label])
    first = inputs[source_label]
    first_manifest, first_row = manifests[source_label], rows[source_label]
    if first_manifest.get("checkpoint_sha256") != bundle.get("source_checkpoint_sha256"):
        raise ValueError("native atlas SOURCE checkpoint differs from the TRAIN metric bundle")
    for label in labels[1:]:
        for key in _MATCHED_ARRAYS:
            if not np.array_equal(first[key], inputs[label][key]):
                raise ValueError(f"native atlas endpoint {label!r} differs in {key}")
        for key in _MATCHED_MANIFESTS:
            if first_manifest.get(key) != manifests[label].get(key):
                raise ValueError(f"native atlas endpoint {label!r} differs in manifest {key}")
        for key in ("sample_id", "dataset_index", "contract_sha256", "point_count", "valid_count", "physical_measure"):
            if first_row.get(key) != rows[label].get(key):
                raise ValueError(f"native atlas endpoint {label!r} differs in condition {key}")
    draw_count = first["draws_model"].shape[0]
    if (isinstance(member, bool) or not isinstance(member, int) or member < 0 or member >= draw_count):
        raise ValueError("native atlas member index is outside the saved draws")
    if include_ensemble_mean and draw_count < 2:
        raise ValueError("an ensemble mean requires at least two saved members")

    representations = [f"member_{member}"]
    if include_ensemble_mean:
        representations.append(f"ensemble_mean_{draw_count}_draws")
    predictions = []
    for label in labels:
        draws = inputs[label]["draws_physical"].astype(np.float64)
        views = [draws[member]]
        if include_ensemble_mean:
            views.append(draws.mean(axis=0))
        predictions.append(np.stack(views))
    predictions = np.stack(predictions)
    target = first["target_physical"].astype(np.float64)
    mask = first["physical_mask"]
    weights = first["physical_weights"].astype(np.float64)
    field_names = tuple(str(value) for value in first["field_names"])
    field_units = tuple(str(value) for value in first["field_units"])
    metrics = {}
    for label_index, label in enumerate(labels):
        metrics[label] = {}
        for view_index, view in enumerate(representations):
            metrics[label][view] = {}
            for field_index, name in enumerate(field_names):
                rel, rmse = _relative_l2_and_rmse(
                    predictions[label_index, view_index, :, field_index],
                    target[:, field_index], weights, mask,
                )
                metrics[label][view][name] = {"relative_l2": rel, "rmse": rmse,
                                              "units": field_units[field_index]}
    occupation, scores = _occupation_arrays(state, bundle, inputs, member)
    metadata = {
        "schema": "native_atlas_v1", "sample_id": str(first["sample_id"].item()),
        "sample_label": str(sample_label or first["sample_id"].item()),
        "source_label": source_label, "endpoint_labels": list(labels),
        "representations": representations, "draw_count": draw_count, "member": member,
        "draw_seeds": first["draw_seeds"].tolist(), "field_names": list(field_names),
        "field_units": list(field_units), "logical_shape": first["logical_shape"].tolist(),
        "metric_bundle_sha256": bundle["sha256"], "bundle_file_sha256": _file_sha256(bundle_path),
        "normalizer_sha256": bundle["normalizer_digest"],
        "native_measure_sha256": bundle["measure"]["sha256"],
        "physical_measure": first_row.get("physical_measure"),
        "support_count": int(mask.sum()), "native_count": int(mask.size),
        "condition_contract_sha256": first_row["contract_sha256"],
        "inputs": {label: {"payload": str(paths[label]), "payload_sha256": _file_sha256(paths[label]),
                           "checkpoint_sha256": manifests[label].get("checkpoint_sha256"),
                           "manifest_sha256": _file_sha256(paths[label].parent / "manifest.json")}
                   for label in labels},
        "fidelity": metrics,
        "state": {"score_definition": state.score_definition, "units": state.units,
                  "score_units": "squared_model_units", "transport_cost": "squared_1d_W2",
                  "score_scope": f"paired_member_{member}_only", "ensemble_mean_scored": False,
                  "scores": scores, "pair_bank": occupation["state_pairs"].tolist(),
                  "pair_occupation": "weighted_raw_shared_bins_and_endpoint_weighted_midrank",
                  "pair_family_raw_bounds": "saved_per_endpoint_from_frozen_pair_diagnostics"},
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    data_path = output / "native_atlas_data.npz"
    if data_path.exists():
        raise FileExistsError("native atlas output already exists; choose a new output directory")
    np.savez_compressed(
        data_path,
        metadata=np.asarray(json.dumps(metadata, sort_keys=True, allow_nan=False)),
        target_physical=target, predictions_physical=predictions,
        raw_coordinates=first["raw_coordinates"], physical_mask=mask, physical_weights=weights,
        **occupation,
    )
    result = render_saved_native_atlas(data_path, output)
    result["payload"] = str(data_path)
    return result


def _rectilinear_edges(coordinates: np.ndarray, shape: tuple[int, ...]):
    if len(shape) != 2 or int(np.prod(shape)) != len(coordinates):
        return None
    grid = coordinates[:, :2].reshape(*shape, 2)
    x, y = grid[0, :, 0], grid[:, 0, 1]
    if (min(shape) < 2 or not np.isfinite(grid).all() or np.any(np.diff(x) <= 0)
            or np.any(np.diff(y) <= 0) or not np.array_equal(grid[..., 0], np.broadcast_to(x, shape))
            or not np.array_equal(grid[..., 1], np.broadcast_to(y[:, None], shape))):
        return None

    def edges(axis):
        return np.concatenate(([axis[0]], (axis[:-1] + axis[1:]) / 2, [axis[-1]]))

    return edges(x), edges(y)


def _field_panel(axis, coordinates, values, mask, shape, geometry, *, cmap, norm):
    if geometry is not None:
        field = np.ma.array(values.reshape(shape), mask=~mask.reshape(shape))
        plotted = axis.pcolormesh(*geometry, field, cmap=cmap, norm=norm, shading="flat", rasterized=True)
    else:
        plotted = axis.scatter(
            coordinates[mask, 0], coordinates[mask, 1], c=values[mask], cmap=cmap, norm=norm,
            s=max(1.0, min(16.0, 3500.0 / max(int(mask.sum()), 1))),
            edgecolors="none", rasterized=True,
        )
    axis.set_aspect("equal", adjustable="box")
    axis.tick_params(labelsize=7, length=2)
    axis.set_xlabel("x (raw coordinates)", fontsize=8)
    axis.set_ylabel("y (raw coordinates)", fontsize=8)
    return plotted


def _metric_label(prediction, target, weights, mask, units):
    relative, rmse = _relative_l2_and_rmse(prediction, target, weights, mask)
    if relative is not None:
        return f"Area rel L2 {relative * 100:.2f}%"
    suffix = " (unit unknown)" if units == "unknown" else f" {units}"
    return f"Area RMSE {rmse:.4g}{suffix}"


def _figure_paths(directory: Path, stem: str) -> tuple[Path, Path]:
    return directory / f"{stem}.png", directory / f"{stem}.pdf"


def _save_figure(figure, png: Path, pdf: Path, *, dpi: int):
    figure.savefig(pdf, bbox_inches="tight", facecolor="white")
    figure.savefig(png, dpi=dpi, bbox_inches="tight", facecolor="white")


def _safe_token(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._-")
    return result or "field"


def _render_fidelity(metadata, data, figure_dir: Path, *, dpi: int):
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    target = data["target_physical"]
    predictions = data["predictions_physical"]
    coordinates = data["raw_coordinates"]
    mask = data["physical_mask"].astype(bool)
    weights = data["physical_weights"]
    labels, views = metadata["endpoint_labels"], metadata["representations"]
    fields, units = metadata["field_names"], metadata["field_units"]
    shape = tuple(metadata["logical_shape"])
    geometry = _rectilinear_edges(coordinates, shape)
    plots = []
    for field_index, (field_name, unit) in enumerate(zip(fields, units)):
        ref = target[:, field_index]
        selected_values = [ref[mask]]
        residual_max = 0.0
        for endpoint in predictions:
            for prediction in endpoint:
                values = prediction[:, field_index]
                selected_values.append(values[mask])
                residual_max = max(residual_max, float(np.abs(values[mask] - ref[mask]).max()))
        low = min(float(value.min()) for value in selected_values)
        high = max(float(value.max()) for value in selected_values)
        if low == high:
            padding = max(abs(low) * .01, 1e-6)
            low, high = low - padding, high + padding
        residual_max = max(residual_max, np.finfo(np.float64).eps)
        value_norm = Normalize(vmin=low, vmax=high)
        residual_norm = Normalize(vmin=-residual_max, vmax=residual_max)
        for view_index, view in enumerate(views):
            # Put each checkpoint on its own row so elongated physical domains
            # remain large enough to inspect alongside their signed residuals.
            rows = len(labels) + 1
            fig, axes = plt.subplots(rows, 2, figsize=(11, max(6, rows * 1.9)),
                                     squeeze=False, layout="constrained")
            reference_image = _field_panel(axes[0, 0], coordinates, ref, mask, shape, geometry,
                                           cmap="viridis", norm=value_norm)
            axes[0, 0].set_title("Reference", fontsize=10, fontweight="bold")
            axes[0, 1].axis("off")
            axes[0, 1].text(
                .03, .90, f"Physical-area support\n{int(mask.sum())} / {len(mask)} nodes\n\n"
                "Residual = prediction − reference\nShared color scales across checkpoints and views",
                transform=axes[0, 1].transAxes, va="top", fontsize=9,
            )
            residual_image = None
            for endpoint_index, label in enumerate(labels):
                value = predictions[endpoint_index, view_index, :, field_index]
                residual = value - ref
                row = endpoint_index + 1
                _field_panel(axes[row, 0], coordinates, value, mask, shape, geometry,
                             cmap="viridis", norm=value_norm)
                residual_image = _field_panel(
                    axes[row, 1], coordinates, residual, mask, shape, geometry,
                    cmap="RdBu_r", norm=residual_norm,
                )
                axes[row, 0].set_title(label, fontsize=10, fontweight="bold")
                axes[row, 1].set_title(f"{label} signed residual", fontsize=9)
                metric = _metric_label(value, ref, weights, mask, unit)
                for axis in axes[row]:
                    axis.text(.02, .98, metric, transform=axis.transAxes, ha="left", va="top", fontsize=7,
                              bbox={"facecolor": "white", "edgecolor": "none", "alpha": .85, "pad": 1.5})
            unit_label = "unit unknown" if unit == "unknown" else unit
            fig.colorbar(reference_image, ax=list(axes[:, 0]), location="bottom", shrink=.86, pad=.04,
                         label=f"{field_name} ({unit_label})")
            fig.colorbar(residual_image, ax=list(axes[1:, 1]), location="bottom", shrink=.86, pad=.04,
                         label=f"Prediction − reference ({unit_label})")
            view_label = (f"member {metadata['member']}" if view.startswith("member_")
                          else f"ensemble mean (J={metadata['draw_count']})")
            fig.suptitle(f"{metadata['sample_label']} · {field_name} · {view_label}", fontsize=13, fontweight="bold")
            stem = f"fidelity_{field_index:02d}_{_safe_token(field_name)}_{_safe_token(view)}"
            png, pdf = _figure_paths(figure_dir, stem)
            _save_figure(fig, png, pdf, dpi=dpi)
            plt.close(fig)
            caption = (f"{field_name}, {view_label}: physical reference, matched predictions and signed residuals. "
                       "Each annotation is computed on the displayed physical-area support; scales are shared across checkpoints and views.")
            plots.append({"preview": f"figures/{png.name}", "pdf": f"figures/{pdf.name}",
                          "alt": f"{field_name} native reference and checkpoint reconstructions for {view_label}",
                          "caption": caption})
    return plots


def _render_marginals(metadata, data, figure_dir: Path, *, dpi: int):
    import matplotlib.pyplot as plt

    fields = [str(name) for name in data["state_field_names"]]
    labels = ["Reference", *metadata["endpoint_labels"]]
    histograms, edges = data["marginal_occupation"], data["marginal_edges"]
    fig, axes = plt.subplots(len(fields), 1, figsize=(10, max(3.0, 2.1 * len(fields))),
                             squeeze=False, layout="constrained")
    colors = ["#202833", "#6B7280", "#157A94", "#D26931", "#7662A8", "#28815D"]
    for field_index, field in enumerate(fields):
        axis = axes[field_index, 0]
        centers = (edges[field_index, 1:] + edges[field_index, :-1]) / 2
        for index, label in enumerate(labels):
            axis.step(centers, histograms[field_index, index], where="mid", label=label,
                      color=colors[index % len(colors)], linewidth=1.8 if index == 0 else 1.3,
                      linestyle="-" if index == 0 else "--" if index == 1 else "-")
        axis.set_title(field, loc="left", fontsize=10, fontweight="bold")
        axis.set_xlabel("Field value (fixed TRAIN model units)", fontsize=9)
        axis.set_ylabel("Area mass / bin", fontsize=9)
        axis.grid(alpha=.15)
        axis.tick_params(labelsize=8)
    axes[0, 0].legend(ncol=min(len(labels), 4), fontsize=8, frameon=False, loc="upper right")
    fig.suptitle(f"{metadata['sample_label']} · State marginals · member {metadata['member']}",
                 fontsize=13, fontweight="bold")
    png, pdf = _figure_paths(figure_dir, "state_marginals")
    _save_figure(fig, png, pdf, dpi=dpi)
    plt.close(fig)
    return {"preview": f"figures/{png.name}", "pdf": f"figures/{pdf.name}",
            "alt": "Physical-area weighted State marginal occupations for matched checkpoints",
            "caption": ("Per-field weighted occupation uses a shared binning across the reference and matched member predictions in fixed TRAIN model units. "
                        "These histograms describe State marginals; the mixed-projection W2-squared mean is recorded separately in the numerical metadata.")}


def _render_pairs(metadata, data, figure_dir: Path, *, dpi: int):
    import matplotlib.pyplot as plt
    from matplotlib.colors import PowerNorm

    labels = ["Reference", *metadata["endpoint_labels"]]
    pairs = data["state_pairs"]
    raw, ranks, edges = data["pair_raw_occupation"], data["pair_own_rank_occupation"], data["pair_edges"]
    plots = []
    for pair_index, pair in enumerate(pairs):
        left, right = map(str, pair)
        fig, axes = plt.subplots(2, len(labels), figsize=(max(9, 2.9 * len(labels)), 6.4),
                                 squeeze=False, layout="constrained")
        raw_norm = PowerNorm(gamma=.35, vmin=0, vmax=max(float(raw[pair_index].max()), np.finfo(float).eps))
        rank_norm = PowerNorm(gamma=.35, vmin=0, vmax=max(float(ranks[pair_index].max()), np.finfo(float).eps))
        bounds = (edges[pair_index, 0, 0], edges[pair_index, 0, -1],
                  edges[pair_index, 1, 0], edges[pair_index, 1, -1])
        for column, label in enumerate(labels):
            image_raw = axes[0, column].imshow(raw[pair_index, column].T, origin="lower", extent=bounds,
                                               aspect="auto", cmap="magma", norm=raw_norm)
            image_rank = axes[1, column].imshow(ranks[pair_index, column].T, origin="lower", extent=(0, 1, 0, 1),
                                                aspect="equal", cmap="magma", norm=rank_norm)
            axes[0, column].set_title(label, fontsize=10, fontweight="bold")
            axes[0, column].set_xlabel(f"{left} (model units)", fontsize=8)
            axes[0, column].set_ylabel(f"{right} (model units)", fontsize=8)
            axes[1, column].set_xlabel(f"{left} own rank", fontsize=8)
            axes[1, column].set_ylabel(f"{right} own rank", fontsize=8)
            for axis in axes[:, column]:
                axis.tick_params(labelsize=7)
        fig.colorbar(image_raw, ax=list(axes[0]), location="right", shrink=.83, label="Area mass / bin (power scale)")
        fig.colorbar(image_rank, ax=list(axes[1]), location="right", shrink=.83, label="Area mass / bin (power scale)")
        fig.suptitle(f"{metadata['sample_label']} · {left} / {right} occupation · member {metadata['member']}",
                     fontsize=13, fontweight="bold")
        stem = f"state_pair_{pair_index:02d}_{_safe_token(left)}_{_safe_token(right)}"
        png, pdf = _figure_paths(figure_dir, stem)
        _save_figure(fig, png, pdf, dpi=dpi)
        plt.close(fig)
        plots.append({"preview": f"figures/{png.name}", "pdf": f"figures/{pdf.name}",
                      "alt": f"Weighted raw and own-rank occupation for {left} and {right}",
                      "caption": (f"{left}–{right}: raw-value occupation uses shared model-unit bins; own-rank occupation uses each endpoint's weighted midranks. "
                                  "Both use the same physical support and represent the selected member; a shared power color scale reveals low-mass bins.")})
    return plots


def render_saved_native_atlas(
    payload_path: str | Path,
    output_dir: str | Path,
    *,
    dpi: int = 150,
) -> dict[str, Any]:
    """Render PDF/PNG figures and a Markdown index from saved atlas data only."""
    import matplotlib

    matplotlib.use("Agg")
    source = Path(payload_path).resolve()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    retained = output / "native_atlas_data.npz"
    if source != retained.resolve():
        if retained.exists() and _file_sha256(retained) != _file_sha256(source):
            raise FileExistsError("a different native atlas payload already exists in the output")
        if not retained.exists():
            shutil.copy2(source, retained)
    with np.load(retained, allow_pickle=False) as stored:
        metadata = json.loads(str(stored["metadata"].item()))
        if metadata.get("schema") != "native_atlas_v1":
            raise ValueError("unsupported native atlas payload schema")
        data = {key: np.array(stored[key], copy=True) for key in stored.files if key != "metadata"}
    figure_dir = output / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    plots = _render_fidelity(metadata, data, figure_dir, dpi=dpi)
    plots.append(_render_marginals(metadata, data, figure_dir, dpi=dpi))
    plots.extend(_render_pairs(metadata, data, figure_dir, dpi=dpi))
    report = {**metadata, "payload": retained.name, "payload_sha256": _file_sha256(retained),
              "figures": plots, "index": "INDEX.md"}
    report_path = output / "native_atlas.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    view_text = (
        f"Member {metadata['member']} and the J={metadata['draw_count']} ensemble mean are separate fidelity views. "
        if len(metadata["representations"]) > 1 else f"Fidelity views show member {metadata['member']}. "
    )
    lines = [
        "# Native fidelity and State atlas", "",
        f"Sample: {metadata['sample_label']} (`{metadata['sample_id']}`). Physical support: {metadata['support_count']} of {metadata['native_count']} nodes.", "",
        view_text + "State values use only the selected member. Their input fields use fixed TRAIN model units, and the squared transport costs use squared model units.", "",
        "[Numerical payload](native_atlas_data.npz) · [Provenance and metrics](native_atlas.json)", "",
        "## Frozen weighted State values", "",
        ("The v3 family value is the weighted sum of the marginal W2-squared mean and the mixed-projection W2-squared mean. "
         "Pair occupations are separate diagnostics and do not enter this value."), "",
        "| Endpoint | Family value | Marginal W2-squared mean | Mixed-projection W2-squared mean | Directions |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for label in metadata["endpoint_labels"]:
        score = metadata["state"]["scores"][label]
        marginal, mixed = score["marginal_mean"], score["mixed_projection_mean"]
        marginal_text = "—" if marginal is None else f"{marginal:.6g}"
        mixed_text = "—" if mixed is None else f"{mixed:.6g}"
        lines.append(f"| {label} | {score['family_total']:.6g} | {marginal_text} | "
                     f"{mixed_text} | {score['mixed_projection_count']} |")
    weights = metadata["state"]["scores"][metadata["endpoint_labels"][0]]["component_weights"]
    weight_text = ", ".join(f"`{key}` = {value:g}" for key, value in weights.items())
    lines.extend(("", f"Component weights: {weight_text}. Costs are in squared fixed TRAIN model units.", ""))
    for plot in plots:
        lines.extend((f"![{plot['alt']}]({plot['preview']})", "", plot["caption"], "",
                      f"[PDF master]({plot['pdf']})", ""))
    index = output / "INDEX.md"
    index.write_text("\n".join(lines), encoding="utf-8")
    return {"index": str(index), "report": str(report_path), "payload": str(retained),
            "figures": plots}

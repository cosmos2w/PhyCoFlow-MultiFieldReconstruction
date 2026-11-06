"""Saved-member native Scale atlas using one frozen TRAIN metric bundle.

The atlas renders a selected stochastic member, while scores use all saved
conditional members. Its plotted-band errors use the displayed analysis support
and the saved native physical weights. No model inference or refitting occurs.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch

from ..coherence.families.multiscale_spectrum.statistics import band_fields
from ..training.run_store import file_sha256
from .canonical_coherence import _json_value, content_sha256
from .native_coherence import _digest_arrays
from .native_scoring import load_native_metric_bundle


def _band_error(prediction, reference, weights, mask):
    """Relative L2, or weighted RMSE for an exactly zero reference band."""
    prediction = np.asarray(prediction, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    weights, mask = np.asarray(weights, dtype=np.float64), np.asarray(mask, dtype=bool)
    if (
        prediction.shape != reference.shape
        or prediction.shape != weights.shape
        or mask.shape != weights.shape
    ):
        raise ValueError("band error fields, physical weights and mask must align")
    if (
        not mask.any()
        or not np.isfinite(prediction[mask]).all()
        or not np.isfinite(reference[mask]).all()
    ):
        raise ValueError("band error requires finite fields on a nonempty displayed support")
    if not np.isfinite(weights[mask]).all() or np.any(weights[mask] <= 0):
        raise ValueError(
            "band error requires positive finite physical weights on the displayed support"
        )
    w = weights[mask] / weights[mask].sum()
    difference = prediction[mask] - reference[mask]
    mse = float(np.sum(w * difference**2))
    power = float(np.sum(w * reference[mask] ** 2))
    return (
        ("relative L2", float(np.sqrt(mse / power))) if power > 0 else ("RMSE", float(np.sqrt(mse)))
    )


def _read_native_member(path, bundle, member):
    required = (
        "draws_model",
        "target_model",
        "model_coordinates",
        "raw_coordinates",
        "native_ids",
        "physical_mask",
        "physical_weights",
        "field_names",
        "field_units",
        "normalizer_sha256",
        "sample_id",
        "logical_shape",
        "observation_coords",
        "observation_values",
        "observation_field_ids",
        "observation_indices",
        "observation_mask",
        "draw_seeds",
    )
    with np.load(path, allow_pickle=False) as data:
        missing = sorted(set(required) - set(data.files))
        if missing:
            raise ValueError(f"native Scale payload is missing {missing}")
        item = {key: np.asarray(data[key]).copy() for key in required}
    if not np.array_equal(item["model_coordinates"], bundle["coordinates"][0].cpu().numpy()):
        raise ValueError("native Scale model coordinates differ from the frozen bundle")
    for key, bundle_key in (
        ("raw_coordinates", "raw_coordinates"),
        ("native_ids", "native_ids"),
        ("physical_mask", "native_mask"),
    ):
        if not np.array_equal(item[key], bundle[bundle_key].cpu().numpy()):
            raise ValueError(f"native Scale {key} differs from the frozen bundle")
    expected = np.where(
        bundle["native_mask"].cpu().numpy(),
        bundle["native_weights"].cpu().numpy().astype(np.float64),
        0.0,
    )
    expected /= expected.sum()
    if not np.allclose(item["physical_weights"], expected, rtol=1e-12, atol=1e-15):
        raise ValueError("native Scale physical weights differ from the frozen bundle")
    if (
        tuple(map(str, item["field_names"])) != tuple(bundle["data_spec"]["field_names"])
        or tuple(map(str, item["field_units"])) != tuple(bundle["data_spec"]["field_units"])
        or str(item["normalizer_sha256"].item()) != bundle["normalizer_digest"]
    ):
        raise ValueError(
            "native Scale field order, units or normalizer differ from the frozen bundle"
        )
    draws, target = item["draws_model"], item["target_model"]
    if draws.ndim != 3 or target.ndim != 2 or draws.shape[1:] != target.shape or draws.shape[0] < 2:
        raise ValueError("native Scale requires multiple conditional draws and one aligned target")
    if not 0 <= member < draws.shape[0]:
        raise ValueError("selected native Scale member index is outside the saved ensemble")
    if (
        item["draw_seeds"].shape != (draws.shape[0],)
        or len(set(item["draw_seeds"].tolist())) != draws.shape[0]
    ):
        raise ValueError("native Scale requires one distinct seed for every saved draw")
    return item


def _matched_manifests(paths, items):
    """Validate saved sampler/checkpoint receipts when they accompany the panel."""
    locations = [Path(path).parent / "manifest.json" for path in paths]
    available = [path.is_file() for path in locations]
    if any(available) and not all(available):
        raise ValueError("native Scale endpoints must all supply their panel manifests or none do")
    if not any(available):
        return None
    manifests = [json.loads(path.read_text(encoding="utf-8")) for path in locations]
    if any(
        manifest.get("schema") != "native_coherence_inference_v1"
        or manifest.get("query_policy") != "full_native"
        for manifest in manifests
    ):
        raise ValueError("native Scale requires full-native inference manifests")
    keys = (
        "schema",
        "query_policy",
        "sampler",
        "sensor_protocol",
        "sensor_manifest_sha256",
        "field_names",
        "field_units",
        "normalizer_sha256",
        "dataset_split",
    )
    for manifest in manifests[1:]:
        for key in keys:
            if manifest.get(key) != manifests[0].get(key):
                raise ValueError(f"native Scale endpoint sampler contract differs: {key}")
    records = []
    digest_keys = (
        "native_ids",
        "model_coordinates",
        "raw_coordinates",
        "physical_mask",
        "physical_weights",
        "target_model",
        "observation_coords",
        "observation_values",
        "observation_indices",
        "observation_field_ids",
        "observation_mask",
        "draw_seeds",
    )
    for path, location, manifest, item in zip(paths, locations, manifests, items):
        rows = [
            row for row in manifest.get("conditions", ()) if row.get("payload") == Path(path).name
        ]
        if (
            len(rows) != 1
            or rows[0].get("sample_id") != str(item["sample_id"].item())
            or rows[0].get("contract_sha256") != _digest_arrays(*(item[key] for key in digest_keys))
        ):
            raise ValueError("native Scale payload does not match its panel manifest")
        records.append(
            {
                "manifest_path": str(location.resolve()),
                "manifest_sha256": file_sha256(location),
                "checkpoint_path": manifest.get("checkpoint_path"),
                "checkpoint_sha256": manifest.get("checkpoint_sha256"),
                "condition_contract_sha256": rows[0]["contract_sha256"],
            }
        )
    return {"sampler": manifests[0].get("sampler"), "records": records}


@torch.no_grad()
def render_native_scale(bundle_path, payload_paths: Mapping[str, str | Path], output_dir, member=0):
    """Save and render matched native Scale maps and diagnostics from saved draws.

    ``payload_paths`` maps endpoint labels to matched condition NPZs. All maps
    show ``member``; scores use every independent saved draw for that endpoint.
    """
    if not isinstance(payload_paths, Mapping) or not payload_paths:
        raise ValueError("native Scale requires at least one named endpoint payload")
    labels = tuple(str(label) for label in payload_paths)
    if len(set(labels)) != len(labels) or any(
        not label or label == "Reference" for label in labels
    ):
        raise ValueError("native Scale endpoint labels must be nonempty, unique and not Reference")
    source_indices = [index for index, label in enumerate(labels) if label.casefold() == "source"]
    if len(source_indices) > 1:
        raise ValueError("native Scale allows at most one SOURCE endpoint")
    if isinstance(member, bool) or not isinstance(member, int):
        raise TypeError("member must be an integer index")
    bundle, families = load_native_metric_bundle(bundle_path)
    family = families["multiscale_spectrum"]
    if family.objective != "signed_band_energy_score" or not family.mean_augmented:
        raise ValueError("native Scale atlas requires the frozen mean-augmented conditional family")
    identity = content_sha256(family.state_artifact())
    items = [_read_native_member(Path(path), bundle, member) for path in payload_paths.values()]
    manifest_receipts = _matched_manifests(tuple(payload_paths.values()), items)
    if source_indices:
        if manifest_receipts is None:
            raise ValueError("native Scale SOURCE requires its inference manifest")
        source_receipt = manifest_receipts["records"][source_indices[0]]
        if source_receipt["checkpoint_sha256"] != bundle["source_checkpoint_sha256"]:
            raise ValueError("native Scale SOURCE checkpoint differs from the TRAIN metric bundle")
    first = items[0]
    matched_keys = (
        "sample_id",
        "target_model",
        "logical_shape",
        "observation_coords",
        "observation_values",
        "observation_field_ids",
        "observation_indices",
        "observation_mask",
        "draw_seeds",
    )
    for item in items[1:]:
        for key in matched_keys:
            if not np.array_equal(item[key], first[key]):
                raise ValueError(f"native Scale matched endpoint contract differs: {key}")
    target = torch.from_numpy(first["target_model"])[None]
    coordinates = torch.from_numpy(first["model_coordinates"])[None]
    family._require_ready(target, coordinates)
    target_stats = family._statistics(target)
    geometry = family._geometry_tensors(family.compute_dtype)

    def maps(statistics):
        selected = band_fields(statistics["coefficients"], geometry["basis"], geometry["filters"])
        output = np.full(
            (selected.shape[0], selected.shape[1], len(first["native_ids"]), selected.shape[-1]),
            np.nan,
        )
        output[:, :, family.analysis_index.cpu().numpy(), :] = selected.cpu().numpy()
        return output

    reference_map = maps(target_stats)[0]
    analysis_mask = np.zeros(len(first["native_ids"]), dtype=bool)
    analysis_mask[family.analysis_index.cpu().numpy()] = True
    analysis_mask &= first["physical_mask"].astype(bool)
    if not analysis_mask.any():
        raise ValueError("native Scale analysis and physical masks do not overlap")
    component = family.settings.component("signed_band_cospectrum")
    pairs = () if component is None else component.pairs
    reference_gram = target_stats["gram"][0].cpu().numpy()
    reference_total = target_stats["fluctuation_power"][0].cpu().numpy()

    def unresolved(stats):
        total = stats["fluctuation_power"][0].cpu().numpy()
        power = torch.diagonal(stats["gram"][0], dim1=-2, dim2=-1).cpu().numpy().sum(axis=0)
        return total - power

    band_maps, means, unresolved_values, grams, scores = [], [], [], [], []
    means.append(target_stats["piece_means"][0].cpu().numpy())
    unresolved_values.append(unresolved(target_stats))
    grams.append(
        np.asarray(
            [
                [reference_gram[s, i, j] for i, j in pairs]
                for s in range(len(family.settings.bands.names))
            ]
        )
    )
    for item in items:
        draws = torch.from_numpy(item["draws_model"])
        selected = draws[member : member + 1]
        result = family(
            selected,
            target,
            coordinates=coordinates,
            context={
                "conditional_draws": tuple(
                    draws[i : i + 1] for i in range(len(draws)) if i != member
                ),
                "score_all_conditional_draws": True,
            },
        )
        statistics = family._statistics(selected)
        band_maps.append(maps(statistics)[0])
        means.append(statistics["piece_means"][0].cpu().numpy())
        unresolved_values.append(unresolved(statistics))
        gram = statistics["gram"][0].cpu().numpy()
        grams.append(
            np.asarray(
                [[gram[s, i, j] for i, j in pairs] for s in range(len(family.settings.bands.names))]
            )
        )
        primary = result.component_results[
            "multiscale_spectrum.signed_band_energy_score.conditional"
        ].diagnostics
        joint = result.diagnostics["signed_band"]["joint_energy_score"]
        factor = 1.0 - (1.0 - family.settings.almost_fair_alpha) / len(draws)
        scores.append(
            {
                "per_band_score": primary["per_band"],
                "per_band_attraction": primary["attraction_per_band"],
                "per_band_repulsion": primary["repulsion_per_band"],
                "band_weights": primary["band_weights"],
                "per_band_mean_pair_separation": [
                    2.0 * value / factor for value in primary["repulsion_per_band"]
                ],
                "joint_score": float(
                    result.component_results[
                        "multiscale_spectrum.joint_spectral_mean_energy_score.conditional"
                    ].scalar_loss
                ),
                "joint_attraction": joint["attraction"],
                "joint_repulsion": joint["repulsion"],
                "joint_mean_pair_separation": 2.0 * joint["repulsion"],
                "family_score": float(result.scalar_loss),
            }
        )
    if content_sha256(family.state_artifact()) != identity:
        raise RuntimeError("native Scale visualization changed the frozen family artifact")
    metadata = {
        "version": "native_scale_atlas_v1",
        "bundle_sha256": bundle["sha256"],
        "payloads": {
            label: {"path": str(Path(path).resolve()), "sha256": file_sha256(path)}
            for label, path in zip(labels, payload_paths.values())
        },
        "sample_id": str(first["sample_id"].item()),
        "labels": labels,
        "field_names": family.field_names,
        "field_units": family.units,
        "bands": family.settings.bands.names,
        "member": member,
        "draw_count": len(first["draws_model"]),
        "logical_shape": first["logical_shape"].tolist(),
        "mask": "native physical mask intersected with frozen Scale analysis support",
        "physical_measure": bundle["measure"],
        "displayed_valid_points": int(analysis_mask.sum()),
        "means_definition": "piecewise mass-weighted means of the displayed selected member",
        "unresolved_definition": "fluctuation power minus summed represented band power; no clipping",
        "cospectrum_definition": "signed within-state band Gram, not dataset covariance or energy transfer",
        "pairs": [[family.field_names[i], family.field_names[j]] for i, j in pairs],
        "scores": scores,
        "reference_fluctuation_power": reference_total.tolist(),
        "piece_mass_fractions": (family.piece_measures.sum(1) / family.piece_measures.sum())
        .cpu()
        .tolist(),
        "score_interpretation": "all independent saved draws; maps and field diagnostics use selected member",
        "sampler_receipts": manifest_receipts,
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = output_dir / "scale_atlas.npz"
    np.savez_compressed(
        payload,
        metadata=np.asarray(json.dumps(_json_value(metadata), sort_keys=True, allow_nan=False)),
        reference_bands=reference_map,
        member_bands=np.stack(band_maps),
        raw_coordinates=first["raw_coordinates"],
        physical_weights=first["physical_weights"],
        display_mask=analysis_mask,
        piece_means=np.stack(means),
        unresolved_power=np.stack(unresolved_values),
        signed_cospectrum=np.stack(grams),
    )
    return render_native_scale_payload(payload, output_dir)


def render_native_scale_payload(payload_path, output_dir):
    """Re-render an atlas entirely from its saved plot payload, without inference."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with np.load(payload_path, allow_pickle=False) as data:
        meta = json.loads(str(data["metadata"].item()))
        if meta.get("version") != "native_scale_atlas_v1":
            raise ValueError("unknown native Scale atlas payload version")
        reference = data["reference_bands"].copy()
        predictions = data["member_bands"].copy()
        coordinates = data["raw_coordinates"].copy()
        weights = data["physical_weights"].copy()
        mask = data["display_mask"].astype(bool)
        means = data["piece_means"].copy()
        unresolved = data["unresolved_power"].copy()
        cospectrum = data["signed_cospectrum"].copy()
    labels, fields, bands = meta["labels"], meta["field_names"], meta["bands"]
    shape = tuple(int(x) for x in meta["logical_shape"])
    if len(shape) != 2 or np.prod(shape) != len(mask) or min(shape) < 2:
        raise ValueError("native Scale map rendering requires a declared 2-D rectangular grid")
    if (
        reference.shape != (len(bands), len(mask), len(fields))
        or predictions.shape != (len(labels), *reference.shape)
        or coordinates.shape != (len(mask), 2)
    ):
        raise ValueError("native Scale atlas maps do not align with their saved dimensions")
    grid = coordinates.reshape(*shape, 2)
    x, y = grid[0, :, 0], grid[:, 0, 1]
    if (
        not np.isfinite(grid).all()
        or np.any(np.diff(x) <= 0)
        or np.any(np.diff(y) <= 0)
        or not np.allclose(grid[..., 0], x[None, :])
        or not np.allclose(grid[..., 1], y[:, None])
    ):
        raise ValueError("native Scale maps require increasing rectilinear physical coordinates")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stored_payload = output_dir / "scale_atlas.npz"
    if Path(payload_path).resolve() != stored_payload.resolve():
        shutil.copy2(payload_path, stored_payload)
    figures = []
    errors = {}
    grid_mask = mask.reshape(shape)
    for band_index, band_name in enumerate(bands):
        columns = 1 + 2 * len(labels)
        fig, axes = plt.subplots(
            len(fields),
            columns,
            squeeze=False,
            figsize=(2.6 * columns, 1.45 * len(fields) + 1.3),
            layout="constrained",
        )
        for field_index, field_name in enumerate(fields):
            target = reference[band_index, :, field_index]
            maps = predictions[:, band_index, :, field_index]
            # Shared signed limits for truth and every model; a second shared
            # signed scale applies to all model-minus-reference panels.
            limit = float(
                np.max(np.abs(np.concatenate((target[mask], maps[:, mask].reshape(-1))))) or 1.0
            )
            residuals = maps - target[None]
            error_limit = float(np.max(np.abs(residuals[:, mask])) or 1.0)
            target_grid = np.ma.masked_where(~grid_mask, target.reshape(shape))
            truth = axes[field_index, 0].pcolormesh(
                x,
                y,
                target_grid,
                cmap="RdBu_r",
                vmin=-limit,
                vmax=limit,
                shading="nearest",
                rasterized=True,
            )
            axes[field_index, 0].set_title(f"{field_name} · reference", fontsize=9)
            for endpoint, label in enumerate(labels):
                pred = maps[endpoint]
                kind, value = _band_error(pred, target, weights, mask)
                errors[f"{label}|{band_name}|{field_name}"] = {
                    "metric": kind,
                    "value": value,
                    "units": meta["field_units"] if kind == "RMSE" else "dimensionless",
                }
                metric = f"{kind} {value:.3g}" + (
                    f" ({meta['field_units']})" if kind == "RMSE" else ""
                )
                pred_ax, error_ax = axes[field_index, 1 + 2 * endpoint : 3 + 2 * endpoint]
                pred_ax.pcolormesh(
                    x,
                    y,
                    np.ma.masked_where(~grid_mask, pred.reshape(shape)),
                    cmap="RdBu_r",
                    vmin=-limit,
                    vmax=limit,
                    shading="nearest",
                    rasterized=True,
                )
                error_plot = error_ax.pcolormesh(
                    x,
                    y,
                    np.ma.masked_where(~grid_mask, residuals[endpoint].reshape(shape)),
                    cmap="RdBu_r",
                    vmin=-error_limit,
                    vmax=error_limit,
                    shading="nearest",
                    rasterized=True,
                )
                pred_ax.set_title(f"{label} · member {meta['member']}\n{metric}", fontsize=8)
                error_ax.set_title(f"{label} − reference\n{metric}", fontsize=8)
            for axis in axes[field_index]:
                axis.set_aspect("equal")
                axis.tick_params(labelsize=7)
            colorbar = fig.colorbar(
                truth,
                ax=axes[field_index, 0],
                orientation="horizontal",
                fraction=0.09,
                pad=0.08,
                ticks=[-limit, limit],
                format="%.2g",
            )
            colorbar.ax.tick_params(labelsize=7)
            colorbar.set_label(meta["field_units"], fontsize=7)
            if labels:
                colorbar = fig.colorbar(
                    error_plot,
                    ax=axes[field_index, 2::2].tolist(),
                    orientation="horizontal",
                    fraction=0.09,
                    pad=0.08,
                    ticks=[-error_limit, error_limit],
                    format="%.2g",
                )
                colorbar.ax.tick_params(labelsize=7)
                colorbar.set_label("difference", fontsize=7)
        fig.suptitle(
            f"{band_name} signed-band fields · {meta['sample_id']} · selected member {meta['member']}\nPhysical-area weighted errors on the displayed valid analysis support",
            fontsize=12,
        )
        fig.supxlabel("Raw x coordinate (dataset units)", fontsize=8)
        fig.supylabel("Raw y coordinate (dataset units)", fontsize=8)
        stem = f"scale_band_{band_index:02d}"
        for suffix in ("pdf", "png"):
            fig.savefig(output_dir / f"{stem}.{suffix}", dpi=180, bbox_inches="tight")
        plt.close(fig)
        figures.append(
            (
                stem,
                f"{band_name} signed-band target, member predictions and signed errors; panel annotations use physical-mass weighting.",
            )
        )

    # Score lines are ensemble scores; they are never calculated from the plotted
    # member or from an ensemble-mean map.
    fig, axes = plt.subplots(1, 5, figsize=(19, 3.8), layout="constrained")
    palette = ("#356b9a", "#bb6748", "#638447", "#8a6793", "#a28a47")
    for i, (label, score) in enumerate(zip(labels, meta["scores"])):
        color = palette[i % len(palette)]
        for axis, key in zip(
            axes[:4],
            (
                "per_band_score",
                "per_band_attraction",
                "per_band_mean_pair_separation",
                "per_band_repulsion",
            ),
        ):
            axis.plot(bands, score[key], marker="o", label=label, color=color)
        positions = np.arange(3) + (i - (len(labels) - 1) / 2) * 0.8 / len(labels)
        axes[4].bar(
            positions,
            [score["joint_score"], score["joint_attraction"], score["joint_repulsion"]],
            width=0.8 / len(labels),
            label=label,
            color=color,
        )
    for axis, title in zip(
        axes[:4], ("Signed-band score", "Attraction", "Mean pair separation", "Weighted repulsion")
    ):
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=25)
        axis.grid(axis="y", alpha=0.2)
    axes[4].set_xticks(np.arange(3), ["Score", "Attraction", "Repulsion"], rotation=25)
    axes[4].set_title("Joint signed-band + piece-mean score")
    axes[0].set_ylabel("Represented normalized distance")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(
        f"Conditional Scale scores · {meta['draw_count']} independent members · {meta['sample_id']}"
    )
    for suffix in ("pdf", "png"):
        fig.savefig(output_dir / f"scale_scores.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(fig)
    figures.append(
        (
            "scale_scores",
            "Conditional signed-band and joint scores show attraction, mean pair separation and weighted repulsion separately.",
        )
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 4), layout="constrained")
    all_labels = ["Reference", *labels]
    positions = np.arange(len(fields))
    for i, label in enumerate(all_labels):
        color = "#30343b" if i == 0 else palette[(i - 1) % len(palette)]
        axes[0].plot(positions, unresolved[i], marker="o", label=label, color=color)
        for piece in range(means.shape[1]):
            axes[1].plot(
                positions,
                means[i, piece],
                marker="o",
                linestyle="-" if piece == 0 else "--",
                label=f"{label}, piece {piece}",
                color=color,
            )
    for axis in axes:
        axis.set_xticks(positions, fields, rotation=25)
        axis.grid(axis="y", alpha=0.2)
    axes[0].set(title="Unresolved fluctuation power", ylabel=f"{meta['field_units']}²")
    axes[1].set(title="Piece means", ylabel=meta["field_units"])
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].legend(frameon=False, fontsize=7)
    fig.suptitle(f"Selected member diagnostics · {meta['sample_id']}")
    for suffix in ("pdf", "png"):
        fig.savefig(output_dir / f"scale_means_unresolved.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(fig)
    figures.append(
        (
            "scale_means_unresolved",
            "Piece means and unresolved fluctuation power describe the selected member and reference.",
        )
    )

    if meta["pairs"]:
        limit = float(np.max(np.abs(cospectrum)) or 1.0)
        fig, axes = plt.subplots(
            1,
            len(all_labels),
            figsize=(3.2 * len(all_labels), max(3.5, 0.32 * len(meta["pairs"]) + 1.8)),
            squeeze=False,
            layout="constrained",
        )
        for index, (axis, label) in enumerate(zip(axes[0], all_labels)):
            image = axis.imshow(
                cospectrum[index].T, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto"
            )
            axis.set_title(label)
            axis.set_xticks(range(len(bands)), bands, rotation=30, ha="right")
            axis.set_yticks(range(len(meta["pairs"])), [" × ".join(pair) for pair in meta["pairs"]])
            axis.tick_params(labelsize=8)
        fig.colorbar(image, ax=axes[0].tolist(), label=f"Signed band Gram ({meta['field_units']}²)")
        fig.suptitle("Declared field-pair signed co-spectrum · selected member")
        for suffix in ("pdf", "png"):
            fig.savefig(
                output_dir / f"scale_signed_cospectrum.{suffix}", dpi=180, bbox_inches="tight"
            )
        plt.close(fig)
        figures.append(
            (
                "scale_signed_cospectrum",
                "Signed within-state band Gram values use the declared field-pair bank and a shared color scale.",
            )
        )
    report = {
        **meta,
        "band_errors": errors,
        "piece_means": means.tolist(),
        "unresolved_power": unresolved.tolist(),
        "signed_cospectrum": cospectrum.tolist(),
        "artifacts": {
            "payload": stored_payload.name,
            "figures": {name: {"png": f"{name}.png", "pdf": f"{name}.pdf"} for name, _ in figures},
        },
    }
    report_path = output_dir / "scale_atlas.json"
    report_path.write_text(
        json.dumps(_json_value(report), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    index = [
        f"# Native Scale atlas: {meta['sample_id']}",
        "",
        "Scores use all independent members; maps use the selected member.",
        "",
        "[Saved numerical payload](scale_atlas.npz) · [Numerical report](scale_atlas.json)",
        "",
    ]
    for stem, caption in figures:
        index.extend([f"![{caption}]({stem}.png)", "", f"{caption} [PDF master]({stem}.pdf)", ""])
    index_path = output_dir / "README.md"
    index_path.write_text("\n".join(index), encoding="utf-8")
    return {
        "payload": str(stored_payload),
        "report": str(report_path),
        "index": str(index_path),
        "figures": {
            name: {"png": str(output_dir / f"{name}.png"), "pdf": str(output_dir / f"{name}.pdf")}
            for name, _ in figures
        },
    }

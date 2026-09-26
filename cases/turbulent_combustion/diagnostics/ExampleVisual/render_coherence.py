"""Render matched explanatory views from the standard coherence evaluation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from phycoflow_reconstruction.coherence.families.topology.persistence import (
    cubical_diagrams,
    sliced_diagram_distance,
)
from phycoflow_reconstruction.config import load_config

ROOT = Path(__file__).resolve().parent
COHERENCE = ROOT / "postprocessing" / "coherence"
EXPLANATORY = ROOT / "explanatory"
SOURCE_COLOR = "#687786"
POST_COLOR = "#16647F"
REFERENCE_COLOR = "#1C3445"


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {key: np.asarray(payload[key]) for key in payload.files}


def _save(figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=250, bbox_inches="tight", facecolor="white")
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    import matplotlib.pyplot as plt

    plt.close(figure)


def _check_matched(base: dict[str, np.ndarray], post: dict[str, np.ndarray]) -> None:
    if not np.array_equal(base["sample_ids"], post["sample_ids"]):
        raise ValueError("source and A+B+C sample IDs do not match")


def _render_global() -> dict[str, object]:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    directory = COHERENCE / "global_distribution"
    base = _load(directory / "metrics-base.npz")
    post = _load(directory / "metrics.npz")
    _check_matched(base, post)
    if not np.array_equal(base["component_names"], post["component_names"]):
        raise ValueError("global-distribution component names differ")
    base_values = np.column_stack((base["weighted_component_totals"], base["family_total"]))
    post_values = np.column_stack((post["weighted_component_totals"], post["family_total"]))
    labels = ("Marginal W2", "Pairwise SWD", "Joint/top-tail SWD", "Family total")
    fig, ax = plt.subplots(figsize=(8.0, 3.4))
    y = np.arange(len(labels))
    for values, color, offset, label in (
        (base_values, SOURCE_COLOR, -0.13, "Source"),
        (post_values, POST_COLOR, 0.13, "A+B+C"),
    ):
        quartiles = np.quantile(values, (0.25, 0.5, 0.75), axis=0)
        for index in range(len(labels)):
            ax.plot(quartiles[[0, 2], index], [y[index] + offset] * 2, color=color, lw=4, alpha=0.55)
        ax.scatter(quartiles[1], y + offset, color=color, s=40, edgecolors="white", lw=0.6, label=label)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlabel("Configured weighted discrepancy (model units; log scale)")
    ax.set_title("Global-distribution terms across matched validation snapshots", loc="left")
    ax.grid(axis="x", color="#DDE3E7", lw=0.7)
    handles, legend_labels = ax.get_legend_handles_labels()
    fig.legend(handles[:2], legend_labels[:2], frameon=False, ncol=2, loc="lower center")
    fig.subplots_adjust(left=0.23, right=0.97, top=0.86, bottom=0.22)
    _save(fig, EXPLANATORY / "global_weighted_terms.png")

    extra = directory / "global_distribution_extra"
    base_joint = _load(extra / "joint_pdf_metrics-base.npz")
    post_joint = _load(extra / "joint_pdf_metrics.npz")
    if not np.array_equal(base_joint["pair_labels"], post_joint["pair_labels"]):
        raise ValueError("global joint-density pair labels differ")
    if not np.array_equal(base_joint["x_edges"], post_joint["x_edges"]) or not np.array_equal(
        base_joint["y_edges"], post_joint["y_edges"]
    ):
        raise ValueError("global joint-density bins differ")
    pair_index = list(post_joint["pair_labels"]).index("CO–T")
    edges_x = post_joint["x_edges"][pair_index]
    edges_y = post_joint["y_edges"][pair_index]
    area = np.diff(edges_x)[:, None] * np.diff(edges_y)[None, :]
    masses = (
        post_joint["reference_probability_mass"][pair_index],
        base_joint["reconstruction_probability_mass"][pair_index],
        post_joint["reconstruction_probability_mass"][pair_index],
    )
    densities = [mass / area for mass in masses]
    positive = np.concatenate([value[value > 0] for value in densities])
    norm = LogNorm(vmin=max(float(positive.max()) * 1e-4, float(positive.min())), vmax=float(positive.max()))
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.3), sharex=True, sharey=True)
    titles = ("Reference", "Source reconstruction", "A+B+C reconstruction")
    for axis, density, title in zip(axes, densities, titles):
        mesh = axis.pcolormesh(edges_x, edges_y, np.ma.masked_less_equal(density.T, 0), cmap="magma", norm=norm, shading="auto")
        axis.set_title(title, fontsize=9.5)
        axis.set_xlabel("CO (model units)")
        axis.set_facecolor("#F5F6F7")
    axes[0].set_ylabel("T (model units)")
    fig.colorbar(mesh, ax=axes, label="Probability density", fraction=0.025, pad=0.025)
    base_js = float(base_joint["jensen_shannon_divergence_bits"][pair_index])
    post_js = float(post_joint["jensen_shannon_divergence_bits"][pair_index])
    fig.suptitle(f"CO–T joint density · matched bins · JSD source {base_js:.3f} → A+B+C {post_js:.3f} bits", fontsize=11.0)
    fig.subplots_adjust(left=0.07, right=0.87, top=0.79, bottom=0.19, wspace=0.10)
    _save(fig, EXPLANATORY / "global_CO_T_joint_density.png")
    return {"matched_samples": len(base["sample_ids"]), "CO_T_jsd_source_bits": base_js, "CO_T_jsd_ABC_bits": post_js}


def _render_cross() -> dict[str, object]:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import PercentFormatter

    directory = COHERENCE / "cross_spectrum"
    base = _load(directory / "metrics-base.npz")
    post = _load(directory / "metrics.npz")
    _check_matched(base, post)
    for name in ("ensemble_sample_ids", "query_indices", "pair_labels", "graph_band_names"):
        if not np.array_equal(base[name], post[name]):
            raise ValueError(f"source and A+B+C {name} differ")
    labels = tuple(str(value).replace("–", "–") for value in post["pair_labels"])
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.1), sharey=True)
    for axis, (term, title) in zip(
        axes,
        (("same_frequency", "Same-frequency coupling"), ("cross_frequency", "Cross-band coupling")),
    ):
        base_rows = base[f"{term}_coherence_score_by_ensemble"]
        post_rows = post[f"{term}_coherence_score_by_ensemble"]
        for index, label in enumerate(labels):
            position = len(labels) - index - 1
            source_mean = float(base_rows[:, index].mean())
            post_mean = float(post_rows[:, index].mean())
            axis.plot([source_mean, post_mean], [position, position], color="#A8B2B9", lw=1.25, zorder=2)
            axis.scatter(base_rows[:, index], np.full(len(base_rows), position - 0.10), s=18, color=SOURCE_COLOR, alpha=0.38, zorder=3)
            axis.scatter(post_rows[:, index], np.full(len(post_rows), position + 0.10), s=18, color=POST_COLOR, alpha=0.38, zorder=3)
            axis.scatter([source_mean], [position - 0.10], s=48, marker="o", color=SOURCE_COLOR, edgecolors="white", lw=0.7, zorder=4)
            axis.scatter([post_mean], [position + 0.10], s=48, marker="s", color=POST_COLOR, edgecolors="white", lw=0.7, zorder=4)
        axis.set_yticks(np.arange(len(labels))[::-1], labels)
        axis.set_xlim(0.0, 1.0)
        axis.xaxis.set_major_formatter(PercentFormatter(1.0))
        axis.set_title(title, loc="left", fontsize=10)
        axis.set_xlabel("Bounded agreement score")
        axis.grid(axis="x", color="#DDE3E7", lw=0.7)
        axis.set_axisbelow(True)
    fig.legend(
        handles=(
            Line2D([], [], marker="o", linestyle="none", color=SOURCE_COLOR, label="Source mean"),
            Line2D([], [], marker="s", linestyle="none", color=POST_COLOR, label="A+B+C mean"),
            Line2D([], [], marker="o", linestyle="none", color="#A8B2B9", alpha=0.6, label="Individual 32-snapshot ensembles"),
        ),
        loc="lower center", ncol=3, frameon=False, fontsize=8.2,
    )
    fig.suptitle("Configured spectral terms by field pair", fontsize=11.5)
    fig.subplots_adjust(left=0.12, right=0.98, top=0.80, bottom=0.22, wspace=0.12)
    _save(fig, EXPLANATORY / "cross_pair_scores.png")

    if not np.allclose(
        base["graph_band_energy_fraction_reference_mean"],
        post["graph_band_energy_fraction_reference_mean"], atol=1e-6,
    ):
        raise ValueError("matched reference graph-band profiles differ")
    fields = tuple(str(value) for value in post["field_names"])
    bands = tuple(str(value) for value in post["graph_band_names"])
    profiles = (
        (post["graph_band_energy_fraction_reference_mean"], REFERENCE_COLOR, "Reference", "-", "o"),
        (base["graph_band_energy_fraction_reconstruction_mean"], SOURCE_COLOR, "Source", "--", "s"),
        (post["graph_band_energy_fraction_reconstruction_mean"], POST_COLOR, "A+B+C", "-", "D"),
    )
    fig, axes = plt.subplots(2, 3, figsize=(9.2, 5.3), sharex=True, sharey=True)
    for field_index, field in enumerate(fields):
        axis = axes.flat[field_index]
        for values, color, label, linestyle, marker in profiles:
            axis.plot(np.arange(len(bands)), values[:, field_index], color=color, label=label,
                      linestyle=linestyle, marker=marker, ms=4, lw=1.5)
        axis.set_title(field, loc="left")
        axis.set_xticks(np.arange(len(bands)), bands)
        axis.yaxis.set_major_formatter(PercentFormatter(1.0))
        axis.set_ylim(0, 1)
        axis.grid(axis="y", color="#DDE3E7", lw=0.65)
        if field_index % 3 == 0:
            axis.set_ylabel("Energy fraction")
    axes.flat[-1].axis("off")
    handles, legend_labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, ncol=3, loc="lower center", frameon=False)
    fig.suptitle("Graph-Fourier band energy · reference, source, A+B+C", fontsize=11.5)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.86, bottom=0.14, wspace=0.20, hspace=0.34)
    _save(fig, EXPLANATORY / "cross_band_three_way.png")
    reference = post["graph_band_energy_fraction_reference_mean"]
    deviations = (
        100 * (base["graph_band_energy_fraction_reconstruction_mean"] - reference),
        100 * (post["graph_band_energy_fraction_reconstruction_mean"] - reference),
    )
    limit = max(float(np.max(np.abs(value))) for value in deviations)
    fig, axes = plt.subplots(1, 2, figsize=(10.1, 3.1), sharex=True, sharey=True)
    for axis, values, title in zip(axes, deviations, ("Source − reference", "A+B+C − reference")):
        image = axis.imshow(values, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
        axis.set_xticks(np.arange(len(fields)), fields)
        axis.set_yticks(np.arange(len(bands)), bands)
        axis.set_title(title, loc="left", fontsize=10.2)
        for row in range(len(bands)):
            for column in range(len(fields)):
                axis.text(column, row, f"{values[row, column]:+.1f}", ha="center", va="center", fontsize=8.2)
    axes[0].set_ylabel("Graph-Fourier band")
    fig.colorbar(image, ax=axes, label="Energy fraction difference (percentage points)",
                 fraction=0.025, pad=0.025)
    fig.suptitle("Graph-Fourier band energy deviations · same reference and scale", fontsize=11.5)
    fig.subplots_adjust(left=0.10, right=0.86, top=0.80, bottom=0.14, wspace=0.12)
    _save(fig, EXPLANATORY / "cross_band_error.png")
    return {"ensemble_count": int(post["ensemble_count"]), "ensemble_size": int(post["ensemble_size"]), "pair_labels": labels}


def _physical_grid(normalized_grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    with np.load(ROOT / "source" / "fullgrid_validation_frame8000.npz", allow_pickle=False) as payload:
        normalized = np.asarray(payload["query_coords"])[0]
        physical = np.asarray(payload["query_coords_physical"])[0]
    lower = physical.min(axis=0)
    upper = physical.max(axis=0)
    if not np.allclose(normalized.min(axis=0), 0.0, atol=1e-6) or not np.allclose(
        normalized.max(axis=0), 1.0, atol=1e-6
    ):
        raise ValueError("full-grid normalized coordinates do not span [0,1]")
    grid = lower + normalized_grid * (upper - lower)
    return grid[..., 0], grid[..., 1]


def _topology_diagrams(ref: np.ndarray, post: np.ndarray):
    import torch

    banks = []
    descriptors = []
    for field_index, field in enumerate(("CO", "T")):
        mean = float(ref[field_index].mean())
        scale = max(float(ref[field_index].std()), 1e-4)
        normalized_ref = (ref[field_index] - mean) / scale
        normalized_post = (post[field_index] - mean) / scale
        descriptors.append((normalized_ref, normalized_post))
        for direction, sign in (("sublevel", 1), ("superlevel", -1)):
            banks.extend((sign * normalized_ref, sign * normalized_post))
    diagrams = cubical_diagrams(
        torch.as_tensor(np.stack(banks), dtype=torch.float32),
        periodic=False, dimensions=(0, 1),
    )
    return diagrams, descriptors


def _render_topology() -> dict[str, object]:
    import matplotlib.pyplot as plt
    import torch
    from matplotlib.colors import ListedColormap, Normalize

    directory = COHERENCE / "topology"
    topology_config = load_config(ROOT / "source" / "resolved_config.yaml")["coherence"]["families"]["topology"]
    geometry_config = topology_config["geometry"]
    persistence_config = topology_config["persistence"]
    mutual_config = topology_config["components"]["mutual"]
    if (tuple(geometry_config["grid_shape"]), tuple(topology_config["fields"]),
        tuple(topology_config["filtration"]["directions"])) != (
        (32, 128), ("CO", "T"), ("sublevel", "superlevel")
    ):
        raise ValueError("the pinned topology raster, fields, or directions changed")
    post = _load(directory / "metrics.npz")
    base = _load(directory / "metrics-base.npz")
    _check_matched(base, post)
    ref = post["representative_reference_fields"]
    pred = post["representative_reconstruction_fields"]
    rep = int(post["representative_index"])
    sample_id = str(post["sample_ids"][rep])
    fields = tuple(str(value) for value in post["field_names"])
    x, y = _physical_grid(post["grid_coordinates"])
    extent = (float(x.min()), float(x.max()), float(y.min()), float(y.max()))
    fig, axes = plt.subplots(2, 3, figsize=(11.0, 4.2), sharex=True, sharey=True)
    for row, field in enumerate(fields):
        low = float(min(ref[row].min(), pred[row].min()))
        high = float(max(ref[row].max(), pred[row].max()))
        level = float(np.quantile(ref[row], 0.5))
        error = np.abs(pred[row] - ref[row])
        for column, values in enumerate((ref[row], pred[row], error)):
            axis = axes[row, column]
            image = axis.imshow(values, extent=extent, origin="lower", interpolation="nearest",
                                cmap="viridis" if column < 2 else "magma",
                                norm=Normalize(vmin=low if column < 2 else 0, vmax=high if column < 2 else max(float(error.max()), 1e-8)),
                                aspect="equal")
            if column < 2:
                axis.contour(x, y, values, levels=(level,), colors="white", linewidths=0.8,
                             linestyles="solid" if column == 0 else "dashed")
            if row == 0:
                axis.set_title(("Reference", "A+B+C", "Absolute difference")[column])
            if column == 0:
                axis.set_ylabel(f"{field} · y (dataset units)")
            if row == 1:
                axis.set_xlabel("x (dataset units)")
            axis.tick_params(labelsize=7.6, labelleft=column == 0, labelbottom=row == 1)
            if column in (1, 2):
                fig.colorbar(image, ax=axis, fraction=0.034, pad=0.015)
    fig.suptitle("Configured 32×128 topology raster · matched model-unit fields", fontsize=11.2)
    fig.subplots_adjust(left=0.10, right=0.98, top=0.84, bottom=0.13, wspace=0.23, hspace=0.22)
    _save(fig, EXPLANATORY / "topology_field_maps.png")

    diagrams, descriptors = _topology_diagrams(ref, pred)
    distance_terms = []
    for field_index, field in enumerate(fields):
        fig, axes = plt.subplots(2, 2, figsize=(7.7, 6.3))
        for direction_index, direction in enumerate(("sublevel", "superlevel")):
            base_index = field_index * 4 + direction_index * 2
            reference_diagrams = diagrams[base_index]
            prediction_diagrams = diagrams[base_index + 1]
            for dimension in (0, 1):
                axis = axes[direction_index, dimension]
                left = reference_diagrams[dimension].finite.detach().numpy()
                right = prediction_diagrams[dimension].finite.detach().numpy()
                both = np.concatenate((left, right)) if len(left) or len(right) else np.array([[0, 1]])
                lower = float(min(both.min(), 0.0))
                upper = float(max(both.max(), 0.0))
                pad = max(0.05 * (upper - lower), 0.05)
                axis.plot((lower - pad, upper + pad), (lower - pad, upper + pad), color="#B2BBC2", lw=0.8)
                axis.scatter(left[:, 0], left[:, 1], s=10, color=REFERENCE_COLOR, alpha=0.38, label="Reference")
                axis.scatter(right[:, 0], right[:, 1], s=9, marker="x", color="#BD7813", alpha=0.47, label="A+B+C")
                axis.set_xlim(lower - pad, upper + pad)
                axis.set_ylim(lower - pad, upper + pad)
                axis.set_aspect("equal", adjustable="box")
                axis.set_title(f"{direction} · H{dimension}", loc="left", fontsize=9.2)
                axis.set_xlabel("Birth (reference-standardized)")
                axis.set_ylabel("Death (reference-standardized)")
                axis.text(0.97, 0.05, f"finite {len(left)} / {len(right)}\nessential {len(reference_diagrams[dimension].essential)} / {len(prediction_diagrams[dimension].essential)}",
                          transform=axis.transAxes, ha="right", va="bottom", fontsize=7.2,
                          bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85})
                distance_terms.append(float(sliced_diagram_distance(
                    prediction_diagrams[dimension], reference_diagrams[dimension],
                    projections=int(persistence_config["projections"]),
                    essential_weight=float(persistence_config["essential_weight"]),
                    normalization=int(np.prod(geometry_config["grid_shape"])),
                )))
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, ncol=2, loc="lower center", frameon=False)
        fig.suptitle(f"{field} persistence diagrams · all positive finite bars", fontsize=11.2)
        fig.subplots_adjust(left=0.12, right=0.98, top=0.89, bottom=0.13, wspace=0.28, hspace=0.31)
        _save(fig, EXPLANATORY / f"topology_diagrams_{field}.png")

    self_distance = float(np.mean(distance_terms))
    scored_self_distance = float(post["objective_component_distances"][rep, 0])
    if not np.isclose(self_distance, scored_self_distance, rtol=2e-3, atol=2e-6):
        raise ValueError(f"diagram self-distance parity failed: {self_distance} vs {scored_self_distance}")

    quantiles = post["quantiles"]
    q_indices = [int(np.where(np.isclose(quantiles, q))[0][0]) for q in (0.25, 0.5, 0.75)]
    for field_index, field in enumerate(fields):
        ref_std, pred_std = descriptors[field_index]
        row_index = next(
            index for index, (metric, direction) in enumerate(zip(post["filtration_metric"], post["filtration_direction"]))
            if metric == f"self.{field}" and direction == "sublevel"
        )
        fig, axes = plt.subplots(3, 2, figsize=(8.7, 5.3), sharex=True, sharey=True)
        for row, quantile_index in enumerate(q_indices):
            level = float(np.quantile(ref_std, float(quantiles[quantile_index])))
            for column, values in enumerate((ref_std, pred_std)):
                axis = axes[row, column]
                mask = values <= level
                axis.imshow(mask, extent=extent, origin="lower", interpolation="nearest", aspect="equal",
                            cmap=ListedColormap(("#F3F5F6", REFERENCE_COLOR if column == 0 else "#B97816")), vmin=0, vmax=1)
                counts = (post["reference_betti"] if column == 0 else post["reconstruction_betti"])[rep, row_index, :, quantile_index]
                axis.text(0.98, 0.92, f"H0 {counts[0]} · H1 {counts[1]}", transform=axis.transAxes,
                          ha="right", va="top", fontsize=8.1,
                          bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 1.5})
                if row == 0:
                    axis.set_title("Reference" if column == 0 else "A+B+C")
                if column == 0:
                    axis.set_ylabel(f"q{int(quantiles[quantile_index] * 100)} · y")
                if row == 2:
                    axis.set_xlabel("x (dataset units)")
                axis.tick_params(labelsize=7.4, labelbottom=row == 2, labelleft=column == 0)
        fig.suptitle(f"{field} sublevel filtration · reference-defined thresholds", fontsize=11.2)
        fig.subplots_adjust(left=0.12, right=0.98, top=0.85, bottom=0.12, wspace=0.08, hspace=0.33)
        _save(fig, EXPLANATORY / f"topology_filtration_{field}.png")

    # Recreate the three configured CO–T mutual lines from the recorded Sobol seed.
    q = torch.quasirandom.SobolEngine(4, scramble=True, seed=int(mutual_config["seed"])).draw(
        int(mutual_config["lines"])
    )
    direction = float(mutual_config["direction_floor"]) + q[:, :2]
    direction /= direction.norm(dim=1, keepdim=True)
    offset = float(mutual_config["offset_range"]) * (2 * q[:, 2:] - 1)
    offset -= offset.mean(dim=1, keepdim=True)
    ref_stack = np.stack([value[0] for value in descriptors])
    pred_stack = np.stack([value[1] for value in descriptors])
    fig, axes = plt.subplots(3, 2, figsize=(8.7, 5.3), sharex=True, sharey=True)
    mutual_rows = [
        index for index, (kind, direction_name) in enumerate(zip(post["filtration_kind"], post["filtration_direction"]))
        if kind == "mutual" and direction_name == "sublevel"
    ]
    if len(mutual_rows) != int(mutual_config["lines"]):
        raise ValueError("configured CO–T mutual line count differs from saved metrics")
    for line_index, row_index in enumerate(mutual_rows):
        vector = direction[line_index].numpy()[:, None, None]
        shift = offset[line_index].numpy()[:, None, None]
        ref_line = ((ref_stack - shift) / vector).max(axis=0)
        pred_line = ((pred_stack - shift) / vector).max(axis=0)
        level = float(np.quantile(ref_line, 0.5))
        for column, values in enumerate((ref_line, pred_line)):
            axis = axes[line_index, column]
            axis.imshow(values <= level, extent=extent, origin="lower", interpolation="nearest",
                        aspect="equal", cmap=ListedColormap(("#F3F5F6", REFERENCE_COLOR if column == 0 else "#B97816")), vmin=0, vmax=1)
            counts = (post["reference_betti"] if column == 0 else post["reconstruction_betti"])[
                rep, row_index, :, q_indices[1]
            ]
            axis.text(0.98, 0.92, f"H0 {counts[0]} · H1 {counts[1]}", transform=axis.transAxes,
                      ha="right", va="top", fontsize=8.1,
                      bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 1.5})
            if line_index == 0:
                axis.set_title("Reference" if column == 0 else "A+B+C")
            if column == 0:
                axis.set_ylabel(f"Line {line_index + 1} · y")
            if line_index == 2:
                axis.set_xlabel("x (dataset units)")
            axis.tick_params(labelsize=7.4, labelbottom=line_index == 2, labelleft=column == 0)
    fig.suptitle("CO–T mutual filtration · each configured line at reference q50", fontsize=11.2)
    fig.subplots_adjust(left=0.12, right=0.98, top=0.85, bottom=0.12, wspace=0.08, hspace=0.33)
    _save(fig, EXPLANATORY / "topology_mutual_CO_T.png")
    return {
        "representative_sample_id": sample_id,
        "representative_index": rep,
        "self_diagram_distance_recomputed": self_distance,
        "self_diagram_distance_scored": scored_self_distance,
        "raster": [32, 128],
        "physical_extent": list(extent),
    }


def render() -> None:
    import matplotlib

    matplotlib.use("Agg")
    EXPLANATORY.mkdir(exist_ok=True)
    report = {
        "source": "standard post-processing metrics from the frozen epoch-1440 checkpoint",
        "global_distribution": _render_global(),
        "cross_spectrum": _render_cross(),
        "topology": _render_topology(),
    }
    (EXPLANATORY / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    render()

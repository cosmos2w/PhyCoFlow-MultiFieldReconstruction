"""Additional figures rendered from saved, run-local coherence metrics.

These views explain the standard set evaluation; they never run inference or
change the scored coherence objectives. Source comparisons use the copied
``metrics-base.npz`` payload produced by the matched set evaluator.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..coherence.families.topology.persistence import (
    cubical_diagrams,
    sliced_diagram_distance,
)
from ..coherence.families.topology.persistence_objective import PersistenceTopologyObjective
from ..coherence.families.topology.spatial_persistence import spatial_diagram_distance
from .coherence_set import _save_publication_figure as save_spectral_figure
from .topology_set import _save_publication_figure as save_topology_figure

SOURCE_COLOR = "#687786"
POST_COLOR = "#16647F"
REFERENCE_COLOR = "#1C3445"


def _load_payload(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {key: np.asarray(payload[key]).copy() for key in payload.files}


def _check_cross_pair(base: dict[str, np.ndarray], post: dict[str, np.ndarray]) -> None:
    for key in (
        "sample_ids", "selected_sample_ids", "ensemble_sample_ids", "query_indices",
        "pair_labels", "graph_band_names", "field_names",
    ):
        if not np.array_equal(base[key], post[key]):
            raise ValueError(f"cross-spectrum explanatory comparison has different {key}")


def render_cross_pair_scores(
    base: dict[str, np.ndarray], post: dict[str, np.ndarray], output_path: str | Path
) -> dict[str, Any]:
    """Show each configured pair's ensemble spread and score change."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import PercentFormatter

    _check_cross_pair(base, post)
    labels = tuple(str(value) for value in post["pair_labels"])
    terms = (
        ("same_frequency", "Same-frequency coupling"),
        ("cross_frequency", "Cross-frequency coupling"),
    )
    active = []
    for term, title in terms:
        source = np.asarray(base[f"{term}_coherence_score_by_ensemble"], dtype=np.float64)
        trained = np.asarray(post[f"{term}_coherence_score_by_ensemble"], dtype=np.float64)
        if source.shape != trained.shape or source.ndim != 2 or source.shape[1] != len(labels):
            raise ValueError(f"{term} ensemble scores do not align with configured pairs")
        if source.size:
            if not np.isfinite(source).all() or not np.isfinite(trained).all():
                raise ValueError(f"{term} ensemble scores must be finite")
            if np.any((source < 0) | (source > 1)) or np.any((trained < 0) | (trained > 1)):
                raise ValueError(f"{term} ensemble scores must lie in [0,1]")
            active.append((term, title, source, trained))
    if not active:
        raise ValueError("cross-spectrum has no active pair-score terms")

    fig, axes = plt.subplots(1, len(active), figsize=(5.4 * len(active), max(4.3, 0.62 * len(labels) + 2.1)), squeeze=False)
    deltas: dict[str, dict[str, float]] = {}
    for axis, (term, title, source, trained) in zip(axes[0], active):
        deltas[term] = {}
        for index, label in enumerate(labels):
            position = len(labels) - index - 1
            source_mean = float(source[:, index].mean())
            trained_mean = float(trained[:, index].mean())
            delta = 100.0 * (trained_mean - source_mean)
            deltas[term][label] = delta
            axis.plot((source_mean, trained_mean), (position, position), color="#AFB9C1", lw=1.25, zorder=2)
            axis.scatter(source[:, index], np.full(len(source), position - 0.11), s=19, color=SOURCE_COLOR, alpha=0.42, zorder=3)
            axis.scatter(trained[:, index], np.full(len(trained), position + 0.11), s=19, color=POST_COLOR, alpha=0.42, zorder=3)
            axis.scatter(source_mean, position - 0.11, s=48, color=SOURCE_COLOR, edgecolor="white", linewidth=0.7, zorder=4)
            axis.scatter(trained_mean, position + 0.11, s=48, marker="s", color=POST_COLOR, edgecolor="white", linewidth=0.7, zorder=4)
            axis.text(
                1.025, position, f"{delta:+.1f} pp",
                ha="left", va="center", fontsize=7.6, color="#263846",
            )
        axis.set_yticks(np.arange(len(labels))[::-1], labels)
        axis.set_xlim(0, 1.22)
        axis.set_ylim(-0.5, len(labels) - 0.5)
        axis.set_xticks(np.linspace(0, 1, 6))
        axis.axvline(1.0, color="#BBC5CC", linewidth=0.8, linestyle="--")
        axis.xaxis.set_major_formatter(PercentFormatter(1.0))
        axis.set_title(title, loc="left", fontsize=10.2)
        axis.set_xlabel("Bounded agreement score")
        axis.grid(axis="x", color="#DDE3E7", linewidth=0.7)
        axis.set_axisbelow(True)
        axis.tick_params(labelsize=8.1)
    fig.suptitle("Configured spectral scores by field pair · labels show post − source", fontsize=11.2)
    fig.legend(
        handles=(
            Line2D([], [], marker="o", linestyle="none", color=SOURCE_COLOR, label="Source mean"),
            Line2D([], [], marker="s", linestyle="none", color=POST_COLOR, label="Post-training mean"),
            Line2D([], [], marker="o", linestyle="none", color="#9EACB6", alpha=0.5, label="Individual ensembles"),
        ),
        loc="lower center", ncol=3, frameon=False, fontsize=8.1,
    )
    fig.subplots_adjust(left=0.13, right=0.985, top=0.81, bottom=0.20, wspace=0.25)
    save_spectral_figure(fig, output_path)
    return {"pair_count": len(labels), "ensemble_count": int(post["ensemble_count"]), "score_change_percentage_points": deltas}


def render_cross_band_error(
    base: dict[str, np.ndarray], post: dict[str, np.ndarray], output_path: str | Path
) -> dict[str, Any]:
    """Compare graph-band energy errors against one matched reference profile."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _check_cross_pair(base, post)
    reference = np.asarray(post["graph_band_energy_fraction_reference_mean"], dtype=np.float64)
    base_reference = np.asarray(base["graph_band_energy_fraction_reference_mean"], dtype=np.float64)
    if not np.allclose(reference, base_reference, rtol=0, atol=1e-6):
        raise ValueError("cross-band source and post references do not match")
    source = np.asarray(base["graph_band_energy_fraction_reconstruction_mean"], dtype=np.float64)
    trained = np.asarray(post["graph_band_energy_fraction_reconstruction_mean"], dtype=np.float64)
    fields = tuple(str(value) for value in post["field_names"])
    bands = tuple(str(value) for value in post["graph_band_names"])
    if reference.shape != source.shape or reference.shape != trained.shape or reference.shape != (len(bands), len(fields)):
        raise ValueError("cross-band energy profiles do not align")
    if not all(np.isfinite(value).all() for value in (reference, source, trained)):
        raise ValueError("cross-band energy profiles must be finite")
    source_error = 100.0 * (source - reference)
    trained_error = 100.0 * (trained - reference)
    limit = max(0.1, float(np.max(np.abs(source_error))), float(np.max(np.abs(trained_error))))
    fig, axes = plt.subplots(1, 2, figsize=(10.2, max(3.3, 0.55 * len(bands) + 1.7)), sharex=True, sharey=True)
    for axis, values, title in zip(axes, (source_error, trained_error), ("Source − reference", "Post-training − reference")):
        image = axis.imshow(values, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
        axis.set_xticks(np.arange(len(fields)), fields)
        axis.set_yticks(np.arange(len(bands)), bands)
        axis.set_title(title, loc="left", fontsize=10.0)
        for row in range(len(bands)):
            for column in range(len(fields)):
                axis.text(
                    column, row, f"{values[row, column]:+.1f}", ha="center", va="center",
                    fontsize=8.1, color="white" if abs(values[row, column]) > 0.55 * limit else "#17212B",
                )
    axes[0].set_ylabel("Graph-Fourier band")
    fig.colorbar(image, ax=axes, label="Energy fraction error (percentage points)", fraction=0.025, pad=0.025)
    source_mae = float(np.abs(source_error).mean())
    trained_mae = float(np.abs(trained_error).mean())
    fig.suptitle(f"Graph-band energy diagnostic · mean absolute error {source_mae:.2f} → {trained_mae:.2f} pp", fontsize=11.1)
    fig.subplots_adjust(left=0.10, right=0.86, top=0.78, bottom=0.15, wspace=0.12)
    save_spectral_figure(fig, output_path)
    return {"mean_absolute_error_percentage_points": {"base": source_mae, "post_training": trained_mae}}


def render_topology_diagrams(
    payload: dict[str, np.ndarray], report: dict[str, Any], output_dir: str | Path,
    *, role: str, suffix: str = "",
) -> dict[str, Any]:
    """Plot exact diagrams for the payload's saved representative raster.

    The per-panel distance reuses the configured diagram metric and is checked
    against the saved self-persistence component before figures are published.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    settings = report["objective"]["settings"]
    fields = tuple(str(value) for value in payload["field_names"])
    objective = PersistenceTopologyObjective(settings, fields)
    if objective.weights.get("self.persistence", 0.0) <= 0:
        return {"skipped": "self persistence is disabled"}
    index = int(payload["representative_index"])
    sample_id = str(payload["sample_ids"][index])
    reference = torch.as_tensor(payload["representative_reference_fields"][None], dtype=torch.float32)
    prediction = torch.as_tensor(payload["representative_reconstruction_fields"][None], dtype=torch.float32)
    if reference.shape != prediction.shape or reference.shape[1] != len(fields):
        raise ValueError("representative topology fields do not align")
    target = objective._descriptor_fields(reference)
    mean = target.mean((-2, -1), keepdim=True)
    scale = target.std((-2, -1), keepdim=True, unbiased=False).clamp_min(objective.scale_floor)
    reference_bank, _, _, reference_metrics = objective._filtrations((target - mean) / scale)
    prediction_bank, _, _, prediction_metrics = objective._filtrations(
        (objective._descriptor_fields(prediction) - mean) / scale
    )
    if reference_metrics != prediction_metrics or tuple(reference_metrics) != tuple(str(value) for value in payload["filtration_metric"]):
        raise ValueError("saved topology filtration rows differ from configured objective")
    row_directions = tuple(str(value) for value in payload["filtration_direction"])
    row_indices = [i for i, metric in enumerate(reference_metrics) if metric.startswith("self.")]
    diagrams_ref = cubical_diagrams(
        reference_bank[row_indices, 0], periodic=objective.periodic, dimensions=objective.dimensions,
        locations=objective.distance == "spatial_wasserstein",
    )
    diagrams_pred = cubical_diagrams(
        prediction_bank[row_indices, 0], periodic=objective.periodic, dimensions=objective.dimensions,
        locations=objective.distance == "spatial_wasserstein",
    )
    normalization = int(reference.shape[-2] * reference.shape[-1])
    distances: dict[tuple[int, int], float] = {}
    for local, row in enumerate(row_indices):
        for dimension in objective.dimensions:
            kwargs = {"essential_weight": objective.essential_weight, "normalization": normalization}
            if objective.distance == "sliced_wasserstein":
                value = sliced_diagram_distance(
                    diagrams_pred[local][dimension], diagrams_ref[local][dimension],
                    projections=objective.projections, **kwargs,
                )
            else:
                value = spatial_diagram_distance(
                    diagrams_pred[local][dimension], diagrams_ref[local][dimension],
                    periodic=objective.periodic, spatial_weight=objective.spatial_weight,
                    spatial_mode=objective.spatial_mode,
                    max_assignment_size=objective.max_assignment_size, **kwargs,
                )
            distances[(row, dimension)] = float(value)
    score_names = tuple(str(value) for value in payload["objective_component_names"])
    saved = float(payload["objective_component_distances"][index, score_names.index("topology.self.persistence")])
    recomputed = float(np.mean(tuple(distances.values())))
    if not np.isclose(saved, recomputed, rtol=2e-3, atol=2e-6):
        raise ValueError(f"topology diagram distance parity failed: saved={saved}, recomputed={recomputed}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures: dict[str, str] = {}
    for field in objective.self_fields:
        rows = [row for row in row_indices if reference_metrics[row] == f"self.{field}"]
        if len(rows) != len(objective.directions):
            raise ValueError(f"self topology filtration rows are missing {field}")
        fig, axes = plt.subplots(len(rows), len(objective.dimensions), figsize=(4.0 * len(objective.dimensions), 3.05 * len(rows)), squeeze=False)
        for direction_index, row in enumerate(rows):
            local = row_indices.index(row)
            direction = row_directions[row]
            for dimension_index, dimension in enumerate(objective.dimensions):
                axis = axes[direction_index, dimension_index]
                ref_diagram = diagrams_ref[local][dimension]
                pred_diagram = diagrams_pred[local][dimension]
                ref_points = ref_diagram.finite.detach().cpu().numpy()
                pred_points = pred_diagram.finite.detach().cpu().numpy()
                both = np.concatenate((ref_points, pred_points)) if len(ref_points) or len(pred_points) else np.asarray([[0.0, 1.0]])
                lower = float(min(both.min(), 0.0))
                upper = float(max(both.max(), 0.0))
                pad = max(0.05 * (upper - lower), 0.05)
                axis.plot((lower - pad, upper + pad), (lower - pad, upper + pad), color="#B2BBC2", linewidth=0.8)
                axis.scatter(ref_points[:, 0], ref_points[:, 1], s=11, color=REFERENCE_COLOR, alpha=0.4, label="Reference")
                axis.scatter(pred_points[:, 0], pred_points[:, 1], s=11, marker="x", color="#BD7813", alpha=0.5, label=role)
                axis.set_xlim(lower - pad, upper + pad)
                axis.set_ylim(lower - pad, upper + pad)
                axis.set_aspect("equal", adjustable="box")
                axis.set_title(f"{direction} · H{dimension} · distance={distances[(row, dimension)]:.5f}", loc="left", fontsize=9.0)
                axis.set_xlabel("Birth (reference-standardized)")
                axis.set_ylabel("Death (reference-standardized)")
                axis.text(
                    0.97, 0.05,
                    f"finite {len(ref_points)} / {len(pred_points)}\nessential {len(ref_diagram.essential)} / {len(pred_diagram.essential)}",
                    transform=axis.transAxes, ha="right", va="bottom", fontsize=7.2,
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85},
                )
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, ncol=2, loc="lower center", frameon=False)
        fig.suptitle(
            f"{role} · {field} persistence · {sample_id}\nself-persistence distance across fields={saved:.5f} / raster vertex",
            fontsize=10.2,
        )
        fig.subplots_adjust(left=0.12, right=0.98, top=0.85, bottom=0.14, wspace=0.30, hspace=0.36)
        path = output_dir / f"{field}{suffix}.png"
        save_topology_figure(fig, path)
        figures[field] = path.name
    return {
        "sample_id": sample_id,
        "representative_index": index,
        "saved_self_persistence_distance": saved,
        "recomputed_self_persistence_distance": recomputed,
        "per_panel_distance": {
            f"{reference_metrics[row]}|{row_directions[row]}|h{dimension}": distances[(row, dimension)]
            for row in row_indices
            for dimension in objective.dimensions
        },
        "figures": figures,
    }


def render_saved_coherence_explanatory(
    evaluation_dir: str | Path, *, families: tuple[str, ...] | None = None
) -> Path:
    """Add the selected explanatory suite to an existing set evaluation."""
    evaluation_dir = Path(evaluation_dir).resolve()
    coherence_dir = evaluation_dir / "coherence"
    if not coherence_dir.is_dir():
        raise FileNotFoundError(f"set evaluation has no coherence metrics: {coherence_dir}")
    selected = set(families) if families is not None else {p.name for p in coherence_dir.iterdir() if p.is_dir()}
    summary: dict[str, Any] = {"source": "saved set-level coherence payloads", "artifacts": {}}
    comparison_path = evaluation_dir / "comparison_report.json"
    comparison = json.loads(comparison_path.read_text()) if comparison_path.is_file() else None

    cross_dir = coherence_dir / "cross_spectrum"
    if "cross_spectrum" in selected and (cross_dir / "metrics-base.npz").is_file():
        base = _load_payload(cross_dir / "metrics-base.npz")
        post = _load_payload(cross_dir / "metrics.npz")
        pair_path = cross_dir / "cross_pair_scores.png"
        band_path = cross_dir / "cross_band_error.png"
        paths = {}
        if any(np.asarray(post[f"{term}_coherence_score_by_ensemble"]).size for term in ("same_frequency", "cross_frequency")):
            summary["cross_pair_scores"] = render_cross_pair_scores(base, post, pair_path)
            paths["cross_pair_scores"] = pair_path.name
        if np.asarray(post["graph_band_energy_fraction_reference_mean"]).size:
            summary["cross_band_error"] = render_cross_band_error(base, post, band_path)
            paths["cross_band_error"] = band_path.name
        if paths:
            summary["artifacts"]["cross_spectrum"] = paths
            report_path = cross_dir / "report.json"
            report = json.loads(report_path.read_text())
            report.setdefault("artifacts", {})["explanatory_figures"] = paths
            report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
            if comparison is not None:
                comparison.setdefault("artifacts", {})["paired_explanatory_cross_spectrum"] = {
                    name: str((cross_dir / filename).relative_to(evaluation_dir)) for name, filename in paths.items()
                }

    topology_dir = coherence_dir / "topology"
    if "topology" in selected and (topology_dir / "metrics.npz").is_file():
        diagram_dir = topology_dir / "diagrams"
        topology_results = {}
        for role, payload_name, report_name, suffix in (
            (
                "Post-training" if (topology_dir / "metrics-base.npz").is_file() else "Evaluated run",
                "metrics.npz", "report.json", "",
            ),
            ("Base source", "metrics-base.npz", "report-base.json", "-base"),
        ):
            payload_path = topology_dir / payload_name
            report_path = topology_dir / report_name
            if not payload_path.is_file() or not report_path.is_file():
                continue
            payload = _load_payload(payload_path)
            report = json.loads(report_path.read_text())
            result = render_topology_diagrams(payload, report, diagram_dir, role=role, suffix=suffix)
            key = "post_training" if not suffix else "base"
            topology_results[key] = result
            if "figures" in result:
                report.setdefault("artifacts", {})["diagrams"] = {
                    field: str(Path("diagrams") / filename) for field, filename in result["figures"].items()
                }
                report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
                summary["artifacts"].setdefault("topology_diagrams", {})[key] = {
                    field: str(Path("diagrams") / filename) for field, filename in result["figures"].items()
                }
                if comparison is not None:
                    comparison.setdefault("artifacts", {}).setdefault(key, {}).setdefault("topology_diagrams", {}).update(
                        {field: str((diagram_dir / filename).relative_to(evaluation_dir)) for field, filename in result["figures"].items()}
                    )
        if topology_results:
            summary["topology_diagrams"] = topology_results
            if "base" in topology_results and "post_training" in topology_results:
                summary["topology_diagrams"]["representative_samples_match"] = (
                    topology_results["base"].get("sample_id") == topology_results["post_training"].get("sample_id")
                )

    report_path = coherence_dir / "explanatory_report.json"
    report_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    if comparison is not None:
        comparison.setdefault("artifacts", {})["explanatory_report"] = str(report_path.relative_to(evaluation_dir))
        comparison_path.write_text(json.dumps(comparison, indent=2, sort_keys=True) + "\n")
    return report_path

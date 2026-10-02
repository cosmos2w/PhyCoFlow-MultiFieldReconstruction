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


def _decode_covariance_block(value: Any) -> np.ndarray:
    if isinstance(value, dict) and "real" in value:
        real = np.asarray(value["real"], dtype=np.float64)
        imaginary = np.asarray(value.get("imag", np.zeros_like(real)), dtype=np.float64)
        if real.shape != imaginary.shape:
            raise ValueError("signed covariance block real and imaginary parts are misaligned")
        return real + 1j * imaginary
    return np.asarray(value, dtype=np.float64)


def render_v4_signed_covariance_blocks(
    post: dict[str, Any],
    output_path: str | Path,
    *,
    base: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Plot representative signed normalized covariance blocks from saved v4 diagnostics."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if post.get("definition") != "second_order_blocks_v4":
        raise ValueError("signed block explanation requires a second_order_blocks_v4 payload")
    if base is not None:
        if base.get("definition") != "second_order_blocks_v4":
            raise ValueError("base signed block explanation has a different definition")
        for key in ("basis_sha256", "calibration_sha256", "primary_aggregation"):
            if base.get(key) != post.get(key):
                raise ValueError(f"base and post-training signed blocks have different {key}")

    aggregation = str(post["primary_aggregation"])
    post_records = post["reports"][aggregation]["ensembles"]
    base_records = [] if base is None else base["reports"][aggregation]["ensembles"]
    if base is not None and [record["sample_ids"] for record in base_records] != [
        record["sample_ids"] for record in post_records
    ]:
        raise ValueError("base and post-training signed block ensemble memberships differ")
    field_names = tuple(str(name) for name in post["field_names"])
    band_names = tuple(str(name) for name in post["band_names"])

    selected: dict[str, dict[str, Any]] = {}
    view_rows: list[tuple[str, str, np.ndarray, np.ndarray, np.ndarray]] = []
    for component, block_key in (
        ("same_frequency", "same_frequency_blocks"),
        ("cross_frequency", "cross_frequency_blocks"),
    ):
        candidates: list[dict[str, Any]] = []
        for ensemble_index, post_record in enumerate(post_records):
            post_blocks = post_record["diagnostics"].get(block_key, [])
            source_blocks = (
                []
                if base is None
                else base_records[ensemble_index]["diagnostics"].get(block_key, [])
            )
            if base is not None and [
                (block["fields"], block["bands"]) for block in source_blocks
            ] != [(block["fields"], block["bands"]) for block in post_blocks]:
                raise ValueError(f"paired {component} signed block identities do not match")
            for block_index, post_block in enumerate(post_blocks):
                post_generated = _decode_covariance_block(
                    post_block["generated_normalized_block"]
                )
                post_reference = _decode_covariance_block(
                    post_block["reference_normalized_block"]
                )
                if base is None:
                    source_generated = post_reference
                    reference_difference = post_generated - post_reference
                    selection_norm = float(np.linalg.norm(reference_difference))
                else:
                    source_block = source_blocks[block_index]
                    source_generated = _decode_covariance_block(
                        source_block["generated_normalized_block"]
                    )
                    source_reference = _decode_covariance_block(
                        source_block["reference_normalized_block"]
                    )
                    if not np.allclose(source_reference, post_reference, rtol=0.0, atol=1.0e-6):
                        raise ValueError(
                            f"paired {component} blocks have different ground-truth covariances"
                        )
                    reference_difference = post_generated - source_generated
                    selection_norm = float(np.linalg.norm(reference_difference))
                candidates.append(
                    {
                        "ensemble_index": ensemble_index,
                        "block_index": block_index,
                        "post_generated": post_generated,
                        "source_generated": source_generated,
                        "reference": post_reference,
                        "selection_norm": selection_norm,
                        "post_record": post_record,
                        "post_block": post_block,
                        "source_block": None if base is None else source_blocks[block_index],
                    }
                )
        if not candidates:
            continue
        choice = max(candidates, key=lambda item: item["selection_norm"])
        block = choice["post_block"]
        left_field, right_field = (int(value) for value in block["fields"])
        left_band, right_band = (int(value) for value in block["bands"])
        source_matrix = choice["source_generated"]
        post_matrix = choice["post_generated"]
        difference = post_matrix - source_matrix if base is not None else post_matrix - choice["reference"]
        matrices = (source_matrix, post_matrix, difference)
        component_selected = {
            "ensemble_index": int(choice["ensemble_index"]),
            "sample_ids": list(choice["post_record"]["sample_ids"]),
            "fields": [field_names[left_field], field_names[right_field]],
            "bands": [band_names[left_band], band_names[right_band]],
            "selection_norm": choice["selection_norm"],
            "post_training_raw_block_loss": float(block["frobenius_squared"]),
            "base_raw_block_loss": (
                None
                if choice["source_block"] is None
                else float(choice["source_block"]["frobenius_squared"])
            ),
        }
        selected[component] = component_selected
        complex_values = any(np.iscomplexobj(matrix) for matrix in matrices)
        parts = ("real", "imaginary") if complex_values else ("real",)
        titles = (
            ("Reference", "Generated", "Generated − reference")
            if base is None
            else ("Base generated", "Post-training generated", "Post-training − base")
        )
        for part in parts:
            if part == "real":
                displayed = tuple(np.asarray(matrix).real for matrix in matrices)
            else:
                displayed = tuple(np.asarray(matrix).imag for matrix in matrices)
            view_rows.append((component, part, *displayed))

    if not view_rows:
        raise ValueError("v4 signed covariance payload contains no eligible blocks to explain")
    figure, axes = plt.subplots(
        len(view_rows),
        3,
        figsize=(9.8, max(3.2, 2.6 * len(view_rows))),
        squeeze=False,
    )
    image = None
    for row_index, (component, part, first, second, difference) in enumerate(view_rows):
        limit = max(
            float(np.max(np.abs(first))),
            float(np.max(np.abs(second))),
            float(np.max(np.abs(difference))),
            1.0e-12,
        )
        for column_index, (axis, matrix, title) in enumerate(
            zip(axes[row_index], (first, second, difference), titles)
        ):
            image = axis.imshow(matrix, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
            axis.set_title(
                f"{component.replace('_', ' ')}\n{title} · {part}",
                loc="left",
                fontsize=8.5,
            )
            axis.set_xlabel("right-band mode")
            axis.set_ylabel("left-band mode" if column_index == 0 else "")
            axis.tick_params(labelsize=7.3)
    if image is not None:
        figure.colorbar(image, ax=axes.ravel().tolist(), label="Signed normalized covariance", fraction=0.025, pad=0.02)
    title = (
        "Signed normalized covariance blocks · source to post-training"
        if base is not None
        else "Signed normalized covariance blocks · reference and reconstruction"
    )
    figure.suptitle(f"{title} · {aggregation.replace('_', ' ')}", fontsize=11.0)
    figure.subplots_adjust(
        left=0.12,
        right=0.90,
        bottom=0.05,
        top=0.91,
        hspace=0.55,
        wspace=0.30,
    )
    save_spectral_figure(figure, output_path)
    plt.close(figure)
    return {
        "definition": "second_order_blocks_v4",
        "aggregation": aggregation,
        "comparison": "base_to_post_training" if base is not None else "reference_to_reconstruction",
        "displayed_quantity": "signed real and imaginary parts of normalized covariance blocks",
        "selected_blocks": selected,
    }


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
    cross_report_path = cross_dir / "report.json"
    cross_report = (
        json.loads(cross_report_path.read_text(encoding="utf-8"))
        if cross_report_path.is_file()
        else {}
    )
    if (
        "cross_spectrum" in selected
        and cross_report.get("definition") == "second_order_blocks_v4"
    ):
        artifact_name = cross_report.get("artifacts", {}).get(
            "signed_covariance_blocks_json", "covariance_blocks_v4.json"
        )
        post_path = cross_dir / artifact_name
        if not post_path.is_file():
            raise FileNotFoundError(f"v4 signed covariance artifact is missing: {post_path}")
        post_blocks = json.loads(post_path.read_text(encoding="utf-8"))
        base_path = cross_dir / "covariance_blocks_v4-base.json"
        base_blocks = (
            json.loads(base_path.read_text(encoding="utf-8"))
            if base_path.is_file()
            else None
        )
        explanatory_path = cross_dir / "signed_covariance_blocks_explanatory_v4.png"
        explanatory = render_v4_signed_covariance_blocks(
            post_blocks,
            explanatory_path,
            base=base_blocks,
        )
        summary["cross_spectrum_signed_covariance_v4"] = explanatory
        summary["artifacts"]["cross_spectrum"] = {
            "signed_covariance_blocks_v4": explanatory_path.name,
            "raw_metrics_csv": cross_report.get("artifacts", {}).get(
                "covariance_metrics_csv"
            ),
            "full_signed_blocks_json": artifact_name,
        }
        cross_report.setdefault("artifacts", {}).setdefault("explanatory_figures", {})[
            "signed_covariance_blocks_v4"
        ] = explanatory_path.name
        cross_report_path.write_text(
            json.dumps(cross_report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        base_report_path = cross_dir / "report-base.json"
        if base_blocks is not None and base_report_path.is_file():
            base_report = json.loads(base_report_path.read_text(encoding="utf-8"))
            base_report.setdefault("artifacts", {}).setdefault("explanatory_figures", {})[
                "signed_covariance_blocks_v4"
            ] = explanatory_path.name
            base_report_path.write_text(
                json.dumps(base_report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        if comparison is not None:
            comparison.setdefault("artifacts", {})[
                "paired_explanatory_cross_spectrum"
            ] = {
                "signed_covariance_blocks_v4": str(
                    explanatory_path.relative_to(evaluation_dir)
                )
            }

    if (
        "cross_spectrum" in selected
        and cross_report.get("definition") != "second_order_blocks_v4"
        and (cross_dir / "metrics-base.npz").is_file()
    ):
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

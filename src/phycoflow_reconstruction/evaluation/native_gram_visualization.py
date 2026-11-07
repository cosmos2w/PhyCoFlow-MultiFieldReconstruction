"""Matched native Gram panels from saved members and a frozen TRAIN bundle."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from ..training.run_store import file_sha256
from .native_gram import score_native_gram_payloads
from .native_scale_visualization import _matched_manifests, _read_native_member
from .native_scoring import load_native_metric_bundle


def _matched_items(bundle, payload_paths):
    labels = list(payload_paths)
    if not labels or any(not isinstance(label, str) or not label or label == "Reference" for label in labels):
        raise ValueError("native Gram endpoints need nonempty names distinct from Reference")
    if len({label.casefold() for label in labels}) != len(labels):
        raise ValueError("native Gram endpoint names must be unique, ignoring case")
    items = [_read_native_member(path, bundle, 0) for path in payload_paths.values()]
    receipts = _matched_manifests(tuple(payload_paths.values()), items)
    for index, label in enumerate(labels):
        if label.casefold() == "source" and (
            receipts is None
            or receipts["records"][index]["checkpoint_sha256"] != bundle["source_checkpoint_sha256"]
        ):
            raise ValueError("native Gram SOURCE requires the matching frozen SOURCE manifest")
    matched = ("sample_id", "target_model", "logical_shape", "observation_coords", "observation_values",
               "observation_field_ids", "observation_indices", "observation_mask", "draw_seeds")
    for item in items[1:]:
        for key in matched:
            if not np.array_equal(item[key], items[0][key]):
                raise ValueError(f"native Gram matched endpoint contract differs: {key}")
    return items, receipts


def _save_figure(fig, path, dpi):
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight", dpi=dpi)


def _physical_panel(reports, labels, fields, bands, units, path, dpi):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = plt.get_cmap("tab10").colors
    count = len(labels)
    fig, axes = plt.subplots(2, len(fields), figsize=(max(3.0 * len(fields), 8.2), 7.2),
                             squeeze=False, layout="constrained")
    fig.get_layout_engine().set(rect=(0, .07, 1, .79))
    centers = np.arange(len(bands), dtype=float)
    width = .8 / (count + 1)
    reference = reports[0]["rows"][0]["physical"]
    for f, field in enumerate(fields):
        ax, skill = axes[:, f]
        ax.bar(centers - .4 + width / 2, np.asarray(reference["target_power"])[:, f],
               width, color="0.75", label="Reference")
        for r, (label, report) in enumerate(zip(labels, reports)):
            physical = report["rows"][0]["physical"]
            x = centers - .4 + width * (r + 1.5)
            pattern = np.asarray(physical["mean_pattern_power"])[:, f]
            variation = np.asarray(physical["member_variation_power"])[:, f]
            ax.bar(x, pattern, width, color=colors[r % 10], label=label)
            ax.bar(x, variation, width, bottom=pattern, color=colors[r % 10],
                   edgecolor="white", linewidth=.4, hatch="///")
            member = np.asarray(physical["represented_band_member_mse"])[:, f]
            ensemble = np.asarray(physical["represented_band_ensemble_mean_mse"])[:, f]
            shift = (r - (count - 1) / 2) * .8 / max(count, 1)
            part = .8 / max(count, 1) / 2
            skill.bar(centers + shift - part / 2, ensemble, part, color=colors[r % 10])
            skill.bar(centers + shift + part / 2, member, part, color=colors[r % 10],
                      alpha=.45, edgecolor=colors[r % 10])
        ax.set_title(field)
        ax.set_ylabel(f"Band power ({units[f]})²")
        skill.set_ylabel(f"Band MSE ({units[f]})²")
        # Matplotlib treats stacked-bar bottoms as sticky edges, which can
        # otherwise clip a small variation component at large absolute power.
        power_max = max(float(np.max(np.asarray(reference["target_power"])[:, f])), *(
            float(np.max(np.asarray(report["rows"][0]["physical"]["mean_member_power"])[:, f]))
            for report in reports
        ))
        ax.set_ylim(0, 1.12 * power_max if power_max > 0 else 1)
        for axis in (ax, skill):
            axis.set_xticks(centers, bands)
            axis.ticklabel_format(axis="y", style="sci", scilimits=(-2, 3), useMathText=True)
            axis.grid(axis="y", alpha=.18)
            axis.set_axisbelow(True)
            axis.margins(y=.12)
    fig.suptitle("Native represented power, ensemble variation and skill", fontsize=14, y=.995)
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, frameon=False, fontsize=8, loc="upper center",
               bbox_to_anchor=(.5, .93), ncol=len(legend_labels))
    fig.text(.5, .015, "Power: solid = mean pattern, hatch = variation  |  "
             "MSE: dark = ensemble mean, light = member mean", ha="center", fontsize=8)
    _save_figure(fig, path, dpi)
    plt.close(fig)


def _coupling_panel(reports, labels, fields, bands, units, pair, path, dpi):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = plt.get_cmap("tab10").colors
    left, right = (fields.index(name) for name in pair) if pair else (None, None)
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.4), layout="constrained")
    x = np.arange(len(bands), dtype=float)
    width = .8 / (len(labels) + 1)
    first = reports[0]["rows"][0]["physical"]
    for axis, key, title, ylabel in (
        (axes[0, 0], "target_gram", "Signed Gram (hatched = variation)", f"{units[left]} × {units[right]}" if pair else ""),
        (axes[0, 1], "rho_regularized_target", "TRAIN-floor regularized rho", "Dimensionless"),
    ):
        if pair is None:
            axis.axis("off")
            axis.text(.5, .5, "Mutual block unavailable\n(single-field family)", ha="center", va="center")
            continue
        reference = np.asarray(first[key])[:, left, right]
        axis.bar(x - .4 + width / 2, reference, width, color="0.75", label="Reference")
        for i, (label, report) in enumerate(zip(labels, reports)):
            physical = report["rows"][0]["physical"]
            position = x - .4 + width * (i + 1.5)
            if key == "target_gram":
                value = np.asarray(physical["mean_pattern_gram"])[:, left, right]
                variation = np.asarray(physical["member_variation_gram"])[:, left, right]
                axis.bar(position, value, width, color=colors[i % 10], label=label)
                axis.bar(position, variation, width, bottom=value, color=colors[i % 10],
                         edgecolor="white", linewidth=.4, hatch="///")
            else:
                value = np.asarray(physical["rho_regularized_mean_member_gram"])[:, left, right]
                axis.bar(position, value, width, color=colors[i % 10], label=label)
        axis.axhline(0, color="0.3", linewidth=.7)
        axis.set_xticks(x, bands)
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=.18)
        axis.set_axisbelow(True)
        if key == "target_gram":
            edges = [endpoint for bar in axis.patches for endpoint in (bar.get_y(), bar.get_y() + bar.get_height())]
            low, high = min(0., *edges), max(0., *edges)
            if high == low:
                axis.set_ylim(-1, 1)
            else:
                margin = .12 * (high - low)
                axis.set_ylim(low - margin if low < 0 else 0, high + margin if high > 0 else 0)
    if pair:
        axes[0, 0].legend(frameon=False, fontsize=8)
        axes[0, 1].set_ylim(-1.05, 1.05)
    positions = np.arange(len(labels), dtype=float)
    for i, report in enumerate(reports):
        means = report["means"]
        axes[1, 0].bar(positions[i] - .2, means["anchor_total"], .4, color="0.65",
                       label="Frozen v4 anchor" if i == 0 else None)
        axes[1, 0].bar(positions[i] + .2, means["candidate_total"], .4, color=colors[i % 10],
                       label="Gram candidate total" if i == 0 else None)
        axes[1, 1].bar(positions[i] - .2, means["gram_self_score"], .4, color=colors[i % 10],
                       label="Self-power energy score" if i == 0 else None)
        if "gram_mutual_score" in means:
            axes[1, 1].bar(positions[i] + .2, means["gram_mutual_score"], .4, color=colors[i % 10],
                           alpha=.45, label="Signed mutual energy score" if i == 0 else None)
    for axis in axes[1]:
        axis.set_xticks(positions, labels)
        axis.legend(frameon=False, fontsize=8)
        axis.grid(axis="y", alpha=.18)
        axis.set_axisbelow(True)
        axis.margins(y=.25)
    axes[1, 0].set_ylabel("Frozen Scale-score units")
    axes[1, 1].set_ylabel("Standardized descriptor-score units")
    axes[1, 0].set_title("Anchor and augmented candidate")
    axes[1, 1].set_title("Unweighted auxiliary blocks")
    title = f"Native signed coupling: {pair[0]} × {pair[1]}" if pair else "Native self-power score"
    fig.suptitle(title, fontsize=14)
    _save_figure(fig, path, dpi)
    plt.close(fig)


def render_native_gram(bundle_path, payload_paths: Mapping[str, str | Path], output_dir,
                       *, field_pair=None, dpi=160):
    """Score matched saved endpoints, then render physical and objective diagnostics.

    There is no inference or fitting. A one-field family displays its self
    score and explicitly marks mutual quantities unavailable.
    """
    if not isinstance(payload_paths, Mapping):
        raise TypeError("native Gram payload_paths must map endpoint names to paths")
    bundle, families = load_native_metric_bundle(bundle_path)
    family = families["multiscale_spectrum"]
    if not hasattr(family, "gram_extension"):
        raise ValueError("native Gram visualization requires a candidate Gram bundle")
    fields = list(family.field_names)
    bands = list(family.settings.bands.names)
    pair = list(field_pair or fields[:2]) if len(fields) > 1 or field_pair else None
    if pair is not None and (len(pair) != 2 or pair[0] == pair[1] or any(field not in fields for field in pair)):
        raise ValueError("native Gram field_pair must name two distinct fitted fields")
    if isinstance(dpi, bool) or not isinstance(dpi, int) or dpi < 40:
        raise ValueError("native Gram dpi must be an integer of at least 40")
    items, manifests = _matched_items(bundle, payload_paths)
    labels = list(payload_paths)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    reports = []
    for index, path in enumerate(payload_paths.values()):
        reports.append(score_native_gram_payloads(bundle_path, [path], output / f"scores_{index:02d}.json"))
    units = reports[0]["rows"][0]["physical"]["field_units"]
    if any(report["rows"][0]["physical"]["field_names"] != fields for report in reports):
        raise ValueError("native Gram score field order differs")
    _physical_panel(reports, labels, fields, bands, units, output / "native_gram_physical", dpi)
    _coupling_panel(reports, labels, fields, bands, units, pair, output / "native_gram_coupling_scores", dpi)
    inputs = {label: {"path": str(Path(path).resolve()), "sha256": file_sha256(path),
                      "scores": f"scores_{index:02d}.json",
                      "scores_sha256": file_sha256(output / f"scores_{index:02d}.json")}
              for index, (label, path) in enumerate(payload_paths.items())}
    receipt = {"version": "native_gram_visualization_v1", "inference_performed": False,
               "fitting_performed": False, "sample_id": str(items[0]["sample_id"].item()),
               "metric_bundle_sha256": bundle["sha256"], "metric_bundle_file_sha256": file_sha256(bundle_path),
               "anchor_bundle_sha256": bundle.get("anchor_bundle_sha256"), "inputs": inputs,
               "manifests": manifests, "field_pair": pair, "field_names": fields, "bands": bands,
               "fields_and_bands_use": "represented v4 spectral embedding on frozen analysis support",
               "rho": reports[0]["rows"][0]["physical"]["rho_regularization"],
               "scores": {label: report["means"] for label, report in zip(labels, reports)},
               "figures": ["native_gram_physical", "native_gram_coupling_scores"]}
    receipt_path = output / "native_gram_rendering.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    index = output / "INDEX.md"
    index.write_text("# Native Gram, power and skill\n\n"
        "All panels use matched saved members, a frozen TRAIN bundle and the v4 represented spectral support.\n\n"
        "![Native represented band power, variation and skill](native_gram_physical.png)\n\n"
        "Power is physical-unit band Gram diagonal; stack height is mean member power. MSE compares represented band fields with the same reference. [PDF master](native_gram_physical.pdf).\n\n"
        "![Native signed coupling, regularized rho and score contributions](native_gram_coupling_scores.png)\n\n"
        "Signed Gram stacks show mean-pattern and hatched member-variation contributions; rho uses the TRAIN floor. The v4 anchor and augmented candidate are distinct definitions. [PDF master](native_gram_coupling_scores.pdf).\n\n"
        "[Numerical provenance and score files](native_gram_rendering.json).\n")
    return {"index": str(index), "receipt": str(receipt_path), "inputs": inputs}

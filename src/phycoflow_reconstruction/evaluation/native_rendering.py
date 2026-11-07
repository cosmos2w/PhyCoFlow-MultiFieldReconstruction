"""Render a matched native SSS atlas entirely from immutable saved members."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .finite_primary_visualization import (
    render_finite_primary_payload,
    save_finite_primary_payload,
)
from .native_scale_visualization import render_native_scale
from .native_scoring import load_native_metric_bundle
from .native_visualization import render_native_atlas


def _shape_index(label, member, files):
    lines = [f"# {label}: native Shape, member {member} (zero-based)", "",
             "Finite/essential persistence and positional event overlap are separate diagnostics.", ""]
    for png in (p for p in files if p.suffix == ".png"):
        if png.stem == "shape_native_event":
            description = f"{label} native joint excursion agreement, missing and excess regions"
            caption = ("Frozen TRAIN thresholds define the displayed joint excursion; physical-area "
                       "missing/excess and IoU measure positional agreement separately from persistence.")
        else:
            filtration = png.stem.removeprefix("shape_").replace(".", " ").replace("_", " ")
            description = f"{label} native H0/H1 persistence and Betti curves for {filtration}"
            caption = (f"The {filtration} filtration uses the frozen native vertex complex; finite "
                       "and essential raw costs and scales are displayed separately.")
        lines.extend([f"![{description}]({png.name})", "",
                      f"{caption} [PDF master]({png.with_suffix('.pdf').name}).", ""])
    return "\n".join(lines)


def _validate_event_config(event_config, bundle):
    if event_config is None:
        return
    for key in ("source_checkpoint_sha256", "metric_bundle_sha256"):
        expected = bundle["source_checkpoint_sha256" if key.startswith("source") else "sha256"]
        if event_config.get(key) != expected:
            raise ValueError(f"event thresholds disagree with the frozen {key}")
    if event_config.get("fit_split") != "train":
        raise ValueError("native event thresholds require declared TRAIN fitting")
    if event_config.get("field_space") != "decoded_dataset":
        raise ValueError("native event thresholds require field_space: decoded_dataset")
    if (event_config.get("normalizer_digest", bundle["normalizer_digest"])
            != bundle["normalizer_digest"]):
        raise ValueError("event thresholds disagree with the frozen normalizer")
    if (tuple(event_config.get("field_order", bundle["data_spec"]["field_names"]))
            != tuple(bundle["data_spec"]["field_names"])):
        raise ValueError("event thresholds disagree with the frozen field order")


def render_native_coherence(bundle_path, payload_paths, output_dir, *, member=0,
                            event_config=None, include_ensemble_mean=True,
                            gram_bundle_path=None, gram_field_pair=None):
    """Validate matched payloads, then render fidelity, State, Scale and Shape.

    ``event_config`` is optional case-provided physical signed thresholds fitted
    on TRAIN, with ``field_space: decoded_dataset``. It must identify the same
    frozen SOURCE and metric bundle. This function never chooses thresholds or
    runs model inference.
    """
    output = Path(output_dir)
    bundle, families = load_native_metric_bundle(bundle_path)
    _validate_event_config(event_config, bundle)
    if gram_field_pair is not None and gram_bundle_path is None:
        raise ValueError("native Gram field pair requires a Gram auxiliary bundle")
    if gram_bundle_path is not None:
        gram_bundle, _ = load_native_metric_bundle(gram_bundle_path)
        if gram_bundle.get("anchor_bundle_sha256") != bundle["sha256"]:
            raise ValueError("native Gram bundle must identify this frozen v4 yardstick")
    output.mkdir(parents=True, exist_ok=True)
    atlas = render_native_atlas(bundle_path, payload_paths, output / "fidelity_state",
                               member=member, include_ensemble_mean=include_ensemble_mean)
    scale = render_native_scale(bundle_path, payload_paths, output / "scale", member=member)
    gram = None
    if gram_bundle_path is not None:
        from .native_gram_visualization import render_native_gram

        gram = render_native_gram(gram_bundle_path, payload_paths, output / "gram",
                                  field_pair=gram_field_pair)
    shape = {}
    for ordinal, (label, path) in enumerate(payload_paths.items()):
        directory = output / "shape" / f"endpoint_{ordinal:02d}"
        directory.mkdir(parents=True, exist_ok=True)
        with np.load(path, allow_pickle=False) as payload:
            event = None
            if event_config is not None:
                logical_shape = tuple(int(n) for n in payload["logical_shape"])
                if len(logical_shape) != 2:
                    raise ValueError("native event rendering requires a declared 2D raster")
                event = {
                    "reference": payload["target_physical"].reshape(*logical_shape, -1).transpose(2, 0, 1),
                    "prediction": payload["draws_physical"][member].reshape(*logical_shape, -1).transpose(2, 0, 1),
                    "coordinates": payload["raw_coordinates"].reshape(*logical_shape, -1),
                    "valid_mask": payload["physical_mask"].reshape(logical_shape),
                    "weights": payload["physical_weights"].reshape(logical_shape),
                    "field_names": payload["field_names"].tolist(),
                    "pair": event_config["pair"], "signs": event_config["signs"],
                    "thresholds": event_config["thresholds"],
                    "units": event_config.get("units", "dataset physical units unspecified"),
                    "measure": "native physical-area nodal weights",
                }
            saved = save_finite_primary_payload(
                families["topology"], torch.from_numpy(payload["draws_model"][member])[None],
                torch.from_numpy(payload["target_model"])[None],
                coordinates=torch.from_numpy(payload["model_coordinates"])[None],
                output_path=directory / "native_shape_payload.npz", event=event,
                sample_id=str(payload["sample_id"].item()), endpoint=str(label),
            )
        files = render_finite_primary_payload(saved, directory)
        index = directory / "INDEX.md"
        index.write_text(_shape_index(label, member, files))
        shape[str(label)] = {"payload": str(saved), "index": str(index)}
    report = {"metric_bundle_sha256": bundle["sha256"], "member_index_zero_based": member,
              "fidelity_state": atlas, "scale": scale, "shape": shape,
              "event_config": event_config, "inference_performed": False}
    if gram is not None:
        report["gram"] = gram
    (output / "native_rendering.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    lines = ["# Matched full-native SSS atlas", "",
             "All views use saved matched members and one frozen TRAIN metric bundle.", "",
             f"[Fidelity and State]({Path(atlas['index']).resolve().relative_to(output.resolve()).as_posix()})", "",
             f"[Scale]({Path(scale['index']).resolve().relative_to(output.resolve()).as_posix()})", ""]
    if gram is not None:
        lines.extend(["[Native Gram and skill](gram/INDEX.md)", ""])
    for ordinal, label in enumerate(payload_paths):
        lines.extend([f"[{label}: Shape](shape/endpoint_{ordinal:02d}/INDEX.md)", ""])
    (output / "INDEX.md").write_text("\n".join(lines))
    return report

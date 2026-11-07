"""Native quadrature relation family, backed by an existing native TRAIN fit."""
from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

import torch

from ...data.manifest import dataset_fingerprint
from ...evaluation.canonical_coherence import content_sha256
from .multiscale_relation import (
    RELATION_DEFINITION,
    MultiscaleRelationFamily,
    parse_relation_extension_config,
)
from .native_quadrature import ADAPTER_ID, NativeQuadrature


def _read_pinned(path, expected):
    if not isinstance(path, str) or not path or not isinstance(expected, str) or len(expected) != 64:
        raise ValueError("native relation artifact path and SHA-256 are required")
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("native relation artifact file SHA-256 mismatch")
    return raw


def parse_native_relation_config(config, field_names):
    p_config = copy.deepcopy(dict(config))
    extension = p_config.get("relation_extension")
    if not isinstance(extension, Mapping) or extension.get("definition") != ADAPTER_ID:
        raise ValueError("native relation definition mismatch")
    extension = dict(extension)
    keys = ("adapter_path", "adapter_sha256", "native_reference_path", "native_reference_sha256")
    native_files = {key: extension.pop(key, None) for key in keys}
    for name in ("adapter", "native_reference"):
        path, digest = native_files[name + "_path"], native_files[name + "_sha256"]
        if (not isinstance(path, str) or not path or not isinstance(digest, str)
                or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest)):
            raise ValueError("native relation requires artifact paths and lowercase SHA-256 digests")
    extension["definition"] = RELATION_DEFINITION
    r_config = copy.deepcopy(p_config)
    r_config["relation_extension"] = extension
    parse_relation_extension_config(r_config, field_names)
    return r_config, native_files


class NativeRelationFamily(MultiscaleRelationFamily):
    """R scoring with native-basis quadrature and a frozen native R calibration.

    Config pins an adapter JSON file and a native fitted R artifact (.pt) by
    SHA-256. Fit only loads those artifacts; it never constructs a query basis.
    """
    score_definition_version = ADAPTER_ID
    requires_full_grid_calibration = False

    def __init__(self, config, data_spec, normalizer):
        p_config = copy.deepcopy(dict(config))
        r_config, self.native_files = parse_native_relation_config(p_config, data_spec.field_names)
        super().__init__(r_config, data_spec, normalizer)
        if self.manifest.factors:
            raise ValueError("native relation does not support third-order factor descriptors")
        self.spec = replace(self.spec, metadata={**self.spec.metadata, "score_definition_version": ADAPTER_ID})
        self.native_config = p_config
        self.native_adapter = None

    def _bind(self, adapter, r_artifact):
        if adapter.calibration_identity != self.native_files["native_reference_sha256"]:
            raise ValueError("native adapter calibration identity does not match native reference")
        state = r_artifact.get("state_dict", {})
        for key in ("normalization_offset", "normalization_scale"):
            value = state.get(key)
            if not isinstance(value, torch.Tensor) or not torch.equal(
                value.detach().cpu(), getattr(self, key).detach().cpu()
            ):
                raise ValueError("native reference normalization differs from the current dataset")
        super().load_state_artifact(r_artifact)
        def matches(tensor, value):
            return torch.equal(tensor.detach().cpu(), torch.as_tensor(value, dtype=tensor.dtype))
        if (not matches(self.mass, adapter.mass) or not matches(self.basis, adapter.basis)
                or not matches(self.filters, adapter.filters)
                or not matches(self.piece_index, adapter.pieces)
                or not matches(self.analysis_index, adapter.node_ids)):
            raise ValueError("adapter geometry or basis differs from native reference")
        self.native_adapter = adapter

    def fit_reference_tensors(self, *args, **kwargs):
        raise RuntimeError("native relation requires its pinned native TRAIN reference artifact")

    def fit_training_reference(self, dataset, *, query_indices, device=None):
        if (getattr(dataset, "split_name", None) != "train"
                or getattr(getattr(dataset, "selection", None), "split", None) != "train"):
            raise ValueError("native relation reference is TRAIN-only")
        if self.native_adapter is not None:
            raise RuntimeError("native relation reference is already bound")
        adapter_bytes = _read_pinned(self.native_files["adapter_path"], self.native_files["adapter_sha256"])
        reference_bytes = _read_pinned(self.native_files["native_reference_path"], self.native_files["native_reference_sha256"])
        import io
        adapter_artifact = json.loads(adapter_bytes)
        adapter = NativeQuadrature.from_artifact(
            adapter_artifact, metric_identity=adapter_artifact["metric_identity"],
            calibration_identity=self.native_files["native_reference_sha256"])
        r_artifact = torch.load(io.BytesIO(reference_bytes), map_location="cpu", weights_only=True)
        record = r_artifact.get("calibration", {})
        path = getattr(dataset, "path", None)
        if (path is None or record.get("split") != "train"
                or record.get("dataset_fingerprint") != dataset_fingerprint(path)):
            raise ValueError("native reference TRAIN dataset identity mismatch")
        if device is not None:
            self.to(device)
        query = torch.as_tensor(query_indices, dtype=torch.long).cpu()
        first = dataset[0]
        adapter.validate_coordinates(query.numpy(), first.coordinates[query])
        if first.valid_points is not None and not bool(first.valid_points[query].all()):
            raise ValueError("native relation query contains invalid points")
        self._bind(adapter, r_artifact)

    def _require_ready(self, fields, coordinates):
        if self.native_adapter is None or not self.fitted:
            raise RuntimeError("native relation needs its frozen native reference and adapter")
        if (coordinates is None or coordinates.ndim != 3
                or coordinates.shape[:2] != fields.shape[:2]):
            raise ValueError("native relation coordinates must align with [B,N]")
        if not torch.equal(coordinates, coordinates[:1].expand_as(coordinates)):
            raise ValueError("native relation coordinates must match for every sample")
        self.native_adapter.validate_coordinates(self.native_adapter.query_ids, coordinates[0])

    def _statistics(self, fields):
        if self.native_adapter is None:
            raise RuntimeError("native relation adapter is not bound")
        fields = fields.to(self.compute_dtype)
        if self.units == "physical_units":
            fields = fields * self.normalization_scale.to(fields) + self.normalization_offset.to(fields)
        return self.native_adapter.statistics(fields[..., self.field_ids])

    def forward(self, *args, **kwargs):
        result = super().forward(*args, **kwargs)
        result.diagnostics["relation_extension"]["definition"] = ADAPTER_ID
        result.diagnostics["native_quadrature"] = {"adapter_sha256": self.native_files["adapter_sha256"],
                                                   "native_reference_sha256": self.native_files["native_reference_sha256"]}
        return result

    def state_artifact(self):
        if self.native_adapter is None:
            raise RuntimeError("native relation adapter is not bound")
        return {"definition": ADAPTER_ID, "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "adapter_implementation_sha256": hashlib.sha256(Path(__file__).with_name("native_quadrature.py").read_bytes()).hexdigest(),
                "config": copy.deepcopy(self.native_config),
                "adapter": self.native_adapter.state_artifact(), "native_reference": super().state_artifact()}

    def load_state_artifact(self, artifact):
        if (not isinstance(artifact, Mapping) or set(artifact) - {"frozen_training_context"} != {"definition", "implementation_sha256", "adapter_implementation_sha256", "config", "adapter", "native_reference"}
                or artifact["definition"] != ADAPTER_ID or artifact["config"] != self.native_config
                or artifact["implementation_sha256"] != hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
                or artifact["adapter_implementation_sha256"] != hashlib.sha256(Path(__file__).with_name("native_quadrature.py").read_bytes()).hexdigest()):
            raise ValueError("native relation artifact identity or config mismatch")
        # Require the pinned files as well as the embedded state, so resume cannot
        # silently substitute a different calibration or query panel.
        adapter_bytes = _read_pinned(self.native_files["adapter_path"], self.native_files["adapter_sha256"])
        reference_bytes = _read_pinned(self.native_files["native_reference_path"], self.native_files["native_reference_sha256"])
        import io
        original = torch.load(io.BytesIO(reference_bytes), map_location="cpu", weights_only=True)
        if content_sha256(original) != content_sha256(artifact["native_reference"]):
            raise ValueError("native relation embedded reference differs from pinned file")
        serialized = json.loads(adapter_bytes)
        if serialized != artifact["adapter"]:
            raise ValueError("native relation embedded adapter differs from pinned file")
        adapter = NativeQuadrature.from_artifact(serialized, metric_identity=serialized["metric_identity"],
                                                  calibration_identity=self.native_files["native_reference_sha256"])
        self._bind(adapter, artifact["native_reference"])

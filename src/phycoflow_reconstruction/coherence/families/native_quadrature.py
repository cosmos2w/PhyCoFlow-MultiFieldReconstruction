"""Native-basis, piece-preserving mass-lumped sparse quadrature.

This adapter changes only statistic extraction. It does not fit a basis or a
calibration, and it does not alter the historical spectrum family. Coordinates
must already use the declared physical distance metric, including any scaling.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch

from .multiscale_spectrum.statistics import band_gram

ADAPTER_ID = "self_rho_native_quadrature_v1"


def _digest(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


class NativeQuadrature:
    """Frozen native geometry and selected native IDs in model-output order.

    ``node_ids`` identifies native analysis sites in the caller's original
    domain. ``query_ids`` must be unique IDs from that list. Distance ties are
    resolved by the smallest native ID, independent of query ordering.
    """

    def __init__(self, *, coordinates: Any, model_coordinates: Any, node_ids: Any, mass: Any, pieces: Any,
                 basis: Any, filters: Any, query_ids: Any, metric_identity: str,
                 calibration_identity: str):
        x = np.asarray(coordinates, dtype=np.float64)
        model_x = np.asarray(model_coordinates)
        m = np.asarray(mass, dtype=np.float64)
        phi = np.asarray(basis, dtype=np.float64)
        filt = np.asarray(filters, dtype=np.float64)
        raw_ids, raw_q, raw_p = np.asarray(node_ids), np.asarray(query_ids), np.asarray(pieces)
        if any(a.ndim != 1 or not np.issubdtype(a.dtype, np.integer)
               for a in (raw_ids, raw_q, raw_p)):
            raise ValueError("node IDs, query IDs, and piece labels must be integer vectors")
        ids, q, p = (a.astype(np.int64) for a in (raw_ids, raw_q, raw_p))
        n = len(ids)
        if (not n or x.ndim != 2 or x.shape[0] != n or x.shape[1] == 0
                or model_x.ndim != 2 or model_x.shape[0] != n or model_x.shape[1] == 0
                or m.shape != (n,) or p.shape != (n,) or phi.ndim != 2
                or phi.shape[0] != n or phi.shape[1] == 0 or filt.ndim != 2
                or filt.shape[1] != phi.shape[1] or filt.shape[0] == 0):
            raise ValueError("incompatible native geometry, basis, or filters")
        if (not all(np.isfinite(a).all() for a in (x, model_x, m, phi, filt))
                or np.any(m <= 0) or not np.isclose(m.sum(), 1.0, rtol=1e-8, atol=1e-10)):
            raise ValueError("geometry must be finite with positive, normalized native mass")
        if len(np.unique(ids)) != n or not len(q) or len(np.unique(q)) != len(q):
            raise ValueError("native IDs and nonempty query IDs must be unique")
        if not metric_identity or not calibration_identity:
            raise ValueError("metric and calibration identities are required")
        labels = np.unique(p)
        if not np.array_equal(labels, np.arange(len(labels))):
            raise ValueError("piece labels must be contiguous from zero")
        index = {int(v): i for i, v in enumerate(ids)}
        try:
            qi = np.array([index[int(v)] for v in q], dtype=np.int64)
        except KeyError as exc:
            raise ValueError("query ID is not a native analysis site") from exc
        if not np.array_equal(np.unique(p[qi]), labels):
            raise ValueError("every native piece must have a selected site")
        # Avoid a large N x Q allocation. Sorted candidates make ties deterministic.
        weights = np.zeros(len(q), dtype=np.float64)
        for label in labels:
            native = np.flatnonzero(p == label)
            selected = np.flatnonzero(p[qi] == label)
            selected = selected[np.argsort(q[selected], kind="stable")]
            for start in range(0, len(native), 256):
                chunk = native[start:start + 256]
                distance = ((x[chunk, None, :] - x[qi[selected]][None, :, :]) ** 2).sum(-1)
                closest = selected[np.argmin(distance, axis=1)]
                np.add.at(weights, closest, m[chunk])
        self.coordinates, self.model_coordinates = x, model_x.copy()
        self.node_ids, self.mass, self.pieces = ids, m, p
        self.basis, self.filters, self.query_ids, self.query_index = phi, filt, q, qi
        self.query_weights = weights
        self.metric_identity, self.calibration_identity = metric_identity, calibration_identity
        self.piece_mass = np.bincount(p, weights=m, minlength=len(labels))

    def validate_coordinates(self, query_ids: Any, coordinates: Any) -> None:
        """Require the exact stored model coordinates and selected ID order."""
        ids = np.asarray(query_ids)
        coords = coordinates.detach().cpu().numpy() if isinstance(coordinates, torch.Tensor) else np.asarray(coordinates)
        if not np.array_equal(ids, self.query_ids) or not np.array_equal(
            coords, self.model_coordinates[self.query_index]
        ):
            raise ValueError("query IDs or model coordinates do not match native selection")

    def statistics(self, fields: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return score-compatible statistics for fields [batch, query, field]."""
        if fields.ndim != 3 or fields.shape[1] != len(self.query_ids) or not fields.is_floating_point():
            raise ValueError("fields must be floating point [batch, selected sites, fields]")
        def tensor(value: np.ndarray) -> torch.Tensor:
            return torch.as_tensor(value, dtype=fields.dtype, device=fields.device)
        w = tensor(self.query_weights)
        piece = torch.as_tensor(self.pieces[self.query_index], device=fields.device)
        mask = torch.nn.functional.one_hot(piece, len(self.piece_mass)).to(fields.dtype).T
        piece_weights = mask * w[None, :]
        means = torch.einsum("pq,bqc->bpc", piece_weights, fields) / tensor(self.piece_mass)[None, :, None]
        fluctuations = fields - means[:, piece]
        coefficients = torch.einsum("q,qk,bqc->bkc", w, tensor(self.basis[self.query_index]), fluctuations)
        filters = tensor(self.filters)
        return {"coefficients": coefficients, "piece_means": means,
                "gram": band_gram(coefficients, filters.square()),
                "second_moment": torch.einsum("q,bqc->bc", w, fields.square()) / w.sum(),
                "fluctuation_power": torch.einsum("q,bqc->bc", w, fluctuations.square()) / w.sum()}

    def diagnostics(self) -> dict[str, Any]:
        """Restricted native-basis Gram and its numerical rank/conditioning."""
        restricted = self.basis[self.query_index]
        gram = restricted.T @ (self.query_weights[:, None] * restricted)
        piece_means = np.stack([
            (self.query_weights[self.pieces[self.query_index] == k, None]
             * restricted[self.pieces[self.query_index] == k]).sum(axis=0) / self.piece_mass[k]
            for k in range(len(self.piece_mass))
        ])
        centered = restricted - piece_means[self.pieces[self.query_index]]
        centered_gram = restricted.T @ (self.query_weights[:, None] * centered)
        eig = np.linalg.eigvalsh(gram)
        tol = np.finfo(gram.dtype).eps * max(gram.shape) * max(float(np.max(np.abs(eig))), 1.0)
        rank = int(np.count_nonzero(eig > tol))
        centered_eig = np.linalg.eigvalsh(centered_gram)
        centered_tol = np.finfo(gram.dtype).eps * max(gram.shape) * max(float(np.max(np.abs(centered_eig))), 1.0)
        centered_rank = int(np.count_nonzero(centered_eig > centered_tol))
        piece_weights = np.bincount(self.pieces[self.query_index], weights=self.query_weights,
                                    minlength=len(self.piece_mass))
        return {"restricted_basis_gram": gram, "centered_restricted_basis_gram": centered_gram, "rank": rank, "modes": gram.shape[0],
                "condition": float(eig[-1] / eig[0]) if rank == len(eig) else float("inf"),
                "centered_rank": centered_rank,
                "centered_condition": (float(centered_eig[-1] / centered_eig[0])
                                       if centered_rank == len(centered_eig) else float("inf")),
                "piece_mass": self.piece_mass.copy(), "sampled_piece_mass": piece_weights,
                "piece_mass_error": piece_weights - self.piece_mass}

    def state_artifact(self) -> dict[str, Any]:
        """Serialize all geometry, query, and calibration identity with a strict digest."""
        payload = {"adapter_id": ADAPTER_ID, "coordinates": self.coordinates.tolist(),
                   "model_coordinates": self.model_coordinates.tolist(),
                   "node_ids": self.node_ids.tolist(), "mass": self.mass.tolist(),
                   "pieces": self.pieces.tolist(), "basis": self.basis.tolist(),
                   "filters": self.filters.tolist(), "query_ids": self.query_ids.tolist(),
                   "query_weights": self.query_weights.tolist(), "metric_identity": self.metric_identity,
                   "calibration_identity": self.calibration_identity}
        return {**payload, "sha256": _digest(payload)}

    @classmethod
    def from_artifact(cls, artifact: Mapping[str, Any], *, metric_identity: str,
                      calibration_identity: str) -> NativeQuadrature:
        expected = {"adapter_id", "coordinates", "model_coordinates", "node_ids", "mass", "pieces", "basis",
                    "filters", "query_ids", "query_weights", "metric_identity",
                    "calibration_identity", "sha256"}
        if set(artifact) != expected or artifact.get("adapter_id") != ADAPTER_ID:
            raise ValueError("native quadrature artifact identity mismatch")
        payload = {k: v for k, v in artifact.items() if k != "sha256"}
        if (_digest(payload) != artifact["sha256"] or artifact["metric_identity"] != metric_identity
                or artifact["calibration_identity"] != calibration_identity):
            raise ValueError("native quadrature artifact digest or contract mismatch")
        args = {k: v for k, v in payload.items() if k not in {"adapter_id", "query_weights"}}
        instance = cls(**args)
        if not np.array_equal(instance.query_weights, np.asarray(artifact["query_weights"])):
            raise ValueError("native quadrature restored weights mismatch")
        return instance

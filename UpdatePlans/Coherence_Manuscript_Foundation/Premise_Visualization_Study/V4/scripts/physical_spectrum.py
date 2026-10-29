"""Native-grid spectral descriptors copied from V2 without algorithmic changes.

Provenance: ../../scripts/v2/spectrum_analysis.py and
../../scripts/v2/spectrum_octave_analysis.py. This descriptive estimator is
not the graph-basis training objective.
"""
from __future__ import annotations
from typing import Any
from pathlib import Path
import hashlib
import itertools
import numpy as np
FIELD_NAMES = ('CH4','CO','T','U_1','p')
PAIRS = tuple(itertools.combinations(range(5), 2))
BANDS = ('low','mid','high')
KAPPA_EDGES = np.asarray([0.,4.,8.,16.,32.,64.,np.inf])
BAND_NAMES = ('kappa_0_4','kappa_4_8','kappa_8_16','kappa_16_32','kappa_32_64','kappa_64_inf')

def _sha256_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).view(np.uint8)).hexdigest()

def _trapezoid_weights(axis: np.ndarray) -> np.ndarray:
    delta = np.diff(axis)
    if not np.all(delta > 0):
        raise ValueError("native coordinate axes must be strictly increasing")
    weights = np.empty(axis.size, dtype=np.float64)
    weights[0] = delta[0] / 2.0
    weights[-1] = delta[-1] / 2.0
    weights[1:-1] = (delta[:-1] + delta[1:]) / 2.0
    return weights

def _make_transform(coordinates: np.ndarray) -> dict[str, Any]:
    if coordinates.ndim != 3 or coordinates.shape[-1] != 2:
        raise ValueError(f"expected shared coordinates [y,x,2], got {coordinates.shape}")
    coords = np.asarray(coordinates, dtype=np.float64)
    x = coords[0, :, 0]
    y = coords[:, 0, 1]
    separability_error = max(
        float(np.max(np.abs(coords[:, :, 0] - x[None, :]))),
        float(np.max(np.abs(coords[:, :, 1] - y[:, None]))),
    )
    coordinate_scale = max(float(np.ptp(x)), float(np.ptp(y)), 1.0)
    if separability_error > 1.0e-6 * coordinate_scale:
        raise ValueError(f"native coordinate tensor is not a separable rectangle: {separability_error}")

    xw = _trapezoid_weights(x)
    yw = _trapezoid_weights(y)
    lx, ly = float(x[-1] - x[0]), float(y[-1] - y[0])
    dx_max, dy_max = float(np.max(np.diff(x))), float(np.max(np.diff(y)))
    # This conservative cutoff uses the largest native gap, avoiding claims
    # above the local Nyquist support at the sparsest part of a stretched axis.
    mx_max = max(1, int(np.floor(lx / (2.0 * dx_max))))
    my_max = max(1, int(np.floor(ly / (2.0 * dy_max))))
    mx = np.arange(-mx_max, mx_max + 1, dtype=np.int32)
    my = np.arange(-my_max, my_max + 1, dtype=np.int32)
    fx = mx.astype(np.float64) / lx
    fy = my.astype(np.float64) / ly
    nx = fx / (mx_max / lx)
    ny = fy / (my_max / ly)
    rho = np.sqrt(ny[:, None] ** 2 + nx[None, :] ** 2) / np.sqrt(2.0)
    band_edges = np.asarray([0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0 + 1.0e-12])
    band_id = np.digitize(rho, band_edges[1:-1], right=False).astype(np.int8)
    positive_half = (my[:, None] > 0) | ((my[:, None] == 0) & (mx[None, :] > 0))
    nonzero = positive_half & (rho <= 1.0 + 1.0e-12)
    masks = np.stack([nonzero & (band_id == j) for j in range(len(BANDS))])
    counts = masks.sum(axis=(1, 2))
    if np.any(counts == 0):
        raise ValueError(f"empty spectral band support: {counts.tolist()}")

    # Mean removal uses the native trapezoidal area measure. Each Hann taper is
    # normalized to weighted mean one, so it changes boundary contribution but
    # does not impose a method-specific amplitude scale.
    wx = 0.5 - 0.5 * np.cos(2.0 * np.pi * (x - x[0]) / lx)
    wy = 0.5 - 0.5 * np.cos(2.0 * np.pi * (y - y[0]) / ly)
    wx *= float(np.sum(xw) / np.sum(xw * wx))
    wy *= float(np.sum(yw) / np.sum(yw * wy))
    x_integral = xw * wx / lx
    y_integral = yw * wy / ly
    tx = np.exp(-2j * np.pi * fx[:, None] * (x - x[0])[None, :]) * x_integral[None, :]
    ty = np.exp(-2j * np.pi * fy[:, None] * (y - y[0])[None, :]) * y_integral[None, :]
    return {
        "x": x,
        "y": y,
        "x_weights": xw,
        "y_weights": yw,
        "lx": lx,
        "ly": ly,
        "dx_max": dx_max,
        "dy_max": dy_max,
        "mx": mx,
        "my": my,
        "fx": fx,
        "fy": fy,
        "rho": rho,
        "band_edges": band_edges,
        "band_id": band_id,
        "masks": masks,
        "mode_counts_halfplane": counts,
        "tx": tx,
        "ty": ty,
        "positive_half": positive_half,
        "nonzero": nonzero,
        "coordinates_sha256": _sha256_array(coords),
        "separability_max_abs_error": separability_error,
    }

def _transform_block(block: np.ndarray, tr: dict[str, Any]) -> np.ndarray:
    """Transform [case,y,x,field] values to native-coordinate Fourier modes."""
    values = np.asarray(block, dtype=np.float64)
    if values.ndim != 4 or values.shape[1:3] != (tr["y"].size, tr["x"].size):
        raise ValueError(f"field block must be [B,{tr['y'].size},{tr['x'].size},F], got {values.shape}")
    xw, yw = tr["x_weights"], tr["y_weights"]
    area_weights = yw[:, None] * xw[None, :]
    means = np.einsum("byxf,yx->bf", values, area_weights, optimize=True) / area_weights.sum()
    centered = values - means[:, None, None, :]
    # The transform matrices already include normalized Hann * trapezoid
    # weights and domain-length factors; applying weights to `centered` here
    # would double-window the map.
    temp = np.einsum("ay,byxf->baxf", tr["ty"], centered, optimize=True)
    coefficients = np.einsum("baxf,mx->bamf", temp, tr["tx"], optimize=True)
    return coefficients

def _physical_band_transform(coordinates: np.ndarray) -> dict[str, Any]:
    """Use legacy native-grid NUDFT matrices with a fixed kappa band map."""
    tr = _make_transform(coordinates)
    kappa = tr["lx"] * np.sqrt(tr["fy"][:, None] ** 2 + tr["fx"][None, :] ** 2)
    band_id = np.digitize(kappa, KAPPA_EDGES[1:-1], right=False).astype(np.int8)
    masks = np.stack([tr["nonzero"] & (band_id == bi) for bi in range(len(BAND_NAMES))])
    counts = masks.sum(axis=(1, 2)).astype(int)
    if np.any(counts == 0):
        raise ValueError(f"empty physical wavelength band(s): {counts.tolist()}")
    tr.update({"kappa": kappa, "physical_band_id": band_id, "physical_masks": masks, "physical_mode_counts": counts})
    return tr

def _reduce(coeff: np.ndarray, tr: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    batch = coeff.shape[0]
    auto = np.zeros((batch, len(FIELD_NAMES), len(BAND_NAMES)), dtype=np.float64)
    cross = np.zeros((batch, len(PAIRS), len(BAND_NAMES)), dtype=np.complex128)
    for bi, mask in enumerate(tr["physical_masks"]):
        z = coeff[:, mask, :]
        auto[:, :, bi] = 2.0 * np.sum(np.abs(z) ** 2, axis=1)
        for pi, (left, right) in enumerate(PAIRS):
            cross[:, pi, bi] = 2.0 * np.sum(z[:, :, left] * np.conjugate(z[:, :, right]), axis=1)
    return auto, cross

def _measure(path: Path, tr: dict[str, Any], chunk_size: int = 8) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        fields = archive["fields"]
        expected = (100, tr["y"].size, tr["x"].size, len(FIELD_NAMES))
        if fields.shape != expected:
            raise ValueError(f"unexpected fields shape in {path.name}: {fields.shape} != {expected}")
        auto = np.empty((100, len(FIELD_NAMES), len(BAND_NAMES)), dtype=np.float64)
        cross = np.empty((100, len(PAIRS), len(BAND_NAMES)), dtype=np.complex128)
        means = np.empty((100, len(FIELD_NAMES)), dtype=np.float64)
        area_weights = tr["y_weights"][:, None] * tr["x_weights"][None, :]
        for start in range(0, 100, chunk_size):
            stop = min(start + chunk_size, 100)
            block = fields[start:stop]
            means[start:stop] = np.einsum("byxf,yx->bf", block, area_weights, optimize=True) / area_weights.sum()
            coeff = _transform_block(block, tr)
            auto[start:stop], cross[start:stop] = _reduce(coeff, tr)
    return auto, cross, means

def _fractions(auto: np.ndarray) -> np.ndarray:
    return auto / np.maximum(auto.sum(axis=-1, keepdims=True), np.finfo(float).tiny)

def _window_transform(base: dict[str, Any], window: str) -> dict[str, Any]:
    """Create a same-grid transform with rectangular or weighted-Hann taper."""
    tr = dict(base)
    x, y = base["x"], base["y"]
    xw, yw = base["x_weights"], base["y_weights"]
    lx, ly = base["lx"], base["ly"]
    if window == "hann":
        wx = (0.5 - 0.5 * np.cos(2.0 * np.pi * (x - x[0]) / lx))
        wy = (0.5 - 0.5 * np.cos(2.0 * np.pi * (y - y[0]) / ly))
        wx *= float(xw.sum() / np.sum(xw * wx))
        wy *= float(yw.sum() / np.sum(yw * wy))
    elif window == "rectangular":
        wx, wy = np.ones_like(x), np.ones_like(y)
    else:
        raise ValueError(window)
    tr["tx"] = np.exp(-2j * np.pi * base["fx"][:, None] * (x - x[0])[None, :]) * ((xw * wx / lx)[None, :])
    tr["ty"] = np.exp(-2j * np.pi * base["fy"][:, None] * (y - y[0])[None, :]) * ((yw * wy / ly)[None, :])
    return tr

"""Rerender the matched A/B graph cross-spectrum score summary from pinned NPZs."""

from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path
from time import perf_counter

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
COHERENCE_ROOT = REPOSITORY_ROOT / "cases" / "turbulent_combustion" / "diagnostics" / "coherence"
SOURCE_ROOT = COHERENCE_ROOT / "source" / "ab_balanced_train_last"
OUTPUT_ROOT = COHERENCE_ROOT / "ab_balanced_train_last"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {key: payload[key].copy() for key in payload.files}


def main() -> None:
    started = perf_counter()
    import sys

    sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

    from phycoflow_reconstruction.evaluation.coherence_set import (
        render_cross_spectrum_paired_score_bars,
    )

    provenance_path = SOURCE_ROOT / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    for filename, expected_hash in provenance["input_sha256"].items():
        input_path = SOURCE_ROOT / filename
        if _sha256(input_path) != expected_hash:
            raise ValueError(f"pinned A/B cross-spectrum input checksum mismatch: {filename}")

    source = _load_npz(SOURCE_ROOT / "source_metrics.npz")
    post = _load_npz(SOURCE_ROOT / "post_metrics.npz")
    output_path = OUTPUT_ROOT / "cross_spectrum_source_post_coherence.png"
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    render_cross_spectrum_paired_score_bars(
        source,
        post,
        output_path,
        title="Matched cross-spectrum coherence",
        subtitle="Resolved-config terms · train · 12 matched groups × 16 · mean ±1 SD",
        component_filter=("same_frequency", "cross_frequency"),
        include_family_aggregate=False,
    )
    render_manifest = {
        "source_provenance": "../source/ab_balanced_train_last/provenance.json",
        "input_sha256": provenance["input_sha256"],
        "command": (
            "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 "
            "PYTHONPATH=src conda run -n phycoflow_env python "
            "cases/turbulent_combustion/diagnostics/coherence/"
            "render_ab_train_last_cross_spectrum.py"
        ),
        "device": "cpu",
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "script_elapsed_seconds": round(perf_counter() - started, 3),
    }
    (OUTPUT_ROOT / "render_manifest.json").write_text(
        json.dumps(render_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"figure": str(output_path), "manifest": str(OUTPUT_ROOT / "render_manifest.json")}, indent=2))


if __name__ == "__main__":
    main()

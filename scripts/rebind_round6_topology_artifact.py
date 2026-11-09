"""Explicitly rebind one frozen finite-primary artifact to the Round 6 source.

The caller supplies the independently configured family and expected geometry
identity. The normal strict loader remains the final validator. This helper is
only for the identical legacy query population, never for expanded banks.
"""

from __future__ import annotations

import hashlib
from copy import deepcopy
from pathlib import Path

import torch

from phycoflow_reconstruction.coherence.families.topology.persistence import persistence_source


def rebind_topology_artifact(
    input_path, output_path, *, expected_sha256, family, expected_geometry_sha256
):
    input_path, output_path = Path(input_path), Path(output_path)
    if output_path.exists() or input_path.resolve() == output_path.resolve():
        raise ValueError("output must be a new path")
    digest = hashlib.sha256(input_path.read_bytes()).hexdigest()
    if digest != expected_sha256:
        raise ValueError("input artifact SHA-256 mismatch")
    artifact = torch.load(input_path, map_location="cpu", weights_only=True)
    if (
        family.version != "4"
        or artifact.get("family") != "topology"
        or artifact.get("version") != "4"
    ):
        raise ValueError("rebind requires identical finite-primary topology")
    if (
        artifact.get("config") != family.config
        or tuple(artifact.get("field_names", ())) != family.field_names
        or artifact.get("target_use") != family.target_use
        or artifact.get("units") != family.units
        or artifact.get("periods") != family.periods
    ):
        raise ValueError("topology config, field, units or geometry contract mismatch")
    if (
        not isinstance(expected_geometry_sha256, str)
        or len(expected_geometry_sha256) != 64
        or artifact.get("geometry_sha256") != expected_geometry_sha256
    ):
        raise ValueError("topology geometry identity mismatch")
    state = artifact["state_dict"]
    for name in ("normalization_offset", "normalization_scale"):
        if not torch.equal(state[name].cpu(), getattr(family, name).detach().cpu()):
            raise ValueError("topology normalization mismatch")
    old = artifact.get("scientific_source")
    current = persistence_source(finite_primary=True)
    if not isinstance(old, dict) or not isinstance(old.get("source_sha256"), dict):
        raise TypeError("missing historical implementation binding")
    if {k: v for k, v in old.items() if k != "source_sha256"} != {
        k: v for k, v in current.items() if k != "source_sha256"
    } or set(old["source_sha256"]) != set(current["source_sha256"]):
        raise ValueError("persistence backend or source inventory differs")
    rebound = deepcopy(artifact)
    rebound["scientific_source"] = current
    rebound["round6_rebinding"] = {
        "input_sha256": digest,
        "old_scientific_source": deepcopy(old),
        "reason": "identical finite-primary population with selected execution support",
    }
    family.load_state_artifact(rebound)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("xb") as handle:
        torch.save(rebound, handle)
    return hashlib.sha256(output_path.read_bytes()).hexdigest()

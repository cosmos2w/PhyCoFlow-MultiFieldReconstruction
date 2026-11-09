#!/usr/bin/env python3
"""Rebind a verified frozen Scale artifact after the Round 6 sparse implementation change.

This changes only implementation binding and records the exact input artifact hash.
The normal strict loader still validates every calibration and data contract.
"""
import argparse
import hashlib
from pathlib import Path

import torch

from phycoflow_reconstruction.coherence.families.multiscale_relation import (
    RELATION_DEFINITION,
    _candidate_implementation_identity,
)
from phycoflow_reconstruction.coherence.families.multiscale_spectrum.family import (
    _implementation_identity,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--expected-sha256", required=True)
    args = parser.parse_args()
    if args.output.exists() or args.input.resolve() == args.output.resolve():
        raise ValueError("output must be a new path")
    digest = hashlib.sha256(args.input.read_bytes()).hexdigest()
    if digest != args.expected_sha256:
        raise ValueError("input artifact SHA-256 mismatch")
    artifact = torch.load(args.input, map_location="cpu", weights_only=True)
    if artifact.get("relation_extension_definition") != RELATION_DEFINITION:
        raise ValueError("input is not the expected relation Scale artifact")
    old_base = artifact.get("implementation_identity")
    old_relation = artifact.get("relation_extension_implementation_identity")
    if not isinstance(old_base, dict) or not isinstance(old_relation, dict):
        raise TypeError("input is missing its old strict implementation bindings")
    current = _implementation_identity()
    if old_base.get("backends") != current["backends"]:
        raise ValueError("backend differs from the frozen artifact")
    artifact["implementation_identity"] = current
    artifact["relation_extension_implementation_identity"] = _candidate_implementation_identity()
    artifact["round6_rebinding"] = {"input_sha256": digest, "old_implementation_identity": old_base,
                                     "old_relation_identity": old_relation,
                                     "reason": "explicit sparse Scale execution"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as handle:
        torch.save(artifact, handle)
    print(hashlib.sha256(args.output.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()

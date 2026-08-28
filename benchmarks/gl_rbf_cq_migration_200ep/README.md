# GL-RBF/CQ execution protocol

This directory retains the reviewed protocol and recipes for comparing the legacy `pointcloud_ffm`/GL-RBF-enhanced path with the `gl_rbf_cq` path. It is a reproduction contract, not a checked-in run archive.

## Scope

The three arms use the same turbulent-combustion field order, training-only normalization, T-only random-uniform sensor range (192--384), seed 42, B40/Q4096 training budget, optimizer settings, checkpoint milestones, and evaluation policy. Arm B uses legacy multi-head attention; Arm C uses cached K/V. Arm A is the existing project-owned GL-RBF-enhanced baseline and is not silently treated as the CQ architecture.

The authorized B40 common batch is part of the protocol: the originally requested B128 setting exhausted GPU 0 memory during backward. No long run is part of repository CI or the local smoke suite.

## Tracked contract and recipes

- `PROTOCOL.yaml` records identities, scientific settings, resource adjustment, normalization provenance, and required evidence.
- `configs/` contains the three launch configurations used to reproduce the arms. The configs reference the shared model/runtime implementation.
- `comparison/generate_comparison.py` joins locally produced evidence into a deterministic summary; its generated CSV/JSON outputs are ignored.
- `execution/benchmark_execution.py` is an opt-in same-state GPU probe for cached-K/V projection counts and resource/timing deltas.
- `scripts/compute_training_normalization.py` produces the checked, training-only normalization artifact when the local payload is available.
- `downstream_train_normalization.json` and `migration/initialization_identity.json` are small immutable contract fixtures consumed by compatibility tests.

The historical run summaries, telemetry, fixed validation manifest, HTML/JSON reports, and milestone CSV are deliberately not tracked. Generate those into the ignored `generated/`, `reports/`, `execution/`, or local run directories when doing a formal investigation. Git history retains the earlier outputs.

## Reproduction outline

From the standalone repository root, after installing optional dependencies and linking the local turbulent-combustion payload:

```bash
python scripts/benchmarks/build_fixed_benchmark_manifest.py \
  --config benchmarks/gl_rbf_cq_migration_200ep/configs/B_gl_rbf_cq_legacy_mha_200ep.yaml \
  --output benchmarks/gl_rbf_cq_migration_200ep/generated/fixed_validation_manifest.json

# Run the three case-local base-training commands with the protocol configs.
# Keep checkpoints and telemetry under cases/turbulent_combustion/runs/.

python benchmarks/gl_rbf_cq_migration_200ep/comparison/generate_comparison.py \
  --output-dir benchmarks/gl_rbf_cq_migration_200ep/generated/comparison
```

The comparison generator requires the locally produced arm evidence named in its source. It validates hashes and matched sensor/query identities before writing output. The protocol's one-update checks are covered by the shared CPU contract tests; use the GPU-0 smoke script for quick model-level checks.

## Interpretation

Benchmark outputs from one seed, one dataset, or a resource-adjusted run are engineering/compatibility evidence rather than general scientific rankings. Report field order, normalization and sensor hashes, checkpoint choice, EMA selection, generation steps, device, and all declared caveats with any externally shared result.

# Contributing

PhyCoFlow is a shared research workspace. Keep each change within a clear contract boundary, preserve scientific meaning, and make the resulting run traceable for the next coworker.

## Workflow

1. Start from an up-to-date `main` and create a focused task branch. Short-lived pilot branches (for example `pilot/<topic>`) are fine for checking the coworking workflow itself.
2. Keep the branch reviewable: separate layout, model, config, and artifact changes into logically scoped commits.
3. Open a pull request to `main` using the repository template. Do not merge your own research change without review.
4. Explain checkpoint, config, data, and artifact compatibility impact in the PR description.

Do not add coworker-named source directories or a second installable package. All reusable Python belongs under `src/phycoflow_reconstruction/`.

## Where new work belongs

- New reconstruction model: add a core/adapter pair under `src/phycoflow_reconstruction/models/`, register the public name in the model registry, and add its shared config under `configs/models/`.
- New coherence family: add it under `src/phycoflow_reconstruction/coherence/families/`, register the family, and define its contract/geometry/reference provenance. Compose families through the shared post-training lifecycle.
- New case: add `cases/<case>/` with metadata, a thin `run.py`, dataset and sensor configs, case-owned coherence/physics settings, and diagnostics. Do not copy general models or trainers into the case.
- New dataset: document the contract under `datasets/<case>/README.md` and `datasets/SCHEMA.md`; keep payloads, links, caches, and large manifests local. Add a small immutable fixture only when a test genuinely requires it.
- New plotting/benchmark helper: use `scripts/visualization/` or `scripts/benchmarks/`. Generated reports, figures, telemetry, and run summaries belong under ignored run/output locations.

Case-specific sensor layouts, normalization sources, physical geometry, coherence weights, and benchmark schedules remain case-local when they carry scientific meaning. Generic optimizer/runtime/evaluation/checkpoint policy belongs in `configs/defaults/`; do not repeat byte-identical model definitions under every case.

## Contracts and compatibility

Prefer the common `ObservationBatch`, `LossBundle`, `ReconstructionBatch`, `ModelCapabilities`, `CaseSpec`, and shared trainer/evaluator over a bespoke trainer. General modules must not import named cases or the historical Demo50 source. Randomness must be explicit and seeded; evaluation sensor manifests and query indices must be persisted for matched comparisons.

Treat model names, parameter prefixes, field order, normalization statistics, sensor semantics, checkpoint selection, EMA state, and coherence/physics definitions as compatibility contracts. GL-RBF/CQ changes additionally need the hard gates for state keys/shapes, seeded initialization, gradients, microbatching, cached-K/V execution, geometry/reconstruction caches, observation consistency, source sampling, EMA, and strict checkpoint loading. Never hide a compatibility failure with `strict=False` or by weakening a numerical regression test.

External code requires a pinned upstream revision, license note, and attribution in [docs/provenance.md](docs/provenance.md). Do not vendor a historical repository as a monolith.

## Required checks

Run these from the repository root before opening a PR:

```bash
ruff check src tests scripts cases benchmarks
pytest
```

Also run the relevant config/case contract and smoke checks for your change:

- model changes: registry, state-dict, native-loss/backward, reconstruction, and checkpoint tests;
- data/sensor changes: dataset validation, split, manifest, and case tests;
- coherence changes: family contract, composition, observation-consistency, and short post-training tests;
- training/evaluation changes: checkpoint lifecycle, resume, rollout, and representative evaluation tests;
- physics changes: direct-PINN and physics post-training tests on a valid case;
- flow/GL-RBF/CQ changes: all point-cloud hard-gate tests;
- local GPU acceptance: `python scripts/smoke/models.py --device cuda:0` when GPU 0 and optional dependencies are available.

Cloud CI is intentionally CPU-oriented and does not run long training or require a GPU. Do not add generated benchmark outputs to make a test pass. If a payload or optional dependency is absent, report a clean skip with the reason and retain the contract coverage.

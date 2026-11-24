# Contributing

PhyCoFlow is a shared research workspace. Keep each change within a clear contract boundary, preserve scientific meaning, and make the resulting run traceable for the next coworker.

## Workflow

1. Check the repository root, branch and `git status` before pulling or changing branches. Preserve local work; use `git pull --ff-only` when an update is needed, and create a focused branch when the task requires one.
2. Keep the change reviewable: separate layout, model, config and artifact changes into logically scoped commits.
3. Open a pull request to `main` using the repository template. Do not merge your own research change without review.
4. Explain checkpoint, config, data, and artifact compatibility impact in the PR description.

Do not add coworker-named source directories or a second installable package. Put reusable library code under `src/phycoflow_reconstruction/` and thin command or visualization entry points under `scripts/`. Keep case-owned scientific choices in their case and frozen execution copies with their campaign. Extend the existing contracts and tools before introducing a new framework, scheduler or compatibility layer.

## Where new work belongs

- New reconstruction model: add a core/adapter pair under `src/phycoflow_reconstruction/models/`, register the public name in the model registry, and add its shared config under `configs/models/`.
- New coherence family: add it under `src/phycoflow_reconstruction/coherence/families/`, register the family, and define its contract/geometry/reference provenance. Compose families through the shared post-training lifecycle.
- New case: add `cases/<case>/` with metadata, a thin `run.py`, dataset and sensor configs, case-owned coherence/physics settings, and diagnostics. Do not copy general models or trainers into the case.
- New dataset: document the contract under `datasets/<case>/README.md` and `datasets/SCHEMA.md`; keep payloads, links, caches, and large manifests local. Add a small immutable fixture only when a test genuinely requires it.
- New plotting/benchmark helper: use `scripts/visualization/` or `scripts/benchmarks/`. Routine generated reports, figures, telemetry and run summaries remain with the ignored campaign. An intentionally selected, portable report bundle is a publication exception described in [experiment artifact organization](docs/Experiment_Artifacts.md).

Case-specific sensor layouts, normalization sources, physical geometry, coherence weights, and benchmark schedules remain case-local when they carry scientific meaning. Generic optimizer/runtime/evaluation/checkpoint policy belongs in `configs/defaults/`; do not repeat byte-identical model definitions under every case.

## Contracts and compatibility

Prefer the common `ObservationBatch`, `LossBundle`, `ReconstructionBatch`, `ModelCapabilities`, `CaseSpec`, and shared trainer/evaluator over a bespoke trainer. General modules must not import named cases or the historical Demo50 source. Randomness must be explicit and seeded; evaluation sensor manifests and query indices must be persisted for matched comparisons.

Treat model names, parameter prefixes, field order, normalization statistics, sensor semantics, checkpoint selection, EMA state, and coherence/physics definitions as compatibility contracts. GL-RBF/CQ changes additionally need the hard gates for state keys/shapes, seeded initialization, gradients, microbatching, cached-K/V execution, geometry/reconstruction caches, observation consistency, source sampling, EMA, and strict checkpoint loading. Never hide a compatibility failure with `strict=False` or by weakening a numerical regression test.

Keep GPU selection in runtime configuration, separate from frozen scientific protocols; do not hard-code physical GPU indices or UUID allowlists in code or frozen contracts. Record the actual device and environment and honor current exclusions. Use epochs for training schedules. Saved recording definitions own ordered columns, their types, units and meanings, and loss-component accumulation order across resume; incoming configuration validates scientific content. JSON key order alone is not a scientific change, and ordinary resume or an authorized device or horizon change does not rewrite logs. Changes to resume or recording need a fresh-process save, reload, resume and append comparison against uninterrupted training.

External code requires a pinned upstream revision, license note, and attribution in [docs/provenance.md](docs/provenance.md). Do not vendor a historical repository as a monolith.

## Required checks

Run these from the repository root before opening a PR:

```bash
ruff check src tests scripts cases benchmarks tools
pytest
```

Also run the relevant config/case contract and smoke checks for your change:

- model changes: registry, state-dict, native-loss/backward, reconstruction, and checkpoint tests;
- data/sensor changes: dataset validation, split, manifest, and case tests;
- coherence changes: family contract, composition, observation-consistency, and short post-training tests;
- training/evaluation changes: checkpoint lifecycle, resume, rollout, and representative evaluation tests;
- physics changes: direct-PINN and physics post-training tests on a valid case;
- flow/GL-RBF/CQ changes: all point-cloud hard-gate tests;
- local GPU acceptance: when required and a compatible permitted GPU and optional dependencies are available, select the device at runtime and run `python scripts/smoke/models.py --device cuda:0`; here `cuda:0` is the logical device within the selected visibility, not a frozen physical index.

Cloud CI is intentionally CPU-oriented and does not run long training or require a GPU. Do not add generated benchmark outputs to make a test pass. If a payload or optional dependency is absent, report a clean skip with the reason and retain the contract coverage.

For portable plans and reports, install the `docs` extra and run `python tools/docs/check_markdown_links.py UpdatePlans/_Reports/Coherence_Adaptive_Repair --portable` from the checkout root, substituting the relevant publication bundle. The broader local report tree also contains intentionally nonportable historical material; CI checks the tracked report tree in a clean checkout. Validate figure decoding, rendered pages and PDF links as described in [experiment artifact organization](docs/Experiment_Artifacts.md), and check the selected publication again in a clean Git export. Review the outgoing tracked files, including previously tracked files beneath ignored directories, and keep machine-specific file and directory aliases and raw campaign payloads out of Git.

Also run `python tools/docs/check_workspace_layout.py`. Never create root review/report aliases or put generated report workspaces in the current reading libraries. Use a unique task-owned temporary directory for small scratch and DATA2 cache storage for large/resumable work. Promote unique results and job/report verification evidence before cleaning confirmed disposable files; preserve unknown or pre-existing material. The location and deletion rules are defined in [the temporary-file lifecycle](docs/Experiment_Artifacts.md#temporary-files-and-cleanup).

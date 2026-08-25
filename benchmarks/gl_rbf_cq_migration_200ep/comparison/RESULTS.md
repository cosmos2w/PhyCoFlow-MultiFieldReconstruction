# GL_rbf_CQ migration benchmark results

## Answer first

The fixed-manifest comparison uses the common B40/Q4096, seed-42, T-only
192--384-sensor protocol. Δ is **later minus earlier**, so a negative quality
delta is an error reduction. B and C use their configured EMA checkpoints for
evaluation; A is the raw/configured legacy checkpoint. At epoch 200:

| Effect (epoch 200 fixed manifest) | Δ MSE | Δ mean relative L2 | Δ worst relative L2 |
|---|---:|---:|---:|
| Migration: B − A | +0.010720 (+2.77%) | +0.009784 (+1.62%) | +0.026053 (+3.12%) |
| Execution: C − B | +0.021666 (+5.45%) | +0.017984 (+2.93%) | +0.021887 (+2.55%) |
| Total latest model: C − A | +0.032387 (+8.37%) | +0.027768 (+4.59%) | +0.047940 (+5.75%) |

- **Migration effect (B−A):** parameters increase
  +14.31%,
  while formal peak allocation/reservation fall
  -83.45%/
  -84.54%
  and steady mean epoch/step time fall
  -54.01%/
  -54.66%.
  Epoch-200 mean relative L2 is +1.62%
  higher error than A.
- **Execution effect (C−B):** parameter count is unchanged. In the matched
  same-state probe, cached K/V reduces median whole-step time
  -3.16%
  and peak allocated/reserved memory
  -2.48%/
  -2.53%,
  with numerical differences below 3e-7. Its independent 40,000-step quality
  delta is accumulated trajectory drift, not a causal execution-quality effect.
- **Total latest-model effect (C−A):** formal peak allocation/reservation fall
  -83.85%/
  -84.87%,
  steady mean epoch/step time fall
  -55.13%/
  -56.79%,
  and epoch-200 mean relative L2 is
  +4.59% higher error.

## Formal resources and timing

| Arm | Parameters | Peak allocated GiB | Peak reserved GiB | Steady epoch s | Steady mean step ms |
|---|---:|---:|---:|---:|---:|
| A (legacy_downstream) | 4,778,118 | 38.37 | 46.19 | 59.725977 | 297.325 |
| B (legacy_mha) | 5,461,817 | 6.35 | 7.14 | 27.470796 | 134.818 |
| C (cached_kv) | 5,461,817 | 6.20 | 6.99 | 26.796824 | 128.482 |

The formal telemetry covers epochs 2--200 for steady statistics. B/C also have
a matched 50-step controlled probe: legacy MHA measured exactly four K/V
projection calls per step; cached-K/V measured exactly one. Probe whole-step
mean/median are 138.363/
127.663 ms for B and
135.723/
123.632 ms for C. The recorded
probe C−B median whole-step change is
-3.16%;
probe allocated memory changes from 6.33 to
6.18 GiB.

## Fixed-manifest convergence and time-to-threshold

The table reports the first available fixed checkpoint at or below each
threshold and exact cumulative epoch wall time from the immutable telemetry;
“not reached” means no listed checkpoint met it.

| Metric threshold | A | B | C |
|---|---:|---:|---:|
| mse_normalized ≤ 0.7 | e40 / 2388.9 s | e20 / 547.4 s | e20 / 534.2 s |
| mse_normalized ≤ 0.6 | e100 / 5972.0 s | e40 / 1100.4 s | e40 / 1070.3 s |
| mse_normalized ≤ 0.5 | e150 / 8957.7 s | e150 / 4122.7 s | e150 / 4020.2 s |
| mean_relative_l2 ≤ 0.75 | e100 / 5972.0 s | e60 / 1650.0 s | e60 / 1607.3 s |
| mean_relative_l2 ≤ 0.7 | e150 / 8957.7 s | e100 / 2746.8 s | e150 / 4020.2 s |
| mean_relative_l2 ≤ 0.65 | e200 / 11945.6 s | e200 / 5493.4 s | e200 / 5359.0 s |
| worst_field_relative_l2 ≤ 1 | e150 / 8957.7 s | e100 / 2746.8 s | e100 / 2676.9 s |
| worst_field_relative_l2 ≤ 0.9 | e200 / 11945.6 s | e200 / 5493.4 s | e200 / 5359.0 s |

The complete matched-milestone data, including per-field relative L2, parameter
count, memory, timing, checkpoint SHA, and report SHA, is in
[`milestones.csv`](milestones.csv). The structured effect decomposition is in
[`final_summary.json`](final_summary.json).

## Reproducibility identity

| Arm | Run path | Launch head | Config source / semantic / resolved | e200 checkpoint / report |
|---|---|---|---|---|
| A | `Cases/turbulent_combustion/runs/gl_rbf_cq_migration_200ep_A_legacy_gl_rbf_enh/20260825T054028Z_0c41ff4a` | `326c02129f6481781fe71456acb1e638d11830c3` | `9fb98efb8faf` / `0c41ff4a5bc3` / `2c0a4ba9ec06` | `75f5dd07a0212a6b894753be72c29e71ab8db1dfcc1c3b9241890b7b27e7c53c` / `0d076da98a9b3a28b2f9dd5b9f09607fa97fa487df8482da6d7a77772b4aa0ec` |
| B | `Cases/turbulent_combustion/runs/gl_rbf_cq_migration_200ep_B_gl_rbf_cq_legacy_mha/20260825T094100Z_93b0b3f4` | `603fcbf79d0e0dab6c7c84a22e67c5bab4ecc394` | `adeee8c3111b` / `93b0b3f40bbe` / `f0c79cfd6c18` | `ed53f7e1c0c26918f06e67ce6a85c09af5d91ffb87eb11722fbf42979f51f255` / `5f17d9690cb434a73a15a37f87e6529a59ebe72678dec6e66dc99b371713f22e` |
| C | `Cases/turbulent_combustion/runs/gl_rbf_cq_migration_200ep_C_gl_rbf_cq_cached_kv/20260825T094108Z_245fd571` | `603fcbf79d0e0dab6c7c84a22e67c5bab4ecc394` | `4d1c34461e34` / `245fd57156b5` / `475aa4994a85` | `a452fa4ae35d65599af2fbc5dec7fa0e5664514ff8fd72af1494721093deefd6` / `1719dd370b24f5c035a9554ed5f3850dccde5afb7e8359e25d1a5929968f95fb` |

All arms share dataset fingerprint
`8c49936567eced7ab94887c336b9b35aaf7ec70dea7479aee83434ff970455d5`, normalization artifact
`b7e31a1497e8d8b76e0ac8c9e7744d202e52581050825bb299f0a8870c353a39`, normalizer digest
`50c5e65e563fee03f4c15df336e064753981f8555a06e7465a4722553b3746f1`, fixed sensor manifest
`2071583f79e30f17bc586d907da184b5c79dfc82c01b4d652ccf05652e2c2b6f`, fixed-manifest file
`f2510f7dd76daaf15f61eab796d299d9ecb0c274e933609c2c21262fc66f3fd9`, and query-index hash
`16c263270389ab8665563c1d6fdcab3c2f193df481eed2b2717b20dfe7b40a5a`. The validation branch is
`validation/proj-multifield-gl-rbf-cq`; A was frozen from
`release/gl-rbf-cq-portable-prep`, and B/C launched from the same
portable-prep head recorded above.

The input evidence hashes used by the generator are:

| Evidence file | SHA-256 |
|---|---|
| `0_demo_TurbulentCombustion/GL_rbf_CQ_UPDATE_GUIDE.md` | `a29dad4766e3b02801e409b421d75946c45c0f4ee803475b9c9928c4ea114ab8` |
| `Proj_MultiFieldReconstruction/benchmarks/gl_rbf_cq_migration_200ep/PROTOCOL.yaml` | `d874d460d882095b761e3d3c19b22c26d321bd759267c7ca512867e09edf5528` |
| `Proj_MultiFieldReconstruction/benchmarks/gl_rbf_cq_migration_200ep/README.md` | `51ef30775b7632fc0185f3456337b7f512b5b61a98cfeeb12f9d335b214d5894` |
| `Proj_MultiFieldReconstruction/benchmarks/gl_rbf_cq_migration_200ep/baseline/A_performance.json` | `279a7b77cc92139dd56bfb4279cb4e9b8005bb9c9b57f5ca69574b8e14cd34cf` |
| `Proj_MultiFieldReconstruction/benchmarks/gl_rbf_cq_migration_200ep/configs/A_legacy_gl_rbf_enh_200ep.yaml` | `9fb98efb8fafa56a58d358391e6bc8be6da0fd02ffee9f1ce24f9d22826d3996` |
| `Proj_MultiFieldReconstruction/benchmarks/gl_rbf_cq_migration_200ep/configs/B_gl_rbf_cq_legacy_mha_200ep.yaml` | `adeee8c3111b5a9e24c91c0eeeedce4fd898556761a43c8eff08e655bd4f1663` |
| `Proj_MultiFieldReconstruction/benchmarks/gl_rbf_cq_migration_200ep/configs/C_gl_rbf_cq_cached_kv_200ep.yaml` | `4d1c34461e34bf9d418aa03d61cdeb7b3afbd6cde02d78b029c4fce4ccf28922` |
| `Proj_MultiFieldReconstruction/benchmarks/gl_rbf_cq_migration_200ep/execution/B_vs_C_execution.json` | `afcaf7519a42955ac96960e758ea5c5206ea9d65b9b4931c62fca2992a37c1a9` |
| `Proj_MultiFieldReconstruction/benchmarks/gl_rbf_cq_migration_200ep/migration/correctness_gates.json` | `e94c51190568b3f76d5f093e6cc4f156d387d9b735be75fb893c5c1e3c772e2b` |
| `Proj_MultiFieldReconstruction/benchmarks/gl_rbf_cq_migration_200ep/migration/initialization_identity.json` | `2351d9fbc09262d60908c360e2df992e3ca7095cdcc18af03e589fef0b4e5dcc` |
| `Proj_MultiFieldReconstruction/benchmarks/gl_rbf_cq_migration_200ep/runs_summary/B_summary.json` | `d0d6e229581b40471ae7aea0ca77165aa817a144aa979f9a3f4760300f6fd3ce` |
| `Proj_MultiFieldReconstruction/benchmarks/gl_rbf_cq_migration_200ep/runs_summary/C_summary.json` | `e6d67a063fd7f9b94135919009a7653e26e14823542566eba61f24572be51e77` |

The per-run artifact SHA maps and all milestone checkpoint/report hashes are
retained in `final_summary.json`; no run directory or checkpoint is copied into
the repository.

## Migration gates and guide findings

The migration gate artifact reports status **passed**:
55 passed;
13 passed;
142 passed, 1 opt-in GPU test skipped in the default invocation; and the opt-in GPU
legacy-equivalence test passed. The seeded B/C initialization state is
identical (`c80df4b84d3c1b992510da6d7ada6e274ebbde133890cbeb980e56dc92e1978c`),
with 148 state keys. Controlled numerical evidence records maximum loss and
gradient-norm differences of
2.384e-07 and
2.917e-07,
respectively.

The portable guide is present at
`0_demo_TurbulentCombustion/GL_rbf_CQ_UPDATE_GUIDE.md` and hashes to
`a29dad4766e3b02801e409b421d75946c45c0f4ee803475b9c9928c4ea114ab8`.
The migration exposed three release-blocking documentation gaps, all fixed on
this validation branch in commit `47de065a5b80a871297620e9703fb0bff528dff4`:

- **Strict tensor loading did not protect positional field semantics.** The guide now requires exact field identity/order and normalization checks before strict load, rejects same-width semantic mismatches, and documents the fresh-start path with matched B/C initialization.
- **The manifest snapshot could be silently rewritten by downstream tooling.** The guide now requires byte-identical copy/checksum evidence and excludes the vendored snapshot from formatters and linters.
- **Generic trainer/evaluator lifecycle integration was underspecified.** The guide now specifies condition-context reuse, exact loss scaling, no double backward, strict EMA auxiliary state, and configured/live evaluation selection without model-name branches.

The following evidence-schema improvements remain useful but did not block or
invalidate this benchmark:

- The guide defines tensor and model integration, but not a downstream fixed-manifest artifact contract (milestone list, metric names, checkpoint/report hashes, or immutable sensor/query digests). Recommended follow-up: Add a machine-readable evaluation schema and require fixed-manifest checkpoint/report hashes plus normalized MSE, mean, worst, and per-field relative L2 at every declared milestone.
- The guide asks for seeded loss/gradient and reconstruction comparisons but does not define the downstream RF bridge/data metadata identity or an all-gradient evidence schema. Recommended follow-up: Specify the adapter training_loss contract, query-index metadata, RF bridge seed/draw recording, state hash, and numerical tolerances in a JSON evidence template.
- The guide does not quantify the legacy-to-cached execution contract; it says cached K/V is preferred but does not require exact projection-call counts or a matched probe protocol. Recommended follow-up: Require an instrumented legacy_mha versus cached_kv probe with expected/observed calls (4 versus 1 here), memory, phase timing, same tensors, and numerical deltas.
- The guide does not define a machine-readable OOM/resource-adjustment record for an authorized common-batch fallback. Recommended follow-up: Define an adjustment record that preserves the requested protocol, failed attempt, authorization, chosen common replacement, and identical application across arms.

## Reproduction

From `Proj_MultiFieldReconstruction/`, run:

```text
rtk env CUDA_VISIBLE_DEVICES= python benchmarks/gl_rbf_cq_migration_200ep/comparison/generate_comparison.py
```

The generator is standard-library-only, reads only the frozen summaries,
execution/gate evidence, configs, guide, and immutable telemetry, and writes
the three comparison artifacts in this directory. It does not modify the
untracked benchmark Markdown or any run artifact.

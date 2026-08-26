# GL_rbf_CQ downstream migration benchmark

This directory contains the tracked protocol and Arm-A pre-migration inputs for
the three-arm 200-epoch `Proj_MultiFieldReconstruction` benchmark. Large run
directories and checkpoints remain under `Cases/turbulent_combustion/runs/`.

The frozen comparison artifacts preserve the historical
`0_demo_TurbulentCombustion/GL_rbf_CQ_UPDATE_GUIDE.md` path and checksum.
Regenerating the comparison with `comparison/generate_comparison.py` therefore
requires that optional local-only historical demo tree; normal project imports,
training, and tests do not require it.

Arm A is the corrected legacy `pointcloud_ffm`/`EnhancedGLRBFTopK` baseline. It
uses only the historical capacity corrections: hidden/latent width 256, 128
latents, 8 heads, 4 latent blocks, top-k 32, field embedding width 128, 32
Fourier bands, and 256 RFF features. It does not use any GL_rbf_CQ feature.

All arms share the downstream dataset, field order, normalization, trainer,
optimizer, seed 42, B40/Q4096, and T-only random-uniform 192--384 sensor
protocol. The originally requested B128 setting OOMed during backward on the
48-GiB GPU; the authorized largest stable common batch is B40 for A/B/C.
`fixed_validation_manifest.json` is generated before the Arm-A run
and must be reused for every post-hoc milestone evaluation. Its current
checksum is `2071583f79e30f17bc586d907da184b5c79dfc82c01b4d652ccf05652e2c2b6f`;
it persists both T-only sensor pairs and the shared Q4096 query indices.

The common field normalizer is fit only on the downstream chronological train
frames `0:8000` and stored in `downstream_train_normalization.json`. The
artifact checksum is `b7e31a1497e8d8b76e0ac8c9e7744d202e52581050825bb299f0a8870c353a39`;
the float32 `FieldNormalizer` digest is
`50c5e65e563fee03f4c15df336e064753981f8555a06e7465a4722553b3746f1`.

The opt-in `benchmark_telemetry` config section writes machine-readable
per-step/per-epoch timing and CUDA peak-memory evidence into each run's
`metrics/` directory. It is disabled for all existing project configs.

Arm-A evidence is frozen in `baseline/` before migration. The exact protocol
smoke limitation, authorized B40 adjustment, completed 200-epoch formal run,
immutable checkpoint hashes, timing/memory telemetry, finite-gradient audit,
and fixed-manifest convergence are recorded in `A_performance.json` and
`A_BASELINE_SUMMARY.md`.

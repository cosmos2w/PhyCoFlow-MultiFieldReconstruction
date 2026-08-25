# GL_rbf_CQ downstream migration benchmark

This directory contains the tracked protocol and Arm-A pre-migration inputs for
the three-arm 200-epoch `Proj_MultiFieldReconstruction` benchmark. Large run
directories and checkpoints remain under `Cases/turbulent_combustion/runs/`.

Arm A is the corrected legacy `pointcloud_ffm`/`EnhancedGLRBFTopK` baseline. It
uses only the historical capacity corrections: hidden/latent width 256, 128
latents, 8 heads, 4 latent blocks, top-k 32, field embedding width 128, 32
Fourier bands, and 256 RFF features. It does not use any GL_rbf_CQ feature.

All arms share the downstream dataset, field order, normalization, trainer,
optimizer, seed 42, B128/Q4096, and T-only random-uniform 192--384 sensor
protocol. `fixed_validation_manifest.json` is generated before the Arm-A run
and must be reused for every post-hoc milestone evaluation. Its current
checksum is `2071583f79e30f17bc586d907da184b5c79dfc82c01b4d652ccf05652e2c2b6f`;
it persists both T-only sensor pairs and the shared Q4096 query indices.

The opt-in `benchmark_telemetry` config section writes machine-readable
per-step/per-epoch timing and CUDA peak-memory evidence into each run's
`metrics/` directory. It is disabled for all existing project configs.

Arm-A evidence is frozen in `baseline/` before migration. The exact protocol
smoke limitation is recorded in `A_performance.json`; a formal run must not be
started without resolving the documented B128/Q4096 GPU-memory gate.

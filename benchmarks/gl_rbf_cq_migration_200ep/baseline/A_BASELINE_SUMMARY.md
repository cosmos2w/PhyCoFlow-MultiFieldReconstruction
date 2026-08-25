# Arm A — corrected legacy GL_rbf_ENH baseline

## Freeze status

This baseline is frozen before any portable `GL_rbf_CQ` migration. The
benchmark branch is `validation/proj-multifield-gl-rbf-cq`; the pre-migration
commit is recorded in `A_performance.json`. The exact B128/Q4096 diagnostic
smoke reached CUDA OOM during backward on GPU 0, so no formal 200-epoch run
was launched and no benchmark setting was silently reduced. An allocator-
identical retry also requested 96 GiB; expandable segments were tested and
fragmentation was ruled out.

## Model and protocol

- Downstream model: `pointcloud_ffm` with `backbone: gl_rbf_enh`.
- Historical capacity corrections only: hidden/latent 256, 128 latents, 8
  heads, 4 latent blocks, top-k 32, field embedding 128, Fourier bands 32
  (max frequency 64), RFF features 256, RFF length scale 0.15.
- Dataset/field order: downstream turbulent combustion `[CH4, CO, T, U_1, p]`;
  coordinate dimension 2; existing normalization and split contracts.
- Sensors: random-uniform T-only, 192--384 valid sensors, seed 42.
- Training: 200 epochs, batch size 128, query points 4096, lr 1e-4,
  weight decay 1e-6, gradient clip 1.0, deterministic seed 42.
- Training preview is disabled; immutable checkpoints are saved at epochs
  `1, 20, 40, 60, 100, 150, 200`, and fixed-manifest evaluation is post-hoc.

## Structural differences from historical 0_demo GL_rbf_ENH

The downstream `EnhancedGLRBFTopK` implementation remains the project-owned
legacy implementation. It is not parameter-for-parameter substituted with the
0_demo implementation and does not receive CQ, FiLM, measurement/support,
EMA, persistent Top-K, or cached-K/V features. The actual parameter count and
formal-run timings/memory are recorded in `A_performance.json` after the run.

## Evidence

- Config: `../configs/A_legacy_gl_rbf_enh_200ep.yaml`
- Protocol: `../PROTOCOL.yaml`
- Fixed manifest: `../fixed_validation_manifest.json`
- Diagnostic evidence: `A_performance.json` and `A_metrics.csv`

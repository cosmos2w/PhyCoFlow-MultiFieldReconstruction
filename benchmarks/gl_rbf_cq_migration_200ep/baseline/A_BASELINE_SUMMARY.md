# Arm A — corrected legacy GL_rbf_ENH baseline

## Freeze status

This baseline is frozen before any portable `GL_rbf_CQ` migration. The
benchmark branch is `validation/proj-multifield-gl-rbf-cq`; the pre-migration
commit is recorded in `A_performance.json`. The exact B128/Q4096 diagnostic
smoke at the originally requested B128/Q4096 reached CUDA OOM during backward
on GPU 0. An allocator-identical retry also requested 96 GiB; expandable
segments were tested and fragmentation was ruled out. Per the authorized
benchmark adjustment, the largest stable common batch is B40; A/B/C all use
B40/Q4096 and no other scientific setting is reduced.

The first B40 formal attempt stopped at step 945 after the shared trainer's
float32 global gradient-norm reduction overflowed. Its timestamped run remains
preserved as failed evidence and its weights are not reused. Commit `d101fe6f`
repairs the generic clipping implementation by accumulating the same global L2
norm in float64 and rejecting genuinely non-finite gradients. The model, data,
optimizer, learning rate, and configured clipping bound are unchanged; the
formal run restarts from seed 42 at that commit.

That restart passed the earlier boundary but safely stopped before the optimizer
at step 1,259 when float32 backward itself became non-finite. Commit `cf44ffce`
adds a common opt-in `2^-64` power-of-two backward scale and algebraically
unscales it as part of the same global clip. The final clipped gradient is
mathematically unchanged, and A/B/C use the identical safeguard. The next
formal attempt again starts from seed 42 rather than either failed run.

The `2^-64` attempt then stopped safely at step 938 when even a scaled
derivative became non-finite. The common safeguard is therefore frozen at
`2^-120`, which remains exactly representable and safely invertible in
float32 while extending representable raw derivatives to approximately
`4e74`. The run again restarts from seed 42 with no failed weights reused.

The `2^-120` run completed 44 epochs and preserved checkpoints 1/20/40 before
another scaled derivative overflowed. The common scale is now `2^-140`, near
the lower end of exactly representable float32 powers of two, with unscale and
clip multiplication performed in float64. The new run again starts from seed
42; no prior weights are reused.

## Model and protocol

- Downstream model: `pointcloud_ffm` with `backbone: gl_rbf_enh`.
- Historical capacity corrections only: hidden/latent 256, 128 latents, 8
  heads, 4 latent blocks, top-k 32, field embedding 128, Fourier bands 32
  (max frequency 64), RFF features 256, RFF length scale 0.15.
- Dataset/field order: downstream turbulent combustion `[CH4, CO, T, U_1, p]`;
  coordinate dimension 2; existing normalization and split contracts.
- Sensors: random-uniform T-only, 192--384 valid sensors, seed 42.
- Training: 200 epochs, batch size 40, query points 4096, lr 1e-4,
  weight decay 1e-6, gradient clip 1.0, deterministic seed 42.
- Training preview is disabled; immutable checkpoints are saved at epochs
  `1, 20, 40, 60, 100, 150, 200`, and fixed-manifest evaluation is post-hoc.

## Structural differences from historical 0_demo GL_rbf_ENH

The downstream `EnhancedGLRBFTopK` implementation remains the project-owned
legacy implementation. It is not parameter-for-parameter substituted with the
0_demo implementation and does not receive CQ, FiLM, measurement/support,
EMA, persistent Top-K, or cached-K/V features. The actual parameter count and
diagnostic evidence are recorded in `A_performance.json`; final timings and
memory are added after the successful run.

## Evidence

- Config: `../configs/A_legacy_gl_rbf_enh_200ep.yaml`
- Protocol: `../PROTOCOL.yaml`
- Fixed manifest: `../fixed_validation_manifest.json`
- Diagnostic evidence: `A_performance.json` and `A_metrics.csv`

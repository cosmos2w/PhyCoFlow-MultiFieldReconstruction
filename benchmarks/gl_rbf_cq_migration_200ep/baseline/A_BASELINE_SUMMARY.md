# Arm A — corrected legacy GL_rbf_ENH baseline

## Freeze status

This baseline is frozen before any portable `GL_rbf_CQ` migration. The fresh
seed-42 formal run at source HEAD `326c02129f6481781fe71456acb1e638d11830c3`
completed all 200 epochs and 40,000 optimizer steps. Its exact run directory is
`Cases/turbulent_combustion/runs/gl_rbf_cq_migration_200ep_A_legacy_gl_rbf_enh/20260825T054028Z_0c41ff4a`.
The exact B128/Q4096 diagnostic
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

The `2^-140` attempt was rejected after all gradients through step 290 were
zero. Short real-data B40/Q4096 calibrations found an early-step float32
boundary: `2^-130` was finite, while `2^-131` and `2^-132` underflowed. A
seed-42 run at fixed `2^-130` nevertheless overflowed at step 219, showing that
no observed fixed scale remained viable as gradient magnitude evolved.

Commit `35473774` tested an exact-RNG adaptive controller, but the corresponding
formal run exhausted every float32 scale through `2^-149` at step 366. This
proved end-of-loss scaling was treating a symptom rather than the cause.

The root cause was the downstream dataset config's identity normalization.
Pressure near `1.14e5` contributed about `1.30e10` per-field MSE, while raw
temperature entered the sensor projection and four attention reinjections.
The pressure-dominated loss stayed near `2.6e9` as the conditioning Jacobian
grew beyond float32 range. A matched initial-batch probe reduced loss from
`2.59578e9` to `2.13649` and global gradient norm from `2.19442e5` to `1.27095`
when the fields were standardized.

Commit `26258a68` adds a checksummed, downstream-owned normalizer artifact fit
only on chronological training frames `0:8000`. It verifies dataset
fingerprint, field order, method, training-split provenance, and checksum; the
shared HDF5 is unchanged. A fresh 500-step B40/Q4096 GPU-0 smoke passed the
prior failure horizon with ordinary scale-1 backward, loss falling from
`2.10978` to `1.029996`, finite gradient norms in `[0.07685, 22.91773]`, and
zero retries. Its truncated weights are not reused. This exact normalization
and full-scale backward protocol is shared by A/B/C, and the formal run again
starts from seed 42.

## Model and protocol

- Downstream model: `pointcloud_ffm` with `backbone: gl_rbf_enh`.
- Historical capacity corrections only: hidden/latent 256, 128 latents, 8
  heads, 4 latent blocks, top-k 32, field embedding 128, Fourier bands 32
  (max frequency 64), RFF features 256, RFF length scale 0.15.
- Dataset/field order: downstream turbulent combustion `[CH4, CO, T, U_1, p]`;
  coordinate dimension 2; downstream chronological split and verified
  training-only mean/std normalization (artifact `b7e31a14`, normalizer digest
  `50c5e65e`).
- Sensors: random-uniform T-only, 192--384 valid sensors, seed 42.
- Training: 200 epochs, batch size 40, query points 4096, lr 1e-4,
  weight decay 1e-6, gradient clip 1.0, deterministic seed 42, ordinary
  scale-1 backward.
- Training preview is disabled; immutable checkpoints are saved at epochs
  `1, 20, 40, 60, 100, 150, 200`, and fixed-manifest evaluation is post-hoc.

## Formal outcome

- Status: `completed`, 40,000/40,000 steps; 4,778,118 trainable parameters.
- Endpoint training loss: `0.3798768520`; endpoint integration normalized MSE:
  `0.3130363226`.
- Every one of the 40,000 recorded losses and gradient norms is finite.
  Backward scale stayed at `1.0`, adaptive scaling remained disabled, and the
  retry count was zero.
- Recorded loss range: `0.3039245903--2.1097817421`; gradient-norm range:
  `0.0768486428--41.5614932435`.
- Epochs 2--200 averaged `59.72598 s` wall and `59.55917 s` training-only;
  sampled steady-state steps averaged `0.297325 s`.
- Peak CUDA allocation/reservation: `41,202,117,632` / `49,595,547,648` bytes.
- Formal config source SHA-256: `9fb98efb8fafa56a58d358391e6bc8be6da0fd02ffee9f1ce24f9d22826d3996`;
  resolved config SHA-256: `2c0a4ba9ec061e1397b57861284e63a3c20c12f95345290665ba678a08b8e035`;
  dataset fingerprint: `8c49936567eced7ab94887c336b9b35aaf7ec70dea7479aee83434ff970455d5`.
- Fixed validation sensor-manifest digest:
  `2071583f79e30f17bc586d907da184b5c79dfc82c01b4d652ccf05652e2c2b6f`;
  fixed query-index SHA-256:
  `16c263270389ab8665563c1d6fdcab3c2f193df481eed2b2717b20dfe7b40a5a`.

Fixed-manifest convergence uses the same 20 validation samples, T-only sensor
pairs, Q4096 query indices, seed, and 32-step reconstruction at every milestone:

| Epoch | normalized MSE | mean relative L2 | worst-field relative L2 |
|---:|---:|---:|---:|
| 1 | 1.401128 | 1.157376 | 1.528529 |
| 20 | 0.708934 | 0.822769 | 1.084258 |
| 40 | 0.648933 | 0.785265 | 1.070286 |
| 60 | 0.651148 | 0.787882 | 1.108059 |
| 100 | 0.580767 | 0.744052 | 1.030048 |
| 150 | 0.471722 | 0.667964 | 0.932756 |
| 200 | 0.386913 | 0.604817 | 0.833840 |

The epoch-200 immutable checkpoint SHA-256 is
`75f5dd07a0212a6b894753be72c29e71ab8db1dfcc1c3b9241890b7b27e7c53c`.
Complete checkpoint, report, per-field relative-L2, timing, memory, and artifact
hash evidence is recorded in `A_performance.json`.

## Structural differences from historical 0_demo GL_rbf_ENH

The downstream `EnhancedGLRBFTopK` implementation remains the project-owned
legacy implementation. It is not parameter-for-parameter substituted with the
0_demo implementation and does not receive CQ, FiLM, measurement/support,
EMA, persistent Top-K, or cached-K/V features. The actual parameter count,
diagnostic evidence, final timings, memory, and fixed-manifest convergence are
recorded in `A_performance.json`.

## Evidence

- Config: `../configs/A_legacy_gl_rbf_enh_200ep.yaml`
- Protocol: `../PROTOCOL.yaml`
- Fixed manifest: `../fixed_validation_manifest.json`
- Normalization: `../downstream_train_normalization.json`
- Diagnostic evidence: `A_performance.json` and `A_metrics.csv`

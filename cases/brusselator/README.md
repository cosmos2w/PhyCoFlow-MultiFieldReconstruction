# Brusselator Case

The first formal multi-field benchmark reconstructs coupled periodic `u,v`
states, primarily inferring `v` from sparse `u` and then testing the reverse.

`configs/base/` contains one plain-data/native-loss configuration per planned
applicable adapter. Train `latent_fm_stage1.yaml` first, then pass its immutable
checkpoint to `latent_fm_stage2.yaml` with
`--override model.stage1_checkpoint=<path>`.

Base configs vary sparse `u` counts from 64 to 256 during training. The fixed
`u_only_random.yaml` protocol is retained for deterministic manifest-based
evaluation.

`configs/posttrain/global_distribution_reference.yaml` is the target-free
coherence template. It inherits dataset, model, and observation sections from
an explicit source run and fits a training-only reference bank. The separate
`global_distribution_paired.yaml` template is labeled paired-supervised for
controlled ablations; it must not be reported as target-free refinement.

`configs/posttrain/phase6_common.yaml` defines the matched target-free protocol
for all compatible Phase-4 adapters. The nine small `phase6_*.yaml` children
name immutable sources and output experiments. Their strict matrix and
one-step GPU1 integration report live in `comparisons/phase6_gpu1_matrix.yaml`
and `comparisons/phase6_gpu1_summary.{json,md}`. PINN and latent-FM stage 1 are
listed there with explicit capability reasons instead of synthetic results.

Phase 7 adds the verified periodic reaction-diffusion provider. Use
`configs/direct_physics/pinn.yaml` for direct physics-informed training and
`configs/posttrain/physics_periodic.yaml` for an immutable physics-refined
child. Both require complete-grid queries and paired finite-difference temporal
context. The frozen Phase-8 sensor selection is stored at
`../../benchmarks/v0_integration/brusselator_u128_validation.json`.

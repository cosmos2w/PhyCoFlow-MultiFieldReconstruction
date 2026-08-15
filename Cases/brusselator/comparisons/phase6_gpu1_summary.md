# Phase 6 comparison: phase6-global-distribution-gpu1

These one-step children verify matched cross-model post-training behavior; their metric changes are pipeline diagnostics, not tuned scientific results.

| Model | Family | Params (M) | MSE before | MSE after | MSE reduction | Coherence reduction | Time/step (s) | Inference (ms) | Peak GPU (MiB) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| coordinate_mlp | deterministic_point | 0.042 | 0.0208271 | 0.0209353 | -0.000108134 | 0.0011692 | 0.2442 | 0.451 | 63.4 |
| mlp_rbf | deterministic_point | 0.043 | 0.0191498 | 0.0190842 | 6.56247e-05 | 0.00157166 | 0.3115 | 1.045 | 63.5 |
| deeponet | deterministic_point | 0.073 | 0.0155179 | 0.015343 | 0.000174884 | 0.00631332 | 0.2088 | 0.551 | 59.3 |
| senseiver | deterministic_point | 0.617 | 0.0171017 | 0.0214757 | -0.00437406 | 0.0160522 | 0.2959 | 3.188 | 94.4 |
| geofno | deterministic_grid_operator | 0.603 | 0.0140054 | 0.0139827 | 2.26647e-05 | 0.00107193 | 0.4366 | 51.259 | 591.0 |
| diffusion_pde | grid_generative | 0.042 | 113305 | 111457 | 1847.92 | 5297.94 | 0.3521 | 5.178 | 429.4 |
| latent_fm_stage2 | latent_generative | 0.079 | 0.0173865 | 0.017408 | -2.15061e-05 | 0.00161982 | 0.3903 | 4.747 | 168.1 |
| pointcloud_ffm_fno | direct_rectified_flow | 0.269 | 0.717969 | 0.718192 | -0.000223279 | 0.000771999 | 0.4085 | 13.948 | 694.5 |
| pointcloud_ffm_gl_rbf_enh | direct_rectified_flow | 0.794 | 0.629183 | 0.610855 | 0.018328 | 0.133067 | 0.2204 | 7.111 | 619.7 |

## Unsupported combinations

- `pinn` — `no_plain_base_run`: PINN is intentionally restricted to direct_physics with a verified case PhysicsProvider, so Phase 4 has no ordinary data-loss source checkpoint.
- `latent_fm_stage1` — `not_inference_model`: Stage 1 is the autoencoder prerequisite for latent_fm stage 2, not a deployable sparse-reconstruction model; the stage-2 checkpoint is paired above.

## Shared protocol

```json
{
  "cuda_device": "NVIDIA RTX 6000 Ada Generation",
  "cuda_visible_devices": "1",
  "dataset_fingerprint": "460e6e3c6209e0a002c02620ba17522c5b550c67fb808f8694d95171acaaa6cb",
  "evaluation_seed": 2027,
  "evaluation_split": "validation",
  "query_indices_sha256": "430740a692cd59e7041421830b768a30707aef2a7f772c520e97d97bad650a6f",
  "reference_bank_sha256": "ec0feb28a77b77b275af7f2d2fcb72b6d86f18bc236ef1730166a24d96ec6160",
  "sensor_manifest_sha256": "9d8537ec41395cab2594798a99f7c5ae2c781ca7790813613e1ecf46f39c9711",
  "target_use": "training_reference",
  "training_seed": 42
}
```

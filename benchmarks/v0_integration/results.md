# Phase 8 integration release

Matched one-update Brusselator checks on one validation trajectory. These rows demonstrate stage wiring, metric aggregation, and traceability; they are not tuned scientific performance estimates.

| Ablation | Stage | Updates | MSE | PDE-u | PDE-v | Time (ms) |
|---|---|---:|---:|---:|---:|---:|
| plain_coordinate_mlp | base_training | 1 | 0.0228697 | 1.44229 | 1.02075 | 9.491 |
| global_distribution_posttrain | post_training_data_coherence | 1 | 0.0229844 | 1.4437 | 1.02138 | 1.884 |
| periodic_physics_posttrain | post_training_pde | 1 | 0.0228188 | 1.39238 | 1.02025 | 4.591 |
| direct_pinn | direct_physics | 1 | 0.0283017 | 2.08998 | 1.45664 | 1.567 |

All rows use one validation trajectory and one optimizer update; standard errors are therefore intentionally unavailable. This table proves pipeline comparability only.

# Upstream References and Provenance

This project uses upstream software in one of three explicit ways: a pinned
library dependency, a clean-room local implementation guided by publications
and public interfaces, or a version-locked extraction from this repository's
own historical demo. No upstream repository is vendored as a monolith.

The repository revisions below were reviewed on 2026-08-14. A future upgrade
must update this file and rerun model-contract and real-case integration checks.

| Reference | Reviewed revision | License | Use in this project |
|---|---|---|---|
| [NeuralOperator](https://github.com/neuraloperator/neuraloperator) | `00b7d86f8d74ff0af55da53eb585fe26df9c71f0`; package `2.0.0` | MIT | Pinned runtime dependency; `GeoFNORegressor` and the PointCloudFFM FNO backbone wrap `neuralop.models.FNO`. |
| [Senseiver](https://github.com/OrchardLANL/Senseiver) | `e443eb0ca188b5e83a0e6ce5afb8db7d90393de1` | BSD-3-Clause | Architectural/paper reference only. The local sparse-attention adapter is independently organized around the shared project contract. |
| [DiffusionPDE](https://github.com/jhhuangchloe/DiffusionPDE) | `1e2bc8b9e312f3a936630a30d2f49aedabf0cea7` | CC BY-NC-SA 4.0 | Conceptual partial-observation reference only. No source was copied; the compact value/mask-conditioned denoiser is a clean local implementation. |
| [DeepXDE](https://github.com/lululxvi/deepxde) | `91bda9aafb8b1c1ea7e932c232ad4880a088f7f3` | LGPL-2.1 | Conceptual API reference for PINN and DeepONet. It is not a dependency and no code was copied. |
| [Meta Flow Matching](https://github.com/facebookresearch/flow_matching) | `11568d37f8d5a080e12aa7b5305d9c35ae07d136` | CC BY-NC 4.0 | Mathematical reference only. The local 1-RF interpolation, Euler integration, and losses are independently implemented. |

## Repository-local compatibility source

`models/compatibility/legacy_tc_demo50.py` is a focused extraction of the
architecture and RF/RFF behavior required by
`0_demo_TurbulentCombustion/Save_TrainedModel/ffm_tc_pointcloud_DemoN50_20260706_084857`.
It has no runtime import from the demo, is absent from the new-model registry,
and is accepted only after strict key/shape loading and fixed-seed equivalence
against the repository-local historical implementation. `pykeops==2.3` is
pinned solely for that checkpoint's original neighbor-search path.

The Phase-5 `global_distribution` estimators and endpoint-consistency behavior
are focused, typed refactors of repository-local
`0_demo_TurbulentCombustion/src/{coherence_dist.py,direct_coherence_loss.py,obs_consistency.py}`.
They have no runtime import from the demo. A compatibility test compares every
refactored component numerically with the historical implementation.

`conflictfree==0.1.8` is an optional post-training dependency used only when
`optimization.gradient_balance: config` is selected. Weighted-sum training
does not import it.

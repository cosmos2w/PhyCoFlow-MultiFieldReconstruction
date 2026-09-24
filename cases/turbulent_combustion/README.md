# Turbulent Combustion Case

Sparse measurable combustion fields condition reconstruction of all five fields. The single long trajectory uses chronological frames 0–7999 for train, 8000–8999 for validation, and 9000–9999 for test.

The legacy clock resets at frame 4000; sample identity and splitting therefore use saved frame indices while retaining the raw clock as provenance.

New PointCloudFFM runs use `GL_rbf_ENH/topk_rbf`; Demo 50 uses the isolated legacy compatibility path with explicit `CO,T,U_0,U_1,p` field mapping. Run `python import_demo50.py` here to perform the strict non-destructive import and write its local compatibility manifest.

`configs/posttrain/demo50_global_distribution.yaml` maps the historical flat direct-coherence settings into the structured Phase-5 schema. It preserves the paired-supervised self/mutual/cross estimators, two-step clean rollout, endpoint-smooth observation consistency, data-retention loss, and optional ConFIG update while writing only to a new child run. The compatibility dataset uses the checkpoint's actual positional fields `CO,T,U_0,U_1,p` and explicitly retains the stored constant third coordinate required by Demo50's encoders.

The canonical loader verifies 403 unique x positions, 100 unique y positions, one constant z coordinate, and all 40,300 unique `(x,y)` pairs. It reorders the stored permutation to ascending y then x, giving logical shape `(100,403)`. New base and training-reference post-training templates live in `configs/base/` and `configs/posttrain/`; the compatibility configuration remains separate.

## Sliced-persistence readiness profiles

`configs/readiness/C_sliced_persistence_smoke.yaml` is a one-epoch, topology-only smoke profile with a 0.1% training fraction and batch size 1. `configs/readiness/ABC_sliced_persistence_50ep_gpu1.yaml` preserves the `AB_balanced.yaml` global-distribution and cross-spectrum settings, adds nonperiodic CO/T cubical persistence with sliced Wasserstein distance, and runs exactly 50 epochs on GPU 1 with a 1% training fraction and batch size 8. It selects checkpoints every five epochs using topology with 5% total and per-field source-relative MSE budgets, logs validation loss every five epochs, and renders a reconstruction preview at epoch 50. Native-grid topology evaluation is disabled because this profile evaluates a reduced 4096-point query set. Both profiles pin the original GL-RBF/CQ source run to `last.pt` and use live evaluation weights for persistence rollout agreement; A+B explicitly keep `self_spectrum` disabled.

`configs/readiness/ABC_sliced_persistence_ABscale_50ep_gpu1.yaml` repeats the 50-epoch A+B+C test with the completed A+B run's batch size 32, training fraction 0.25, 32-sample validation panel, and reconstruction-preview interval of 200 epochs. It keeps the A+B learning rate, data/coherence weights, and family calibration. This is a formal-scale test profile, not a 5000-epoch launch configuration.

Validate from the repository root:

```bash
python cases/turbulent_combustion/run.py validate --config configs/readiness/C_sliced_persistence_smoke.yaml
python cases/turbulent_combustion/run.py validate --config configs/readiness/ABC_sliced_persistence_50ep_gpu1.yaml
python cases/turbulent_combustion/run.py validate --config configs/readiness/ABC_sliced_persistence_ABscale_50ep_gpu1.yaml
```

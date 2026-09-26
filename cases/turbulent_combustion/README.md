# Turbulent Combustion Case

Sparse measurable combustion fields condition reconstruction of all five fields. The single long trajectory uses chronological frames 0–7999 for train, 8000–8999 for validation, and 9000–9999 for test.

The [diagnostic figure gallery](diagnostics/README.md) defines the roles,
comparison rules, and reviewable examples for training, reconstruction,
distribution, spectral, and topology figures.

The legacy clock resets at frame 4000; sample identity and splitting therefore use saved frame indices while retaining the raw clock as provenance.

New PointCloudFFM runs use `GL_rbf_ENH/topk_rbf`; Demo 50 uses the isolated legacy compatibility path with explicit `CO,T,U_0,U_1,p` field mapping. Run `python import_demo50.py` here to perform the strict non-destructive import and write its local compatibility manifest.

`configs/posttrain/demo50_global_distribution.yaml` maps the historical flat direct-coherence settings into the structured Phase-5 schema. It preserves the paired-supervised self/mutual/cross estimators, two-step clean rollout, endpoint-smooth observation consistency, data-retention loss, and optional ConFIG update while writing only to a new child run. The compatibility dataset uses the checkpoint's actual positional fields `CO,T,U_0,U_1,p` and explicitly retains the stored constant third coordinate required by Demo50's encoders.

The canonical loader verifies 403 unique x positions, 100 unique y positions, one constant z coordinate, and all 40,300 unique `(x,y)` pairs. It reorders the stored permutation to ascending y then x, giving logical shape `(100,403)`. New base and training-reference post-training templates live in `configs/base/` and `configs/posttrain/`; the compatibility configuration remains separate.

## MIMONet baseline

`configs/base/mimonet_5000ep.yaml` adds the released MIMONet branch--trunk operator as a five-field deterministic baseline. It uses exactly 256 random temperature sensors: the value and location of each sensor enter the two branches, while the trunk receives query coordinates and predicts all five fields. The released 256-dimensional basis, multiplicative fusion, ReLU activations, 512-wide branch layers, and 256-wide trunk layers are retained. Its configured training and validation query count is 4,096. The repository trainer uses AdamW at a fixed $10^{-4}$ learning rate with $10^{-6}$ weight decay; it does not use the demo wrapper's Adam plus cosine schedule.

The 256-sensor input is the specified information budget for this profile. Existing baseline configs should be checked individually before claiming a matched sensor budget, because some use a 192--384 sensor range. Dataset and normalization assets remain local and are configured through the existing case contract.

```bash
python cases/turbulent_combustion/run.py validate \
  --config configs/base/mimonet_5000ep.yaml
python cases/turbulent_combustion/run.py train-base \
  --config configs/base/mimonet_5000ep.yaml \
  --override runtime.device=cuda:1
```

The architecture and source citation are documented in [docs/models.md](../../docs/models.md#34-mimonet-sparse-input-operator); its paper is Kobayashi et al., *Nature Communications* (2026), [10.1038/s41467-026-77463-7](https://doi.org/10.1038/s41467-026-77463-7), with code release [10.5281/zenodo.21986357](https://doi.org/10.5281/zenodo.21986357).

## Sliced-persistence readiness profiles

`configs/readiness/C_sliced_persistence_smoke.yaml` is a one-epoch, topology-only smoke profile with a 0.1% training fraction and batch size 1. `configs/readiness/ABC_sliced_persistence_50ep_gpu1.yaml` preserves the `AB_balanced.yaml` global-distribution and cross-spectrum settings, adds nonperiodic CO/T cubical persistence with sliced Wasserstein distance, and runs exactly 50 epochs on GPU 1 with a 1% training fraction and batch size 8. It selects checkpoints every five epochs using topology with 5% total and per-field source-relative MSE budgets, logs validation loss every five epochs, and renders a reconstruction preview at epoch 50. Native-grid topology evaluation is disabled because this profile evaluates a reduced 4096-point query set. Both profiles pin the original GL-RBF/CQ source run to `last.pt` and use live evaluation weights for persistence rollout agreement; A+B explicitly keep `self_spectrum` disabled.

`configs/readiness/ABC_sliced_persistence_ABscale_50ep_gpu1.yaml` repeats the 50-epoch A+B+C test with the completed A+B run's batch size 32, training fraction 0.25, 32-sample validation panel, and reconstruction-preview interval of 200 epochs. It keeps the A+B learning rate, data/coherence weights, and family calibration. This is a formal-scale test profile, not a 5000-epoch launch configuration.

`configs/readiness/ABC_sliced_persistence_formal_5000ep_gpu1.yaml` is the fresh A+B+C formal profile: 5000 configured epochs, optimizer and coherence batch size 32, and training fraction 0.15 on GPU 1. This gives 38 updates and 1216 coherence sample presentations per reported epoch; the superseded batch-16 attempt gave 75 updates and 1200 presentations. It retains the source `last.pt`, topology-with-fidelity checkpoint selection, 32-sample validation panel, and 200-epoch reconstruction interval. Earlier A+B+C test runs and the superseded batch-16 formal attempt are archived under `runs/bk/`.

Cubical-persistence post-training uses up to four CPU workers for independent GUDHI pairings by default; `PHYCOFLOW_TOPOLOGY_WORKERS` overrides this count. The AB-scale continuation from step 660 uses `PHYCOFLOW_TOPOLOGY_REFERENCE_CACHE=24000`; from the epoch-25 checkpoint at step 1575 it uses `PHYCOFLOW_TOPOLOGY_REFERENCE_CACHE=100000` so the 8000-frame training set's ten reference filtrations per frame can remain cached. These settings only affect execution; the persistence definition, diagrams, and gradients are unchanged. Pair extraction remains on the CPU, while sliced diagram distances run on the configured PyTorch device.

Validate from the repository root:

```bash
python cases/turbulent_combustion/run.py validate --config configs/readiness/C_sliced_persistence_smoke.yaml
python cases/turbulent_combustion/run.py validate --config configs/readiness/ABC_sliced_persistence_50ep_gpu1.yaml
python cases/turbulent_combustion/run.py validate --config configs/readiness/ABC_sliced_persistence_ABscale_50ep_gpu1.yaml
python cases/turbulent_combustion/run.py validate --config configs/readiness/ABC_sliced_persistence_formal_5000ep_gpu1.yaml
```

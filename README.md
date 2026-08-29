# PhyCoFlow Multi-Field Reconstruction

PhyCoFlow reconstructs complete physical states from sparse, multi-field measurements. The primary research workflow studies whether data-driven physical-coherence post-training improves the coherence of a reconstructed state while preserving the immutable source checkpoint and the supervised data contract.

The standard lifecycle is:

```text
dataset contract → case + sensors → shared model → base checkpoint
                                               ↓
                              coherence post-training → evaluation
```

Physics-informed training and historical compatibility routes are available, but remain explicit alternatives to this primary workflow.

## Start here

Commands below run from this repository root. Install the shared package in a virtual environment or conda environment, then validate whichever local data payloads are present:

```bash
conda env create -f environment.yml
conda activate phycoflow_reconstruction
python -m pip install -e '.[dev]'
python scripts/data/validate_dataset.py --all
```

Datasets are intentionally local. Link a payload into the canonical catalog without copying it:

```bash
python scripts/data/link_dataset.py \
  --case brusselator \
  --source /absolute/path/to/brusselator.h5
```

The optional `operator` extra provides `neuraloperator` for GeoFNO and the FNO PointCloudFFM backbone. The `posttrain` extra provides ConFIG gradient balancing, and `legacy` provides the optional Demo50 neighbor-search path.

## Repository architecture

The reusable implementation is under `src/phycoflow_reconstruction/`:

- `contracts.py` defines dataset, observation, reconstruction, loss, capability, coherence, and physics boundaries;
- `data/` handles payload adapters, normalization, splits, manifests, and sensor protocols;
- `models/` contains deterministic, generative, operator, flow, and isolated historical-compatibility adapters;
- `coherence/` contains reference banks, family composition, observation consistency, and the global-distribution, cross-spectrum, and topology families;
- `training/` owns base training, coherence/physics post-training, direct physics, checkpoints, rollout, previews, and monitoring;
- `evaluation/`, `physics/`, `config/`, and `cli.py` provide shared evaluation, case-independent physics interfaces, config composition, and command routing.

The point-cloud flow package keeps the low-level tensor core separate from the project adapter:

```text
models/flows/pointcloud/core/       model math, geometry, attention, priors,
                                   tensor training/reconstruction primitives
models/flows/pointcloud/runtime/   builder, EMA, checkpoint and tensor runtime
models/flows/pointcloud/adapters/  project contracts, lifecycle, registry glue
```

See [docs/architecture.md](docs/architecture.md) for dependency direction and [docs/models.md](docs/models.md) for model capabilities and stages.

Each `cases/<case>/` directory owns only scientific meaning and launch profiles: field metadata, physical diagnostics/providers, dataset selection, sensor protocols, coherence/physics settings, and a thin `run.py`. Generic models and trainers never import a named case.

Generic model fragments live once in `configs/models/`; shared runtime, optimization, evaluation, and checkpoint defaults live in `configs/defaults/`. Case launch profiles compose those fragments and carry only scientifically meaningful overrides. Configuration ownership and composition rules are described in [docs/configuration.md](docs/configuration.md).

## Dataset contract

`datasets/` is a local catalog, not a place to commit normal payloads. Track the schema and reproducibility instructions, while keeping HDF5/PT/NumPy payloads, links, and derived caches local. The canonical contract requires a dense field state, coordinates, time/condition metadata, field order, logical shape, and declared train/validation/test split semantics. See:

- [datasets/README.md](datasets/README.md) for the catalog;
- [datasets/SCHEMA.md](datasets/SCHEMA.md) for the accepted HDF5/PT structure;
- [docs/reproducibility.md](docs/reproducibility.md) for normalization, lineage, and release rules.

Validate an individual payload or all known catalog entries:

```bash
python scripts/data/validate_dataset.py datasets/brusselator/brusselator.h5
python scripts/data/validate_dataset.py --all
```

Missing optional local payloads are reported by validation; they are never fabricated by the repository.

## Case workflow

Choose a case and inspect its README, dataset config, sensor protocol, and base launch profiles:

```bash
cd cases/brusselator
python run.py validate --config configs/dataset.yaml
```

Sensor definitions live under `cases/<case>/configs/sensors/`. Build a fixed manifest when comparing runs:

```bash
python run.py build-manifest \
  --config configs/base/coordinate_mlp.yaml \
  --split validation --max-samples 8 \
  --output manifests/validation_sensors.json
```

Manifests are local/generated and should not be committed except for a small, immutable benchmark fixture explicitly covered by a contract.

## Select and train a model

The registry preserves these public model names:

`coordinate_mlp`, `mlp_rbf`, `pinn`, `deeponet`, `senseiver`, `geofno`, `diffusion_pde`, `latent_fm`, `pointcloud_ffm`, and `gl_rbf_cq`.

Run a short base-training smoke from a case directory:

```bash
python run.py train-base \
  --config configs/base/coordinate_mlp.yaml \
  --override runtime.device=cpu \
  --max-steps 1
```

Normal training omits `--max-steps`. Point models consume sparse observation tokens; grid/operator models rasterize observations and their support mask. Diffusion and flow models retain their native noise/velocity objectives.

DiffusionPDE supports two interchangeable denoising backbones. The maintained `configs/models/diffusion_pde.yaml` profile selects a time-conditioned, multiscale U-Net with configurable channel multipliers, residual depth, timestep embedding, coarse-level attention, attention heads, and dropout. The original three-convolution implementation remains available as the lightweight, checkpoint-compatible `plain_cnn` option:

```bash
# Maintained conditional U-Net profile.
python run.py train-base --config configs/base/diffusion_pde.yaml

# Select the legacy plain CNN without editing the shared model fragment.
python run.py train-base --config configs/base/diffusion_pde.yaml \
  --override model.backbone=plain_cnn
```

Both backbones use the same cosine noise schedule, noise-prediction loss, and deterministic DDIM-style reconstruction with observed values clamped after each sampling update. They require complete two-dimensional target grids. The U-Net uses considerably more accelerator memory, especially when attention is enabled; set the case-level training batch size accordingly.

Latent flow has an explicit two-stage lifecycle:

```bash
python run.py train-base --config configs/base/latent_fm_stage1.yaml
python run.py train-base --config configs/base/latent_fm_stage2.yaml \
  --override model.stage1_checkpoint=runs/<stage1>/<run-id>/checkpoints/best.pt
```

Stage 2 strictly loads and freezes the Stage-1 autoencoder. It is the sparse reconstruction source; Stage 1 is a prerequisite checkpoint only.

## Coherence post-training

Post-training creates a child run from a completed, immutable base run. The source checkpoint, dataset, model, and observations are loaded as a verified lineage; the source run is never modified:

```bash
python run.py post-train \
  --config configs/posttrain/global_distribution_reference.yaml \
  --override source_run=runs/<base-experiment>/<run-id>
```

Available coherence families are `global_distribution`, `cross_spectrum`, and `topology`; a declared composition can run multiple families over one differentiable reconstruction. Reference banks are fit from the training split, and paired-supervised mode is labeled as such. Use a fixed query policy for geometry-based families and keep sensor manifests matched across runs.

The cleaned GL-RBF/CQ path supports the same lifecycle while preserving its state-dict keys, seeded behavior, cached-K/V execution, query microbatching, geometry/reconstruction caches, EMA state, and observation consistency.

For a physics post-training route, use a case that exposes a differentiable `PhysicsProvider` (currently Brusselator):

```bash
python run.py post-train \
  --config configs/posttrain/physics_periodic.yaml \
  --override source_run=runs/<base-experiment>/<run-id>
```

Direct PINN training is a separate route:

```bash
python run.py train-direct --config configs/direct_physics/pinn.yaml
```

## Evaluation and local outputs

Evaluate any base or child run with a sensor config or fixed manifest:

```bash
python run.py evaluate-run \
  --run runs/<experiment>/<run-id> \
  --checkpoint best \
  --sensor-config configs/sensors/u_only_random.yaml \
  --split validation --max-samples 8 \
  --report-name validation
```

The evaluator records normalized/physical errors, observed/unobserved metrics, sample/query/sensor identities, diagnostics, timing, and provenance. Generated checkpoints, manifests, reports, previews, figures, caches, and histories stay under `cases/<case>/runs/` and are ignored by Git.

### Single-snapshot reconstruction figure

From the repository root, the simple form is:

```bash
python cases/<case>/run.py visualize-run --run runs/<experiment>/<run-id>
```

With no optional arguments, this loads `best.pt`, selects snapshot `0` relative to the test split, and uses the sparse observation protocol and generation settings in `resolved_config.yaml`. It always reconstructs the complete grid and writes a 300-DPI PNG.

The fully explicit command used for the Senseiver example below is:

```bash
python cases/turbulent_combustion/run.py visualize-run \
  --run runs/tc_senseiver_5000ep/20260828T190145Z_fea0fc25 \
  --checkpoint last \
  --split test \
  --snapshot-index 0 \
  --generation-steps 4 \
  --device cuda:2 \
  --contour-levels 20 \
  --weight-selection configured
```

- `--snapshot-index` is relative to the selected split.
- Omit `--sensor-config` and `--sensor-manifest` to use the observation protocol recorded by the run. Supplying either option overrides or replays that protocol.
- `--contour-levels` controls both filled and thin grey contours. Colorbars are continuous with four labeled ticks.
- Figure size, spacing, and text scale adapt to the physical-domain aspect ratio and field count.
- A CUDA-memory warning appears before inference when the selected device may be tight; choose a different device with `--device`.

Outputs are stored under `evaluation/reconstruction_<split>_<snapshot>_<checkpoint>/`: `reconstruction.png`, `report.json`, `sensor_manifest.json`, `query_indices.pt`, and the portable plotting payload `reconstruction.npz`. The former duplicate `reconstruction.pt` is no longer written.

![Senseiver turbulent-combustion reconstruction on the first test snapshot](docs/assets/reconstruction_examples/senseiver_test_snapshot_0000_last.png)

### Multi-snapshot reconstruction statistics

Add `--eval-set` to render each field's physical-space relative $L_2$ distribution as a violin plot overlaid with individual sample points. The quick form is:

```bash
python cases/<case>/run.py visualize-run \
  --run runs/<experiment>/<run-id> --eval-set test
```

The fully explicit command used for the Senseiver example below is:

```bash
python cases/turbulent_combustion/run.py visualize-run \
  --run runs/tc_senseiver_5000ep/20260828T190145Z_fea0fc25 \
  --checkpoint best \
  --eval-set test \
  --eval-samples 200 \
  --generation-steps 4 \
  --device cuda:2 \
  --weight-selection configured
```

`--eval-set` accepts `train`, `validation`, or `test`. The default limit is 200 deterministic, evenly spaced samples across that split; request more with `--eval-samples 500` or the complete split with `--eval-samples all`. Statistical distribution plots use a logarithmic vertical scale by default; select the normal linear scale with `--stat-scale linear`. For snapshot datasets each sample is one snapshot; trajectory datasets are evaluated one trajectory at a time. The model and checkpoint load once while full-grid samples stream individually to bound memory.

Outputs are written under `evaluation/reconstruction_set_<split>_<checkpoint>/` and include the 300-DPI violin/scatter figure, summary report, per-sample CSV, reusable NPZ metrics, and streaming sensor manifest.

<img src="docs/assets/reconstruction_examples/senseiver_test_best_relative_l2_violin.png" alt="Senseiver test-set relative L2 distributions" width="65%">

### Multi-snapshot physical-coherence statistics

Add `--eval-coherence` to the same set evaluation to calculate paired reconstruction-to-ground-truth coherence metrics alongside the default relative-$L_2$ results. The shortest command for both currently supported statistical coherence families is:

```bash
python cases/<case>/run.py visualize-run \
  --run runs/<experiment>/<run-id> \
  --eval-set test \
  --eval-coherence global_distribution cross_spectrum
```

Use one family name when only that evaluation is needed. `--eval-coherence` requires `--eval-set`, accepts `global_distribution`, `cross_spectrum`, or both, and writes each family into `evaluation/reconstruction_set_<split>_<checkpoint>/coherence/<family>/` so figures and numerical artifacts remain separated.

#### Global-distribution coherence

The fully explicit global-distribution command for the Senseiver example is:

```bash
python cases/turbulent_combustion/run.py visualize-run \
  --run runs/tc_senseiver_5000ep/20260828T190145Z_fea0fc25 \
  --checkpoint best \
  --eval-set test \
  --eval-samples 200 \
  --eval-coherence global_distribution \
  --stat-scale log \
  --generation-steps 4 \
  --device cuda:2 \
  --weight-selection configured
```

Global-distribution evaluation is calculated per snapshot and produces separate violin/scatter figures for marginal field distributions, pairwise field distributions, and joint top-tail distributions. Each figure includes its sub-terms, weighted component total, and family total. Use `--stat-scale linear` when a normal vertical scale is preferred; the default is logarithmic. The family directory also contains `metrics.csv`, `metrics.npz`, and `report.json`.

<img src="docs/assets/reconstruction_examples/senseiver_test_best_global_distribution_marginal.png" alt="Senseiver test-set marginal field-distribution coherence discrepancies" width="72%">

#### Cross-spectrum coherence

The fully explicit cross-spectrum command for the same checkpoint is:

```bash
python cases/turbulent_combustion/run.py visualize-run \
  --run runs/tc_senseiver_5000ep/20260828T190145Z_fea0fc25 \
  --checkpoint best \
  --eval-set test \
  --eval-samples 200 \
  --eval-coherence cross_spectrum \
  --generation-steps 4 \
  --device cuda:2 \
  --weight-selection configured
```

Cross-spectrum evaluation pools all selected snapshots into one ensemble while retaining only compact graph-Fourier coefficients. The modern horizontal-bar figures report same-frequency and cross-frequency field-pair scores on a fixed linear range from 0 to 1, where 1 means exact spectral agreement; `--stat-scale` therefore does not alter these score charts. Raw mean-squared spectral discrepancies and normalized coherence scores are both retained in `metrics.csv`, `metrics.npz`, and `report.json`. If spectral-band energy is enabled in the run's coherence configuration, its figure is generated in the same family directory.

<img src="docs/assets/reconstruction_examples/senseiver_test_best_cross_spectrum_same_frequency.png" alt="Senseiver test-set same-frequency spectral coherence scores" width="72%">

> **Pending topology evaluation:** the topology coherence family is available for training and post-training objectives, but its intuitive set-level statistical evaluation and visualization are intentionally pending. That work will be refined as a separate task before `topology` is exposed through `visualize-run --eval-coherence`.

During training, the fixed validation objective and qualitative reconstruction use independent `evaluation.preview.loss_every_epochs` and `reconstruct_every_epochs` cadences. Validation loss is added to `loss_history.png` and selects `best.pt`; periodic recovery writes only `last.pt`, plus explicitly requested epoch checkpoints. Re-render a portable preview payload with:

```bash
python scripts/visualization/training_reconstruction_preview.py \
  --payload cases/<case>/runs/<experiment>/<run-id>/evaluation/training_preview/latest_reconstruction.npz
```

Run the complete local model smoke matrix on GPU 0 when available (or pass `--device cpu`):

```bash
python scripts/smoke/models.py --device cuda:0
```

This matrix uses tiny synthetic inputs, one loss/backward/update, and one reconstruction step. It is not a performance benchmark and is not required by cloud CI.

## Benchmarks and reproducibility

`benchmarks/` tracks protocols, canonical configs, source scripts, and only small immutable fixtures or concise validation summaries. Routine telemetry, HTML/JSON/CSV reports, large manifests, plots, and run summaries are generated locally and ignored. Use [benchmarks/README.md](benchmarks/README.md) for the reproduction contract and [docs/provenance.md](docs/provenance.md) for pinned upstream references and compatibility provenance.

The one-step integration workflow is available as:

```bash
bash scripts/smoke/reproduce_brusselator_integration.sh
```

## Contributing

Start with [CONTRIBUTING.md](CONTRIBUTING.md), then read the architecture and configuration docs before adding code. Work on a task branch, keep datasets and runs local, add contract tests for changed boundaries, and open a PR to `main`. The local acceptance checks are:

```bash
ruff check src tests scripts cases benchmarks
pytest
```

GPU smoke, long experiments, and formal benchmark regeneration are local acceptance activities rather than mandatory GitHub CI jobs.

# Multi-Field Reconstruction

This project reconstructs complete physical fields from partial, sparse
measurements. It provides a common data/model/training pipeline while keeping
field definitions, sensor layouts, physical diagnostics, equation losses, and
working configs inside each case.

The main operational routes are:

```text
validated dataset + sensor protocol
              |
              v
       supervised base model -------------------------+
              |                                       |
              v                                       v
       checkpoint evaluation            immutable source checkpoint
                                                      |
                                      +---------------+---------------+
                                      |                               |
                                      v                               v
                            coherence post-training         physics post-training
                                      |                               |
                                      +---------------+---------------+
                                                      |
                                                      v
                                             checkpoint evaluation

validated dataset + case PhysicsProvider
              |
              v
       direct physics-informed training
              |
              v
       checkpoint evaluation
```

Data-driven coherence and physics post-training are separate child-run modes.
The source checkpoint remains unchanged.

## Setup

From the project root:

```bash
cd Proj_MultiFieldReconstruction
conda env create -f environment.yml
conda activate phycoflow_reconstruction
python -m pip install -e .
python scripts/validate_dataset.py --all
```

If a dataset is stored elsewhere, link it instead of copying it:

```bash
python scripts/link_dataset.py \
  --case brusselator \
  --source /absolute/path/to/brusselator.h5
```

An existing compatible environment can use:

```bash
python -m pip install -e . --no-deps
```

GeoFNO and the FNO PointCloudFFM backbone require the optional maintained
`neuraloperator` dependency. ConFIG gradient balancing requires the optional
`conflictfree` package.

## Code and configuration organization

### Reusable package

`src/phycoflow_reconstruction/` contains the case-independent implementation:

- `contracts.py`: dataset, sparse observation, reconstruction, loss, model
  capability, coherence, and physics interfaces;
- `data/`: HDF5/data adapters, normalization, sensor protocols, manifests,
  rasterization, and training-batch sources;
- `models/`: deterministic point models, GeoFNO, diffusion, latent flow, and
  PointCloudFFM;
- `coherence/`: reference banks, family composition, observation consistency,
  global distribution, cross-spectrum, and topology;
- `training/`: base, coherence post-training, physics post-training, direct
  physics training, checkpointing, previews, and gradient balancing;
- `evaluation/`: common reconstruction metrics, case diagnostics, traceable
  checkpoint evaluation, and post-training comparison utilities;
- `config/` and `cli.py`: recursive YAML composition, validation, dotted
  overrides, and the command dispatcher used by every case.

The top-level `configs/defaults/` directory contains reusable runtime, logging,
and evaluation fragments. `configs/schema/` documents the three training
contracts: `base_training`, `post_training`, and `direct_physics`.

### Case-owned layer

Each `Cases/<case>/` directory contains:

- `case.py`: registered field order, units, mesh, logical shape, reconstruction
  unit, and optional physics/diagnostics factories;
- `run.py`: a thin launcher that binds the shared CLI to that case;
- `configs/dataset.yaml`: dataset path, split policy, normalization, coordinate
  interpretation, and field metadata;
- `configs/sensors/*.yaml`: observed fields, counts or count ranges, sharing,
  stride, and seed;
- `configs/base/*.yaml`: one model definition layered over common base defaults;
- `configs/coherence/*.yaml`: scientific definitions and weights for individual
  coherence families;
- `configs/posttrain/*.yaml`: child-run objectives, rollout, trainable scope,
  evaluation, and source checkpoint settings;
- `configs/direct_physics/*.yaml`: direct equation-informed training when the
  case supplies a differentiable `PhysicsProvider`;
- `diagnostics.py` or `physics.py`: physical calculations that belong to the
  case rather than the reusable package;
- ignored `runs/`: resolved configs, manifests, checkpoints, histories,
  evaluation reports, and figures.

All commands should be launched from the selected case directory. Relative
dataset, reference-bank, and source paths are resolved there.

## Configuration composition

A config can recursively include defaults:

```yaml
defaults:
  - ../dataset.yaml
  - ../sensors/u_only_random.yaml
  - ../coherence/global_distribution_reference.yaml

stage: post_training
case: brusselator
source_run: null
source_checkpoint: last.pt
```

Files are merged in order, then the current file overrides inherited values.
CLI overrides use dotted keys and YAML value parsing:

```bash
python run.py train-base \
  --config configs/base/pointcloud_ffm.yaml \
  --override optimization.epochs=300 \
  --override runtime.device=cuda:1
```

For a native post-training run, `inherit_base_config: true` reloads `dataset`,
`model`, and `observations` from the source run's `resolved_config.yaml`. The
child config therefore controls only refinement-specific settings while the
source architecture and data contract remain exact. Config validation rejects
unknown keys, incompatible stages, invalid geometry requirements, missing
source checkpoints, and a post-training config that mixes coherence with
physics.

## Supported model families

| Config name | Representation | Base objective | Main requirements |
|---|---|---|---|
| `coordinate_mlp` | query points | masked field MSE | arbitrary query points |
| `mlp_rbf` | query points with local RBF features | masked field MSE | sensor/query coordinates |
| `deeponet` | token branch and coordinate trunk | masked field MSE | arbitrary query points |
| `senseiver` | latent attention | masked field MSE | arbitrary query points |
| `geofno` | regular grid | masked field MSE | complete 1-D/2-D grid, `neuraloperator` |
| `diffusion_pde` | regular grid | noise-prediction MSE | complete 2-D grid |
| `latent_fm` | regular latent grid | autoencoder MSE, then latent flow MSE | complete 2-D grid and Stage-1 checkpoint |
| `pointcloud_ffm` | point or FNO flow | rectified-flow velocity MSE | differentiable sampling; complete grid for FNO |
| `pinn` | query points | dense data MSE plus case PDE loss | `direct_physics` and active case provider |

See [ModelExplain.md](ModelExplain.md) for the complete equations, conditioning
paths, post-training objectives, and current limitations.

## End-to-end working procedure

### 1. Validate the dataset

Start in the case directory and validate its catalog entry:

```bash
cd Cases/brusselator
python run.py validate --config configs/dataset.yaml
```

Validation should precede training because field order, logical shape,
coordinates, split availability, and stored statistics determine every later
checkpoint and physics calculation.

### 2. Select a sensor protocol

Choose a file under `configs/sensors/`. A sensor config determines which fields
are visible and how many positions are measured. Training offsets the protocol
seed by optimizer step; evaluation uses a fixed seed or a saved manifest.

For an auditable fixed evaluation set, build a manifest:

```bash
python run.py build-manifest \
  --config configs/base/pointcloud_ffm.yaml \
  --split validation \
  --max-samples 8 \
  --output manifests/validation_sensors.json
```

Reuse that file with `evaluate-run --sensor-manifest` when comparing models.

### 3. Train a base model

Every normal model starts with `stage: base_training`:

```bash
python run.py train-base --config configs/base/pointcloud_ffm.yaml
```

Available Brusselator and Kolmogorov templates cover coordinate MLP, MLP-RBF,
DeepONet, Senseiver, GeoFNO, DiffusionPDE, latent flow, and both PointCloudFFM
backbones. KS and the other cases provide the compatible subset under their
own `configs/base/` directories.

Point models may train on `model.query_points` sampled target locations. Grid
models require the complete logical grid. The trainer calls the adapter's
native loss, so diffusion and flow models retain their noise/velocity
objectives rather than being converted to direct endpoint regression.

#### Latent flow

Latent flow requires two base runs:

```bash
python run.py train-base --config configs/base/latent_fm_stage1.yaml

python run.py train-base \
  --config configs/base/latent_fm_stage2.yaml \
  --override model.stage1_checkpoint=runs/<stage1-experiment>/<run-id>/checkpoints/best.pt
```

Stage 2 verifies that the source checkpoint is `latent_fm` Stage 1, loads only
the autoencoder, freezes it, and trains the sparse-conditioned latent velocity.
Stage 1 alone is not a sparse-reconstruction source for post-training.

### 4. Monitor, checkpoint, and resume

The common checkpoint policy saves complete epoch-boundary states:

```yaml
checkpointing:
  enabled: true
  every_epochs: 5
  save_epoch_one: true
```

Each due save atomically refreshes `checkpoints/last.pt`; `latest.pt` is a
relative symlink to it. `best.pt` follows fixed validation-preview MSE when the
preview is enabled and training loss otherwise. Base and data-driven coherence
training can resume only with the same resolved config:

```bash
python run.py train-base \
  --config configs/base/pointcloud_ffm.yaml \
  --resume runs/<experiment>/<run-id>
```

For data-driven post-training, use the same `post-train` command and its
original child config with `--resume`.

An optional fixed-sample reconstruction preview is configured independently:

```yaml
evaluation:
  generation_steps: 8
  preview:
    enabled: true
    every_epochs: 25
    split: validation
    sample_index: 0
    query_points: null
    generation_steps: 8
    seed: 2027
    keep_history: false
```

The latest PNG, SVG, PDF, metrics, figure contract, and portable physical-field
arrays are written under `evaluation/training_preview/`. Re-render the portable
payload without loading a model or dataset:

```bash
python figures/scripts/training_reconstruction_preview.py \
  --payload Cases/<case>/runs/<experiment>/<run-id>/evaluation/training_preview/latest_reconstruction.npz
```

### 5. Evaluate the base checkpoint

Use the common evaluator with either a sensor config or a fixed manifest:

```bash
python run.py evaluate-run \
  --run runs/<experiment>/<run-id> \
  --checkpoint best \
  --sensor-config configs/sensors/u_only_random.yaml \
  --split validation \
  --max-samples 8 \
  --report-name base_validation
```

For matched comparisons, keep checkpoint choice, dataset split, sample count,
sensor manifest, query count, generation steps, seed, and device consistent.
The report contains normalized and physical-unit errors, observed/unobserved
errors, timing, peak CUDA memory, available uncertainty summaries, case
diagnostics, hashes, sample IDs, and plotting payloads.

### 6. Run data-driven coherence post-training

A coherence child run requires a completed immutable base run:

```bash
python run.py post-train \
  --config configs/posttrain/global_distribution_reference.yaml \
  --override source_run=runs/<base-experiment>/<run-id>
```

The child inherits the source dataset, model, and observation settings. It
loads `source_checkpoint`, selects the configured trainable scope, and combines
the adapter's native base loss with scheduled coherence loss.

The available families are:

- `global_distribution`: marginal Wasserstein, pairwise sliced Wasserstein,
  and joint top-tail sliced Wasserstein descriptors;
- `cross_spectrum`: graph same-frequency coherence, cross-band energy coupling,
  and optional band-power matching;
- `topology`: self and mutual fibered Betti-curve matching on a declared
  two-dimensional raster.

Families can run separately or share one differentiable reconstruction. For
cross-spectrum and topology, set:

```yaml
coherence:
  compute_budget:
    query_policy: fixed_shared
```

The fixed point set and ordering define the graph basis or raster map.
Cross-frequency statistics require a coherence batch size of at least three;
larger ensembles give more stable covariance estimates.

Each family must declare one reference policy:

- `training_reference` loads or fits a versioned bank from the training split;
- `paired_supervised` compares against the current training target and is
  supervised structural regularization.

In both modes the dense target is removed from the model rollout. It enters
only the declared post-reconstruction comparison. Family settings, geometry,
reference provenance, field mapping, units, hashes, and state are stored with
the child run.

### 7. Run physics-informed training

Brusselator provides the active differentiable physics implementation. It
decodes predictions to physical units and evaluates periodic reaction-diffusion
residuals using a spectral Laplacian and explicitly labeled paired
finite-difference temporal derivatives.

#### Direct physics-informed training

Train a coordinate PINN from initialization with dense data and PDE objectives:

```bash
python run.py train-direct --config configs/direct_physics/pinn.yaml
```

This route uses `stage: direct_physics`, requires `model.name: pinn`, and does
not start from a base checkpoint.

#### Physics post-training

Refine a differentiable base model while retaining endpoint data accuracy:

```bash
python run.py post-train \
  --config configs/posttrain/physics_periodic.yaml \
  --override source_run=runs/<base-experiment>/<run-id>
```

The objective combines endpoint MSE and the same case-owned physics loss.
Physics post-training and direct-physics resume are not currently supported;
start a new immutable run if those configurations change.

Kolmogorov, KS, turbulent combustion, and mass transport currently contribute
physical diagnostics during evaluation, but they do not expose a trainable
`PhysicsProvider`. Their configs should therefore use base training and
data-driven coherence, followed by `evaluate-run` for case diagnostics.

### 8. Evaluate and compare refined checkpoints

Evaluate the child exactly as a base run:

```bash
python run.py evaluate-run \
  --run runs/<posttrain-experiment>/<run-id> \
  --checkpoint best \
  --sensor-manifest manifests/validation_sensors.json \
  --split validation \
  --max-samples 8 \
  --report-name refined_validation
```

Interpret coherence, reconstruction, and physics metrics together. A lower
coherence loss does not by itself imply lower pointwise error or satisfaction
of unmeasured governing equations.

## Case-specific entry points

| Case | Reconstructed fields | Geometry/task | Trainable physics |
|---|---|---|---|
| `brusselator` | $u,v$ | periodic $192\times192$ snapshots | reaction-diffusion residual and positivity |
| `kolmogorov` | $u,v,p$ | periodic $256\times256$ snapshots | diagnostics only |
| `ks` | $u$ | periodic $401\times256$ space-time trajectories | diagnostics only |
| `turbulent_combustion` | `CH4`, `CO`, `T`, `U_1`, `p` | reordered $100\times403$ snapshots | diagnostics only |
| `mass_transport_fluid` | $u_x,u_y,c$ | nonperiodic $32\times32$ fixture | diagnostics only |

The historical turbulent-combustion Demo50 checkpoint is isolated behind a
compatibility adapter and cannot be selected for new base training. To import
and verify it without changing the original files:

```bash
cd Cases/turbulent_combustion
python import_demo50.py --config configs/compatibility/demo50.yaml
```

It can then serve as the immutable source for the provided Demo50 post-training
configs, including the all-family coherence workflow:

```bash
python run.py post-train \
  --config configs/posttrain/demo50_all_coherence.yaml
```

## Run artifacts

A native run is stored under
`Cases/<case>/runs/<experiment>/<run-id>/`. Important contents are:

- `resolved_config.yaml`: fully merged, path-resolved run contract;
- `run_manifest.json` and `status.json`: lineage, hashes, data strategy, and
  completion status;
- `history.jsonl`: per-step losses and gradient diagnostics;
- `checkpoints/{last,latest,best}.pt`: formal model/optimizer states;
- `artifacts/`: sensor manifests, reference banks, coherence family states,
  geometry bases, and provenance;
- `evaluation/`: before/after or named checkpoint reports, query indices,
  physical arrays, and plotting payloads;
- `evaluation/training_preview/`: optional fixed-sample qualitative monitoring.

Post-training directories identify their parent and record source hashes before
and after work. This lineage is the basis for confirming that a refinement run
did not mutate its source.

## Practical rules for trustworthy comparisons

- Split by the declared sample/trajectory unit and fit normalization or
  reference banks only on training data.
- Preserve field order and units from the dataset through sensors, checkpoints,
  coherence artifacts, physics providers, and plots.
- Compare runs with a shared sensor manifest and query-index set.
- Use full-grid queries for grid models and for physics providers that require
  differential operators.
- Treat `paired_supervised` coherence as supervised training, not target-free
  refinement.
- Report generation steps and random seed for diffusion and flow models.
- Inspect both observed and unobserved errors; final sensor clamping can make
  observed error small without improving the unobserved field.
- Use `--max-steps` only when intentionally truncating a command; a truncated
  run is not a completed training result.

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for artifact and environment
requirements and [UPSTREAM.md](UPSTREAM.md) for dependency and clean-room
reference decisions.

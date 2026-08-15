# Project Creation Plan: Data-Driven Multi-Field Reconstruction

## 0. Purpose and status of this document

This document is the implementation plan for turning
`Proj_MultiFieldReconstruction/` into the formal shared workspace for the
data-driven multi-field reconstruction project. 
At the time this document is created, no project code,
configuration, data links, tests, or additional documentation should be added.

The scientific goal is to reconstruct complete multi-field physical states
from flexible, partial, sparse measurements and then determine whether a
post-training physical-coherence framework improves those reconstructions
across model families and datasets. The central comparison is not merely
whether a reconstruction agrees with itself within one field. It is whether
measured fields can support physically credible inference of unmeasured fields,
including systems for which a complete and tractable PDE model is unavailable.

The project should answer four questions:

1. How much can deterministic regression recover from the sparse observations?
2. Does generative modeling improve recovery, uncertainty, or ambiguity over
   direct regression?
3. Can the same coherence-refinement interface improve point models, grid
   operators, diffusion models, latent models, and direct point-field flows?
4. When PDE information is available, how does data-driven coherence compare
   with PDE-based post-training and direct physics-informed training?

This plan is written so that a later Goal-mode run can execute it phase by
phase and stop at explicit validation gates.

---

## 1. Scope, terminology, and scientific guardrails

### 1.1 Initial task definition

The default multi-field benchmark unit is one physical state at one time:

- input: a variable-size set of observations containing coordinates, values,
  field identities, and validity masks;
- output: all requested physical fields at all query points;
- observation pattern: random point sensors, structured super-resolution
  samples, or a case-defined mixed pattern;
- conditioning: one or more measurable fields, potentially at different sensor
  counts and locations;
- targets: both observed fields away from sensors and entirely unobserved fields.

For the multi-field cases, the implementation treats
`(trajectory_id, time_index)` as a reconstruction sample. The KS case adds a
second supported reconstruction unit, `space_time_trajectory`: sparse samples
on the joint `(t, x)` domain condition reconstruction of the complete saved
space-time state. This is a reconstruction/interpolation task, not rollout or
future forecasting. Both units share the same observation/query contract.

### 1.2 Meanings that must not be conflated

- **Data fidelity**: agreement with the sparse observations supplied to the
  model. This is a constraint, not by itself evidence of physical coherence.
- **Data-driven coherence**: constraints learned or estimated only from the
  training data/reference statistics, without requiring governing PDEs.
- **Coherence family**: a top-level kind of data-driven physical coherence.
  The initial family is **global distribution coherence**. Future families
  include **cross-spectrum coherence** and **topological coherence**, each with
  its own definitions, inputs, estimators, and multiple components.
- **Global distribution components**: the currently implemented `self`,
  `mutual`, and `cross` terms. These names are subdivisions of global
  distribution coherence only; they are not the top-level taxonomy for the
  complete coherence framework.
- **PDE coherence**: residuals, conservation laws, boundary conditions, or
  constitutive constraints supplied by a case implementation.
- **Base training**: ordinary supervised, diffusion, or flow-matching training.
- **Post-training**: loading a completed base run and optimizing an explicitly
  selected parameter subset or refinement module with additional coherence
  objectives.
- **Direct physics-informed training**: training from initialization with data
  and PDE objectives active in the same run.
- **Inference-time refinement**: optimizing a state or latent sample while the
  trained model remains frozen. This is useful but is a separate experimental
  axis from post-training and must be reported separately.

### 1.3 Leakage and claim rules

These rules are mandatory because otherwise a coherence score can quietly use
information that would not exist in the intended inverse problem:

1. Dataset partitions are made at the trajectory or independent-realization
   level whenever more than one trajectory exists. A single long trajectory is
   valid benchmark data and uses the standard ordered frame split: first 80%
   train, next 10% validation, final 10% test. Frame splits are never shuffled
   across these boundaries.
2. Normalization, reference banks, projection banks derived from data, learned
   coherence models, and thresholds are fit on the training split only.
3. Validation data may select hyperparameters; test data is evaluation-only.
4. A loss comparing a reconstruction with its paired dense target must be
   labeled `paired_supervised`, even if it is distributional. It must not be
   reported as target-free coherence refinement.
5. A target-free data-driven coherence term may use frozen training-reference
   statistics or a learned training-only critic, but never the paired hidden
   validation/test target.
6. PDE metrics may be used for evaluation without being used for training. The
   report must distinguish these two roles.
7. The same fixed evaluation sensor manifests, split membership, normalization,
   and metric implementation must be used across all compared models.

### 1.4 Initial non-goals

- Do not depend on or import code from `0_demo_TurbulentCombustion/` at runtime.
  Explicit, versioned third-party dependencies remain allowed where their
  licenses and interfaces have been reviewed.
- Do not copy large HDF5 or PyTorch datasets into this project.
- Do not bake sensor masks into canonical dense datasets.
- Do not create a separate training loop for every case or every model.
- Do not create many one-off smoke scripts. Use a small parametrized contract
  suite and real case configurations.
- Do not call a coordinate MLP a PINN unless a PDE/boundary residual is active.
  The same network without those terms is a deterministic coordinate regressor.
- Do not claim a benchmark result from the current mass-transport demo; it is
  too small, is not the intended official dataset, and remains an integration
  fixture/future-case scaffold only.

---

## 2. Findings from the existing repository that shape the design

### 2.1 Reusable ideas in `0_demo_TurbulentCombustion/`

The demo already contains useful implementations and behavior to migrate by
review and refactoring:

- deterministic `mlp_rbf`, Senseiver, and grid/irregular Geo-FNO adapters;
- point-cloud rectified flow with IID/RFF priors, enhanced GL-RBF, and FNO
  backbones;
- S3GM-style diffusion, latent flow matching, and SiT implementations;
- variable per-field sparse conditioning;
- observation-consistency operations;
- global distribution coherence with `self`, `mutual`, and `cross` components;
- differentiable clean-rollout post-training;
- RAM full-copy and LoRA post-training paths;
- checkpoint, history, evaluation, and visualization behavior.

The new project must not copy the current large `Model.py`, `model_baseline.py`,
or helper modules as monoliths. Each reused behavior should be extracted behind
the new contracts, given provenance in `ModelExplain.md`, and tested without
importing the demo. Any behavior that is not migrated remains a reference only.

### 2.2 Problems to correct rather than preserve

- The current combustion loader hard-codes combustion field names and indexes
  only trajectory zero.
- It randomly divides time snapshots into train and held-out samples. The new
  loader should instead use the standardized chronological 80/10/10 frame
  split and save its exact boundaries in the run artifacts.
- Dataset loading, sparse sampling, model construction, training, evaluation,
  and plotting are coupled in large files.
- Base training and post-training parameters are partially mixed in the same
  configurations.
- Output roots and run discovery depend on demo numbers and timestamps rather
  than a stable experiment/run manifest.
- Existing global distribution coherence can compare generated and paired
  reference states. The new API must record whether a term is paired-supervised
  or training-reference-based.

### 2.3 Existing data inspected for the initial cases

| Case | Existing source | Shape and fields | Initial role |
|---|---|---|---|
| Turbulent combustion | `0_demo_TurbulentCombustion/Dataset/Merged_CH4COTU1P.h5` | `fields [1,10000,40300,1,1,5]`; `CH4, CO, T, U_1, p`; unstructured metadata | Full benchmark with the standard ordered 80/10/10 frame split |
| Brusselator | `datagen/data/processed/brusselator.h5` | `fields [100,241,36864,1,1,2]`; `u, v`; 192 by 192 periodic grid; 80/10/10 trajectory split | First canonical multi-trajectory benchmark |
| Kolmogorov flow | `datagen/data/processed/kolmogorov.h5` | `fields [100,201,65536,1,1,3]`; `u, v, p`; 256 by 256 periodic grid; auxiliary vorticity; 80/10/10 trajectory split | Initial canonical velocity-to-pressure and multi-field flow benchmark |
| Kuramoto-Sivashinsky (KS) | `datagen/data/processed/ks.h5` | `fields [100,401,256,1,1,1]`; `u`; 1-D periodic grid; 80/10/10 trajectory split | Space-time quasi-super-resolution benchmark with independent spatial/temporal downsampling |
| Mass transport-fluid | `datagen/data/processed/mass_transport_fluid_demo.h5` | `fields [1,3,1024,1,1,3]`; `u_x, u_y, concentration`; 32 by 32 nonperiodic grid; train only | Future-case scaffold and loader integration fixture; no official benchmark dataset yet |

The processed Brusselator, Kolmogorov, KS, and mass-transport files already
carry schema version `1.0`, physical coordinates, field names/units,
conditions, provenance, and training statistics. The three canonical datasets
also carry disjoint trajectory split groups. The project loader should consume
this contract rather than introduce a conflicting format.

`datagen/data` is already a symbolic link to external storage. Links created
inside the new project should point to the repository-visible paths above, not
hard-code `/data/wanglz/...`, so a collaborator can replace only the final link.

---

## 3. Target directory layout

Create the following layout incrementally. Empty generated directories should
be represented by a short README or `.gitkeep` only when necessary.

```text
Proj_MultiFieldReconstruction/
├── README.md
├── ModelExplain.md
├── ProjectCreate.md
├── CONTRIBUTING.md
├── pyproject.toml
├── environment.yml
├── .gitignore
├── configs/
│   ├── schema/
│   │   ├── base_training.schema.yaml
│   │   ├── post_training.schema.yaml
│   │   └── direct_physics.schema.yaml
│   └── defaults/
│       ├── runtime.yaml
│       ├── evaluation.yaml
│       └── logging.yaml
├── src/
│   └── phycoflow_reconstruction/
│       ├── __init__.py
│       ├── cli.py
│       ├── contracts.py
│       ├── registry.py
│       ├── config/
│       │   ├── load.py
│       │   ├── schema.py
│       │   └── validate.py
│       ├── data/
│       │   ├── h5_dataset.py
│       │   ├── pt_dataset.py
│       │   ├── normalization.py
│       │   ├── observations.py
│       │   ├── sensor_protocols.py
│       │   ├── splits.py
│       │   └── validation.py
│       ├── models/
│       │   ├── common/
│       │   ├── compatibility/
│       │   │   └── legacy_tc_demo50.py
│       │   ├── deterministic/
│       │   │   ├── coordinate_mlp.py
│       │   │   ├── mlp_rbf.py
│       │   │   ├── deeponet.py
│       │   │   └── senseiver.py
│       │   ├── operators/
│       │   │   └── geofno.py
│       │   ├── generative/
│       │   │   ├── diffusion_pde.py
│       │   │   └── latent_fm.py
│       │   └── flows/
│       │       ├── pointcloud_ffm.py
│       │       ├── gl_rbf_enh_topk.py
│       │       ├── fno_backbone.py
│       │       ├── priors.py
│       │       └── integrators.py
│       ├── coherence/
│       │   ├── base.py
│       │   ├── registry.py
│       │   ├── compose.py
│       │   ├── reference_bank.py
│       │   ├── observation.py
│       │   └── families/
│       │       ├── global_distribution/
│       │       │   ├── family.py
│       │       │   └── components/
│       │       │       ├── self_marginal.py
│       │       │       ├── mutual_pairwise.py
│       │       │       └── cross_joint.py
│       │       ├── cross_spectrum/
│       │       │   ├── README.md
│       │       │   └── family.py
│       │       └── topology/
│       │           ├── README.md
│       │           └── family.py
│       ├── training/
│       │   ├── engine.py
│       │   ├── base_training.py
│       │   ├── post_training.py
│       │   ├── direct_physics.py
│       │   ├── gradient_balance.py
│       │   ├── checkpointing.py
│       │   └── run_store.py
│       ├── evaluation/
│       │   ├── evaluator.py
│       │   ├── metrics.py
│       │   ├── benchmark.py
│       │   └── aggregate.py
│       └── utils/
│           ├── reproducibility.py
│           ├── device.py
│           └── provenance.py
├── Dataset/
│   ├── README.md
│   ├── SCHEMA.md
│   ├── turbulent_combustion/
│   │   └── README.md
│   ├── brusselator/
│   │   └── README.md
│   ├── kolmogorov/
│   │   └── README.md
│   ├── ks/
│   │   └── README.md
│   └── mass_transport_fluid/
│       └── README.md
├── Cases/
│   ├── README.md
│   ├── turbulent_combustion/
│   │   ├── README.md
│   │   ├── run.py
│   │   ├── case.py
│   │   ├── preprocess.py
│   │   ├── physics.py
│   │   ├── coherence.py
│   │   ├── visualize.py
│   │   ├── configs/
│   │   │   ├── dataset.yaml
│   │   │   ├── sensors/
│   │   │   ├── coherence/
│   │   │   ├── base/
│   │   │   ├── posttrain/
│   │   │   ├── direct_physics/
│   │   │   └── evaluation/
│   │   └── runs/
│   ├── brusselator/
│   │   └── ...same case contract...
│   ├── kolmogorov/
│   │   └── ...same case contract...
│   ├── ks/
│   │   └── ...same case contract...
│   └── mass_transport_fluid/
│       └── ...same case contract...
├── scripts/
│   ├── validate_dataset.py
│   ├── build_sensor_manifest.py
│   ├── compare_runs.py
│   └── summarize_run.py
└── tests/
    ├── fixtures/
    ├── test_config_contracts.py
    ├── test_dataset_contracts.py
    ├── test_observation_contracts.py
    ├── test_model_contracts.py
    ├── test_coherence_contracts.py
    └── test_run_contracts.py
```

### 3.1 Dependency direction

The allowed dependency direction is:

```text
case run/config/code -> general package contracts -> model/data/coherence modules
                     -> case physics/coherence providers through registered interfaces
```

The general package must never import a named case. A case registers a
`CaseSpec`, optional PDE provider, coherence-family settings/context,
preprocessing, and visualizations. Models receive the general batch contracts
and must never open case-specific files directly.

### 3.2 Why entry points remain inside each case

Users launch from a case directory as requested:

```bash
cd Proj_MultiFieldReconstruction/Cases/brusselator
python run.py train-base --config configs/base/pointcloud_ffm.yaml
python run.py post-train --config configs/posttrain/data_driven.yaml
python run.py evaluate --config configs/evaluation/benchmark.yaml
```

`run.py` is a thin case selector. It changes no global state and delegates to
the installed `phycoflow_reconstruction` package. Training logic and run
storage remain general; case-specific files provide only physical meaning and
configuration.

---

## 4. Stable interfaces to implement first

### 4.1 In-memory sample and batch contracts

Use typed dataclasses or typed dictionaries with runtime validation. The common
point representation is the interoperability boundary:

```text
FieldSample
  values             [N, C]       dense physical state in model or physical units
  coordinates        [N, D]       model coordinates
  coordinates_raw    [N, D]       physical coordinates
  time               scalar
  trajectory_id      scalar/string
  time_index         scalar
  conditions         [P]
  valid_points       [N]          optional for variable meshes
  reconstruction_unit             snapshot | space_time_trajectory
  logical_shape                    reversible spatial or space-time shape

ObservationBatch
  obs_coords         [B, M, D]
  obs_values         [B, M, 1]    one scalar observation per token initially
  obs_field_ids      [B, M]
  obs_valid_mask     [B, M]
  query_coords       [B, Q, D]
  query_valid_mask   [B, Q]
  target_fields      [B, Q, C]    present only in supervised stages/evaluation
  sample_ids         metadata
```

Scalar observation tokens allow fields to use different locations and sensor
counts. A future vector-sensor adapter may expand a colocated multi-field
measurement into multiple scalar tokens without changing models.

For `space_time_trajectory`, `N = T * Nx` in the initial 1-D case and `D = 2`
with `(t, x)` coordinates assembled from `/time` and `/coordinates`. The logical
`[T, Nx, C]` layout and inverse mapping remain available to grid models,
space/time metrics, and visualizers. This extends the same point contract rather
than creating a separate KS-only trainer.

The model protocol should expose:

```text
build(model_config, data_spec) -> model
training_loss(batch, stage_context) -> LossBundle
reconstruct(observations, query_coords, sampling_config) -> ReconstructionBatch
capabilities -> {point/grid, deterministic/generative, differentiable_rollout,
                 structured_grid_required, uncertainty_samples, stages}
```

Adapters, not trainers, handle model-specific reshaping, rasterization,
multi-stage autoencoders, EMA weights, and samplers.

### 4.2 Case contract

Each `case.py` registers one `CaseSpec` containing:

- stable case name and display name;
- dataset profile and accepted schema versions;
- field names, units, observable/unobservable roles, and display settings;
- spatial dimension, mesh type, grid shape/order, coordinate normalization;
- time semantics and default sample granularity;
- reconstruction unit (`snapshot` or `space_time_trajectory`) and reversible
  query layout;
- default normalization policy;
- sensor protocol compatibility;
- optional `PhysicsProvider`, `CoherenceProvider`, and `Visualizer` factories.

Case code may validate extra metadata, but it must not replace the general HDF5
loader when the file already follows the canonical schema.

### 4.3 Coherence family and component contracts

Use two levels. A `CoherenceFamily` owns a scientifically distinct definition
and coordinates one or more `CoherenceComponent` implementations. Components
from one family must not be promoted into the global namespace as if they were
complete coherence categories.

```text
CoherenceFamily
  name                 global_distribution | cross_spectrum | topology | ...
  version
  required_context
  fit(train_loader)    optional family/reference state
  build_components(case_settings)
  aggregate(component_results, family_settings) -> FamilyResult

CoherenceComponent
  name
  version
  family
  component_path       e.g. global_distribution.self.marginal_w2
  supervision         training_reference | paired_supervised | analytic
  differentiable      true | false
  required_context    coordinates, time, boundary, reference_bank, ...
  forward(prediction, observations, context) -> TermResult

TermResult
  per_sample_cost      [B]
  scalar_loss          scalar
  diagnostics         named tensors/scalars
  valid_mask/reason    explicit handling of unavailable terms

FamilyResult
  component_results    mapping[component_path, TermResult]
  per_sample_cost      [B] after configured family aggregation
  scalar_loss          scalar
  diagnostics          family-namespaced summaries
```

The general package stores reusable calculations, schemas, validation, and
family/component registries. Each case stores only scientific settings such as
enabled components, field groups/pairs, channel weights, wavenumber bands,
topological objects, physical units, and loss weights under
`Cases/<case>/configs/coherence/<family>.yaml`; case `coherence.py` supplies
geometry- or field-specific context when required. Composition is config-driven.
A family/component cannot silently denormalize, change units, subsample points,
or use a target. Those choices must appear in the resolved configuration and
serialized family state.

### 4.4 Physics provider contract

PDE support belongs to cases because equations, units, boundaries, gauges, and
auxiliary fields are problem-specific:

```text
PhysicsProvider.residuals(state, coordinates, time, conditions, auxiliary)
PhysicsProvider.boundary_losses(...)
PhysicsProvider.integral_constraints(...)
PhysicsProvider.diagnostics(...)
```

Each returned term declares discretization, units, normalization, whether it is
differentiable, and whether it is valid on irregular points. A missing PDE
provider is a supported state, not an error for data-driven experiments.

---

## 5. Dataset standard

### 5.1 Accepted file types

1. **HDF5 (`.h5`/`.hdf5`) is canonical and preferred** for datasets too large
   for memory, partial reads, multiple trajectories, metadata, and split data.
2. **PyTorch (`.pt`) is supported** for smaller trusted datasets. It must contain
   a documented mapping of tensors and plain metadata matching the logical HDF5
   contract. Arbitrary pickled classes are not accepted.
3. Other formats require a case preprocessor that writes one of the accepted
   formats. Training code must not accumulate format-specific branches.

Dataset binaries and links are local-only. Only the per-dataset `README.md` is
required to synchronize. The project `.gitignore` should ignore all payloads,
links named with data extensions, cached statistics, generated sensor
manifests, and downloads under `Dataset/`, while explicitly allowing Markdown.

### 5.2 HDF5 tensor convention

Retain compatibility with the existing processed schema:

```text
fields       [B, T, Nx, Ny, Nz, C]
coordinates  [Nx, Ny, Nz, 3]
time         [T]
conditions   [B, P]
```

Here `B` is the number of independent trajectories/realizations, `T` is saved
time, `C` is the number of physical fields, and unused spatial dimensions have
size one. Existing flattened 2-D data uses `[B,T,N,1,1,C]` and coordinates
`[N,1,1,3]`; `grid_shape` and `coordinate_order` reconstruct the original grid.
The loader flattens any structured representation to `[N,C]` internally and
retains a reversible index map for grid models and plotting.

Required datasets:

- `/fields`, `/coordinates`, `/time`, `/conditions`;
- `/trajectory_id`, `/seed` when `B > 1` or when available;
- `/splits/train`, `/splits/validation`, `/splits/test` for canonical datasets;
- `/statistics/train_mean`, `/statistics/train_std` or a documented training-only
  normalization alternative;
- `/metadata/json` for schema/provenance not naturally represented as arrays.

Required attributes/metadata:

- `schema_version`, `case_name`, `display_name`, `source_dataset_id`;
- field/state/condition names and units;
- spatial dimension, mesh type, `grid_shape`, coordinate axes/units/order;
- domain and boundary conditions, periodic axes, gauge conventions;
- time units, saved-time meaning, and time-step information when applicable;
- source, generation/processing command, code commit, license/citation, and
  checksum or download location;
- storage dtype, missing-value convention, auxiliary-field descriptions;
- split method/seed and normalization method/split.

Optional groups include `/auxiliary`, `/diagnostics`, and case-specific static
geometry. They are exposed through metadata/context, not automatically treated
as reconstruction targets.

### 5.3 Split policies

- Canonical multi-trajectory datasets: use the stored disjoint trajectory
  splits. Refuse to reshuffle unless writing a new named split manifest.
- Single-trajectory time series such as the current combustion file: use an
  ordered frame split by default—first 80% train, next 10% validation, final
  10% test. Compute boundaries before applying a per-split `time_stride`, never
  shuffle frames across boundaries, and record exact ranges/indices in a local
  split manifest. This is a normal supported split strategy, not a degraded or
  integration-only mode.
- Independent snapshots with no trajectories: allow deterministic item-level
  splits only when the dataset README explicitly establishes independence.
- Demo data without sufficient validation/test samples: mark
  `benchmark_eligible: false`; training or integration may proceed only with an
  explicit acknowledgement in config.

### 5.4 Normalization

Normalization is a named, serializable component and is always fit on the
training split. Initially support:

- per-field mean/std;
- fixed per-field offset and robust 99th-percentile scale for high-dynamic-range
  multiphysics data;
- no normalization.

Models operate in normalized units unless their adapter declares otherwise.
Coherence and physics terms independently declare `model_units` or
`physical_units`. The run stores the exact normalization tensors and their
source split. Never write a new sidecar beside a shared read-only dataset; put
derived statistics inside the run or case cache.

### 5.5 Dataset README template

Every `Dataset/<case>/README.md` must contain:

1. physical background and coupled fields;
2. source, license/citation, generation method, and download instructions;
3. local payload/link names and link setup commands;
4. shapes, dtypes, file size, schema version, and checksum;
5. fields/units, coordinates/domain/mesh/boundaries, time and conditions;
6. split sizes and independence assumptions;
7. normalization/statistics provenance;
8. auxiliary data and known limitations;
9. a validation command and expected concise output;
10. suitable and unsuitable scientific uses.

### 5.6 Initial links to create during implementation

Use relative links and never replace an existing file automatically:

```text
Dataset/turbulent_combustion/Merged_CH4COTU1P.h5
  -> ../../../0_demo_TurbulentCombustion/Dataset/Merged_CH4COTU1P.h5

Dataset/turbulent_combustion/Merged_COTU0U1P.h5
  -> ../../../0_demo_TurbulentCombustion/Dataset/Merged_COTU0U1P.h5

Dataset/brusselator/brusselator.h5
  -> ../../../datagen/data/processed/brusselator.h5

Dataset/kolmogorov/kolmogorov.h5
  -> ../../../datagen/data/processed/kolmogorov.h5

Dataset/ks/ks.h5
  -> ../../../datagen/data/processed/ks.h5

Dataset/mass_transport_fluid/mass_transport_fluid_demo.h5
  -> ../../../datagen/data/processed/mass_transport_fluid_demo.h5
```

The link setup command must support `--check-only`, show resolved targets, and
refuse broken links or target replacement. A collaborator may instead place a
real file at the documented path.

---

## 6. Sparse observation and benchmark protocol

### 6.1 Sensor samplers

Implement samplers behind one registry:

- `random_uniform`: uniform valid-point selection;
- `structured_stride`: regular subgrid/super-resolution conditioning;
- `uniform_spacetime_stride`: independent uniform temporal and spatial strides
  over a complete space-time state;
- `random_grid_offset`: structured stride with randomized phase;
- `region`: points restricted to case-defined measurable regions;
- `mixed`: configurable mixture of the above;
- later: adaptive or learned sensor locations, kept as a separate comparison.

Each protocol specifies:

- conditioned field names, never only positional integers in saved configs;
- per-field minimum/maximum or exact sensor counts;
- whether locations are shared or independent across fields;
- replacement policy, valid-domain mask, observation noise, dropout, and seed;
- whether sensors change by sample, epoch, or remain fixed;
- query-point policy and whether observed points are included in the loss.

`uniform_spacetime_stride` uses separately configurable positive integers
`temporal_downsample_ratio` and `spatial_downsample_ratio`. Its default phase is
zero and its exact retained time/space indices are saved. The query set is the
complete space-time grid. This protocol must not be described as forecasting:
the model reconstructs samples omitted inside the same saved trajectory/window.

Training may sample observations online from a deterministic seed stream.
Validation and test must use persisted, checksum-identified sensor manifests so
every model receives identical observations. Manifests store dataset checksum,
split, sample IDs, protocol, seed, and indices/coordinates; they remain local
generated artifacts unless a small canonical benchmark manifest is deliberately
versioned later.

### 6.2 Core experiment matrix

Every formal case should eventually include at least:

- one measured field -> all fields;
- two or more measured fields -> all fields;
- low, medium, and high sensor budgets expressed both as counts and fractions;
- random sensors and structured super-resolution sensors;
- for space-time reconstruction cases, independent spatial and temporal
  downsampling ratios;
- clean sensors plus one documented noise/dropout stress test;
- interpolation and, where scientifically valid, held-out condition or
  resolution generalization.

Do not launch the full Cartesian product initially. Start with one canonical
protocol per case, validate the pipeline, and then expand through named config
files.

### 6.3 Fair model comparison

All model adapters consume the same `ObservationBatch`. Grid models receive a
general rasterization adapter that provides value maps and binary masks; point
models receive tokens directly. Rasterization and interpolation parameters are
part of the resolved config and cannot use hidden target values.

Report at minimum:

- per-field normalized and physical-unit L1/L2/RMSE;
- observed-sensor residual and unobserved-domain error separately;
- coherence metrics grouped first by family and then by component—for example
  `global_distribution/{self,mutual,cross}`—without flattening the taxonomy;
- case-defined PDE/balance diagnostics when available;
- spectral/gradient/structure metrics where the mesh supports them;
- parameter count, training time, peak memory, and inference cost/NFE;
- for generative models: sample mean error, best-of-K only as a secondary
  oracle diagnostic, spread/calibration, and seed count.

Aggregate first within a trajectory, then across trajectories, so long
trajectories do not dominate the result. Save per-sample results as well as
mean, median, standard deviation, and bootstrap confidence intervals.

---

## 7. Model families and implementation order

### 7.1 Model registry

Each registry entry contains a stable name, model version, adapter class,
capabilities, required optional dependencies, upstream references/licenses,
and supported training stages. A checkpoint records all of these plus the data
and observation contracts.

### 7.2 Planned models

| Class | Registry names | Scientific role | First implementation source |
|---|---|---|---|
| Deterministic point/coordinate | `coordinate_mlp`, `mlp_rbf`, `pinn`, `deeponet`, `senseiver` | Establish whether generation helps beyond direct regression; compare generic data fitting with PDE-informed coordinate fitting | Refactor demo MLP-RBF/Senseiver; implement small common coordinate/DeepONet adapters |
| Deterministic grid/operator | `geofno` | Test whether coherence refinement works with structured and geometry-aware operator regression | Wrap maintained `neuraloperator`; use demo behavior as compatibility reference |
| Grid/latent generative | `diffusion_pde`, `latent_fm` | Test transfer to diffusion and latent/interpolant models | Refactor demo latent FM; independently adapt DiffusionPDE concepts after license review |
| Direct point-field rectified flow | `pointcloud_ffm` with `gl_rbf_enh` (default) or `fno` backbone | Main PhyCoFlow realization on irregular/flattened fields, with an optional structured-grid path | Narrow modular refactor of the required demo PointCloudFFM behavior |

Optional SiT/S3GM adapters may be added after the required matrix works. They
should not delay the first end-to-end comparison.

### 7.3 PointCloudFFM scope and legacy checkpoint compatibility

Keep the new PointCloudFFM surface deliberately narrow:

- default backbone: `gl_rbf_enh`;
- only new GL-RBF gather mode: `topk_rbf`;
- optional backbone: `fno` for compatible regular grids;
- do not implement or expose `mlp_rbf`, `perceiver`, legacy `GL_rbf`,
  `rbf`, `topk_rbf_gate`, `topk_rbf_ptlocal`, or other historical gather modes
  inside the new PointCloudFFM registry at this stage.

The first compatibility target is the existing run:

```text
0_demo_TurbulentCombustion/Save_TrainedModel/
  ffm_tc_pointcloud_DemoN50_20260706_084857/
```

Inspection shows that Demo 50 is `GL_rbf_ENH` with legacy
`gather_mode=topk_rbf_glres`, not the new `topk_rbf` default. Therefore strict
checkpoint support requires one explicit exception to the new-model scope: an
isolated `models/compatibility/legacy_tc_demo50.py` implementation of exactly
the modules and state-dict layout needed by this run. It is not a generally
selectable new backbone or gather mode, and it must not grow into a collection
of old modes. It may be loaded for evaluation and post-training, but new base
training uses `gl_rbf_enh/topk_rbf` or FNO.

The compatibility loader must:

1. accept an explicit legacy run directory and `best.pt` or `last.pt`;
2. read `run_config.yaml`, `args.json`, checkpoint metadata, normalization, and
   source dataset metadata before model construction;
3. map legacy names to a versioned compatibility spec and load the model state
   strictly, reporting every missing/unexpected/shape-mismatched key;
4. preserve the checkpoint's stored mean/std, RF convention, RFF prior,
   coordinate normalization, conditioning, field ordering, and Euler sampling;
5. handle the observed metadata inconsistency safely: Demo 50 trained from
   `Merged_COTU0U1P.h5`, whose HDF5 fields are `CO,T,U_0,U_1,p`, while the
   checkpoint records stale `CH4,CO,T,U_1,p` names. The importer must require an
   explicit verified field mapping and save it; it must never guess silently;
6. save a conversion/compatibility manifest containing source hashes and all
   mappings without rewriting the original checkpoint;
7. compare fixed-seed reconstructions from the legacy and new compatibility
   paths within documented numerical tolerances before it is accepted.

This compatibility path is required early so the new post-training pipeline can
be validated on a mature checkpoint before all new base models finish training.

### 7.4 Upstream references and licensing gate

Use these as references, pin exact revisions in `ModelExplain.md`, and record
licenses before copying or adapting code:

- [NeuralOperator](https://github.com/neuraloperator/neuraloperator) for the
  maintained FNO/operator API. The old
  [Geo-FNO repository](https://github.com/neuraloperator/Geo-FNO) is deprecated
  and should be treated as an algorithm/reference, not the primary dependency.
- [Senseiver](https://github.com/OrchardLANL/Senseiver) for the published sparse
  attention reconstruction architecture.
- [DiffusionPDE](https://github.com/jhhuangchloe/DiffusionPDE) for partial-
  observation diffusion design. Its repository is archived and uses
  CC BY-NC-SA 4.0, so do not copy it into the project until compatibility with
  the intended release/use is approved.
- [DeepXDE](https://github.com/lululxvi/deepxde) as a maintained conceptual/API
  reference for PINN and DeepONet behavior; avoid forcing all project models
  through a second framework unless the dependency is clearly beneficial.
- [Meta flow_matching](https://github.com/facebookresearch/flow_matching) as a
  mathematical and implementation reference for general flow-matching pieces.

For every external source, choose one of: library dependency, clean-room local
implementation from the paper, or properly attributed adaptation. Add an
`UPSTREAM.md`/license entry before merge. Do not use copied code whose license
or provenance is uncertain.

### 7.5 Implementation and immediate-comparison sequence

1. Build the data/observation/run contracts and the narrow PointCloudFFM
   `gl_rbf_enh/topk_rbf` path.
2. Implement the Demo 50 compatibility importer and reproduce a fixed legacy
   reconstruction.
3. Implement the plain data-loss paths for `coordinate_mlp`, `mlp_rbf`,
   `senseiver`, `deeponet`, `geofno`, `latent_fm`, `diffusion_pde`, and
   PointCloudFFM; enable `pinn` only through a case `PhysicsProvider`.
4. Add the optional FNO backbone inside PointCloudFFM after structured-grid
   validation.
5. For each applicable planned model, immediately run a small real-case base
   experiment with only its native data loss and record reconstruction quality.
6. Apply the same first-release global-distribution post-training interface to
   each compatible base run and compare before/after results using fixed sensor
   manifests.
7. Expand training budgets and cases only after the plain-base and post-training
   paths both work end to end.

The first goal is functional performance evidence for every planned family,
not exhaustive hyperparameter tuning. Each adapter must pass the shared model
contract test and one real-case batch before its first base/post-training pair.

---

## 8. Coherence framework

### 8.1 Family hierarchy

The data-driven coherence registry is hierarchical:

```text
data_driven_coherence
├── global_distribution              # available first
│   ├── self
│   │   └── marginal_w2
│   ├── mutual
│   │   └── pairwise_swd
│   └── cross
│       └── joint_topk_swd
├── cross_spectrum                    # future co-worker contribution
│   └── <multiple family-specific components>
└── topology                          # future co-worker contribution
    └── <multiple family-specific components>
```

`self`, `mutual`, and `cross` are meaningful only beneath
`global_distribution`. Cross-spectrum coherence and topological coherence are
peer families, not additional global-distribution components. Their future
contributors may define different internal subdivisions; the framework must
not force them into the self/mutual/cross pattern.

Observation consistency is a reconstruction constraint shared by training and
sampling. Store it beside the family registry, not as a fourth data-driven
coherence family. PDE coherence similarly uses the separate `PhysicsProvider`.

### 8.2 First implemented family: global distribution coherence

Port the existing calculation as one family with three configurable component
groups:

- `global_distribution.self.marginal_w2`: per-field empirical 1-D Wasserstein;
- `global_distribution.mutual.pairwise_swd`: two-field sliced-Wasserstein over
  configured field pairs;
- `global_distribution.cross.joint_topk_swd`: fixed-bank all-field
  sliced-Wasserstein.

The family owns shared validation, denormalization, point subsampling,
projection-bank construction, aggregation, and diagnostic naming. Components
remain independently weighted/enabled. Provide differentiable batched costs and
detached evaluation metrics when mathematically possible. Projection banks must
be fixed, seeded, serialized, and shared across models in a comparison.

General calculation code lives under
`coherence/families/global_distribution/`. Case configuration specifies fields,
pairs, channel weights, physical/model units, reference-bank selection, point
budget, component weights, and case-specific display names. No combustion field
indices or assumptions belong in the general family.

### 8.3 Reserved extension families

Create registry/API space and concise README contracts for these families, but
do not invent their scientific formulas or placeholder losses before the
responsible co-workers contribute them:

- **Cross-spectrum coherence**: will contain several components describing
  frequency-domain relationships between fields. Its context may require grid
  topology, spacing, boundary/periodicity information, detrending/windowing,
  and configurable wavenumber bands.
- **Topological coherence**: will contain several components describing
  topology of individual or coupled fields. Its context may require thresholds,
  filtrations, connectivity, masks, physical domains, and possibly optional
  topology libraries or learned models.

Future families are added through one family registry entry, a family config
schema, reusable general code, focused tests, and per-case settings. They must
not require edits to the post-training loop. Every family/component ships with
a definition, invariances, units, differentiability, computational cost,
failure cases, and a degenerate-example test. Unavailable geometry/time/context
returns an explicit reason rather than a misleading number.

### 8.4 Reference banks

A reference bank is a versioned artifact fit from the training split. It stores:

- dataset/split checksum and sample-selection seed;
- selected fields, units, normalization, and point subsampling;
- frozen statistics, projections, learned weights if any, and
  family/component versions;
- compatibility information for coordinates, resolution, and conditions.

For conditional physical regimes, support stratified or condition-aware banks;
do not compare a generated state against an indiscriminate mixture when the
dataset spans materially different regimes.

### 8.5 Post-training strategies

The common post-trainer must support:

- full-model fine-tuning;
- head-only or named-module fine-tuning;
- LoRA/adapter tuning when the backbone supports it;
- a separately registered output/latent refiner with a frozen base model;
- differentiable rollout losses;
- reward-based/non-differentiable objectives such as the existing RAM concept;
- optional data-retention loss to prevent catastrophic drift;
- `weighted_sum` first, with conflict-aware gradient balancing as an optional
  dependency rather than a mandatory core requirement.

All post-training runs reference an immutable base run/checkpoint, write a new
run, and record trainable parameter names. They never overwrite base artifacts.
The same interface accepts a data-driven term composition or a case PDE
provider. A mixed PDE + data-driven mode may be added, but it must have its own
explicit config and ablation.

### 8.6 First direct-coherence compatibility target

The first post-training implementation must reproduce the behavior represented
by `0_demo_TurbulentCombustion/Save_config/config_pointcloud_ffm_direct_posttrain.yaml`
through the new structured config and run store:

- initialize from a separate pretrained PointCloudFFM run and inherit its base
  data/model/conditioning configuration after validation;
- always write a new child run and leave the source run immutable;
- retain ordinary rectified-flow velocity MSE as the data-retention loss;
- on a configurable schedule, perform a differentiable clean Euler/Heun rollout
  and apply `endpoint_smooth` observation consistency when selected;
- evaluate global distribution coherence with independently weighted
  `self`, `mutual`, and `cross` components;
- support coherence start epoch, warmup, every-N-step scheduling, optional
  interval rescaling, coherence batch size, and point downsampling;
- support `weighted_sum` and optional ConFIG two-objective gradient balancing;
- save resolved inherited keys, source checkpoint hash, component histories,
  gradient diagnostics, and before/after fixed-manifest evaluations.

The compatibility config should map the legacy flat keys into explicit nested
sections instead of preserving a second flat parser:

| Legacy configuration area | New structured destination |
|---|---|
| `training_mode`, `initialization`, `pretrained_*` | `stage`, `source_run`, `source_checkpoint`, `inherit_base_config` |
| `data_loss_weight`, `coherence_loss_weight`, warmup | `objectives.data_retention` and `objectives.coherence` |
| `coherence_every_n_steps`, start/interval/batch/points | `coherence.schedule` and `coherence.compute_budget` |
| rollout solver/steps and observation-consistency keys | `rollout` and `observation_consistency` |
| `coherence_self_*`, `coherence_mutual_*`, `coherence_cross_*` | `coherence.families.global_distribution.components.*` |
| `gradient_balance_mode`, `config_*` | `optimization.gradient_balance` |

Only base model, data, normalization, and conditioning keys may be inherited
from the source run. New post-training optimizer, schedule, coherence, and
output settings come from the child config. The resolved config records the
origin of every inherited or overridden key.

First run this compatibility experiment from the imported Demo 50 checkpoint,
then repeat it on a newly trained `gl_rbf_enh/topk_rbf` base run. Numerical
identity of training curves is not required, but configuration semantics,
gradient flow, checkpoint lineage, and evaluation behavior must be demonstrably
equivalent.

---

## 9. Configuration design

### 9.1 Three standalone training schemas

Keep these stages separate, with no inactive sections hidden in one giant file:

1. `base_training`: dataset, sensors, model, native objective, optimizer,
   schedule, runtime, evaluation, and output.
2. `post_training`: immutable source run/checkpoint, post-training method,
   trainable scope, coherence/physics composition, data-retention loss,
   optimizer/schedule, runtime, evaluation, and new output.
3. `direct_physics`: dataset, sensors, model, data objective, PDE/boundary terms,
   objective balancing, optimizer/schedule, runtime, evaluation, and output.

A stage config may reference `dataset.yaml`, a sensor protocol, and small global
defaults, but the launcher must write one fully resolved, validated YAML into
the run before training. Unknown keys are errors. CLI overrides are recorded
verbatim and applied before validation.

### 9.2 Stable names instead of fragile indices

Saved configs use field names:

```yaml
observations:
  fields:
    T: {count: [192, 384]}
    U_1: {count: [192, 384]}
targets: [CH4, CO, T, U_1, p]

model:
  name: pointcloud_ffm
  backbone: gl_rbf_enh
  gather_mode: topk_rbf

coherence:
  families:
    global_distribution:
      components:
        self: {enabled: true, weight: 1.0}
        mutual: {enabled: true, weight: 1.0}
        cross: {enabled: true, weight: 1.0}
```

The loader resolves names to indices from dataset metadata and records that
mapping. A config is rejected if names, sensor counts, grid requirements, split
eligibility, or model capabilities are incompatible.

Case configs may override the contents of one family but may not flatten
`self`, `mutual`, or `cross` into top-level coherence names. Future
`cross_spectrum` and `topology` blocks sit beside `global_distribution` under
`families` and use their own component schemas.

### 9.3 Dependency groups

`pyproject.toml` should define a minimal core plus optional groups rather than
forcing every collaborator to install every model:

- `core`: PyTorch, NumPy, h5py, PyYAML, tqdm;
- `plot`: matplotlib and scientific image metrics;
- `operator`: neuraloperator and its compatible dependencies;
- `posttrain`: PEFT/conflict-aware optimization packages if selected;
- `dev`: pytest, lint, formatting, typing;
- `all`: reproducible union for benchmark machines.

`environment.yml` pins the validated CUDA/PyTorch environment for this
workspace. The resolved run records package versions, device/GPU, Git commit,
and dirty-worktree state.

---

## 10. Run-directory and artifact contract

Each launch writes only under the active case:

```text
Cases/<case>/runs/<experiment_name>/<run_id>/
├── run_manifest.json
├── resolved_config.yaml
├── command.txt
├── environment.json
├── status.json
├── checkpoints/
│   ├── best.pt
│   └── last.pt
├── metrics/
│   ├── history.csv
│   ├── validation.csv
│   └── test_samples.csv
├── artifacts/
│   ├── normalization.pt
│   ├── split_manifest.json
│   ├── sensor_manifest.json
│   └── coherence_reference.pt
├── evaluation/
└── logs/
```

`run_id` should combine UTC time and a short configuration hash; human meaning
belongs in `experiment_name`. `run_manifest.json` includes parent run/checkpoint,
stage, status, start/end time, dataset checksum, sensor-manifest checksum,
model/coherence versions, seeds, best metric, checkpoint hashes, and provenance.

Checkpoint payloads must contain model/optimizer/scheduler/EMA state as
applicable, epoch/global step, RNG states, normalized data specification, and
resolved config hash. Resume validates compatibility and writes to the same
incomplete run; post-training always creates a child run.

Generated run contents are ignored by Git. A collaborator shares results by an
explicit export/summarization workflow, not by committing checkpoints or CSVs.

---

## 11. Concrete initial cases

For every benchmark-eligible case, create plain data-loss configurations for
all applicable planned models as soon as their adapters exist. Evaluate those
base runs on fixed manifests, then create matched child post-training runs using
the available coherence families. This base-versus-post pair is the immediate
experimental goal; extensive tuning and large ablation grids come afterward.

### 11.1 Turbulent combustion

`Cases/turbulent_combustion/` should define:

- primary dataset: five fields `CH4, CO, T, U_1, p` from
  `Merged_CH4COTU1P.h5`;
- compatibility dataset `Merged_COTU0U1P.h5`, also with five channels but field
  order `CO, T, U_0, U_1, p`; register it as a distinct dataset variant rather
  than inferring its fields from the filename;
- 40,300 spatial points and physical coordinate plotting;
- a documented structured-grid interpretation only after coordinate validation
  confirms the 403 by 100 mapping; point models must not require it;
- standard chronological frame split: frames `[0, 8000)` train,
  `[8000, 9000)` validation, and `[9000, 10000)` test for the current 10,000
  frames, with the general 80/10/10 rule retained if the file length changes;
- initial sensor protocols: `T_only_random`, `T_U1_random`,
  `CO_T_U1_p_random`, and `T_structured_sr`;
- first prediction focus: condition on measurable `T`, or `T + U_1`, and infer
  all five fields, with unmeasured-field metrics called out separately;
- global distribution coherence with case settings for its self/mutual/cross
  components and combustion-specific joint plots;
- new PointCloudFFM base configs use `gl_rbf_enh` with `topk_rbf` by default and
  optionally the FNO backbone; no other new PointCloudFFM modes are exposed;
- a legacy Demo 50 import/evaluation/post-training config with an explicit
  `CO,T,U_0,U_1,p` mapping and fixed reference reconstruction;
- PDE provider absent initially unless verified governing equations and
  discretizations are supplied. The case must still support data-driven work.

The case README should document frame-based 80/10/10 splitting as the intended
protocol for this long single-trajectory dataset. It should distinguish this
task from cross-trajectory generalization without describing the valid
frame-based benchmark itself as a limitation.

### 11.2 Brusselator

`Cases/brusselator/` should define:

- canonical dataset `brusselator.h5`, shape
  `[100,241,36864,1,1,2]`, fields `u, v`, 192 by 192 periodic domain;
- use stored trajectory splits: 80 train, 10 validation, 10 test;
- initial sensor protocols: `u_only_random`, `v_only_random`,
  `u_v_shared_random`, and `u_structured_sr`;
- primary multi-field task: infer `v` from sparse `u` and reconstruct both;
- reverse-direction `v -> u` as a mutual-inference asymmetry experiment;
- the global distribution family with case-specific self/mutual/cross settings;
- optional PDE provider for the two reaction-diffusion residuals, periodic
  boundaries, positivity, and case diagnostics;
- use PDE terms both as evaluation-only metrics and, in separate configs, for
  PDE post-training/direct PINN experiments.

This should be the first full end-to-end case because it has independent
trajectories, meaningful two-field coupling, a regular grid, and verified
metadata.

### 11.3 Kuramoto-Sivashinsky quasi-super-resolution

`Cases/ks/` should define:

- canonical dataset `ks.h5`, shape `[100,401,256,1,1,1]`, field `u`, periodic
  1-D domain of length 60, and stored 80/10/10 trajectory splits;
- reconstruction unit `space_time_trajectory`, with logical target shape
  `[401,256,1]` and point coordinates formed as `(t,x)`;
- default protocol `uniform_spacetime_sr`, which observes
  `u[::temporal_downsample_ratio, ::spatial_downsample_ratio]` and queries the
  full 401 by 256 space-time grid;
- independently tunable positive integer temporal and spatial downsampling
  ratios, saved phases/indices, and validation that at least two observations
  remain on each active axis;
- easy-to-use defaults `temporal_downsample_ratio: 4` and
  `spatial_downsample_ratio: 4`, using uniform downsampling with phase zero;
  optional randomized phase may be added later as a separate named protocol,
  not substituted silently;
- explicit statement that the task reconstructs/interpolates the complete
  saved space-time state from sparse space-time samples and does not predict an
  unseen future interval;
- point-model support through flattened `(t,x)` query tokens and grid-model
  support through a reversible `[time,x]` rasterization;
- data-loss metrics on the full state plus errors grouped by observed versus
  omitted spatial positions and observed versus omitted time frames;
- spatial spectra, temporal spectra, gradients, and KS PDE residuals as
  evaluation metrics where appropriate;
- initially the global distribution coherence family only. Cross-spectrum and
  topological families remain future registry extensions when their definitions
  and implementations are contributed. Because KS has one field, enable the
  global-distribution `self` component by default and mark `mutual`/`cross`
  components unavailable rather than manufacturing multi-field terms.

The first KS sweep should vary spatial and temporal ratios separately before a
small joint grid, so their effects are identifiable. Initial named protocols
should include `(temporal, spatial)` ratios `(2,4)`, `(4,4)`, and `(4,8)`, with
`(4,4)` serving as the default.

### 11.4 Kolmogorov flow

`Cases/kolmogorov/` should define:

- canonical dataset `kolmogorov.h5`, shape
  `[100,201,65536,1,1,3]`, fields `u, v, p`, and a verified 256 by 256
  periodic square grid on `[0,2*pi)^2`;
- use stored trajectory splits: 80 train, 10 validation, 10 test;
- conditions `reynolds_number`, `forcing_amplitude`, and
  `forcing_wavenumber`; the current canonical file holds these fixed at
  `40, 1, 4`, so this first dataset tests reconstruction rather than
  held-out-condition generalization;
- auxiliary solver-native vorticity as evaluation/physics context, not as a
  hidden reconstruction input or an extra target channel;
- initial sensor protocols `u_only_random`, `v_only_random`,
  `velocity_shared_random`, and `velocity_structured_sr`;
- primary task: infer all `u,v,p` fields from sparse colocated velocity
  measurements, with pressure reported explicitly as the unmeasured field;
- asymmetric single-component tasks `u -> (u,v,p)` and `v -> (u,v,p)` to
  quantify cross-field inference rather than assuming both velocity components
  are always measurable;
- PointCloudFFM with `gl_rbf_enh/topk_rbf` as the initial point-field model and
  both GeoFNO and the optional PointCloudFFM FNO backbone on the verified grid;
- the global distribution family with self/mutual/cross settings for `u,v,p`;
  cross-spectrum and topology remain future peer families, although this case
  is a natural later target for both;
- a later case physics provider for incompressible Navier-Stokes momentum,
  zero divergence, periodic boundaries, the zero-mean pressure gauge, and the
  sinusoidal body force. Until that provider is implemented, these are
  evaluation requirements rather than silently active training losses;
- physical diagnostics including per-field error, pressure-gauge error,
  divergence, velocity/vorticity error, kinetic energy and spectra, momentum
  residuals, energy, and enstrophy.

Because one snapshot contains 65,536 points, point models should use
deterministically sampled query subsets during ordinary training and chunked
full-field reconstruction for evaluation. Grid models may consume the full
verified grid. The case README and run manifest must make this distinction
explicit so query subsampling cannot be confused with a change in the target
problem.

### 11.5 Mass transport-fluid (future case scaffold)

`Cases/mass_transport_fluid/` should define:

- demo dataset `mass_transport_fluid_demo.h5`, shape
  `[1,3,1024,1,1,3]`, fields `u_x, u_y, concentration`, 32 by 32 nonperiodic
  grid;
- auxiliary pressure and source field, conditions `A, x0, y0, s`, physical
  units, zero-mean pressure gauge, and nonperiodic boundaries;
- robust offset/99th-percentile normalization support in addition to mean/std;
- initial protocols `concentration_only_random`, `velocity_only_random`, and
  `concentration_structured_sr`;
- case-specific conservation, boundary flux, transport residual, pressure
  gauge, and source-relative visualizations;
- `benchmark_eligible: false` for the current one-trajectory, three-frame demo;
  this file is not the intended official dataset and the case remains a future
  scaffold until that dataset is available.

Use this file only to prove the loader, auxiliary/context interface, physical
unit handling, boundary-aware sensors, and visualization path. Before reporting
model comparisons, generate a named production variant with enough independent
trajectories and stored train/validation/test splits, then add a new dataset
README and config without changing the demo registration.

---

## 12. Documentation deliverables

### 12.1 `README.md`

Keep it operational and extensible:

- motivation and research questions;
- high-level architecture and directory map;
- environment setup and editable install;
- dataset link/setup/validation;
- one base-training, post-training, direct-PDE, and evaluation command;
- run artifact and reproducibility rules;
- contributor entry points: add a model, coherence family/component, dataset,
  or case;
- current support matrix and known limitations.

### 12.2 `ModelExplain.md`

Make this the scientific/method reference:

- formal sparse multi-field reconstruction notation;
- snapshot and space-time quasi-super-resolution observation/query
  representations;
- deterministic, operator, generative, and rectified-flow model summaries;
- hierarchy of data-driven coherence families, with self/mutual/cross documented
  specifically as components of global distribution coherence;
- reserved extension contracts for cross-spectrum and topological coherence;
- exact formula, normalization, estimator, differentiability, and complexity
  for every implemented family and component;
- reference-bank construction and leakage controls;
- post-training algorithms: direct differentiable, reward/RAM, LoRA/refiner;
- PDE post-training versus direct physics-informed training;
- uncertainty and evaluation protocol;
- model/upstream provenance, licenses, deviations, and citations;
- experiment/ablation map.

### 12.3 `CONTRIBUTING.md`

Document ownership boundaries and review checklists. A contribution should be
small enough to add through one registry and shared contract. Require:

- no case imports in general modules;
- no binary datasets or run artifacts in Git;
- typed public interfaces and config validation;
- deterministic seed handling;
- one focused contract test, not a new bespoke smoke suite;
- documentation/provenance for scientific definitions and external code;
- updated support matrix and example case config.

---

## 13. Minimal testing and validation strategy

Use one tiny synthetic HDF5 fixture that exercises two trajectories, multiple
times, two fields, coordinates, splits, statistics, and variable sensors. Tests
should be parametrized across registries.

Required automated checks:

1. HDF5/PT schema validation, field-name resolution, split isolation, and lazy
   multi-worker loading.
2. Sensor reproducibility, per-field counts, masks, no target leakage, and fixed
   evaluation manifests.
3. Every registered model consumes the standard batch and returns the required
   shape/metadata; stochastic models reproduce a fixed seed.
4. Differentiable coherence components have finite gradients; detached
   components declare that they cannot drive gradient post-training.
5. Reference-bank fitting touches training IDs only.
6. Base/post/direct-physics configs reject mixed or irrelevant keys.
7. Checkpoint resume and child-run lineage preserve config/data compatibility.
8. One real metadata-only or one-batch integration check per initial case.
9. Space-time downsampling reconstructs the exact full KS logical shape and
   independently honors temporal/spatial ratios.
10. Demo 50 compatibility loads strictly after explicit field mapping and
    matches one fixed-seed legacy reconstruction within tolerance.

Do not run full training in CI and do not create a separate dry-run script for
each backbone. A single `--validate-only` path may resolve config, dataset,
capabilities, and output location without initializing a long training job.

---

## 14. Step-by-step build plan and gates

### Phase 0 — Freeze decisions and protect scope

- [x] Confirm package name, directory names, and three training-stage meanings.
- [x] Record the leakage, target-use, and benchmark-eligibility rules from this
      document as architecture decisions.
- [x] Verify the worktree and avoid touching unrelated `datagen/` changes.
- [x] Add project `.gitignore` rules before creating links or runs.

**Gate 0:** only the planned project directory is changed; generated/binary
paths are demonstrably ignored.

**Execution record (2026-08-15): Gate 0 passed.** The package and directory
names, stage meanings, target-use/leakage rules, and benchmark eligibility are
frozen in this document and the contributor documentation. A final worktree
audit confirmed that all work from Phases 0--8 is contained under
`Proj_MultiFieldReconstruction/`; the pre-existing modified `datagen/` files
were identified and left untouched. Project-local ignore rules cover all six
linked HDF5 payloads, case run directories, build products, caches, and local
environments while retaining Markdown catalog records and benchmark metadata.

### Phase 1 — Documentation, packaging, and contracts

- [x] Create `README.md`, `ModelExplain.md`, and `CONTRIBUTING.md` skeletons.
- [x] Add `pyproject.toml`, `environment.yml`, package initialization, and the
      three config schemas.
- [x] Implement typed sample, observation, reconstruction, loss, case, model,
      coherence-family/component, and physics contracts.
- [x] Implement registries with duplicate-name/version checks.

**Gate 1:** editable import succeeds; config/contract tests pass without any
dataset or GPU.

**Execution record (2026-08-15): Gate 1 passed and re-audited.** The editable
package imports as version `0.1.0`, the nine model adapters register through one
versioned registry, and the documentation, environment, packaging, typed
contracts, and separated stage schemas are present. The review tightened
nested dataset/observation/optimization/runtime validation and tensor shape,
mask, field-ID, logical-grid, and sample-ID checks. All executable base configs
now pass the strict schema; the two latent Stage-2 templates deliberately fail
until a Stage-1 checkpoint is supplied. These checks run on CPU without data.

### Phase 2 — Dataset catalog and general loading

- [x] Write `Dataset/README.md` and `SCHEMA.md`.
- [x] Add the five detailed dataset READMEs.
- [x] Create and validate the six relative links listed in Section 5.6.
- [x] Implement lazy HDF5 and trusted-PT loaders, reversible grid flattening,
      stored trajectory splits, ordered 80/10/10 frame splits, space-time
      reconstruction units, and normalization.
- [x] Implement a concise dataset validator and dataset/case compatibility
      report.

**Gate 2:** all five data sources pass structural validation; Brusselator,
Kolmogorov, and KS trajectory split IDs are disjoint; combustion resolves to
exact ordered 80/10/10 frame ranges; mass transport is rejected as a formal
benchmark unless explicitly allowed.

**Execution record (2026-08-15): Gate 2 passed and re-audited.** The five
registered sources and all six relative catalog links resolve. Validation
reconfirmed shapes of `(1,10000,40300,1,1,5)` for combustion,
`(100,241,36864,1,1,2)` for Brusselator,
`(100,201,65536,1,1,3)` for Kolmogorov,
`(100,401,256,1,1,1)` for KS, and `(1,3,1024,1,1,3)` for the mass demo.
Stored 80/10/10 trajectory splits are disjoint and exhaustive for the three
100-trajectory sources; combustion uses exact ordered frame ranges
`[0,8000)`, `[8000,9000)`, and `[9000,10000)`. The review added rank, geometry,
statistics, trajectory-ID, time, and split validation, plus one shared format
factory so trusted PT mappings now use the same training/evaluation entry
points. The mass demo remains structurally valid but formally ineligible.

### Phase 3 — Observation protocol and run store

- [x] Implement sensor samplers, query selection, padding/masks, grid
      rasterization, and independent space/time uniform downsampling.
- [x] Implement deterministic evaluation sensor manifests.
- [x] Implement run creation, manifest, provenance, checkpoints, histories,
      resume validation, and child-run lineage.
- [x] Create thin `run.py` entry points for all five cases.

**Gate 3:** the same saved sensor manifest produces byte-identical observation
indices for two different model adapters; an interrupted tiny run resumes
without changing lineage.

**Execution record (2026-08-15): Gate 3 passed and re-audited.** Persisted
sensor evidence rebuilds byte-identical point/field indices across adapters;
fixed/variable per-field counts, shared locations, structured grids, padded
masks, and independent KS time/space strides are covered by contract tests.
Run-store tests verify atomic checkpoint loading, config-hash rejection on an
incompatible resume, and child lineage, while the two-step resumed Phase-4
PointCloudFFM run retains one run ID and ordered history. The review centralized
sensor-config translation for all stages and now rejects empty, duplicate, or
out-of-range manifest observations before tensor indexing.

### Phase 4 — Planned models and plain data-loss baselines

- [x] Implement the narrow PointCloudFFM with default
      `gl_rbf_enh/topk_rbf`, IID/RFF priors, RF objective/sampling, and optional
      FNO backbone. Do not expose other new gather modes.
- [x] Implement the isolated Demo 50 compatibility model/importer, explicit
      field mapping, and fixed-seed equivalence check.
- [x] Implement `coordinate_mlp`, `mlp_rbf`, Senseiver, DeepONet, Geo-FNO,
      latent FM, and an allowed DiffusionPDE adapter; activate PINN only where a
      physics provider exists.
- [x] For every applicable adapter, run a small real-case experiment using only
      its ordinary data/native generative loss and save a common evaluation.
- [x] Validate parameter counts, full-field/space-time chunking, observation
      consistency, deterministic sampling seeds, and optional dependencies.

**Gate 4:** every planned model has at least one successful plain-training
result on an applicable real case; PointCloudFFM defaults to
`gl_rbf_enh/topk_rbf`; optional FNO works on a regular grid; Demo 50 reproduces
one legacy output without runtime imports from the demo.

**Execution record (2026-08-15): Gate 4 passed and re-audited.** Real-data
one-update integration runs exist for coordinate MLP, MLP-RBF, DeepONet,
Senseiver, Geo-FNO, DiffusionPDE, latent FM Stages 1 and 2, and PointCloudFFM
with both GL-RBF-ENH and FNO backbones; their run manifests record parameter
counts, normalization, sensor evidence, checkpoints, and common evaluation.
Kolmogorov and full-space-time KS runs additionally exercise large-grid and
chunked query paths. The default PointCloudFFM config remains exactly
`gl_rbf_enh/topk_rbf`, and unsupported gather modes are rejected. The isolated
Demo50 test was rerun on physical GPU 1 and matched the legacy fixed-seed
reconstruction without a runtime dependency from new code to the demo. The
review also added explicit zero-step and invalid diffusion-width errors. These
runs prove integration only; they are not completed training or performance
evidence.

### Phase 5 — Global distribution coherence and post-training

- [x] Implement the hierarchical family registry and global distribution
      family with self/marginal W2, mutual/pairwise SWD, and cross/joint top-k
      SWD components; keep observation consistency outside the family taxonomy.
- [x] Add reserved family contracts/READMEs for cross-spectrum and topology
      without placeholder scientific calculations.
- [x] Implement training-only reference-bank fitting and serialization.
- [x] Reproduce the direct differentiable post-training semantics of
      `config_pointcloud_ffm_direct_posttrain.yaml`, including scheduled clean
      rollout, inherited base config, endpoint smoothing, component histories,
      weighted sum, and optional ConFIG.
- [x] Run that post-training first from Demo 50 and then from a new
      `gl_rbf_enh/topk_rbf` checkpoint; add reward/RAM and LoRA only afterward.
- [x] Add explicit paired-supervised and target-free configs and assertions.
- [x] Verify nonzero finite gradients and base-run immutability.

**Gate 5:** Demo 50 and one new PointCloudFFM base run each produce immutable
child post-training runs with equivalent legacy semantics, finite gradients,
and fixed-manifest before/after metrics. Target-free configurations do not
access validation/test dense targets.

**Execution record (2026-08-15): Gate 5 passed.** The hierarchical family,
training-reference bank, paired-supervised compatibility mode, differentiable
Euler/Heun rollout, endpoint consistency, weighted-sum/ConFIG updates, child
lineage, fixed-manifest evaluation, and reserved peer-family contracts are
implemented and covered by focused tests. Minimal GPU1 child runs passed for
both Demo50 and a native `gl_rbf_enh/topk_rbf` source; their integration metrics
are pipeline evidence only, not scientific performance claims.

### Phase 6 — Cross-model post-training comparison

- [x] Apply the first-release global distribution post-training composition to
      every compatible plain base run from Phase 4.
- [x] Use identical case splits, sensor manifests, reference banks, evaluation
      seeds, and before/after metrics.
- [x] Record unsupported combinations explicitly instead of adding model-specific
      shortcuts to the trainer.
- [x] Compare gain from post-training as well as absolute final performance,
      runtime, memory, and inference cost.

**Gate 6:** every compatible planned family has a matched plain-base and
post-trained pair, and each unsupported pair has a documented capability reason.

**Execution record (2026-08-15): Gate 6 passed.** Nine compatible Phase-4
Brusselator sources produced immutable one-step GPU1 children through one
differentiable-reconstruction contract: coordinate MLP, MLP-RBF, DeepONet,
Senseiver, GeoFNO, DiffusionPDE, latent FM stage 2, and PointCloudFFM with FNO
and GL-RBF-ENH backbones. A strict comparison artifact verifies identical data,
training bank, sensor and query manifests, seeds, split, and target-use label;
it reports MSE/coherence changes, runtime, warmed-up inference cost, and peak
memory. PINN and latent FM stage 1 have explicit capability reasons. These
short runs establish integration coverage only, not scientific performance.

### Phase 7 — Case completion

- [x] Turbulent combustion: verify coordinate topology, create the standardized
      chronological 80/10/10 frame split, port plots/global-distribution
      settings, and create initial base/post-training configs.
- [x] Brusselator: add periodic physics provider, `u -> v`/`v -> u` protocols,
      and make it the first formal benchmark.
- [x] Kolmogorov: expose the verified periodic grid, velocity-to-pressure and
      single-component protocols, vorticity context, flow diagnostics, and
      initial base/post-training configs.
- [x] KS: add reversible `(t,x)` layout, independently tunable uniform temporal
      and spatial downsampling, space/time metrics, and quasi-super-resolution
      base/post-training configs.
- [x] Mass transport-fluid: add auxiliary fields, robust normalization,
      nonperiodic physics/plots, and integration-only safeguards.

**Gate 7:** each case can run validation, one base update, one reconstruction,
and one evaluation report from its own directory. KS reconstructs the exact
full space-time state for at least two distinct `(temporal, spatial)` ratio
pairs. Mass-transport demo results remain outside formal comparison summaries.

**Execution record (2026-08-15): Gate 7 passed.** All five real payloads pass
their case-local validators and common checkpoint evaluator. Combustion's
40,300 stored points were verified as a complete permuted 100 by 403 Cartesian
grid and canonically reordered; its ordered 8000/1000/1000 split remains based
on frame position. Brusselator has differentiable periodic reaction-diffusion
physics. Kolmogorov exposes vorticity and periodic flow diagnostics. KS
reconstructed all 102,656 `(t,x)` queries for both `(2,4)` and `(4,8)` sensor
ratios. The mass demo exercised robust normalization, auxiliary context,
nonperiodic diagnostics, and visualization while remaining benchmark-ineligible.

### Phase 8 — Benchmark and reproducibility release

- [x] Freeze benchmark sensor manifests and config hashes.
- [x] Run deterministic baselines before expensive generative sweeps.
- [x] Run base, data-driven post-training, PDE post-training where available,
      and direct-PDE ablations with matched budgets.
- [x] Aggregate per-field reconstruction, coherence, physics, uncertainty, and
      compute metrics with trajectory-aware statistics.
- [x] Audit claims, licenses, run lineage, checksums, and documentation.

**Gate 8:** another collaborator can relink data, install the environment,
reproduce one documented result from a case directory, and trace every reported
number to a resolved config, checkpoint, sensor manifest, dataset checksum, and
code commit.

**Execution record (2026-08-15): Gate 8 passed for the integration release.**
The checked-in `benchmarks/v0_integration/` artifact freezes a portable sensor
manifest and matched one-update plain, data-coherence post-training, periodic
PDE post-training, and direct-PINN rows. It aggregates reconstruction,
coherence availability, PDE diagnostics, uncertainty availability, compute,
and trajectory-aware statistics, and traces each row through config,
checkpoint, sensor, query, dataset, and code hashes. `REPRODUCIBILITY.md`, the
safe dataset linker, a case-local reproduction script, CI, and the license/
claim audit document the collaborator workflow. The one-trajectory values are
explicitly integration evidence and cannot support method-ranking claims.

**Cross-phase code-review record (2026-08-15): Phases 5--8 rechecked.** The
full regression suite re-exercises coherence-family composition and leakage
barriers, immutable post-training lineage and matched comparison evidence,
case-owned physics/diagnostics, KS full-space-time reconstruction, and release
aggregation. During review, source-model loading was separated from the
data-coherence trainer, stage-independent dataset/sensor helpers replaced
cross-trainer private imports, optional post-training evaluation settings were
made safe, and empty-split/zero-update failures now report before training.
Lint, package build/install, five real-data validators, the GPU1 Demo50
equivalence test, and the documented one-step reproduction workflow are the
final release checks; exact command outcomes are summarized in the handoff.

---

## 15. Recommended initial execution program

Run two complementary tracks as soon as the shared contracts exist.

### Track A — Fast compatibility validation

1. Register the combustion compatibility dataset `Merged_COTU0U1P.h5` with
   verified fields `CO,T,U_0,U_1,p` and the ordered 80/10/10 frame split.
2. Import Demo 50 through the isolated legacy compatibility path and match one
   fixed-seed legacy reconstruction.
3. Translate `config_pointcloud_ffm_direct_posttrain.yaml` into the structured
   global-distribution post-training config.
4. Create a child post-training run from Demo 50 and verify data loss,
   differentiable rollout, self/mutual/cross component losses, observation
   consistency, gradient balancing, checkpointing, and before/after evaluation.

This provides a quick, high-value validation of the new post-training framework
without waiting for a new large combustion base run.

### Track B — Planned-model base/post comparison

1. Use Brusselator with stored trajectory splits and one fixed `u_only_random`
   manifest as the first common multi-field benchmark.
2. Train every applicable planned model with only its ordinary data/native
   generative objective. PointCloudFFM uses `gl_rbf_enh/topk_rbf` by default;
   its FNO backbone is a separate optional run.
3. Evaluate all plain base runs through the same reconstruction metrics and
   global distribution family/component diagnostics.
4. Post-train every compatible run with the same training-reference global
   distribution composition and re-evaluate on the unchanged manifest.
5. Report both absolute performance and the change caused by post-training.
6. Add Kolmogorov immediately afterward as the first three-field
   velocity-to-pressure flow benchmark, using its stored trajectory splits and
   verified periodic grid.
7. Add KS to validate space-time quasi-super-resolution at independently
   varied temporal/spatial ratios.

These tracks jointly test backward compatibility, plain planned-model
performance, multi-field inference, and cross-backbone post-training. They
should precede broad hyperparameter sweeps, future coherence families, and
large direct-PDE ablations.

---

## 16. Definition of project readiness

The directory is ready for multi-person research work when all of the following
are true:

- project imports and launch commands do not depend on the demo directory;
- dataset binaries remain external/local while synchronized READMEs are
  sufficient to obtain, link, validate, and understand them;
- cases launch locally but share loaders, trainers, models, coherence
  families/components, evaluation, and run storage;
- base training, post-training, and direct physics-informed training have
  separate validated configurations and artifacts;
- every model consumes the same sparse-observation contract;
- every coherence family/component declares target use, units,
  differentiability, and reference provenance;
- self/mutual/cross remain correctly nested under global distribution, while
  cross-spectrum and topology can be added as peer families;
- benchmark splits and sensor manifests prevent cross-model and temporal
  leakage;
- single-trajectory datasets use the standard chronological 80/10/10 frame
  split, Kolmogorov uses disjoint stored trajectory splits, and KS supports
  independent spatial/temporal downsampling;
- run directories are self-describing, resumable, immutable by lineage, and
  ignored by Git;
- contributors can add one model, term, dataset, or case without editing a
  monolithic trainer;
- Demo 50 compatibility post-training and the first Brusselator planned-model
  base/post comparison are reproducible end to end.

These conditions are more important than immediately migrating every existing
model. The project should become broad by adding well-defined registry entries,
not by weakening the common interfaces.

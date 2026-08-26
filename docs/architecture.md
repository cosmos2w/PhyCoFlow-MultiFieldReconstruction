# Architecture

PhyCoFlow is organized around a small set of contracts so that model
experiments can be shared across scientific cases without moving case logic
into the reusable package.

## Dependency direction

```text
datasets/ ──> case data contract ──> ObservationBatch ──> model adapter
                                  │                         │
                                  │                         ├─ native loss
                                  │                         └─ reconstruction
                                  │
                                  └─> case physics/diagnostics

model adapter + contracts ──> shared training / post-training ──> runs/
model adapter + contracts ──> shared evaluation ────────────────> reports/
```

The core package must remain independent of named cases, case directories,
run directories, and the project CLI. Case modules may import shared
contracts and register a `CaseSpec`; the reverse dependency is prohibited.

## Reusable package and point-cloud boundary

All installable code is below the single `phycoflow_reconstruction` namespace.
The point-cloud flow implementation deliberately has two layers:

```text
src/phycoflow_reconstruction/models/flows/pointcloud/
├── core/
│   ├── attention, geometry, priors, observation
│   ├── GL-RBF / GL-RBF-CQ tensor architectures
│   └── tensor training and reconstruction primitives
├── runtime/
│   ├── model builder and config identity resolution
│   ├── EMA and checkpoint auxiliary-state helpers
│   └── cache-aware tensor training/reconstruction runtimes
└── adapters/
    ├── pointcloud_ffm_adapter.py
    └── gl_rbf_cq_adapter.py
```

The core accepts tensors and model configuration only. It does not know about
`CaseSpec`, coherence families, sensor YAML files, or run storage. Adapters
translate `ObservationBatch` and `LossBundle`, expose model capabilities, and
implement the lifecycle hooks used by shared training and evaluation. This
boundary is especially important for GL-RBF/CQ: module relocation must not
rename serialized state keys or alter seeded initialization, gradients, EMA,
cached-K/V execution, query microbatching, or geometry/reconstruction caches.

The registered identities remain stable:
`coordinate_mlp`, `mlp_rbf`, `pinn`, `deeponet`, `senseiver`, `geofno`,
`diffusion_pde`, `latent_fm`, `pointcloud_ffm`, and `gl_rbf_cq`. Historical
Demo50 support is isolated in `models/compatibility/` and is not a modern base
model choice.

## Cases and datasets

`cases/<name>/` owns field order and units, logical geometry, sensor protocols,
dataset selection, physical diagnostics/providers, and case-specific launch
profiles. A case `run.py` is a thin command entry point into the shared CLI.
General model/trainer code must not import case names.

`datasets/` is a catalog and contract location. Large payloads and links stay
local; tracked files describe schema, fields, split policy, and any required
training-only normalization. The loader exposes a normalized `FieldSample`
and the sensor layer constructs a padded `ObservationBatch`, which is the only
data shape consumed by project adapters.

## Configuration composition

Generic model definitions live once in `configs/models/`. Shared runtime,
optimization, evaluation, and checkpoint policy belongs in
`configs/defaults/`. A case launch config composes those fragments with its
dataset and sensor protocol, then adds only case-scoped scientific settings
or an experiment name. Sensor geometry, normalization source, coherence
weights, physical geometry, and benchmark schedules remain case-specific when
they carry scientific meaning.

Config loading resolves defaults recursively, resolves paths relative to the
declaring file, applies dotted command-line overrides, and validates the
stage/model contract. Every tracked canonical YAML must either resolve and
validate or be an explicitly documented incomplete template (for example a
latent Stage-2 config before its source checkpoint is supplied).

## Training, post-training, and artifacts

Base training writes an immutable run lineage and complete checkpoint state.
Data-driven coherence and physics post-training create child runs that record
their source checkpoint/config/data hashes; they never mutate the source. The
shared evaluator accepts either run and produces traceable metrics, optional
portable plotting payloads, and case diagnostics.

Normal run outputs belong under `cases/<case>/runs/`, which is local and
ignored. This includes checkpoints, manifests, histories, telemetry,
reference banks, previews, generated reports, and figures. `benchmarks/`
contains only reproducible protocols, source configs/scripts, and small
immutable fixtures or concise validation summaries suitable for review.

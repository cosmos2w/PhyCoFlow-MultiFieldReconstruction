# Efficient sliced-persistence post-training

The current topology method combines `cubical_persistence`,
`persistence.distance: sliced_wasserstein`, and
`optimization.gradient_balance: topology_regularized`.
[Shared training defaults](configs/defaults/topology_post_training.yaml) define
optimization and fidelity policy. Each case supplies its fields, geometry,
sensors, and persistence components; see the
[active-emulsion example](cases/active_emulsion/configs/posttrain/topology.yaml).

## Objective and updates

Exact cubical pairings identify all positive-persistence finite bars and essential
births for H0/H1. Sliced Wasserstein distances compare the generated and paired
reference diagrams. Self components act on individual fields; mutual components
use fixed positive-line slices of field groups or derived descriptors.

One Adam update minimizes a batch-average topology score plus endpoint and
source-anchor fidelity penalties. Component weights are calibrated once on
training data and persisted with the run. Short final batches use their actual
sample count. The source model is frozen. Fixed validation samples and generation
seeds determine whether topology and total/per-field fidelity satisfy the
configured gates for `best.pt`; an individual update does not guarantee that every
component improves.

`sampling: full_pass` shuffles the selected training snapshots without replacement.
An optional `steps_per_epoch` changes reporting intervals without resetting that
stream. Recovery restores optimizer state, sample position, and random state.

## Setup and launch

Install the project and topology dependency in your environment:

```bash
python -m pip install -e '.[topology,plot]'
python cases/active_emulsion/run.py post-train \
  --config cases/active_emulsion/configs/posttrain/topology.yaml \
  --override source_run=/path/to/completed/native/run
```

The source run must contain `resolved_config.yaml` and the selected checkpoint
(`best.pt` by default). The CLI inherits its dataset, model, and observations;
case-specific topology fields and geometry must match that dataset. Source paths
may be absolute or relative to the case directory. Run data and checkpoints remain
local and are ignored by Git.

For bounded segments and automatic recovery of a matching child run:

```bash
python -m phycoflow_reconstruction.training.segmented \
  --case-dir cases/active_emulsion \
  --config cases/active_emulsion/configs/posttrain/topology.yaml \
  --override source_run=/path/to/completed/native/run \
  --segment-epochs 100 --allocation-hours 24
```

Rerun the same command to resume. `--requeue` explicitly enables Slurm requeue at
the allocation deadline. Request cluster resources and activate your environment
before launch; the shared launcher does not select an account, partition, GPU
model, or environment. Ambiguous children and configuration mismatches are errors.

## Execution controls

For physical GL-RBF/CQ models, `optimization.rollout_execution.context_cache`
accepts `none`, `condition`, `geometry`, or `static_features`. Shared contexts
require `model_mode: eval`, retain gradients, and are rebuilt for each rollout.
`checkpointing: true` trades recomputation for lower activation memory. These
options preserve the model's parameter names, neighbor-search shape, and readout
chunking.

`PHYCOFLOW_TOPOLOGY_WORKERS` controls CPU pairing workers, and
`PHYCOFLOW_TOPOLOGY_REFERENCE_CACHE` bounds the in-memory reference cache. Choose
worker counts within the allocated CPUs and benchmark on the intended hardware.

Optional `training_subset` uses per-snapshot metadata. Configure `strata_keys`
for proportional group allocation and `trajectory_key`, `time_key`, `frame_key`,
and `split_key` for trajectory/time coverage. Defaults are no strata,
`trajectory_id`, `time`, `frame`, and `split`. Every selected row must have split
`train`; selection never reads validation or test targets. The manifest records
the mapping, selected indices, and dataset fingerprint.

## Code and verification

| Responsibility | Module under `src/phycoflow_reconstruction/` |
|---|---|
| Rollout context | `models/compatibility/physical_gl_rbf_cq.py` |
| Pairing and sliced distances | `coherence/families/topology/persistence.py` |
| Persistence components | `coherence/families/topology/persistence_objective.py` |
| Optimizer and calibration | `training/topology_constraints.py` |
| Lifecycle and held-out selection | `training/post_training.py`, `training/topology_selection.py` |
| Sampling and reporting | `data/topology_subset.py`, `training/update_budget.py` |
| Segmented recovery | `training/segmented.py` |

Synthetic tests cover persistence gradients, rollout value/gradient/update parity,
source immutability, short batches, and interrupted versus uninterrupted training.
Run `ruff check src tests scripts cases benchmarks` and `pytest` before review.

Historical Betti-curve and spatial strategies remain explicitly selectable for
compatibility. Completed runs with implementation hashes require their recorded
source for exact recovery; this refactor is intended for new runs. See the
[family guide](src/phycoflow_reconstruction/coherence/families/topology/README.md)
for mathematical definitions and [provenance](docs/provenance.md) for attribution.

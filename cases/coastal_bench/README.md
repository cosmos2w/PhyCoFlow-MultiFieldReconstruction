# CoastalBench reconstruction and post-training

This case reconstructs `u`, `v`, `temp` and `salt` on an ocean point cloud using
GL-RBF/CQ. The maintained workflow prepares a canonical HDF5 input, trains a
source model, and post-trains with spectral coherence alone or with distribution
and topology. All commands below run from the repository root.

## Data contracts

The [dataset catalog](../../datasets/coastal_bench/README.md) describes the two
prepared inputs. Field values are `[trajectory, time, point, 1, 1, field]` and
longitude/latitude are the two active coordinate axes. The logical point-list
shape is not the bounding raster shape.

The original profile uses 30,542 points and the file's stored trajectory splits.
It uses TRAIN-only mean/std normalization, generated without modifying the input:

```bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python scripts/data/link_dataset.py --case coastal_bench --source /path/to/coastal_bench.h5
python cases/coastal_bench/prepare.py
python cases/coastal_bench/run.py validate --config configs/base/gl_rbf_cq.yaml
```

`prepare.py` validates disjoint stored splits, computes training statistics in
float64, and binds the normalization artifact to the input fingerprint. It rejects
changed input selections unless explicitly recomputed. Its artifacts remain local.

The calendar profile expects the prepared 576-training-day, hourly crop at
`datasets/coastal_bench/calendar_576days_hourly/coastal_bench.h5`, or an explicit
`--override dataset.path=/path/to/calendar.h5`. It has 30,389 retained wet points,
24 hourly frames per day and stored TRAIN-only robust normalization (training
median and p99–p1). `auxiliary/native_grid_indices` must contain integer
`[point, 2]` row/column indices in the `[201, 381]` native grid. The spectral and
masked-topology recipes require this calendar geometry. The retained wet mask is
fixed across all states; do not interpolate land into the coherence domain.

This repository does not contain payloads or generated extraction records. Use
an already prepared canonical input matching the selected profile. Stored split
boundaries and a fixed availability mask are data contracts, not proof of
statistical independence. Keep validation/test values out of normalization.

## Base training and recovery

```bash
python cases/coastal_bench/run.py train-base --config configs/base/gl_rbf_cq_calendar.yaml
```

The calendar recipe uses 54 updates per reporting epoch while continuing to
sample the full TRAIN split. The default base-training budget remains one
nominal dataset pass when `optimization.steps_per_epoch` is omitted.

For bounded execution, `train_segment.py` freezes one resolved configuration and
uses a lock plus a run pointer to resume the same optimizer/EMA/RNG state:

```bash
python cases/coastal_bench/train_segment.py --state-dir /path/to/chain \
  --config cases/coastal_bench/configs/base/gl_rbf_cq_calendar.yaml --segment-epochs 100
```

Invoke the same command again to continue. If interrupted before the initial
pointer is written, inspect the created run and set `run_dir.txt` to its absolute
path with a valid `checkpoints/last.pt`; the launcher refuses to silently start a
second model. Use the case CLI for dataset overrides before freezing a run.

## Post-training recipes

Every recipe requires an explicit source run and inherits its dataset, model and
sensor contract. The calendar source must expose `native_grid_indices` and match
the configured geometry. Start with a complete source checkpoint; short runs
marked integration-only are rejected by default.

| Config under `configs/posttrain` | Objective and update policy |
|---|---|
| `data_only.yaml` | Native continuation control; spectral evaluation remains enabled |
| `spectral.yaml` | Native loss plus fair signed-band/joint spectral score, fixed sum |
| `unified.yaml` | Native loss plus distribution, masked topology and spectral score, fixed sum |
| `unified_fidelity.yaml` | Internal ConFIG over the three families with the current SOURCE-relative Lagrangian endpoint-MSE policy |

```bash
python cases/coastal_bench/run.py validate --config configs/posttrain/unified.yaml \
  --override source_run=/path/to/source-run
python cases/coastal_bench/run.py post-train --config configs/posttrain/unified.yaml \
  --override source_run=/path/to/source-run
```

Each reporting epoch is 54 updates of continuous full-pass sampling. Recipes use
16 Euler steps and production `endpoint_smooth` observation guidance. The public
CQ velocity hook follows production query chunk boundaries, including near-tie
sensor selections. The case tests check production parity. At runtime, the external fidelity
policy checks its declared endpoint operator with and without gradients.

The spectral term uses independent draws, fair alpha `1`, joint weight `0.25`
and no coherence-CRPS auxiliary. Its 16/8/4/2 km bands are overlapping spatial
filters. Geometry and normalization scales are frozen from TRAIN references.
Distribution/topology share the first draw. See the
[spectral definition and limitations](../../docs/spectral_post_training.md).

Masked topology maps native wet vertices exactly to the raster. Land vertices
enter at infinity, do not contribute finite bars, and receive zero gradient.
Classes killed only by excluded land remain essential on the wet domain.
Nonperiodic boundaries, zero smoothing and no downsampling are required.

These maintained recipes use the current shared framework. They do not recreate
historical campaign optimizer trajectories. The optional fidelity recipe bounds
per-field endpoint MSE relative to fixed SOURCE/TRAIN estimates; it does not
implement the historical native-loss/ensemble-mean feedback controller. This
stochastic policy is not a guarantee that every validation field stays in budget.
For weighted-sum recipes, checkpoint MSE selection uses the configured preview;
the external-policy recipe uses its validation panel. Compare explicit matched
milestones when assessing scientific changes across policies.

## Evaluate and visualize

Compare source/control/candidate checkpoints on identical frozen geometry,
calibration, states, sensors and seeds. The evaluator checks fingerprints and
metric definitions before combining results. It writes only to the output path:

```bash
python cases/coastal_bench/evaluate.py arms /path/to/evaluation \
  control=/path/to/control-run:epoch_1000 candidate=/path/to/candidate-run:epoch_1000 \
  --control control:epoch_1000 --draws 32 --steps 16
python cases/coastal_bench/visualize.py /path/to/candidate-run --checkpoint epoch_1000 \
  --draws 16 --output /path/to/maps
```

The score report uses fair alpha `1` and all draws by default. Deliberate legacy
score/sampler ablations must be labeled separately. Field CRPS is an evaluation
diagnostic, not an enabled training constraint. `visualize.py` plots the first
recorded condition's reference, ensemble mean, one draw and that draw minus the
mean for each band/field, with common colour limits. Pair maps with scores and
power/spread diagnostics; increased sharpness or power alone is not calibration.

The reusable evaluator is `phycoflow_reconstruction.evaluation.spectral`.
`python -m phycoflow_reconstruction.training.posttrain_rescore` also restores a
run's exact saved panel and scores selected checkpoints through the shared
trainer. Restore verifies source files, family/calibration artifacts and saved
sensor/query identities. Historical experimental-optimizer runs require their
original software; current validation deliberately does not reinterpret them.

## Validation

Synthetic tests cover TRAIN-only normalization, holdout preservation, source-file
immutability, segmented recovery, explicit base budgets, masked-domain Betti
counts and gradients, production rollout parity, and shared three-family updates.
No real dataset or GPU is required for these CPU contract tests:

```bash
pytest tests/test_coastal_*.py tests/test_base_update_budget.py \
  tests/test_topology_masked.py tests/test_multiscale_*.py tests/test_spectral_*.py
```

Install the project's `topology`, `multitask`, `posttrain_constraints` and `plot`
extras for the full workflow. GPU memory and convergence on a full CoastalBench
run still require an actual experiment; unit/integration tests do not establish
those empirical results.

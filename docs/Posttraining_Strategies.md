# Post-training strategies

The framework supports several opt-in ways to combine reconstruction fidelity and coherence objectives. This guide identifies the route boundaries and the portable Run 11f-style recipe; it makes no method-superiority or universal-fidelity claim. For measured outcomes, see [Posttraining results and protocol](Posttraining_Results.md), and for the detailed SOURCE-relative policy mathematics, see [Fidelity-budget post-training](Fidelity_Budget_Posttraining.md).

## Choose a route

| Route | Main selectors | What is combined |
| --- | --- | --- |
| Legacy | Omit `optimization.multitask`; preserve the historical profile and its `optimization.gradient_balance` setting. | Existing objective and gradient-balancing behavior. |
| Scalarized or hierarchical multitask | `family_scalarization` is `fixed_sum`, `stch`, or `dwa`; `outer_aggregation` is `sum`, `torchjd_config`, `upgrad`, or `cagrad`. | Scalarize A/B/T first, then combine that coherence task with the other active task gradients such as native reconstruction retention. |
| Joint multitask | `family_scalarization: {method: none}` plus an `outer_aggregation` selector. | Keep calibrated A/B/T separate so the outer aggregator can combine RF, A, B, and T as distinct active tasks. |
| Scalar external fidelity policy | Select `external_policy.method` as `lagrangian`, `rectify`, or `extended_log_barrier`; select a family scalarizer; leave `outer_aggregation` null or omit it. | Combine the scalarized coherence direction with per-field SOURCE-relative endpoint constraints. |
| Internal A/B/T ConFIG plus external policy | Set `family_scalarization: {method: none}`, `coherence_aggregation: {method: torchjd_config}`, `outer_aggregation: null`, and an `external_policy`. | Apply TorchJD ConFIG to separate A/B/T gradients, then add the selected external fidelity-policy direction. The reported scientific A+B+T sum is not the ConFIG scalar objective. |

The separate component-constrained and topology-regularized modes use the legacy `optimization.gradient_balance` selector and do not compose with `optimization.multitask`. Physics-only post-training also does not accept the multitask namespace.

## Compatibility boundaries

The validator rejects a non-null `outer_aggregation` together with `external_policy`; external policies require `vector_reduction: sum` and `objective_scale: source_mean`. They cannot be combined with a non-default legacy gradient balancer, native data retention, endpoint or source-anchor objectives, component constraints, non-full-model trainable scope, coherence warm-up or interval rescaling, recurring family-gradient diagnostics, or a checkpoint selector other than `reconstruction_mse` or `topology_with_fidelity`.

Internal `coherence_aggregation` is limited to TorchJD ConFIG, requires explicit `family_scalarization.method: none`, requires an external policy, and requires all of global-distribution, cross-spectrum, and topology families to be enabled. Keep the family and objective definitions in the case configuration; the selector does not make arbitrary subsets or a hidden scalarization.

Existing `optimization.multitask` configurations without an external policy retain the hierarchical and joint aggregation routes. Invalid multi-task directions stop with an error instead of falling back to a weighted sum. The internal TorchJD route uses full-width FP64 aggregation and may fail explicitly if a valid direction cannot be constructed.

## Install the primary route dependencies

From the repository root, install the optional extras required by the Lagrangian, internal ConFIG, topology and KeOps example:

```bash
python -m pip install -e '.[posttrain_constraints,topology,keops,multitask]'
python -m pip check
```

`posttrain_constraints` supplies Cooper for the Lagrangian controller; `multitask` supplies TorchJD; `topology` and `keops` support the configured persistence and operator path. The `posttrain` extra is only needed for the older legacy ConflictFree path. Add `plot` if you want to render figures with Matplotlib; it is not needed for training or the CPU-only history summarizer.

## Select an alternative external policy

External-policy selectors compose with either a scalarized coherence loss or the internal A/B/T ConFIG direction. The first block below is a complete scalar external-policy selector map with fixed-sum families. Use a separate base profile whose resolved map omits `coherence_aggregation`, or replace the entire `optimization.multitask` map with this map using a CLI override. YAML defaults merge recursively: omitting the selector from a leaf profile does not remove an inherited selector, and setting it to null is invalid. This block is therefore not a leaf patch to the primary internal-ConFIG profile:

```yaml
optimization:
  multitask:
    family_scalarization: {method: fixed_sum}
    outer_aggregation: null
    external_policy:
      method: rectify
      vector_reduction: sum
      objective_scale: source_mean
```

Use one selected method's parameters only. For `lagrangian`, the method-specific map is `lagrangian: {dual_lr: 0.01, penalty_interval_updates: 1}`. For `extended_log_barrier`, use `extended_log_barrier: {t_initial: 1.0, growth_factor: 1.1, update_interval: 100, t_max: 100.0}`. `rectify` has no method-specific parameter map. The validator rejects unknown/inactive nested keys, so changing a Lagrangian config to another policy requires replacing the whole `external_policy` map; overriding only `method=rectify` leaves the inherited `lagrangian` map and fails validation.

When replacing the entire `optimization.multitask` map, retain the original `fidelity_budget` settings as well as the desired selectors if the allowance and SOURCE calibration must remain matched. Omitting that budget map selects its defaults, including four calibration batches rather than the primary recipe's eight.

Here is the complete selector block for the internal A/B/T ConFIG route with the extended log barrier:

```yaml
optimization:
  multitask:
    family_scalarization: {method: none}
    outer_aggregation: null
    coherence_aggregation: {method: torchjd_config}
    external_policy:
      method: extended_log_barrier
      vector_reduction: sum
      objective_scale: source_mean
      extended_log_barrier: {t_initial: 1.0, growth_factor: 1.1, update_interval: 100, t_max: 100.0}
```

For example, replace the full policy map when changing the primary internal-ConFIG profile from Lagrangian to rectify, preserve the internal ConFIG selector, and choose a separate output family for the new child:

```bash
python cases/turbulent_combustion/run.py validate \
  --config configs/posttrain/fidelity_budget/lagrangian_config_warmup.yaml \
  --override source_run="$SOURCE_RUN" \
  --override 'optimization.multitask.external_policy={method: rectify, vector_reduction: sum, objective_scale: source_mean}'
```

For a new full run with that alternative, also pass a fresh `output.experiment_name`; the primary recipe's default output family identifies the Lagrangian method. The internal-aggregation route still requires A, B and T all enabled. Scalar external policies may instead use the family subset supported by the selected case profile.

## Run 11f-style primary example

The maintained [Lagrangian, internal-ConFIG, warm-up profile](../cases/turbulent_combustion/configs/posttrain/fidelity_budget/lagrangian_config_warmup.yaml) is a portable 1,000-epoch recipe. It declares `source_run: null` so each invocation supplies a concrete SOURCE run, uses `source_checkpoint: last.pt`, and writes children under the `posttrain_lagrangian_config_warmup` output family. Its selectors are `family_scalarization: none`, `coherence_aggregation: torchjd_config`, `outer_aggregation: null`, and dense Lagrangian feedback with `penalty_interval_updates: 1`. Model LR rises linearly from `5e-6` to `5e-5` across the first 76 global successful updates and then stays at `5e-5`.

Existing `run11*` YAML profiles are retained as historical regression and reproduction profiles; use the maintained warm-up profile above as the portable primary recipe. Historical profiles may contain a bound SOURCE, so inspect their resolved provenance before reuse.

Choose a completed SOURCE run that contains the architecture, normalization, dataset and checkpoint intended for the experiment. Set `SOURCE_RUN` to that run directory; `validate` checks the composed config, SOURCE configuration inheritance and dataset without starting training. The bounded smoke loads the checkpoint and checks the native SOURCE status, model identity, field order and normalization.

```bash
SOURCE_RUN=/path/to/completed/source-run
CONFIG=configs/posttrain/fidelity_budget/lagrangian_config_warmup.yaml

python cases/turbulent_combustion/run.py validate \
  --config "$CONFIG" \
  --override source_run="$SOURCE_RUN" \
  --override source_checkpoint=last.pt
```

Run one bounded real-model update under a fresh output identity before starting a full child. This checks the resolved SOURCE, enabled A/B/T families, ConFIG direction, topology path, and external controller on the chosen environment.

```bash
python cases/turbulent_combustion/run.py post-train \
  --config "$CONFIG" \
  --override source_run="$SOURCE_RUN" \
  --override source_checkpoint=last.pt \
  --override output.experiment_name=posttraining_checks/lagrangian_config_warmup_smoke \
  --max-steps 1
```

For the full campaign, use the recipe's default output family and retain its 1,000-epoch horizon. A new invocation creates a distinct child under that family; save the run directory printed by the command as the absolute `CHILD_RUN` path.

```bash
python cases/turbulent_combustion/run.py post-train \
  --config "$CONFIG" \
  --override source_run="$SOURCE_RUN" \
  --override source_checkpoint=last.pt
```

Resume the same child directory with the same SOURCE, checkpoint, config and output identity; `--max-steps` optionally bounds this invocation. The normal checkpoint restores the model optimizer, controller, SOURCE calibration, RNG and update cursor, so the warm-up does not restart.

```bash
CHILD_RUN=/path/to/the/printed/child-run-directory
python cases/turbulent_combustion/run.py post-train \
  --config "$CONFIG" \
  --override source_run="$SOURCE_RUN" \
  --override source_checkpoint=last.pt \
  --resume "$CHILD_RUN"
```

Evaluate a named child checkpoint on validation by replacing `best` with `last` when that is the checkpoint under review. `evaluate-run` is a checkpoint evaluation entry point; it does not reproduce the report's matched fixed-panel comparison by itself.

```bash
python cases/turbulent_combustion/run.py evaluate-run \
  --run "$CHILD_RUN" \
  --checkpoint best \
  --split validation \
  --max-samples 128 \
  --query-points 4096 \
  --generation-steps 2 \
  --weight-selection live \
  --report-name selected_validation
```

Use a fresh output identity for every new child. Do not change a resumed child’s configuration to extend its epoch horizon; horizon extension is a separate explicit operation with its own checkpoint and manifest safeguards.

To extend a stopped child deliberately, choose a larger horizon and call the existing metadata operation once. For example, the following extends a saved 1,000-epoch child to 1,500 epochs; it preserves the pre-extension metadata/checkpoints and changes only the epoch target and its provenance binding. Ensure no process is writing that child while extending it. `CHILD_RUN` must be the absolute path to your child, and the original SOURCE and data must remain available at their recorded locations.

```bash
python - "$CHILD_RUN" <<'PY'
import sys
from phycoflow_reconstruction.training.run_store import RunStore

store = RunStore.extend_epochs(sys.argv[1], 1500)
print(store.run_dir / "resolved_config.yaml")
PY

python cases/turbulent_combustion/run.py post-train \
  --config "$CHILD_RUN/resolved_config.yaml" --resume "$CHILD_RUN"
```

For a new policy, budget, source checkpoint or scientific definition, create a fresh child instead of changing the saved configuration of an existing run.

## Read the fidelity metrics correctly

The training constraint measures pooled, masked endpoint MSE separately for each normalized model output field in model units. Its reference is calibrated from fixed TRAIN SOURCE batches; fixed validation SOURCE metrics form a separate reference for checkpoint selection and assessment. The TRAIN controller can influence validation behavior, but it does not enforce every validation field or guarantee unseen-population fidelity.

Per-field MSE changes are not physical relative-L2 errors. Report per-field and total endpoint MSE alongside the physical relative-L2 metric when that metric is evaluated, and keep raw A/B/T coherence values separate from a scalar diagnostic sum. Under internal ConFIG, the reported A+B+T sum is not the optimized scalar objective.

Checkpoint names answer different questions: `last.pt` is the terminal/recovery state; `best.pt` follows the configured validation selection rule and can remain the SOURCE fallback if no child checkpoint is eligible; `best_fidelity.pt` tracks the lowest aggregate fixed-panel MSE and does not guarantee each field meets its guard. Inspect the checkpoint's selection metadata and global update cursor before reporting it.

A smoke confirms execution, not efficacy. A selected checkpoint that passes one fixed panel is not a universal fidelity guarantee, and a lower topology score does not establish physical causality or end-to-end speedup. Keep TEST locked until a separate selection protocol authorizes its use.

## Further reading

- [Measured results and protocol](Posttraining_Results.md)
- [SOURCE-relative policy mathematics and original smoke evidence](Fidelity_Budget_Posttraining.md)
- [Configuration composition and validation rules](configuration.md)
- [Multitask collaborator quick start](Multitask_Prep.md)
- [Post-training implementation and maintenance guide](Posttraining_Development.md)

Older Run 11 operational notes are optional historical archive evidence and may not be present in a fresh clone.

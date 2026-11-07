# Post-training strategies

The framework supports several ways to combine reconstruction fidelity and coherence objectives. The current combustion profiles below specify the State, Scale and Shape queries and a fresh SOURCE-relative controller. The earlier Run 11f-style recipe remains available for replay. For measured historical outcomes, see [Posttraining results and protocol](Posttraining_Results.md), and for the SOURCE-relative policy mathematics, see [Fidelity-budget post-training](Fidelity_Budget_Posttraining.md).

## Current State, Scale and Shape profiles

Use the complete [stochastic combustion profile](../cases/turbulent_combustion/configs/posttrain/coherence_current.yaml) for the recorded generative SOURCE. Its resolved active families are exactly `global_distribution`, `multiscale_spectrum` and `topology`; `cross_spectrum` is explicitly disabled so recursive YAML inheritance cannot execute both Scale families accidentally. The SOURCE model, sensors and normalization are inherited; the child starts a fresh controller, optimizer and coherence calibration. Historical `cross_spectrum` and `lpc_ordered` recipes retain their existing definitions and checkpoint contracts.

State uses `marginal_mixed_weighted_v3`: paired marginal W2² and a 32-direction mixed mean under case-supplied physical-area masses. Scale uses `signed_spectral_mean_augmented_v4`, two independent conditional draws, alpha 1, joint weight 0.25 and mean weight 1. Shape uses `finite_primary_v1`, the full CO/T sign/line bank, and a 32×128 fixed physical-cell raster. Its fixed Q4096 training panel uses a sparse interpolation followed by physical-cell overlap; this is a surrogate for the native field, not exact full-native quadrature. The case reports its raw coordinate and field units as unknown. Field scales and geometry are frozen from TRAIN. See the [versioned query definitions](coherence_compute_enhancement.md).

`objectives.data_retention` is disabled with weight zero. The fresh TorchJD ConFIG direction combines the three coherence families; dense Lagrangian feedback uses paired TRAIN targets and SOURCE-relative per-field constraints. The model's native `model_loss` path is not called. A displayed sum of family scores is not a scalar objective for this direction. The sampled equal-node, normalized-model-unit endpoint-MSE constraint allows a 10% relative increase per field. The independent validation decision also uses a 10% per-field allowance, but measures decoded-field MSE with native physical-area weights across the declared matched panel and draws. Those are different estimators with the same nominal tolerance.

The [deterministic profile](../cases/turbulent_combustion/configs/posttrain/coherence_deterministic.yaml) selects the separate `paired_signed_spectral_mean_v1` point discrepancy with one draw. It does not represent repeated deterministic outputs as conditional samples. Neither profile establishes a learned benefit until its selected and final checkpoints have been evaluated.

From the repository root, inspect the physical GPU and its current processes before mapping the requested GPU 2 UUID to the profile's logical `cuda:0`:

```bash
nvidia-smi --query-gpu=index,uuid --format=csv,noheader
nvidia-smi -i 2
GPU_UUID="$(nvidia-smi -i 2 --query-gpu=uuid --format=csv,noheader)"
export CUDA_VISIBLE_DEVICES="$GPU_UUID"
export KEOPS_CACHE_FOLDER="$(mktemp -d /tmp/phycoflow-coherence-keops.XXXXXX)"

python cases/turbulent_combustion/run.py validate \
  --config configs/posttrain/coherence_current.yaml
python cases/turbulent_combustion/run.py post-train \
  --config configs/posttrain/coherence_current.yaml
```

For a deterministic SOURCE use `configs/posttrain/coherence_deterministic.yaml` instead. Set a distinct output identity for a new run; an interrupted run resumes from its own checkpoint and frozen family artifacts. Inspect the complete resolved configuration, SOURCE checkpoint and device identity before launching.

Full-native evaluation uses the same frozen TRAIN metric bundle, panel positions, endpoint settings, observations and draw seeds for SOURCE and child. The combustion support contains all 40,300 native nodes and the native Shape complex is 100×403. `evaluate-native-coherence` writes native field-risk arrays and can score saved members with a supplied TRAIN-fitted bundle. Construct that bundle once with [`fit_native_metric_bundle`](../src/phycoflow_reconstruction/evaluation/native_scoring.py) from TRAIN targets and SOURCE endpoints; do not fit a new denominator on validation. The panel file is a JSON list of dataset positions or a manifest containing the selected panel. An endpoint JSON matching the current profile can contain:

```json
{
  "steps": 2,
  "solver": "euler",
  "observation_config": {
    "mode": "endpoint_smooth",
    "strength": 1.0,
    "sigma": 0.05,
    "schedule_power": 2.0,
    "final_clamp": true,
    "chunk_size": 4096
  },
  "weight_selection": "live"
}
```

Supply the same recorded sensor manifest when one is used. For a prepared bundle, endpoint and panel:

```bash
CHILD_RUN=/absolute/path/to/child-run
PANEL_JSON=/absolute/path/to/frozen-validation-panel.json
ENDPOINT_JSON=/absolute/path/to/frozen-endpoint.json
TRAIN_BUNDLE=/absolute/path/to/train-metric-bundle.pt

python cases/turbulent_combustion/run.py evaluate-native-coherence \
  --run "$CHILD_RUN" --checkpoint last --split validation \
  --indices "$PANEL_JSON" --endpoint-config "$ENDPOINT_JSON" \
  --metric-bundle "$TRAIN_BUNDLE" --draws 4 --seed 812711 \
  --device cuda:0 --output "$CHILD_RUN/evaluation/native_validation_last"
```

Run the same command with the SOURCE run and a separate output directory, using the identical panel, endpoint, bundle and draw seeds. `--risk-only` streams native field risk without retaining members; it does not perform family scoring. Record SOURCE, selected and final separately, and do not use TEST for checkpoint selection.

Saved native payloads can be rendered without model inference, using the same TRAIN bundle. Add `--selected` and `--final` condition NPZs for a matched atlas; member indices are zero-based. State and Shape use the displayed member, Scale scores use all saved conditional members, and ensemble-mean fidelity is a separate view. Optional event thresholds must be fitted on TRAIN, declare `field_space: decoded_dataset`, and bind the frozen SOURCE/bundle. Fit signed thresholds directly in decoded space, or convert them with the frozen positive-scale normalizer: for `x_phys = scale * x_model + offset` and event `sign * x >= threshold`, use `threshold_phys = scale * threshold_model + sign * offset`, at the saved fields' decode precision.

```bash
python cases/turbulent_combustion/run.py render-native-coherence \
  --metric-bundle /path/to/native_metric_bundle.pt \
  --source /path/to/source/condition_00000.npz \
  --selected /path/to/selected/condition_00000.npz \
  --final /path/to/final/condition_00000.npz \
  --member 0 --output /path/to/native_atlas
```

For an opt-in self/mutual Gram candidate, retain the original SSS bundle above and add `--gram-bundle /path/to/native_gram_bundle.pt --gram-pair CO T`. The auxiliary bundle must bind that exact v4 yardstick. Its saved-data panels distinguish physical-unit power, mean-pattern/member-variation Gram, represented member/mean skill, TRAIN-floor regularized alignment, and the separate self/mutual scores. This performs no inference or fitting. The augmented candidate total and the unchanged v4 anchor are different definitions; report both. A single-field family retains its self score and explicitly marks mutual quantities unavailable.

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

Internal `coherence_aggregation` is limited to TorchJD ConFIG, requires explicit `family_scalarization.method: none`, requires an external policy, and requires global-distribution, topology, and either multiscale-spectrum or legacy cross-spectrum. Keep the family and objective definitions in the case configuration; the selector does not make arbitrary subsets or a hidden scalarization.

Existing `optimization.multitask` configurations without an external policy retain the hierarchical and joint aggregation routes. Invalid multi-task directions stop with an error instead of falling back to a weighted sum. The internal TorchJD route uses full-width FP64 aggregation and may fail explicitly if a valid direction cannot be constructed.

## Install the primary route dependencies

From the repository root, install the optional extras required by the Lagrangian, internal ConFIG, topology and KeOps example:

```bash
python -m pip install -e '.[posttrain_constraints,topology,keops,multitask]'
python -m pip check
```

`posttrain_constraints` supplies Cooper for the Lagrangian controller; `multitask` supplies TorchJD; `topology` and `keops` support the configured persistence and operator path. The `posttrain` extra is only needed for the older legacy ConflictFree path. Add `plot` if you want to render figures with Matplotlib; it is not needed for training or the CPU-only history summarizer.

## Select an alternative external policy

External-policy selectors compose with either a scalarized coherence loss or the internal A/B/T ConFIG direction. The first block below is a complete scalar external-policy selector map with fixed-sum families. Use a separate base profile whose resolved map omits `coherence_aggregation`, or replace the entire `optimization.multitask` map with this map using a CLI override. YAML defaults merge recursively: omitting the selector from a leaf profile does not remove an inherited selector, and setting it to null is invalid. This block is therefore not a leaf patch to an internal-ConFIG profile:

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

When replacing the entire `optimization.multitask` map, retain the original `fidelity_budget` settings as well as the desired selectors if the allowance and SOURCE calibration must remain matched. Omitting that budget map selects its defaults, including four calibration batches rather than the current profile's eight.

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

For example, replace the full policy map when changing the historical internal-ConFIG profile from Lagrangian to rectify, preserve the internal ConFIG selector, and choose a separate output family for the new child:

```bash
python cases/turbulent_combustion/run.py validate \
  --config configs/posttrain/fidelity_budget/lagrangian_config_warmup.yaml \
  --override source_run="$SOURCE_RUN" \
  --override 'optimization.multitask.external_policy={method: rectify, vector_reduction: sum, objective_scale: source_mean}'
```

For a new full run with that alternative, also pass a fresh `output.experiment_name`; the historical recipe's default output family identifies the Lagrangian method. The internal-aggregation route still requires State, Scale and Shape all enabled. Scalar external policies may instead use the family subset supported by the selected case profile.

## Historical Run 11f-style example

The [Lagrangian, internal-ConFIG, warm-up profile](../cases/turbulent_combustion/configs/posttrain/fidelity_budget/lagrangian_config_warmup.yaml) is the earlier 1,000-epoch legacy-cross-spectrum recipe. It declares `source_run: null` so each invocation supplies a concrete SOURCE run, uses `source_checkpoint: last.pt`, and writes children under the `posttrain_lagrangian_config_warmup` output family. Its selectors are `family_scalarization: none`, `coherence_aggregation: torchjd_config`, `outer_aggregation: null`, and dense Lagrangian feedback with `penalty_interval_updates: 1`. Model LR rises linearly from `5e-6` to `5e-5` across the first 76 global successful updates and then stays at `5e-5`.

Existing `run11*` YAML profiles are retained as historical regression and reproduction profiles. Historical profiles may contain a bound SOURCE, so inspect their resolved provenance before reuse.

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

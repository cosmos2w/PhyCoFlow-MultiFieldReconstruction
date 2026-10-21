# Multitask optimization: collaborator quick start

For current route choices and the Run 11f-style SOURCE-relative workflow, see [Posttraining strategies](Posttraining_Strategies.md); the completed four-run assessment is in [Posttraining results](Posttraining_Results.md). The maintained policy equations and selector constraints are in [Fidelity-budget post-training](Fidelity_Budget_Posttraining.md), and the gradient-aggregation interface described below remains available.

## What changed

Coherence post-training has two choices under `optimization.multitask`: how to group coherence families, and how to aggregate the active task gradients. The existing rectified-flow loss, descriptors, SOURCE calibration, AdamW, evaluation, checkpointing, and SOURCE protection remain in the shared training pipeline.

The coherence families are A (global distribution), B (cross-spectrum), and T (topology). `fixed_sum`, STCH, and DWA combine their calibrated losses into one coherence task; `none` keeps the ordered calibrated A/B/T losses as separate tasks. The outer aggregator combines the active tasks, so `none` can pass RF, A, B, and T together to ConFIG, UPGrad, CAGrad, or Sum. Continue reporting raw A/B/T and reconstruction fidelity separately because scalarized surrogate values differ across methods.

Existing configurations without `optimization.multitask` retain historical behavior. In the new path, `torchjd_config`, `upgrad`, and `cagrad` receive all active task gradients; invalid multi-task directions stop the update without a weighted-sum fallback. If only one task is active, its gradient is applied directly and telemetry records `modular_single_objective`.

## Install

From the repository root, use the existing environment and install the optional methods:

```bash
conda activate phycoflow_reconstruction
python -m pip install -e '.[posttrain,posttrain_constraints,topology,keops,multitask]'
```

TorchJD is pinned to 0.17.1 with the solver extras needed by UPGrad and CAGrad; Cooper 1.0.1 is supplied by `posttrain_constraints` for the Lagrangian policy. Fixed Sum + Sum does not require TorchJD. Dependency attribution is in [provenance.md](provenance.md).

## Choose a strategy

Add this block to a coherence post-training config:

```yaml
optimization:
  multitask:
    family_scalarization:
      method: fixed_sum
    outer_aggregation:
      method: torchjd_config
```

Choose one method from each table. Put optional parameters under the selected method's name.

| Family method | Behavior | Optional settings |
| --- | --- | --- |
| `none` | Keep calibrated A/B/T losses as separate ordered tasks | None |
| `fixed_sum` | Preserve the original calibrated family sum | None |
| `stch` | Smooth Tchebycheff scalarization | `stch: {mu: 0.1}` |
| `dwa` | Adapt weights from two completed reporting epochs | `dwa: {temperature: 2.0}` |

| Outer method | Behavior | Optional settings |
| --- | --- | --- |
| `sum` | Sum all active task gradients | None |
| `torchjd_config` | Always-on TorchJD ConFIG over active tasks | None |
| `upgrad` | TorchJD dual-cone projection aggregation | None |
| `cagrad` | TorchJD conflict-averse aggregation | `cagrad: {c: 0.5}` |

For a four-task ConFIG run, select `family_scalarization.method: none` and `outer_aggregation.method: torchjd_config`. For raw unit-scaled RF/A/B/T, set both objective coefficients and all three family weights to 1.0, with `coherence.family_balance.mode: none`. This leaves descriptor normalization and topology `essential_weight` in place. `fixed_sum` + `sum` with those same unit coefficients computes RF+A+B+T.

The portable [Run 00 recipe](../cases/turbulent_combustion/configs/posttrain/multitask_unit_sum.yaml) and [Run 06 recipe](../cases/turbulent_combustion/configs/posttrain/multitask_joint_config.yaml) use 1,000 reporting epochs and the 32×128 CO/T topology with one mutual line. Supply the original SOURCE path when validating either recipe:

```bash
SOURCE_RUN_PATH="$PWD/cases/turbulent_combustion/runs/tc_gl_rbf_cq_cached_kv_5000ep/20260826T212231Z_7b68c461"
CUDA_VISIBLE_DEVICES=0 python cases/turbulent_combustion/run.py validate \
  --config configs/posttrain/multitask_joint_config.yaml \
  --override "source_run=$SOURCE_RUN_PATH"
```

For joint methods, telemetry records the ordered task names, per-task norms, directional products, and pairwise cosines under `outer_aggregation`; the top-level gradient cosine compares RF with the sum of the family gradients. The logged `total` is a diagnostic sum, not a scalar objective whose gradient equals the ConFIG update.

The parameter values above are defaults, not tuned scientific choices. DWA uses unit weights in its first two reporting epochs and can adapt in epoch three; its history is saved in checkpoints. Remove a method's parameter block when switching to another method: inactive settings are rejected.

Keep outer objective weights under `objectives.data_retention.weight` and `objectives.coherence.weight`. Keep scientific family preferences and SOURCE calibration under `coherence`. Inherited `gradient_balance`, `config_missing_behavior`, and `config_*_grad_scale` settings are ignored by the modular path. Do not combine it with constrained-topology modes or physics-only post-training.

## Run a small check, then configure an experiment

Use the original pretrained SOURCE run available on your workstation. The local example below selects its `last.pt`; point `SOURCE_RUN_PATH` to the corresponding original run if your checkout stores artifacts elsewhere.

```bash
SOURCE_RUN_PATH="$PWD/cases/turbulent_combustion/runs/tc_gl_rbf_cq_cached_kv_5000ep/20260826T212231Z_7b68c461"
CUDA_VISIBLE_DEVICES=2 python cases/turbulent_combustion/run.py post-train \
  --config configs/posttrain/multitask_smoke.yaml \
  --override "source_run=$SOURCE_RUN_PATH" \
  --override source_checkpoint=last.pt \
  --override output.experiment_name=multitask_fixed_sum_config_check \
  --override optimization.multitask.family_scalarization.method=fixed_sum \
  --override optimization.multitask.outer_aggregation.method=torchjd_config \
  --max-steps 1
```

Physical GPU 2 is addressed as `cuda:0` inside this process. The smoke profile uses small batches and differentiable topology at reduced resolution; it checks execution and is unsuitable for scientific comparisons with the full ABC profile. For DWA, use three smoke updates to reach its first adaptive epoch.

Change methods directly with CLI overrides; for example, replace the two selector lines above with:

```bash
  --override optimization.multitask.family_scalarization.method=stch \
  --override optimization.multitask.family_scalarization.stch.mu=0.1 \
  --override optimization.multitask.outer_aggregation.method=cagrad \
  --override optimization.multitask.outer_aggregation.cagrad.c=0.5 \
```

For matched scientific experiments, start from `configs/posttrain/abc_sliced_persistence.yaml`, retain the original SOURCE `last.pt`, set `model.model_ema_eval=false` for live-weight topology evaluation, and agree on the exposure budget before removing the small-step limit. Keep architecture, data split, sampling manifests, SOURCE calibration, full CO+T topology, rollout, optimizer, validation, and checkpoint cadence matched. Use a fresh output name for each arm. Replace `post-train` with `validate` and omit `--max-steps` to check config and data before launch.

To resume, reuse the same source, config, and method parameters and add `--resume /path/to/child-run`. Pass the child run directory, not a checkpoint file; DWA state is recovered automatically.

## Completed comparison status

As of 21 October 2026, Runs 00, 06, 11 and 11f completed their 1,000-epoch / 38,000-update campaigns and were assessed on the matched broader validation panel. Run 11f is the maintained example recipe; the recorded comparison is mixed and does not establish method superiority. See [Posttraining results](Posttraining_Results.md) for selected-versus-final checkpoint outcomes and limits. Earlier launch notes are historical preparation guidance, not pending launch requests.

For any new campaign, select on validation and check total/per-field endpoint fidelity against SOURCE. Report raw A/B/T, native loss, gradient/weight diagnostics, and runtime alongside candidate results. Keep TEST untouched until candidate selection and criteria are fixed; then evaluate it under the separately approved protocol. Do not compare topology scores across different rasters, projections, filtrations, or mutual components.

For configuration rules, see [configuration.md](configuration.md#coherence-post-training-multitask-methods). The two adapters live in [multitask.py](../src/phycoflow_reconstruction/training/multitask.py); the shared trainer is [post_training.py](../src/phycoflow_reconstruction/training/post_training.py).

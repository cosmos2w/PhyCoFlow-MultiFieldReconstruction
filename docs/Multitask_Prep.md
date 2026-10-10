# Multitask optimization: collaborator quick start

## What changed

Coherence post-training now has two independent choices under `optimization.multitask`: how to combine coherence families, and how to combine the coherence gradient with the native reconstruction gradient. The existing rectified-flow loss, descriptors, SOURCE calibration, AdamW, evaluation, checkpointing, and SOURCE protection remain in the shared training pipeline.

The coherence families are A (global distribution), B (cross-spectrum), and T (topology). Family scalarization produces one coherence surrogate from their SOURCE-calibrated losses; outer aggregation combines its gradient with the data-retention gradient. Continue reporting raw A/B/T and reconstruction fidelity separately because different scalarizers produce different surrogate values.

Existing configurations without `optimization.multitask` retain historical behavior. In the new path, `torchjd_config` invokes TorchJD ConFIG whenever both objectives have nonzero coefficients; invalid directions stop the update without a weighted-sum fallback. An explicitly zero objective coefficient selects the sole active gradient, recorded as `modular_single_objective`.

## Install

From the repository root, use the existing environment and install the optional methods:

```bash
conda activate phycoflow_env
python -m pip install -e '.[posttrain,topology,keops,multitask]'
```

TorchJD is pinned to 0.17.1 with the solver extras needed by UPGrad and CAGrad. Fixed Sum + Sum does not require TorchJD. Dependency attribution is in [provenance.md](provenance.md).

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
| `fixed_sum` | Preserve the original calibrated family sum | None |
| `stch` | Smooth Tchebycheff scalarization | `stch: {mu: 0.1}` |
| `dwa` | Adapt weights from two completed reporting epochs | `dwa: {temperature: 2.0}` |

| Outer method | Behavior | Optional settings |
| --- | --- | --- |
| `sum` | Sum the two objective gradients | None |
| `torchjd_config` | Always-on TorchJD ConFIG | None |
| `upgrad` | TorchJD dual-cone projection aggregation | None |
| `cagrad` | TorchJD conflict-averse aggregation | `cagrad: {c: 0.5}` |

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

## Pending experiments and ownership

The existing Fixed Sum + TorchJD ConFIG ABC run is stopped, with checkpoints retained. Its coherence trend is broadly comparable to the historical run, with a modest lag; cross-spectrum validation and CO fidelity still need attention. This establishes an initial reference, not a preferred method.

| Owner | Next task |
| --- | --- |
| Family-scalarization lead | Use Fixed Sum + TorchJD ConFIG as the reference; study STCH sensitivity on the calibrated scale, then DWA weight evolution and recovery under the same outer method. |
| Outer-aggregation lead | Compare legacy conditional ConFIG with always-on TorchJD ConFIG as distinct conditions; then evaluate UPGrad and CAGrad under Fixed Sum. Reuse existing reference runs where appropriate. |
| Evaluation lead | Inspect A/B/T separately, including same-/cross-frequency B components and CO near the fidelity limit. Agree on checkpoint selection and the validation/TEST protocol. |
| All collaborators | Agree on short exposure budgets and seeds; retain resolved configs, SOURCE identity, sampling manifests, and checkpoint lineage. Review individual method effects before combining non-default choices. |

Select candidates on validation and check total/per-field fidelity against SOURCE. Report raw A/B/T, native loss, gradient/weight diagnostics, and runtime alongside candidate results. Keep TEST untouched until candidate selection and criteria are fixed; then evaluate it once. Do not compare topology scores across different rasters, projections, filtrations, or mutual components.

For configuration rules, see [configuration.md](configuration.md#coherence-post-training-multitask-methods). The two adapters live in [multitask.py](../src/phycoflow_reconstruction/training/multitask.py); the shared trainer is [post_training.py](../src/phycoflow_reconstruction/training/post_training.py).

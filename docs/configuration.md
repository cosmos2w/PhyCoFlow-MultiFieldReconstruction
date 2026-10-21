# Configuration

Configuration is a composition layer around the shared contracts. It should make a run reproducible without copying generic model definitions into every case.

## Ownership

Use these locations for new settings:

| Concern | Canonical location | Examples |
|---|---|---|
| Shared runtime policy | `configs/defaults/` | device, workers, deterministic seed |
| Shared optimization policy | `configs/defaults/base_training.yaml` | optimizer, clipping, batch defaults |
| Evaluation/checkpoint policy | `configs/defaults/` | validation loss, reconstruction, checkpoint cadence |
| Generic architecture | `configs/models/` | width, depth, backbone, sampler |
| Dataset contract | `cases/<case>/configs/dataset.yaml` | payload, fields, splits, normalization |
| Sensors | `cases/<case>/configs/sensors/` | fields, counts, sampling seed |
| Coherence/physics | case `coherence/`, `posttrain/`, `direct_physics/` | scientific weights and geometry |
| Local experiments | `cases/<case>/configs/_experiments/` | untracked temporary overrides |

Do not create coworker-named source directories or duplicate a generic model definition under multiple cases. A case may retain a thin `configs/base/<model>.yaml` launch profile for discoverability, but it should compose shared defaults, the case dataset/sensor contract, and one shared model fragment before adding small case-specific overrides.

## Composition example

The exact relative path can vary with a case layout, but the ownership should look like this:

```yaml
defaults:
  - ../../../../configs/defaults/base_training.yaml
  - ../dataset.yaml
  - ../sensors/u_only_random_variable.yaml
  - ../../../../configs/models/coordinate_mlp.yaml

case: brusselator
stage: base_training
optimization:
  batch_size: 4
output:
  experiment_name: coordinate_mlp
```

Defaults are merged in order and the current file wins. CLI overrides use dotted keys with YAML values:

```bash
python run.py train-base \
  --config configs/base/coordinate_mlp.yaml \
  --override runtime.device=cpu \
  --override optimization.batch_size=2
```

Post-training configs should inherit the source run's dataset/model/ observation contract where supported. The child config owns only its objective, rollout, trainable scope, reference policy, and output lineage.

## Coherence post-training multitask methods

Data-driven coherence post-training can opt into the explicit `optimization.multitask` namespace. Its family selector either composes A/B/T into one coherence surrogate or keeps the calibrated family losses as separate tasks; the outer aggregator combines those active tasks with the native data-retention loss. The existing `objectives.data_retention.weight` and `objectives.coherence.weight` remain the top-level objective coefficients, while family weights and descriptor definitions stay under `coherence`.

Both interface blocks may be omitted or left empty; their defaults are `fixed_sum` and `sum`. Each block has a `method` selector and may include only the selected method's nested settings. `family_scalarization.method: none` passes calibrated A/B/T losses as separate ordered tasks, enabling the outer aggregator to process RF, A, B, and T individually. It changes task grouping only: descriptor normalization and topology `essential_weight` remain part of the scientific definition. Parameters default to STCH `mu: 0.1`, DWA `temperature: 2.0`, and CAGrad `c: 0.5`; those are implementation defaults, not selected scientific optima. The validator rejects inactive method maps and extra keys, including redundant `weights` declarations.

```yaml
objectives:
  data_retention: {enabled: true, weight: 0.1}
  coherence: {enabled: true, weight: 1.0}

optimization:
  multitask:
    family_scalarization:
      method: stch
      stch: {mu: 0.1}
    outer_aggregation:
      method: torchjd_config
```

The family method forms are `{method: none}`, `{method: fixed_sum}`, `{method: stch, stch: {mu: 0.1}}`, or `{method: dwa, dwa: {temperature: 2.0}}`; pair one with `outer_aggregation: {method: sum}`, `{method: torchjd_config}`, `{method: upgrad}`, or `{method: cagrad, cagrad: {c: 0.5}}`. STCH, DWA, and CAGrad parameter maps are optional when their method is selected and use the defaults above. Do not add objective or family weights under `optimization.multitask`.

For example, select DWA and CAGrad from a composed config with dotted overrides:

```bash
python cases/turbulent_combustion/run.py validate \
  --config configs/posttrain/abc_sliced_persistence.yaml \
  --override source_run=/path/to/source-run \
  --override optimization.multitask.family_scalarization.method=dwa \
  --override optimization.multitask.family_scalarization.dwa.temperature=2.0 \
  --override optimization.multitask.outer_aggregation.method=cagrad \
  --override optimization.multitask.outer_aggregation.cagrad.c=0.5
```

When `optimization.multitask` is present, the named outer aggregator receives the active task gradients. A zero top-level coefficient disables its objective tasks; if one task remains, its gradient is applied directly and telemetry records `modular_single_objective`. An active task with a genuinely zero gradient stays in the matrix. Invalid multi-task directions fail without fallback. Historical `optimization.gradient_balance: config`, `config_missing_behavior`, and `config_*_grad_scale` fields may arrive from inherited profiles and are ignored by this route; they retain their old behavior when `multitask` is absent. In particular, the legacy conditional ConFIG path is not the same algorithm as `torchjd_config`. Do not combine `multitask` with `component_constrained`, `topology_regularized`, or `component_constraints`; physics-only post-training does not accept this namespace.

## SOURCE-relative external fidelity policies

The optional `optimization.multitask.external_policy` adds one of three endpoint-fidelity policies: `lagrangian`, `rectify`, or `extended_log_barrier`. It composes a selected family scalarization with per-field constraints whose reference is calibrated from the original SOURCE model on fixed TRAIN batches. The complete portable Run 11f-style workflow, including dependencies, source binding, smoke, launch, resume and explicit-checkpoint evaluation commands, is in [Posttraining strategies](Posttraining_Strategies.md); measured outcomes and checkpoint identities are in [Posttraining results](Posttraining_Results.md).

An external policy requires `vector_reduction: sum`, `objective_scale: source_mean`, `objectives.data_retention` disabled with weight zero, `objectives.coherence` enabled with weight one, and `trainable.scope: full_model`. It cannot be combined with endpoint/source-anchor objectives, component constraints, non-default legacy `gradient_balance`, non-null `outer_aggregation`, coherence warm-up/interval rescaling, recurring family-gradient diagnostics, or unsupported checkpoint selection. For the scalar-policy route, choose `family_scalarization` from `none`, `fixed_sum`, `stch`, or `dwa`; the selected family scalarizer must not include inactive method settings.

The separate internal A/B/T ConFIG composition uses `coherence_aggregation: {method: torchjd_config}` with explicit `family_scalarization: {method: none}`, a null or omitted `outer_aggregation`, one external policy, and all three coherence families enabled. It combines the three family gradients with the constraint direction and reports the scientific A+B+T sum separately; that reported sum is not the scalar objective optimized by ConFIG. The current validator rejects other `coherence_aggregation` methods, family scalarization with this selector, or an external policy with an active outer aggregator. Invalid directions stop without weighted-sum fallback.

The configured fidelity metric is pooled masked endpoint MSE per normalized model field, with defaults of a five-percent relative increase, zero absolute allowance and a denominator floor of `1e-8`. The denominator floor scales violations but does not relax the accepted bound. TRAIN SOURCE calibration and validation SOURCE baseline are separate quantities. A training-time SOURCE-relative constraint does not certify each validation panel or held-out sample.

## Training validation and checkpoints

Many modern training profiles enable `evaluation.preview.enabled: true`, which runs a fixed, seeded validation objective at `loss_every_epochs` and plots it beside the training curves in `loss_history.png`; it does not run the generative reconstruction loop. Some post-training profiles, including SOURCE-relative external-policy profiles, disable preview and use the configured fixed-panel validation checkpoint evaluator instead. Check the stage's checkpoint-selection metric and validator rules; preview is not required by every supported post-training path. Qualitative reconstruction figures and payloads run independently at `reconstruct_every_epochs` when enabled.

Every fresh validation loss is compared with the run's best value and may replace `checkpoints/best.pt`. The rolling resumable `checkpoints/last.pt` is refreshed at `checkpointing.every_epochs`, while `checkpointing.epochs` adds immutable requested epoch files. There is no redundant `latest.pt` alias.

## Validation rules

Before opening a PR, discover every tracked YAML under `configs/`, `cases/*/configs/`, and `benchmarks/` and validate it with the shared loader. Check that defaults resolve from canonical paths, all referenced dataset and sensor files exist (or are deliberately local payloads), and stage-specific requirements are enforced. A latent Stage-2 template may intentionally fail until `model.stage1_checkpoint` is supplied; this exception must be clear in its documentation and tests.

Do not use descriptive schema files as a second, dead runtime authority. The retained snapshots are explicitly labeled in [`docs/reference/config_schema/`](reference/config_schema/); executable validation lives in `phycoflow_reconstruction.config.validate` plus its contract tests.

## Compatibility-sensitive values

Treat these as serialized or scientific interfaces: model `name`, backbone identity, parameter prefixes, field order, normalization method/statistics, sensor protocol seed/count semantics, checkpoint selection, EMA policy, and coherence/physics definitions. GL-RBF/CQ aliases remain accepted by the builder; a cleanup must not silently rename them or change defaults.

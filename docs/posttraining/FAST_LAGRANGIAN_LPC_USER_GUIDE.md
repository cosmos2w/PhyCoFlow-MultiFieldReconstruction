# Experimental coherence execution and ordered objectives

The recommended starting point remains dense Lagrangian endpoint-fidelity feedback, with fresh family ConFIG when using all three coherence families. See the [shared strategy guide](../Posttraining_Strategies.md) and [fidelity-budget guide](../Fidelity_Budget_Posttraining.md). The optional components below are development tools, not established improvements in fidelity retention or total training cost. They do not change existing configurations when omitted.

## Use your existing case and SOURCE

Start from a validated post-training configuration for your own case. Retain its dataset, field order, sensors, immutable `source_run`, model settings, normalization, rollout and evaluation protocol. Configure the horizon in epochs. These components have no required filesystem mount, GPU index, experiment directory name or campaign launcher.

```bash
python cases/YOUR_CASE/run.py validate --config PATH_TO_POSTTRAIN_CONFIG
python cases/YOUR_CASE/run.py post-train --config PATH_TO_POSTTRAIN_CONFIG
```

Launch a small CPU smoke run using `--override runtime.device=cpu --max-steps 1` before a full experiment. Resume through the same case CLI with `--resume RUN_DIRECTORY` and the original resolved configuration. Optional controllers save implementation identities and reject incompatible exact-resume attempts; preserve the matching source revision for such research runs.

## Sampled execution

`optimization.fast_lagrangian` enables training-only topology execution controls. This fragment selects dense execution while making the option explicit:

```yaml
optimization:
  fast_lagrangian:
    topology_estimator: full
    family_probability: 1.0
    coefficient_interval: 1
    seed: 700211
```

It requires the external Lagrangian fidelity policy with a penalty on every update and fixed family weights. For scalar aggregation, set `optimization.multitask.family_scalarization.method: fixed_sum` explicitly. An empty mapping selects the same execution defaults; omit the option entirely for the legacy path.

`topology_estimator: p3` or `s2` samples complete topology filtrations. `family_probability: 0.5` samples the topology family on the scalar-sum path; it cannot be combined with internal ConFIG or periodic coefficients. Dense SOURCE calibration, fidelity feedback and validation remain separate from these training estimators. Descriptor support and estimator weighting are checked by the topology provider.

`coefficient_interval` accepts 1, 2 or 4; intervals greater than 1 reuse internal ConFIG coefficients and remain experimental. Keep 1 for fresh gradients. Evaluate any proposed reuse against full execution and independent fidelity checks.

## Ordered objectives

`optimization.lpc_ordered` exposes an experimental scalar objective that prioritizes topology while constraining global-distribution and cross-spectrum losses relative to fixed SOURCE means. It requires the exact ordered global-distribution, cross-spectrum and topology families, `fixed_sum`, dense external Lagrangian feedback, fixed family scales and every-step coherence evaluation.

```yaml
optimization:
  fast_lagrangian:
    topology_estimator: full
    family_probability: 1.0
    coefficient_interval: 1
  lpc_ordered:
    dual_lr: 0.01
    denominator_floor: 1.0e-8
```

These fragments must be merged into an existing compatible profile; they are not complete launch configurations. The saved reference artifact and dual state are required for exact resume and must belong to the same SOURCE.

## Measured execution planning

The generic Python API in `phycoflow_reconstruction.training.execution_strategy` provides `TaskContract`, `CostProfile`, `ExecutionPlan`, `manifest_for_task`, `select_execution_plan`, `preserve_profile_state` and `profile_one_update`. A caller supplies its resolved scientific contract, immutable SOURCE identity, hardware fingerprint and measured TRAIN-only costs/noise. `optimization.execution_strategy.enabled: true` requires the resulting bound task, profile and plan. No local campaign script is required by the installed package.

The runtime validates the plan against the task, retains deterministic decisions at accepted-update cursors, records audit state, and falls back to full execution when its monitoring rules require it. This path supports scalar additive family losses with dense Lagrangian fidelity feedback. It is mutually exclusive with `fast_lagrangian` and `lpc_ordered`. Do not reuse a profile across changed hardware or scientific contracts.

Historical GPU admission, filesystem setup, experiment queues, campaign recovery and one-off report builders remain on the research branch.

## Evaluate through the shared case interface

Use `evaluate-run` for checkpoint metrics and `visualize-run` for reconstruction/coherence figures. Both use the saved run configuration and ordinary local run directories. See the [README evaluation commands](../../README.md#7-evaluation-and-local-outputs) for complete options.

```bash
python cases/YOUR_CASE/run.py evaluate-run --run RUN_DIRECTORY --checkpoint last
python cases/YOUR_CASE/run.py visualize-run --run RUN_DIRECTORY --checkpoint last \
  --eval-set validation --eval-coherence global_distribution cross_spectrum topology
```

Within the post-training trainer's validation panels, optional `evaluation.group_size` fixes groups for batch-dependent coherence metrics and must exactly divide `evaluation.max_samples`. Those panel summaries retain the base generation seed, ordered group seeds and per-group records. Ordinary `evaluate-run` and `visualize-run` do not apply this grouping option. Compare matched observation/query protocols, full metrics, fieldwise fidelity, complete epoch histories and total measured costs before drawing scientific conclusions. TEST remains reserved according to each study's declared protocol.

## Checkpoint compatibility

The additions do not change model tensor schemas or the ordinary checkpoint loader. Existing non-DiffusionPDE base and post-trained models continue to load using their original model configuration, field order and normalization. Do not add these optional settings when loading or resuming an old run. Exact resume also requires the original optimizer/controller state and compatible source implementation. DiffusionPDE's EDM formulation is a separate architecture/objective change and requires fresh training.

# Proj_MultiFieldReconstruction — GL_rbf_CQ downstream migration benchmark

## Goal

Use `Proj_MultiFieldReconstruction` as the first realistic downstream consumer of the portable `GL_rbf_CQ` release core.

The experiment must answer **two different questions**:

1. **Migration effect** — what changes when the downstream project's intended large `GL_rbf_ENH` baseline is upgraded to the latest `GL_rbf_CQ`, while preserving the downstream project's dataset, contracts, trainer, evaluation workflow, field order, and sensor semantics?
2. **Execution effect** — within the migrated `GL_rbf_CQ`, what changes when only the condition-attention execution switches from `legacy_mha + full` to `cached_kv + full`?

The current tiny PointCloudFFM config in `Proj_MultiFieldReconstruction` is a repository-creation placeholder and is **not** the intended baseline.

Work from repository root on:

```text
release/gl-rbf-cq-portable-prep
```

Create a dedicated validation branch, for example:

```text
validation/proj-multifield-gl-rbf-cq
```

Do not modify the portable-prep branch itself.

---

# 1. Source references

Portable release preparation:

```text
branch: release/gl-rbf-cq-portable-prep
source SHA: a7992a5c462458d4689d2312491f0220cd0d4d1a
portable package:
  0_demo_TurbulentCombustion/src/phycoflow_pointcloud/
model defaults:
  0_demo_TurbulentCombustion/configs/gl_rbf_cq_core.yaml
migration guide:
  0_demo_TurbulentCombustion/GL_rbf_CQ_UPDATE_GUIDE.md
release manifest:
  0_demo_TurbulentCombustion/GL_rbf_CQ_RELEASE_MANIFEST.yaml
```

Historical intended large GL-RBF baseline:

```text
0_demo_TurbulentCombustion/
  _CheckNotes/Stage6_clean_ab/F0_ENH_L256_1000ep_b128.yaml
```

Use that historical configuration as the **scale/protocol reference** for the corrected legacy downstream baseline.

Do not import the turbulent-combustion trainer or HDF5 code into `Proj_MultiFieldReconstruction`.

---

# 2. Three-arm benchmark design

Run three independent 200-epoch training jobs.

## Arm A — corrected downstream legacy GL_rbf_ENH

This is the existing `Proj_MultiFieldReconstruction` PointCloudFFM / `EnhancedGLRBFTopK` path, corrected from the tiny placeholder scale to the intended historical scale.

It remains the **old downstream implementation**. Do not replace it with the portable package before this run.

Target scale:

```yaml
model:
  name: pointcloud_ffm
  backbone: gl_rbf_enh

  hidden_dim: 256
  latent_dim: 256
  num_latents: 128
  heads: 8
  latent_blocks: 4

  gather_topk: 32
  rbf_sigma: 0.05
  query_chunk_size: 2048
  query_points: 4096

  prior: rff
```

Where the existing downstream implementation has hard-coded **capacity-related** values that were accidentally reduced during repository creation, expose them as normal config keys and set them to the historical intended values where this can be done without redesigning the model:

```text
field embedding width: 128
Fourier bands: 32
RFF features: 256
RFF length scale: 0.15
```

These are baseline-correction changes, not GL_rbf_CQ migration changes.

Do **not** add CQ, FiLM, measurement/support, EMA, GLRES redesign, persistent Top-K, cached K/V, or other Stage-7/8 features to Arm A.

Do not attempt to force exact parameter-count equality by inventing new legacy modules. Record the actual Arm-A parameter count and any structural differences that remain relative to the historical 0_demo GL_rbf_ENH.

## Arm B — migrated GL_rbf_CQ with legacy attention

Use the new portable `GL_rbf_CQ` scientific model with:

```yaml
condition_attention_execution: legacy_mha
sensor_attention_padding_mode: full
```

All other `GL_rbf_CQ` scientific settings come from:

```text
0_demo_TurbulentCombustion/configs/gl_rbf_cq_core.yaml
```

This is the primary **migration/model-effect** arm.

## Arm C — migrated GL_rbf_CQ with cached K/V

Identical to Arm B except:

```yaml
condition_attention_execution: cached_kv
sensor_attention_padding_mode: full
```

This is the **execution-effect** arm and the latest recommended model.

---

# 3. Comparison logic

Report three comparisons separately.

## Migration effect

```text
B - A
```

Interpret as:

> effect of migrating from the corrected downstream legacy GL_rbf_ENH implementation to the latest GL_rbf_CQ scientific model, under the same downstream data/training protocol.

This comparison includes accepted model-development changes such as compact CQ, stronger condition core, FiLM, measurement/support, and EMA.

It is **not** a pure one-module architectural ablation.

## Execution effect

```text
C - B
```

Interpret as:

> effect of Stage-8 cached sensor K/V when the GL_rbf_CQ scientific model, initialization, data sequence, optimizer, and evaluation protocol are otherwise identical.

This is the clean execution comparison.

## Total latest-model effect

```text
C - A
```

Interpret as:

> practical total change experienced by a downstream user upgrading from the intended legacy model to the latest release-default GL_rbf_CQ.

---

# 4. Common downstream benchmark protocol

The three arms must use the same `Proj_MultiFieldReconstruction` dataset and training infrastructure.

Preserve downstream ownership of:

- dataset loading;
- field order;
- coordinate convention;
- normalization;
- split logic;
- `ObservationBatch`;
- sensor protocol implementation;
- run store;
- checkpoint format;
- training loop structure;
- evaluation metrics.

The benchmark case is:

```text
Proj_MultiFieldReconstruction/Cases/turbulent_combustion/
```

Current downstream field order remains:

```text
[CH4, CO, T, U_1, p]
```

Coordinate dimension remains:

```text
2
```

## Common sensor protocol

Condition on temperature only, matching the intended 0_demo supervision range:

```text
T-only
random-uniform
192–384 valid sensors per sample
seed = 42
```

Use the project's own supported syntax for a sensor-count range. Do not alter the sensor sampler implementation merely to imitate the 0_demo YAML syntax.

## Common training scale

Use for all A/B/C:

```yaml
optimization:
  epochs: 200
  batch_size: 128
  lr: 1.0e-4
  weight_decay: 1.0e-6
  grad_clip: 1.0

runtime:
  seed: 42
  deterministic: true

model:
  query_points: 4096
```

Use the same project-supported worker/data-loading settings for all arms.

Do not silently reduce batch size, query count, sensor count, or model width to make a run fit.

If the corrected Arm A cannot run at B128/Q4096 on the target GPU, stop and report the exact OOM/limitation before changing the scientific benchmark protocol.

## Training schedule

Preserve the existing `Proj_MultiFieldReconstruction` optimizer/trainer semantics.

Do not add the 0_demo cosine scheduler just to imitate that repo.

The benchmark is meant to evaluate model migration **inside the downstream project's own training system**.

---

# 5. Fair evaluation and timing policy

Training-time evaluation/plotting/checkpoint overhead must not dominate epoch timing.

Use the same lightweight policy for all arms.

Recommended:

```text
checkpoint epochs: 1, 20, 60, 100, 150, 200
fixed post-hoc evaluation: 20, 60, 100, 150, 200
training preview: disabled, or identical low-frequency preview for all arms
loss plot refresh: low frequency
```

Prefer post-hoc fixed-manifest evaluation rather than expensive full validation every epoch.

Before Arm A starts:

1. build one fixed validation sensor manifest using the common T-only 192–384 protocol;
2. record its checksum;
3. reuse it for every A/B/C milestone evaluation.

Evaluation randomness, reconstruction steps, solver, and source seed must be identical.

---

# 6. Benchmark instrumentation

Before the first 200-epoch run, add **generic opt-in benchmark telemetry** to the downstream training infrastructure.

It must be disabled by default so existing project behavior is unchanged.

Prefer a small helper/module rather than embedding model-specific logic in `base_training.py`.

Record at minimum:

```text
epoch
epoch wall time
training-only epoch time if separately measurable
steps per epoch
mean/median sampled training-step time
forward/native-loss time
backward time
optimizer time
peak CUDA allocated memory
peak CUDA reserved memory
parameter count
trainable parameter count
```

If practical also record:

```text
data wait/materialization time
process/GPU memory peak
```

For CUDA timing use synchronized CUDA events or another correct GPU timing method.

Reset peak-memory statistics at controlled boundaries.

Do not use profiler tracing during all 200 epochs.

Use short profiler/controlled timing runs separately.

Store machine-readable evidence under something like:

```text
Proj_MultiFieldReconstruction/
  benchmarks/gl_rbf_cq_migration_200ep/
```

Tracked evidence should contain configs, summaries, CSV/JSON, hashes, and analysis scripts.

Large checkpoints and full run directories remain under the project's ignored `runs/` structure.

---

# 7. Phase I — freeze and run the corrected old baseline

Before migrating source code:

1. create the validation branch;
2. audit the current PointCloudFFM implementation;
3. correct only the accidental baseline-scale/config omissions described in Arm A;
4. add generic benchmark telemetry;
5. add the 200-epoch Arm-A config;
6. validate the dataset and config;
7. run focused tests;
8. record the exact branch SHA used for Arm A.

Then launch:

```text
Arm A — corrected legacy GL_rbf_ENH — 200 epochs
```

Preserve:

```text
resolved config
environment.json
run manifest
checkpoint hashes
history
performance telemetry
milestone checkpoints
fixed evaluation outputs
```

Create a tracked `A_BASELINE_SUMMARY.md/json` before beginning migration.

Do not rewrite Arm A after seeing B/C results.

---

# 8. Phase II — migrate GL_rbf_CQ without changing project architecture

Read:

```text
0_demo_TurbulentCombustion/GL_rbf_CQ_UPDATE_GUIDE.md
0_demo_TurbulentCombustion/GL_rbf_CQ_RELEASE_MANIFEST.yaml
```

Use them as the migration contract.

## Package integration

Bring the portable package into the downstream source tree as a package, preserving its relative layout, for example:

```text
Proj_MultiFieldReconstruction/
  src/
    phycoflow_pointcloud/
```

Do not import the combustion demo at runtime.

Do not make the downstream project depend on:

```text
0_demo_TurbulentCombustion/
Dataset/
helpers.py
_CheckNotes/
research/
visualization scripts
```

## Downstream adapter

Do not replace the existing downstream `PointCloudFFM` baseline.

Add a new adapter such as:

```text
phycoflow_reconstruction/models/flows/gl_rbf_cq.py
```

The adapter must preserve the project's existing interfaces:

```text
ObservationBatch
LossBundle
ReconstructionBatch
ModelCapabilities
model registry
common trainer
common evaluator
case-owned config system
```

The adapter should translate `ObservationBatch` directly into the portable core tensor contract.

Do not copy the combustion training loop.

## Registry

Register a new public downstream model name, for example:

```yaml
model:
  name: gl_rbf_cq
```

Keep:

```yaml
model:
  name: pointcloud_ffm
```

working exactly as the old baseline.

Do not silently change an existing model name to the new model.

---

# 9. EMA integration

EMA is part of the final GL_rbf_CQ scientific configuration and must not be dropped during migration.

Integrate EMA into the downstream trainer through a **generic optional model lifecycle interface**, not a hard-coded `if model == gl_rbf_cq` branch.

A suitable pattern is:

```text
after_optimizer_step()
evaluation_weight_context()
training_aux_state_dict()
load_training_aux_state_dict()
```

or an equivalent minimal design consistent with the existing project.

Models that do not implement/use EMA must behave exactly as before.

For GL_rbf_CQ:

```text
EMA decay = 0.999
EMA used for validation/reconstruction/best selection
live + EMA state resumable
```

Test:

```text
save
load
resume
EMA update count
evaluation selection
```

---

# 10. Query microbatch and reconstruction integration

The latest GL_rbf_CQ supports query microbatching and cached-streamed reconstruction.

For the B128/Q4096 benchmark, use the release-default effective query microbatch policy:

```text
effective queries = 4096
query execution microbatch = 2048
reuse condition context = true
```

This is part of the latest model execution and is allowed for B/C.

Arm A remains on its existing downstream training implementation; do not retrofit Stage-5 query microbatching into Arm A merely to make it faster.

Record this difference as part of the practical migration effect.

For reconstruction, use the portable cached-streamed path for B/C through the adapter while returning the project's normal `ReconstructionBatch`.

Preserve downstream observation clamping/consistency semantics where applicable.

---

# 11. Migration correctness gates before B/C runs

Do not launch B/C until these pass.

## Existing project safety

- full existing `Proj_MultiFieldReconstruction` regression suite;
- old `pointcloud_ffm` still builds and trains;
- all unrelated model families still import/build as before.

## Adapter/core correctness

- portable package imports without the combustion demo;
- synthetic 2-D forward/backward;
- synthetic 3-D forward/backward;
- F=5 plus at least one different field count;
- mixed padded sensor counts;
- direct portable-core output equals downstream-adapter output for identical tensors;
- `cached_kv` and `legacy_mha` both construct;
- persistent Top-K/cached-streamed reconstruction works;
- no extra KNN after persistent geometry build where geometry reuse applies;
- EMA save/load/resume passes.

## B/C initialization identity

With seed 42:

- build Arm B and Arm C;
- verify identical state-dict key sets;
- verify all initial parameter/buffer tensors are exactly equal;
- record a state hash.

The only config difference must be:

```text
condition_attention_execution
```

## Data/RNG identity

Verify on several dry batches that B and C see identical:

```text
sample IDs
query indices
sensor indices
sensor field IDs
sensor masks
targets
```

and that the RF source/time draw sequence begins identically.

---

# 12. Phase III — 200-epoch GL_rbf_CQ runs

Launch two independent 200-epoch runs.

## Arm B

```yaml
model:
  name: gl_rbf_cq

condition_attention_execution: legacy_mha
sensor_attention_padding_mode: full
```

## Arm C

Same config except:

```yaml
condition_attention_execution: cached_kv
sensor_attention_padding_mode: full
```

Use identical:

```text
seed
dataset
split
normalization
sensor protocol
query sampling
B128/Q4096
optimizer
EMA
query microbatch
checkpoint schedule
evaluation manifest
evaluation seeds
```

If multiple equivalent GPUs are available, either:

- run B and C sequentially on the same GPU for the cleanest timing comparison; or
- run simultaneously on matched GPUs but perform an additional controlled same-GPU timing benchmark afterward.

Do not compare epoch time across dissimilar GPUs.

---

# 13. Controlled execution benchmark for B vs C

In addition to the 200-epoch runs, run a short controlled B128/Q4096 benchmark on the same idle GPU.

Compare only:

```text
GL_rbf_CQ legacy_mha + full
GL_rbf_CQ cached_kv + full
```

Use identical initialized weights and tensors.

Measure:

```text
condition-context time
forward time
backward time
optimizer time
whole step
peak allocated
peak reserved
K/V projection calls
```

This is the primary evidence for the execution effect.

Expected K/V projection count:

```text
legacy_mha: 4
cached_kv: 1
```

---

# 14. Convergence evaluation

Evaluate milestone checkpoints at:

```text
20, 40, 60, 100, 200
```

using the same fixed validation manifest.

For every arm report:

```text
training RF loss
fixed validation/reconstruction MSE
mean relative L2
per-field relative L2
worst-field relative L2
best milestone
final milestone
```

If the downstream evaluator has additional case diagnostics, keep them but do not use a model-specific metric for only one arm.

For B/C report both when practical:

```text
EMA-selected metric
live-weight diagnostic metric
```

The release/default scientific metric remains EMA.

---

# 15. Convergence-speed metrics

Do not define convergence speed only as "loss after epoch 200."

Report:

## Epoch-based convergence

For common metric thresholds that at least two arms cross:

```text
first epoch reaching threshold
```

## Wall-time convergence

Report:

```text
first cumulative training time reaching the same threshold
```

## Area under convergence curve

Optionally compute a simple trapezoidal summary over the common milestone evaluations.

The most useful questions are:

```text
Does GL_rbf_CQ reach a given reconstruction quality in fewer epochs than legacy GL_rbf_ENH?
Does it reach that quality in less wall time?
Does cached_kv reach essentially the same quality as legacy_mha with lower time/memory?
```

---

# 16. Final result tables

Produce three separate tables.

## Table A — Migration/model effect: B vs A

Include:

```text
parameter count
peak memory
mean epoch time
200-epoch wall time
epoch-200 metric
best metric
epoch/time to shared quality threshold
```

## Table B — Execution effect: C vs B

Include:

```text
condition-context time
whole-step time
mean epoch time
peak memory
K/V projections
milestone metric differences
epoch-200 metric difference
```

## Table C — Practical latest-model effect: C vs A

Summarize the full downstream upgrade.

Do not collapse these three interpretations into one headline percentage.

---

# 17. Important interpretation rules

### Arm A is a corrected legacy downstream baseline

It should use the intended large scale rather than the accidental tiny placeholder.

However, the existing `EnhancedGLRBFTopK` implementation is not guaranteed to be parameter-for-parameter identical to the historical 0_demo GL_rbf_ENH.

Therefore:

- match the major capacity/training knobs;
- record parameter count;
- list remaining structural differences;
- do not invent additional legacy modules solely to force parameter equality.

### B vs C is the clean causal execution test

Because state initialization and scientific configuration are identical, this is where cached-K/V claims should be made.

### A vs B/C is a migration/release comparison

It intentionally includes scientific model improvements and downstream integration changes.

---

# 18. Do not do

Do not:

- modify the dataset format;
- copy the combustion HDF5 loader;
- change field order;
- change normalization between arms;
- silently reduce B/Q/model scale;
- use different sensor protocols;
- change optimizer between arms;
- add new model research features;
- modify `0_demo_TurbulentCombustion` portable source during the benchmark unless a genuine migration defect is found;
- overwrite Arm-A evidence after migration;
- replace the old `pointcloud_ffm` registry entry;
- merge this validation branch to `main` automatically.

If the migration guide is missing something, document the gap and propose a guide fix after the benchmark.

---

# 19. Evidence layout

Use a clear structure such as:

```text
Proj_MultiFieldReconstruction/
  benchmarks/
    gl_rbf_cq_migration_200ep/
      README.md
      PROTOCOL.yaml
      fixed_validation_manifest.json
      configs/
        A_legacy_gl_rbf_enh_200ep.yaml
        B_gl_rbf_cq_legacy_mha_200ep.yaml
        C_gl_rbf_cq_cached_kv_200ep.yaml
      baseline/
        A_BASELINE_SUMMARY.md
        A_metrics.csv
        A_performance.json
      migration/
        correctness.json
        initialization_identity.json
      runs_summary/
        B_summary.json
        C_summary.json
      execution/
        legacy_vs_cached.json
      comparison/
        milestones.csv
        final_summary.json
        RESULTS.md
        figures/
```

Keep large run/checkpoint files in ignored `Cases/.../runs/`.

---

# 20. Completion criteria

The downstream test is complete when:

1. corrected Arm A has completed 200 epochs;
2. its benchmark evidence was frozen before migration;
3. GL_rbf_CQ was integrated without replacing the project's contracts/data/trainer architecture;
4. all existing project tests remain green;
5. B/C initial state and data sequence identity are verified;
6. B and C each complete 200 epochs;
7. fixed-manifest milestone evaluation is complete;
8. memory and epoch/step timing are reported;
9. migration effect and execution effect are analyzed separately;
10. any missing migration-guide instruction is identified;
11. no automatic merge to `main` is performed.

Final report must include:

```text
validation branch + SHA
Arm-A pre-migration SHA
post-migration SHA
all three run directories
all three resolved configs
all three parameter counts
test results
200-epoch completion status
memory comparison
epoch-time comparison
convergence comparison
B-vs-C controlled execution benchmark
migration effect conclusion
execution effect conclusion
recommended downstream default
migration-guide gaps, if any
```

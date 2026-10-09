# PhyCoFlow multitask optimization preparation

This guide documents the bounded preparation of the existing post-training pipeline for controlled multi-objective experiments. It defines the scientific hierarchy, the two optimization interfaces, exact configuration forms, the local topology-lite smoke, and the handoff protocol. The smoke demonstrates execution on a real SOURCE checkpoint; it does not rank algorithms or establish scientific efficacy.

## Scientific problem hierarchy

| Level | Scientific meaning | Treatment in this work |
| --- | --- | --- |
| 0 — acceptance | Native rectified-flow validation and total/per-field reconstruction fidelity relative to SOURCE | Retain existing evaluation, checkpoint selection, and fidelity reporting; add no controller or guarantee. |
| 1 — outer objectives | Native reconstruction loss (D) and scalarized coherence surrogate (C') | Add the explicit two-row outer gradient aggregation interface. |
| 2 — coherence families | A: global distribution; B: cross-spectrum; T: topology | Add Fixed Sum, STCH, and DWA family scalarization. |
| 3 — descriptors | Marginals and joint field statistics, graph-frequency descriptors, and persistent topology components | Preserve current descriptor definitions, targets, weights, geometry, and calibration. |

For family k in A/B/T, let Lₖ be its raw differentiable loss, sₖ its SOURCE-derived initial calibration factor, and wₖ its configured family preference. The calibrated value is xₖ = sₖ × wₖ × Lₖ. Original scientific coherence remains C = Σₖ xₖ, while the selected scalarizer produces C′ = S(x_A, x_B, x_T). The outer task gradients are g_D = a ∇D and g_C′ = b ∇C′, where a and b are the existing `objectives.data_retention.weight` and `objectives.coherence.weight`, including any configured schedule. Calibration factors are fixed after SOURCE initialization; family preferences are fixed configuration values. Neither is a learned controller.

The old `optimization.gradient_balance: config` mode remains a legacy compatibility path. It uses the weighted sum for aligned task gradients, tries legacy ConFIG only when the cosine is negative, and can fall back to the weighted sum when its candidate is rejected. Existing configurations without `optimization.multitask` preserve that behavior and meaning. A configuration with `optimization.multitask` selects one named scalarizer and aggregator explicitly; its `torchjd_config` method calls TorchJD ConFIG on every eligible active update and has no alignment gate or automatic weighted-sum fallback. Legacy conditional ConFIG and always-on TorchJD ConFIG are different experimental conditions.

## The two interfaces

`FamilyScalarizer` receives the ordered differentiable family losses, SOURCE calibration factors, and configured family preferences. Fixed Sum returns C′ = x_A + x_B + x_T, preserving the former fixed weighted aggregate and its gradient. STCH uses TorchJD's smooth Tchebycheff form C′ = μ · log(Σₖ exp(λₖ xₖ / μ)); its defaults are a uniform preference λₖ = 1/K and no ideal-point shift. Its derivative weights are ∂C′/∂xₖ = λₖ · softmaxₖ(λ · x / μ), so for K=3 their sum is 1/3. Decreasing `mu` sharpens the soft maximum. `mu: 0.1` is an illustrative starting value, not a tuned value for this loss scale; no additional normalization is performed. With the smoke's calibrated inputs, STCH reported derivative weights A=0.112577, B=0.112524, T=0.108232, whose sum is one third.

DWA applies C′ₜ = Σₖ λₖ,ₜ xₖ,ₜ. Let qₖ,ₜ be the reporting-epoch mean of calibrated values passed to the scalarizer. At epoch t, λₖ,ₜ = K · exp((qₖ,ₜ₋₁ / qₖ,ₜ₋₂) / T) / Σⱼ exp((qⱼ,ₜ₋₁ / qⱼ,ₜ₋₂) / T). DWA uses unit weights during its first two reporting epochs; the third epoch can use the two earlier means. The installed TorchJD 0.17.1 DWA returns an empty public `state_dict()`, so this minimal adapter follows its published formula and stores family names, the last two completed means, the partial epoch sum/count, temperature, and completed-epoch count. Numeric agreement with TorchJD and partial-epoch serialization were tested on CUDA. Its `step()` is called once at a real completed reporting-epoch boundary; a partial epoch accumulates batch values without finalizing history, and dry runs do not advance it. See the [TorchJD DWA API](https://torchjd.org/stable/docs/scalarization/dwa/) and [original DWA paper](https://openaccess.thecvf.com/content_CVPR_2019/html/Liu_End-To-End_Multi-Task_Learning_With_Attention_CVPR_2019_paper.html).

`OuterAggregator` receives the ordered two-row matrix `G = [g_D; g_C′]`, with one row per objective and columns covering the trainable parameter vector. `sum` returns `g_D + g_C′`. The current modular Sum path computes both task gradients so the same update records per-task diagnostics; it is not a single-backward speed path. `torchjd_config` invokes TorchJD ConFIG on both rows unconditionally for each active update. Under nonzero, nondegenerate inputs, ConFIG's normalized-direction construction cancels positive scalar preweights on the rows; the family weights and scalarizer still change the coherence gradient itself. `upgrad` applies TorchJD's dual-cone projection aggregation with its default equal preference. `cagrad` uses TorchJD CAGrad; `c: 0.5` is an initial example, not a selected optimum. Aggregators may return different direction norms, which are measured rather than renormalized away. Report the two task norms, their cosine, directional products, and the aggregated norm. These values are measured before clipping; `clip_input_norm` is the clipping utility's returned pre-clipping norm. A method's gradient-direction properties do not guarantee a favorable finite AdamW displacement or validation fidelity, and ConFIG is not gradient descent on `aD + bC′`. An undefined or nonfinite modular direction is an explicit failed attempted update; the modular path does not silently substitute a weighted sum.

The optimizer, gradient clipping, EMA lifecycle, evaluation, SOURCE calibration, family computations, and checkpoint manager remain in the existing post-training path. The modular integration exposes family losses from the same coherence-objective call and records family diagnostics from the same live graph; it does not repeat the rollout to obtain the scalarizer input. The original `L_A`, `L_B`, `L_T`, calibrated `x_A`, `x_B`, `x_T`, and `C′` are separate quantities. A lower `C′` alone does not establish that all original families improved.

## SOURCE and historical reference

The inspected SOURCE is `cases/turbulent_combustion/runs/tc_gl_rbf_cq_cached_kv_5000ep/20260826T212231Z_7b68c461/`; all smoke runs selected its original base-training `checkpoints/best.pt` (`global_step=253260`). Its model is GL-RBF/CQ with fields CH4/CO/T/U_1/p, `cached_kv` condition attention, a 100x403 spatial grid, and 4,096 query points. The HDF5 field tensor has shape `(1, 10000, 40300, 1, 1, 5)`: the training split has 8,000 snapshots, validation has 1,000, and TEST has 1,000. Smoke evaluation used validation only; no TEST metrics were computed. The smoke YAML leaves `source_run: null`; pass the inspected local run path on the command line so the file stays portable.

The historical diagnostic/configuration reference is `cases/turbulent_combustion/runs/coherence_fix_ABC_sliced_persistence_formal_5000ep_gpu1/20260924T030724Z_4208231c/`. It used the same original base run as SOURCE and `last.pt`, `full_model`, two Euler rollout steps, objective weights 0.1/1.0, learning rate `5e-5`, batch and coherence batch size 32, and 4,096 fixed-shared points. Its legacy `gradient_balance: config` is conditional ConFIG. Topology used CO and T, 32x128 raster, H0/H1, sublevel/superlevel, 32 projections, and self plus mutual persistence. A 38-batch epoch in its log took about 68–71 seconds. Those settings and times are historical references, not a baseline ranking or a performance guarantee for the smaller smoke raster.

On another workstation, discover the local original SOURCE artifacts before setting `source_run`; inspect the chosen child's `resolved_config.yaml` and checkpoint metadata for architecture, field order, normalization, and checkpoint identity. The historical coherence run supplies configuration and diagnostic context only. Neither directory should be edited or used as the output destination.

```bash
rg --files --hidden --no-ignore \
  cases/turbulent_combustion/runs/tc_gl_rbf_cq_cached_kv_5000ep \
  -g resolved_config.yaml -g best.pt -g last.pt
```

## Topology-lite configuration and execution

The exact tracked smoke profile is [multitask_smoke.yaml](../cases/turbulent_combustion/configs/posttrain/multitask_smoke.yaml). It resolves to the same configuration used for the completed baseline child when supplied the source and output-name overrides; the recursive comparison of resolved YAMLs found no differences. It keeps `inherit_base_config: true`, the original architecture and normalization, `full_model` scope, native rectified-flow loss, two Euler steps, fixed-shared 4,096 points, SOURCE `initial_grad_norm` family calibration, AdamW, EMA state, validation, and standard checkpointing. It sets a bounded smoke learning rate `1e-6`, batch size 4, one reporting update per reporting epoch, three configured reporting epochs, and validation sample count 4. `sampling: full_pass` is required by the existing rules for `steps_per_epoch: 1`. Each reporting epoch consumes one batch from the full-pass sampler; it does not consume a full dataset pass.

A/B remain enabled with their five-field ABC recipe values. T is the actual differentiable `cubical_persistence` family on CO self-persistence only: grid `[16,32]`, H0 sublevel filtration, two sliced projections, and mutual persistence disabled. The fixed shared 4,096 point contract feeds this topology raster; the projected-coordinate collision check remains enforced. Cross-frequency B requires at least three coherence samples, so the batch and coherence batch are both 4. These settings produced a finite, nonzero source-model topology gradient; they are a differentiability/timing check and their metric scale must not be compared with the historical 32x128, H0/H1, 32-projection topology objective.

The source base config has `model_ema_eval: true`. The smoke explicitly sets `model_ema_eval: false` because the existing cubical-persistence rollout contract requires live evaluation weights. The model's EMA state is still preserved in checkpoints and updated after each optimizer step; validation in this smoke used the live model weights. Future comparisons must state and match the evaluation-weight policy.

From the repository root, activate the environment and install the optional methods and their solver dependencies without replacing the installed Torch/CUDA build:

```bash
conda activate phycoflow_env
python -m pip install -e '.[posttrain,topology,keops,multitask]'
python -m pip check
```

The project pins its optional `multitask` extra to TorchJD 0.17.1 with CAGrad and quadprog-projector support. The direct package installation used in this preparation was `python -m pip install 'torchjd[cagrad,quadprog-projector]==0.17.1'`; the repository-extra resolver was also dry-run checked and proposed only the editable project because all requirements were already satisfied. Verified environment versions were Python 3.10.19, PyTorch 2.5.1+cu121, CUDA 12.1, TorchJD 0.17.1, conflictfree 0.1.8, GUDHI 3.13.0, PyKeOps 2.3, Clarabel 0.11.1, CVXPY 1.7.5, and quadprog 0.1.13. The CAGrad path needs its CVXPY/Clarabel dependencies; UPGrad's default projector needs the quadprog support. TorchJD's [installation guide](https://torchjd.org/stable/installation/) documents its optional groups.

If the first PyKeOps extension import fails because its expected cache directory is absent, create only that reported directory and retry; the observed environment repair was:

```bash
mkdir -p /home/wanglz/.cache/keops2.3/Linux_deng-Lambda-Vector_6.8.0-138-generic_p3.10.19_CUDA_VISIBLE_DEVICES_2
```

This preparation used physical GPU 2; because `CUDA_VISIBLE_DEVICES=2` masks the other devices, the process addresses it as `cuda:0`. The following baseline command uses the exact case-local config path convention and one real optimizer update:

```bash
SOURCE_RUN_PATH="/home/wanglz/Desktop/src/PhyCoFlow/Proj_MultiFieldReconstruction/cases/turbulent_combustion/runs/tc_gl_rbf_cq_cached_kv_5000ep/20260826T212231Z_7b68c461"
CUDA_VISIBLE_DEVICES=2 conda run -n phycoflow_env python cases/turbulent_combustion/run.py post-train \
  --config configs/posttrain/multitask_smoke.yaml \
  --override "source_run=$SOURCE_RUN_PATH" \
  --override output.experiment_name=multitask_fixed_sum_live \
  --max-steps 1
```

To resume the stateless baseline, use the same configuration/source/output overrides, pass its printed child directory to `--resume`, and retain `--max-steps 1`; the completed preparation resumed `multitask_fixed_sum_live/20261009T022235Z_263ccac0` for one additional update.

Use these exact selector blocks under the profile's `optimization` mapping. An empty `optimization.multitask: {}` defaults to Fixed Sum + Sum; omitting the namespace entirely selects the legacy path. Keys for inactive methods are omitted and rejected by validation if supplied. Inherited `gradient_balance` and `config_*_grad_scale` keys retain legacy compatibility but do not affect the modular route; outer coefficients belong only to `objectives`. The first two entries are a plain Sum control and an explicit TorchJD ConFIG run; the legacy conditional mode is configured separately under `optimization.gradient_balance` and has no `optimization.multitask` block.

```yaml
# Fixed Sum family scalarization with plain outer Sum.
multitask:
  family_scalarization: {method: fixed_sum}
  outer_aggregation: {method: sum}
```

```yaml
# Fixed Sum with always-on TorchJD ConFIG.
multitask:
  family_scalarization: {method: fixed_sum}
  outer_aggregation: {method: torchjd_config}
```

```yaml
# STCH with always-on TorchJD ConFIG.
multitask:
  family_scalarization: {method: stch, stch: {mu: 0.1}}
  outer_aggregation: {method: torchjd_config}
```

```yaml
# DWA with always-on TorchJD ConFIG.
multitask:
  family_scalarization: {method: dwa, dwa: {temperature: 2.0}}
  outer_aggregation: {method: torchjd_config}
```

```yaml
# Fixed Sum with UPGrad.
multitask:
  family_scalarization: {method: fixed_sum}
  outer_aggregation: {method: upgrad}
```

```yaml
# Fixed Sum with CAGrad; c=0.5 is an initial value.
multitask:
  family_scalarization: {method: fixed_sum}
  outer_aggregation: {method: cagrad, cagrad: {c: 0.5}}
```

For one-update STCH, UPGrad, or CAGrad checks, keep the source and profile fixed and apply these exact method overrides to the baseline CLI shape. Each command writes to its own output name.

```bash
CUDA_VISIBLE_DEVICES=2 conda run -n phycoflow_env python cases/turbulent_combustion/run.py post-train \
  --config configs/posttrain/multitask_smoke.yaml \
  --override "source_run=$SOURCE_RUN_PATH" \
  --override output.experiment_name=multitask_stch_config \
  --override optimization.multitask.family_scalarization.method=stch \
  --override optimization.multitask.family_scalarization.stch.mu=0.1 \
  --override optimization.multitask.outer_aggregation.method=torchjd_config \
  --max-steps 1

CUDA_VISIBLE_DEVICES=2 conda run -n phycoflow_env python cases/turbulent_combustion/run.py post-train \
  --config configs/posttrain/multitask_smoke.yaml \
  --override "source_run=$SOURCE_RUN_PATH" \
  --override output.experiment_name=multitask_upgrad \
  --override optimization.multitask.outer_aggregation.method=upgrad \
  --max-steps 1

CUDA_VISIBLE_DEVICES=2 conda run -n phycoflow_env python cases/turbulent_combustion/run.py post-train \
  --config configs/posttrain/multitask_smoke.yaml \
  --override "source_run=$SOURCE_RUN_PATH" \
  --override output.experiment_name=multitask_cagrad \
  --override optimization.multitask.outer_aggregation.method=cagrad \
  --override optimization.multitask.outer_aggregation.cagrad.c=0.5 \
  --max-steps 1
```

For a two-plus-one DWA recovery check, keep the same source, output experiment name, selector overrides, and profile for both commands. Capture the run-directory printed after the first command and pass that run directory to `--resume`; `--resume` accepts the child run directory, not `checkpoints/last.pt`.

```bash
CUDA_VISIBLE_DEVICES=2 conda run -n phycoflow_env python cases/turbulent_combustion/run.py post-train \
  --config configs/posttrain/multitask_smoke.yaml \
  --override "source_run=$SOURCE_RUN_PATH" \
  --override output.experiment_name=multitask_dwa_resume \
  --override optimization.multitask.family_scalarization.method=dwa \
  --override optimization.multitask.family_scalarization.dwa.temperature=2.0 \
  --override optimization.multitask.outer_aggregation.method=torchjd_config \
  --max-steps 2
```

After the first command prints its new child directory, assign that exact path before running the resume command. Replace the following placeholder with the path printed by your run; the completed preparation used /home/wanglz/Desktop/src/PhyCoFlow/Proj_MultiFieldReconstruction/cases/turbulent_combustion/runs/multitask_dwa_resume/20261009T022535Z_a182b5a8.

```bash
DWA_RUN_DIR="/path/printed/by/the/first/command"
```

```bash
CUDA_VISIBLE_DEVICES=2 conda run -n phycoflow_env python cases/turbulent_combustion/run.py post-train \
  --config configs/posttrain/multitask_smoke.yaml \
  --override "source_run=$SOURCE_RUN_PATH" \
  --override output.experiment_name=multitask_dwa_resume \
  --override optimization.multitask.family_scalarization.method=dwa \
  --override optimization.multitask.family_scalarization.dwa.temperature=2.0 \
  --override optimization.multitask.outer_aggregation.method=torchjd_config \
  --resume "$DWA_RUN_DIR" \
  --max-steps 1
```

For the uninterrupted DWA recovery comparison, use the same source and DWA overrides with a new output name and three real updates:

```bash
CUDA_VISIBLE_DEVICES=2 conda run -n phycoflow_env python cases/turbulent_combustion/run.py post-train \
  --config configs/posttrain/multitask_smoke.yaml \
  --override "source_run=$SOURCE_RUN_PATH" \
  --override output.experiment_name=multitask_dwa_full \
  --override optimization.multitask.family_scalarization.method=dwa \
  --override optimization.multitask.family_scalarization.dwa.temperature=2.0 \
  --override optimization.multitask.outer_aggregation.method=torchjd_config \
  --max-steps 3
```

## Measured execution and verification

All training updates and the GPU topology probe used `phycoflow_env` and physical GPU 2. Eleven actual optimizer updates were performed in total: two Fixed Sum + Sum updates in one run (one initial, one resumed), one STCH + TorchJD ConFIG update, three DWA + TorchJD ConFIG updates split 2+1 across recovery, one UPGrad update, one CAGrad update, and three uninterrupted DWA updates for recovery comparison. Every completed child reported successful SOURCE immutability verification. No formal campaign, method search, or TEST evaluation was launched.

| Actual child run | Methods and completed updates | Local log |
| --- | --- | --- |
| `multitask_fixed_sum_live/20261009T022235Z_263ccac0` | Fixed Sum + Sum, 2 updates (one initial and one resumed) | `/tmp/multitask_fixed_sum_live_20261008.log`; `/tmp/multitask_fixed_sum_resume_20261008.log` |
| `multitask_stch_config/20261009T022442Z_25382298` | STCH(0.1) + TorchJD ConFIG, 1 update | `/tmp/multitask_stch_config_20261008.log` |
| `multitask_dwa_resume/20261009T022535Z_a182b5a8` | DWA(2.0) + TorchJD ConFIG, 3 updates (2 then resume 1) | `/tmp/multitask_dwa_segment1_20261008.log`; `/tmp/multitask_dwa_resume_20261008.log` |
| `multitask_dwa_full/20261009T022930Z_e1ef2978` | DWA(2.0) + TorchJD ConFIG, uninterrupted 3-update comparison | `/tmp/multitask_dwa_full_20261008.log` |
| `multitask_upgrad/20261009T022810Z_05bc38b8` | Fixed Sum + UPGrad, 1 update | `/tmp/multitask_upgrad_20261008.log` |
| `multitask_cagrad/20261009T022848Z_258c2a82` | Fixed Sum + CAGrad(0.5), 1 update | `/tmp/multitask_cagrad_20261008.log` |


The first Fixed Sum batch measured native `D=0.30447456`, raw family losses A=0.01228346, B=0.05103833, T=0.00549458, and SOURCE-calibrated values A=0.01730285, B=0.01716279, T=0.00549458. Fixed Sum therefore produced `C′=C=0.03996022`. On the same initial source batch, STCH produced `C′=0.11431819` and the derivative weights listed above. The unshifted STCH formula includes its smooth-max baseline; its scalar value should not be compared numerically with fixed-sum `C` as though they were the same objective.

UPGrad and CAGrad each completed a real update with Fixed Sum. The recorded starting task-gradient norms were approximately 0.422499 and 3.181692, with cosine 0.096635. UPGrad returned direction norm 1.624921; CAGrad with `c: 0.5` returned 1.973202 and directional products 0.497468 and 5.376334. The plain Sum direction norm was 3.249842. These are update diagnostics from the first batch, not method quality scores. STCH + TorchJD ConFIG completed one update with scalarized `C′=0.11431819`, task-gradient norms 0.422499 and 0.355982, and direction norm 0.575976. All selected methods wrote child checkpoints and emitted finite family and outer-gradient diagnostics.

The DWA segmented run completed two epochs, recovered the same child run directory, and completed its third update. Its third-epoch effective weights in A/B/T order were A=1.16583359, B=0.84444535, T=0.98972106; the checkpoint recorded `global_step=3`, `epochs_completed=3`, the two latest completed means, and no partial epoch. A separate uninterrupted three-update run used the same SOURCE, seed, profile, and method choices. Both checkpoint paths had 12 sample exposures and `dataset_passes=0.0015` over the 8,000-sample training split. Their model, AdamW, EMA, DWA means/weights, and family calibration values agreed within floating-point tolerance, not bitwise: maximum model parameter difference was 4.96e-8, maximum AdamW first-moment difference 6.82e-8, maximum EMA-shadow difference 7.45e-9, the maximum family-scale difference was 1.5e-9, and final DWA weight differences were below 9e-7. The last checkpoint retained `model_ema` state. The initial two epochs used unit DWA weights; adaptation appeared on epoch three as intended. The segmented resume began at an epoch boundary, and both final states had no partial DWA epoch accumulator; mid-epoch accumulator recovery was not exercised. In the early STCH run record, the scalarized coherence gradient norm was written under total_coherence_gradient_norm; the corrected canonical modular field is total_scalarized_coherence_gradient_norm.

A separate source-derived probe replayed the actual batch IDs used by the baseline, loaded the original `best.pt`, retained `full_model`, used two Euler steps and 4,096 fixed-shared points, then timed the configured topology family's actual forward and backward. CO cubical persistence on a 16x32 raster with H0 sublevel, two projections, and no mutual component produced finite topology loss 0.00549458 and full-model gradient norm 1.35939527. Synchronized topology `forward` took 43.3 ms including first lazy geometry-map construction; backward took 94.5 ms. This probe instantiated no optimizer and performed zero optimizer steps; its timing and gradient evidence was saved to /tmp/multitask_topology_probe.json. Its process peak allocation was 1.758 GB. Training update telemetry ranged from 2.67 to 3.26 seconds per update across these tiny runs; measured training peak allocation was 2.64–2.71 GB. Update timing excludes separate calibration, validation, and reporting work; it is not end-to-end epoch time.

The targeted checks passed: 71 focused coherence/calibration/integration/readiness/config/GL-RBF-CQ tests; 32 multitask adapter tests; 20 focused config tests; 17 legacy/source/checkpoint/recovery tests; and a three-test telemetry-label integration rerun in 2.31 seconds. The final config/integration group passed all 23 tests in 2.25 seconds. The 71-test group ran before the third integration test was added. These suites overlap; the combined deduplicated test inventory was 141 distinct tests. Full Ruff checks were clean, executable CLI validation passed, the dependency dry run resolved, and `pip check` reported no broken requirements. The targeted suite commands were:

```bash
CUDA_VISIBLE_DEVICES=2 /home/wanglz/miniconda3/envs/phycoflow_env/bin/python -m pytest \
  tests/test_multitask.py -o addopts= -q

CUDA_VISIBLE_DEVICES=2 /home/wanglz/miniconda3/envs/phycoflow_env/bin/python -m pytest \
  tests/test_multifamily_coherence.py tests/test_coherence_calibration.py \
  tests/test_multitask_integration.py tests/test_coherence_readiness.py \
  tests/test_config_contracts.py tests/test_gl_rbf_cq_configs.py -q

CUDA_VISIBLE_DEVICES=2 /home/wanglz/miniconda3/envs/phycoflow_env/bin/python -m pytest \
  tests/test_multitask_config.py tests/test_multitask_integration.py \
  -o addopts= -q --junitxml=/tmp/multitask_prep_config_integration.xml

CUDA_VISIBLE_DEVICES=2 /home/wanglz/miniconda3/envs/phycoflow_env/bin/python -m pytest \
  tests/test_coherence_contracts.py tests/test_checkpointing.py tests/test_run_contracts.py \
  -o addopts= -q --junitxml=/tmp/multitask_prep_recovery_regression.xml

CUDA_VISIBLE_DEVICES=2 /home/wanglz/miniconda3/envs/phycoflow_env/bin/python -m ruff check \
  src tests scripts cases benchmarks
```

Executable CLI validation passed for the same profile and SOURCE with DWA + CAGrad selected; this combination was configuration-validated and was not another optimizer trial. The command shape was:

```bash
CUDA_VISIBLE_DEVICES=2 conda run -n phycoflow_env python cases/turbulent_combustion/run.py validate \
  --config configs/posttrain/multitask_smoke.yaml \
  --override "source_run=$SOURCE_RUN_PATH" \
  --override optimization.multitask.family_scalarization.method=dwa \
  --override optimization.multitask.family_scalarization.dwa.temperature=2.0 \
  --override optimization.multitask.outer_aggregation.method=cagrad \
  --override optimization.multitask.outer_aggregation.cagrad.c=0.5
```

One early startup attempt inherited `model_ema_eval: true` and was rejected by the existing cubical-persistence live-weight contract before any optimizer update; the tracked profile explicitly sets it false while retaining EMA state and updates. One resume attempt passed the checkpoint file instead of its containing run directory and failed before an update; the corrected run-directory resume completed. One initial PyKeOps import found its cache directory missing; creating the environment-reported directory fixed the import without changing Torch or CUDA. These failures changed no SOURCE artifacts and did not consume the 11-update budget.

## Evaluation and collaborative experiment protocol

The branch is ready for subsequent bounded collaborative experiments with the agreed original SOURCE and matched protocol. Remaining limitations are the small exposure budget, four-sample validation panel, smoke-specific learning rate and topology resolution, modular Sum's two-gradient cost, and lack of efficacy or convergence evidence. The smoke disables preview generation and does not read or evaluate TEST. STCH, UPGrad, and CAGrad each had one update; Sum had an initial and a resumed update. The repeated DWA arm tests state recovery, not convergence. The recovery comparison covers the tested tensor and scalar checkpoint state at this three-update boundary; it does not test every possible mid-epoch interruption or cross-version resume. Numerical parity was within tolerance and not bitwise. The 16x32 topology profile establishes nonzero differentiation, not a useful topology ranking or comparability with the historical raster. No formal campaign, TEST evaluation, method search, or quality conclusion is supported by these runs. For later formal comparison, first agree on the exact original SOURCE checkpoint, data split, and protocol. Use a deliberately short candidate sequence, not a full factorial matrix: (1) historical weighted-sum/conditional-ConFIG compatibility versus Fixed Sum + always-on TorchJD ConFIG; (2) STCH + TorchJD ConFIG; (3) DWA + TorchJD ConFIG; (4) Fixed Sum + UPGrad; (5) Fixed Sum + CAGrad. Review those effects before combining a non-default family scalarizer with another outer aggregator. The first two ConFIG conditions have different semantics.

For each future arm, match source checkpoint, architecture, trainable scope, field order, normalization, sensor and validation manifests, batch/exposure budget, query points, rollout, optimizer, clipping, EMA evaluation policy, checkpoint cadence, and seeds. Select checkpoints on validation only; once candidate selection and criteria are fixed, evaluate TEST once. Report native TRAIN/VAL (D), total and per-field endpoint MSE and relative-L2 ratios to SOURCE, and raw A/B/T evaluation independently of (C'). Also record calibrated values, family weights/scales, (C'), outer task norms/cosine/directional products, actual AdamW displacement when inexpensive, runtime/memory, validation selection, source/config identity, and every error or failed update. Record whether evaluation used live weights or EMA. Keep spectral ensembles complete and deterministic at the declared batch size; disclose incomplete or dropped samples. Do not compare topology numbers across different raster, projection, filtration, or mutual-component settings.

Collaborator assignments:

- **Family-scalarization lead:** verify Fixed Sum equivalence to the historical family aggregate; characterize STCH sensitivity on the calibrated loss scale; and check DWA completed-epoch updates and resume state with matched TorchJD ConFIG.
- **Outer-aggregation lead:** compare legacy conditional ConFIG with always-on TorchJD ConFIG as separate modes, then measure UPGrad and CAGrad under Fixed Sum; report task gradients, direction norms/products, actual AdamW displacement, runtime/solver cost, and invalid attempts without hidden fallback.
- **Evaluation lead:** agree on the original SOURCE and validation/TEST protocol, per-field fidelity measures, spectral ensemble handling, and checkpoint selection; review arm-specific A/B/T tables before grouped claims.
- **All collaborators:** use the agreed source/protocol and matched exposure budget, retain each resolved config and checkpoint lineage, and report limitations alongside successful measurements.

## Changed files and implementation map

The 12 implementation, configuration, and targeted-test files are [post_training.py](../src/phycoflow_reconstruction/training/post_training.py), [multitask.py](../src/phycoflow_reconstruction/training/multitask.py), [gradient_balance.py](../src/phycoflow_reconstruction/training/gradient_balance.py), [validate.py](../src/phycoflow_reconstruction/config/validate.py), [post_training.schema.yaml](../docs/reference/config_schema/post_training.schema.yaml), [pyproject.toml](../pyproject.toml), [configuration.md](configuration.md), [test_multitask.py](../tests/test_multitask.py), [test_multitask_config.py](../tests/test_multitask_config.py), [test_multitask_integration.py](../tests/test_multitask_integration.py), [test_multifamily_coherence.py](../tests/test_multifamily_coherence.py), and [multitask_smoke.yaml](../cases/turbulent_combustion/configs/posttrain/multitask_smoke.yaml). This guide, [Multitask_Prep.md](Multitask_Prep.md), is the thirteenth changed file. Related unchanged source references are [coherence_calibration.py](../src/phycoflow_reconstruction/training/coherence_calibration.py) and the [topology family guide](../src/phycoflow_reconstruction/coherence/families/topology/README.md).

## References

- TorchJD primary documentation: [installation](https://torchjd.org/stable/installation/), [scalarization API](https://torchjd.org/stable/docs/scalarization/), [STCH](https://torchjd.org/stable/docs/scalarization/stch/), [DWA](https://torchjd.org/stable/docs/scalarization/dwa/), [aggregation API](https://torchjd.org/stable/docs/aggregation/), [ConFIG](https://torchjd.org/stable/docs/aggregation/config/), [UPGrad](https://torchjd.org/stable/docs/aggregation/upgrad/), and [CAGrad](https://torchjd.org/stable/docs/aggregation/cagrad/).
- Primary method papers: [Smooth Tchebycheff Scalarization](https://arxiv.org/abs/2402.19078), [End-to-End Multi-Task Learning with Attention / DWA](https://openaccess.thecvf.com/content_CVPR_2019/html/Liu_End-To-End_Multi-Task_Learning_With_Attention_CVPR_2019_paper.html), [ConFIG: Towards Conflict-free Training of Physics Informed Neural Networks](https://arxiv.org/abs/2408.11104), [Jacobian Descent for Multi-Objective Optimization](https://arxiv.org/abs/2406.16232), and [Conflict-Averse Gradient Descent for Multi-task Learning](https://proceedings.neurips.cc/paper_files/paper/2021/hash/9d27fdf2477ffbff837d73ef7ae23db9-Abstract.html).

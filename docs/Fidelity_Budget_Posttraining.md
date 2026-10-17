# SOURCE-relative fidelity-budget post-training

This guide describes the scalar external-policy route and its original smoke evidence. For the separate internal A/B/T TorchJD ConFIG composition and the authorized 1,000-epoch Runs 11–13, see [Formal Runs 11–13](Fidelity_Budget_Formal_Runs_11_13.md); that direction-based route uses multiple gradient traversals while retaining one shared reconstruction.

This opt-in extension provides framework readiness for deterministic and generative reconstruction models. It does not establish improved coherence, guaranteed fidelity, preservation of a generative distribution, or a preferred policy. The research branch `codex/fidelity-budget-posttraining` inherits the exact remote feature tip `096237ffb7c4a5c0ad90d869bdd9831cf0de105d` of `cosmos2w/flexible-loss-balancing`, fetched before branching. Historical joint RF/A/B/T, hierarchical scalarization, TorchJD, and legacy modes remain available when `external_policy` is absent.

## Meaning of the budget

One target-free `differentiable_reconstruction()` call supplies the observation-consistent prediction used by both coherence and paired-target endpoint risk. Flow retains its actual iterative rollout; deterministic models retain their direct differentiable reconstruction. Dense paired targets already carried by the coherence batch enter only the losses. These modes use no model-native `training_loss()` replay during optimization or SOURCE calibration, no teacher model, and no extra reconstruction for fidelity. They are supervised by the existing paired data, rather than data-free. Native metrics are unmeasured (`null`), while `data_loss` reports mean endpoint MSE across fields.

For already model-normalized predictions and targets, risk is pooled over valid sample/query entries, not averaged samplewise ratios:

\[
E_f=\frac{\sum_{bq}m_{bq}(\hat u_{bqf}-u_{bqf})^2}{\sum_{bq}m_{bq}},\qquad
\tau_f=(1+\epsilon)E_f^0+\delta,\qquad
a_f=\max(E_f^0,10^{-8}),\qquad h_f=(E_f-\tau_f)/a_f.
\]

Every declared output field is protected separately. Defaults are `relative_increase: 0.05` and `absolute_allowance: 0.0`: each model receives a 5% increase relative to its own original SOURCE endpoint MSE. The denominator floor affects scaling only; it never raises the acceptance bound. Zero SOURCE risk with zero absolute allowance implies a zero bound. Floor activation and actual thresholds are saved and reported. Risk arithmetic accumulates errors/counts in FP64 without normalizing a second time.

The original live SOURCE is measured before any model update, using fixed TRAIN calibration batches and seeds. Defaults are four batches and two generative draws. Deterministic models use one effective draw. Smoke profiles use two batches, with two Flow draws and one Senseiver draw, as a low-precision execution reference. Calibration preserves Torch, Python and NumPy RNG state and does not advance DWA history. SOURCE family gradient calibration remains a separate setup operation; it can evaluate families separately and perform family-gradient traversals, but never evaluates a native-loss branch in the new modes. No persistent teacher or baseline model copy is introduced.

The scientific coherence sum is \(C_{\mathrm{scientific}}=\sum_k s_k w_k L_k\), using the inherited family preferences and SOURCE calibration scales. The family scalarizer supplies \(\Phi\); the external objective uses \(\bar C=\Phi/\max(C^0_{\mathrm{scientific}},10^{-8})\). This automatic scale differs from historical unnormalized objectives and must be disclosed in future comparisons. With `family_scalarization.method: none`, ordered calibrated family tensors stay separate until the explicit external `vector_reduction: sum`. This equals `fixed_sum` in value and gradient, and does not change the meaning of `none` in the older joint aggregator. STCH and DWA remain available; their proxies need not have the scientific-sum scale.

## Three policies

`lagrangian` minimizes \(J=\bar C+\sum_f\lambda_f h_f\) with the current multipliers, then applies \(\lambda_f\leftarrow\max(0,\lambda_f+0.01\,\mathrm{stopgrad}(h_f))\) using the measured pre-update batch. Multipliers start at zero. The implementation uses Cooper 1.0.1's public `Constraint`, `ConstraintState`, `Lagrangian`, and `DenseMultiplier` components, plain PyTorch SGD with `maximize=True`, and Cooper's `post_step_()` projection. Cooper detaches multipliers in the primal contribution and violations in the dual contribution. A detached multiplier-only backward does not traverse the model graph. The existing AdamW owns the single model step; no `roll()` or second model optimizer step is added. Dual state advances only after a completed model update. Analytic tests compare these lower-level simultaneous semantics against Cooper's `SimultaneousOptimizer`. See the [Cooper 1.0.1 optimization API](https://cooper.readthedocs.io/en/stable/optim.html).

Optional `lagrangian.penalty_interval_updates: N` applies the primal fidelity term only at successful model updates N, 2N, …; its default is 1. SOURCE-relative risk measurement and projected dual updates still occur every batch. The restored controller update count determines cadence. No N-fold penalty compensation, additional rollout, teacher or optimizer step is introduced. In the internal ConFIG route, inactive updates omit the constraint-gradient traversal but retain all three coherence gradients and FP64 ConFIG. This sampled penalty provides weaker feedback and can permit substantial drift or multiplier accumulation; it does not guarantee the budgets. The [100-epoch Run 11 sparse candidate](Run11_Sparse_Fidelity_100ep.md) retains the separately tested model-LR warm-up.

`rectify` uses \(J=\bar C\) when \(\max_f h_f\le0\), otherwise \(J=h_{f^*}\), where \(f^*=\arg\max_f h_f\). Ties follow declared field order. It logs `improve` or `repair` and the selected field. It computes the family values once for diagnostics, but a repair backward follows only the selected risk. This borrows the switching principle from [Xu, Liang and Lan, CRPO, ICML 2021](https://proceedings.mlr.press/v139/xu21a.html); it does not implement the RL algorithm or transfer its guarantees to AdamW reconstruction.

`extended_log_barrier` minimizes \(J=\bar C+\sum_f b_t(h_f)\), using the [published extension, equation (3)](https://arxiv.org/html/1904.04205v5):

\[
b_t(h)=\begin{cases}-\log(-h)/t,&h\le-1/t^2,\\ th+(2\log t+1)/t,&h>-1/t^2,\end{cases}
\qquad
b'_t(h)=\begin{cases}-1/(th),&h\le-1/t^2,\\t,&h>-1/t^2.\end{cases}
\]

The implementation uses a safe logarithm argument on inactive branches. Value and first derivative match at the join, including infeasible inputs in the finite linear branch. With `u` completed model updates before the current update, \(t(u)=\min(100,1.0\times1.1^{\lfloor u/100\rfloor})\). The schedule is capped without overflowing and resumes from the saved update count. There is no additional barrier coefficient. Boundary and continuation behavior are exercised with tensors, without extending real runs to 100 epochs.

All three policies build one scalar, perform one ordinary model backward, apply the established finite-gradient check and clipping to optimized model parameters, take one existing AdamW step, then preserve the normal EMA/checkpoint lifecycle. Unused trainable parameters receive zero gradients, matching the old flattened-gradient route under AdamW weight decay, including complex parameters. Nonfinite calculations fail without a fallback to another algorithm. No per-field parameter-gradient matrix, projection solver, new trainer, or model-specific trainable adapter is introduced.

## Configuration and CLI

Use the existing environment. `rectify` and `extended_log_barrier` add no dependencies beyond Torch. The optional Lagrangian dependency is the distribution `cooper-optim`, imported as `cooper`:

```bash
conda activate phycoflow_env
python -m pip install -e '.[posttrain,topology,keops,multitask,posttrain_constraints]'
python -m pip check
```

The execution environment verified Torch 2.5.1+cu121, Cooper 1.0.1, PyKeOps 2.3 and GUDHI 3.13.0. `pip check` reported no broken requirements. Torch/CUDA were not upgraded.

The supported block below attaches to an existing coherence post-training configuration. Model architecture, normalization, observations and SOURCE loading stay with the existing case loader:

```yaml
objectives:
  data_retention: {enabled: false, weight: 0.0}
  coherence: {enabled: true, weight: 1.0}
optimization:
  multitask:
    family_scalarization: {method: fixed_sum} # none/stch/dwa remain available
    outer_aggregation: null
    external_policy:
      method: lagrangian
      vector_reduction: sum
      objective_scale: source_mean
      lagrangian: {dual_lr: 0.01}
    fidelity_budget:
      metric: endpoint_mse_model_units
      relative_increase: 0.05
      absolute_allowance: 0.0
      denominator_floor: 1.0e-8
      calibration_batches: 4
      calibration_draws: 2
      calibration_seed: 700103
```

Select `rectify` without a method-specific block. The barrier block is `extended_log_barrier: {t_initial: 1.0, growth_factor: 1.1, update_interval: 100, t_max: 100.0}` under `external_policy`, with `method: extended_log_barrier`. Relative/absolute allowances are common scalars for all fields in this first public configuration. A simultaneous outer aggregator, legacy gradient balancer, endpoint/source-anchor retention, sparse coherence schedule, or recurring gradient diagnostics is rejected as a contradictory configuration. The route uses full-model training and fixed endpoint or existing topology-with-fidelity checkpoint selection. Paired current targets are required for fidelity; a training-reference coherence bank remains independently supported.

From the repository root, these exact entrypoints use original completed checkpoints and configure one reporting epoch with two updates available. `--max-steps 1` shortens each invocation to one update:

```bash
python cases/turbulent_combustion/run.py post-train \
  --config configs/posttrain/fidelity_budget/smoke_flow.yaml \
  --override source_run=runs/tc_gl_rbf_cq_cached_kv_5000ep/20260826T212231Z_7b68c461 \
  --override output.experiment_name=fidelity_budget_checks/flow_lagrangian \
  --max-steps 1

python cases/turbulent_combustion/run.py post-train \
  --config configs/posttrain/fidelity_budget/smoke_senseiver.yaml \
  --override source_run=runs/tc_senseiver_5000ep/20260828T190145Z_fea0fc25 \
  --override output.experiment_name=fidelity_budget_checks/senseiver_lagrangian \
  --max-steps 1
```

For either SOURCE, use `configs/posttrain/fidelity_budget/rectify.yaml` or `extended_log_barrier.yaml` to select the other policies, with the same `source_run` override. When using a method fragment with an EMA-enabled Flow SOURCE, also pass `--override model.model_ema_eval=false` to select live weights. The Flow entrypoint already declares this option; deterministic SOURCEs need no EMA option. The three thin method profiles share `_common.yaml`; the two SOURCE entrypoints inherit architecture and observations at invocation and use the default Lagrangian selector. To use `none`, add `--override optimization.multitask.family_scalarization.method=none`. To replace a policy on a SOURCE entrypoint, replace the whole mapping, for example `--override 'optimization.multitask.external_policy={method: rectify, vector_reduction: sum, objective_scale: source_mean}'`, avoiding leftover method-specific options from recursive YAML merging.

Resume with the identical configuration and overrides, adding `--resume <exact-child-run-path> --max-steps 1`. `--max-steps` adds at most that many updates from the saved cursor and is still capped by configured `epochs * steps_per_epoch`; it never changes the configured horizon. The ordinary checkpoint saves model/AdamW, existing EMA and RNG, family calibration/state and DWA state when applicable, plus compact SOURCE reference statistics, multiplier/dual-optimizer state and completed policy updates. Resume loads these values rather than recalibrating the child as SOURCE. `artifacts/fidelity_budget_reference.json` is a readable setup report; the recovery checkpoint is authoritative for controller/reference state.

Smoke profiles use batch/coherence/validation batch four, 4,096 fixed shared queries, full-model AdamW at `1e-6`, live evaluation, and genuine CO self cubical persistence on a 16×32 raster with H0/sublevel, two sliced projections and mutual topology disabled. A/B remain active, with cross-frequency ensemble minimum four. Flow uses two Euler steps; Senseiver's direct hook ignores the generic step argument. These are execution settings, not scientific recommendations. Native previews, TEST evaluation and repeated reconstruction figures/full-grid exports are disabled; a final ordinary history figure and fixed validation endpoint check remain.

## Verified execution

All six pretrained smokes ran sequentially on physical GPU 1 (`CUDA_VISIBLE_DEVICES=1`) on 15 October 2026, using original `last.pt` SOURCEs, not post-trained arms. Flow SOURCE `tc_gl_rbf_cq_cached_kv_5000ep/20260826T212231Z_7b68c461` and Senseiver SOURCE `tc_senseiver_5000ep/20260828T190145Z_fea0fc25` were independently inspected as completed historical runs at 315,000 updates / 5,000 epochs. Those historical horizons were not launched or continued in this work. Both use `[CH4, CO, T, U_1, p]` and inherited `mean_std` normalization. Flow's saved EMA lifecycle is retained while evaluating live weights; Senseiver has no EMA state. Flow's historical evaluation used four steps; these child SOURCE references and smokes use the explicitly declared two-step operator.

Every child below declares `optimization.epochs: 1`, `steps_per_epoch: 2`, and receives `--max-steps 1`. The Flow Lagrangian child alone receives a second invocation with `--resume` and the identical one-epoch configuration. Paths are relative to `cases/turbulent_combustion/runs/fidelity_budget_checks/`:

| Model/policy | Child path | Configured epochs | Actual reporting epochs | Completed model updates | Stop/status |
| --- | --- | ---: | ---: | ---: | --- |
| Flow / lagrangian | `flow_lagrangian/20261015T234456Z_ae0be2f0` | 1 | 1.0 | 2 (1 + resumed 1) | configured limit / completed |
| Flow / rectify | `flow_rectify/20261015T234653Z_e542d7b6` | 1 | 0.5 | 1 | max-steps / integration_truncated |
| Flow / extended_log_barrier | `flow_extended_log_barrier/20261015T234720Z_dcc8dec2` | 1 | 0.5 | 1 | max-steps / integration_truncated |
| Senseiver / lagrangian | `senseiver_lagrangian/20261015T235025Z_1526c3bb` | 1 | 0.5 | 1 | max-steps / integration_truncated |
| Senseiver / rectify | `senseiver_rectify/20261015T235048Z_0ff0884c` | 1 | 0.5 | 1 | max-steps / integration_truncated |
| Senseiver / extended_log_barrier | `senseiver_extended_log_barrier/20261015T235110Z_1001f01e` | 1 | 0.5 | 1 | max-steps / integration_truncated |

These are seven genuine model updates in total. Instrumentation wraps the normal case CLI and real models: a raising native-loss sentinel covers setup and training; a hook on the shared prediction measures model-backward traversal; optimizer hooks and saved AdamW step tensors measure actual stepping. Every ordinary update had exactly one shared reconstruction, one prediction backward traversal, one AdamW step, zero native-loss calls and zero teacher calls. A Flow reconstruction used two velocity evaluations; a Senseiver reconstruction used one direct hook call. Setup/evaluation forwards are counted separately. Instrumentation is retained with the ignored child artifacts as `functional_smoke_first.json`, plus `functional_smoke_resume.json` for Flow; the one-time wrapper is saved under the ignored checks root as `functional_smoke_runner.py`. The compact [functional evidence summary](../cases/turbulent_combustion/runs/fidelity_budget_checks/functional_smoke_summary.json) links these local artifacts. Generated runs, figures, checkpoints and diagnostics are not versioned in Git.

Initial TRAIN reference risks and exact 5% semantics are illustrated below (rounded for display; artifacts/checkpoints contain full precision). Neither the field denominator floor nor the objective floor activated. `C0_scientific` was 0.03800324 for Flow and 0.18908408 for Senseiver, with the same estimator/settings within each model's three checks:

| Field | Flow SOURCE risk | Flow upper bound | Senseiver SOURCE risk | Senseiver upper bound |
| --- | ---: | ---: | ---: | ---: |
| CH4 | 0.00929737 | 0.00976224 | 0.00857473 | 0.00900347 |
| CO | 0.05741019 | 0.06028070 | 0.05241434 | 0.05503505 |
| T | 0.01573507 | 0.01652183 | 0.10744928 | 0.11282175 |
| U_1 | 0.03752251 | 0.03939864 | 0.04774133 | 0.05012840 |
| p | 0.01274319 | 0.01338035 | 0.01619199 | 0.01700159 |

On the first real batch, Flow's raw A/B/T values were `[0.01044391, 0.04898662, 0.00598904]`, SOURCE scales `[1.82861613, 0.28136453, 1]`, scientific sum `0.03887003` and scaled primary `1.02280855`. Senseiver's corresponding values were `[0.04685442, 0.10090548, 0.01494504]`, scales `[3.22895390, 0.49995607, 1]`, scientific sum `0.21668412` and scaled primary `1.14596701`. The topology term and its SOURCE calibration gradient were genuine cubical-persistence calculations, rather than a constant stand-in.

In field order, first-batch normalized violations were Flow `[0.0404453, -0.1171205, -0.0058500, -0.0322787, -0.3388343]` and Senseiver `[0.4284842, 0.1258980, -0.0465633, -0.0157635, -0.0661174]`. Lagrangian multipliers advanced from zeros to `[0.000404453, 0, 0, 0, 0]` and `[0.004284842, 0.001258980, 0, 0, 0]`, respectively. Both rectification checks selected `repair`, field `CH4`; their improvement branch is covered by tensor tests. Both barriers used the unmodified default `t=1` with five effective derivative weights of one; no continuation boundary was reached in a real run. Controller branch/schedule coverage comes from focused tensor tests rather than altered real targets, budgets or schedules.

| Model/policy, first update | Actual optimized scalar | Gradient norm before clipping | Clip coefficient | Synchronized shared forward + backward + step (s) | Peak CUDA allocation (GB) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Flow / lagrangian | 1.022809 | 73.45235 | 0.0136143 | 0.267 | 1.768 |
| Flow / rectify | 0.040445 | 47.38792 | 0.0211024 | 0.256 | 1.788 |
| Flow / extended_log_barrier | 5.569170 | 151.67562 | 0.0065930 | 0.279 | 1.768 |
| Senseiver / lagrangian | 1.145967 | 32.70317 | 0.0305781 | 0.226 | 0.271 |
| Senseiver / rectify | 0.428484 | 67.16681 | 0.0148883 | 0.180 | 0.291 |
| Senseiver / extended_log_barrier | 6.571905 | 155.34991 | 0.0064371 | 0.211 | 0.271 |

These one-shot timings exclude process startup, SOURCE loading, calibration, fixed evaluation and reporting, and do not establish relative speed. SOURCE risk/C0 preparation took 0.407–0.450 seconds for Flow and 0.248–0.257 seconds for Senseiver; separate family-gradient calibration took 2.174–2.749 and 1.622–1.697 seconds. Setup used six family-calibration reconstructions plus four risk-reference reconstructions for Flow, or two risk-reference reconstructions for Senseiver, followed by separate contract/evaluation calls. Terminal status timing includes reporting overhead and is not interchangeable with the synchronized update timing. No failed numerical model update occurred.

Fixed validation uses its own original SOURCE measurements. In field order, Flow validation SOURCE risks were `[0.0719810, 0.2369243, 0.0615667, 0.3842469, 0.2165198]`; Senseiver's were `[0.1000967, 0.3013053, 0.1052513, 0.4025285, 0.3305285]`. For the Lagrangian checks, terminal validation/SOURCE ratios were Flow `[1.001784, 1.000272, 1.001569, 1.001409, 1.003798]` after two updates and Senseiver `[1.000067, 0.999576, 1.000109, 1.001548, 1.000642]` after one. Every run's `evaluation/after.json` stores the individual ratios and its separately measured validation SOURCE vector. These tiny, unequal exposures are diagnostics, not a method comparison or population fidelity result. No TEST reconstruction/evaluation was performed.

The real Flow recovery checkpoint retained SOURCE risk/C0/bounds exactly, loaded the nonzero first multiplier, then advanced controller updates 1→2, all 145 AdamW parameter step tensors 1→2, and EMA updates 315001→315002. Learned tensors changed from the original SOURCE in every arm. Restricted checkpoint loading, inherited normalization and the existing SOURCE immutability checks passed. Synthetic integration tests separately check deterministic and iterative models, DWA recovery and calibration RNG preservation. Reporting histories contain epoch means for scalar telemetry and the latest nested field/controller diagnostics; use checkpoint/status counters for exact completed update counts.

Focused checks passed with real numerical computations:

- `tests/test_fidelity_budget.py`: **15 passed**; masked pooling/draws/gradients including invalid NaN padding, zero counts, 5%/near-zero bounds, explicit `none` sum, analytic and high-level Cooper agreement, detached/projection ordering, rectification, safe barrier derivatives, resumed continuation at updates 99/100, complex/unused AdamW behavior with and without clipping, and nonfinite risk/gradient refusal.
- `tests/test_fidelity_budget_config.py`: **30 passed**; new combinations/defaults, invalid combinations, portable composition and deterministic SOURCE inheritance without Flow EMA options.
- `tests/test_fidelity_budget_integration.py`: **3 passed**; genuine fixture-model reconstruction/backward/steps, no native replay through calibration/update/evaluation, SOURCE state, AdamW/DWA recovery, RNG preservation, and the direct-model observation-consistency regression. Fixture base runs declare one epoch and one update; fixture child runs declare two epochs and one update per epoch, split into one-update invocations.
- Existing multitask/joint/config/integration, coherence calibration and history suites: **107 passed**. Existing checkpointing, fidelity-history and rollout-context suites: **21 passed**. Existing topology-persistence suite, including the historical inference contract: **21 passed**. Stable-gradient tests: **8 passed**. These are **205 distinct focused tests** in total, counted once despite development reruns.
- Repository Ruff, `git diff --check` and dependency `pip check` passed. Existing long-horizon recipes were parsed by compatibility tests only; none was launched.

Encountered setup issues were resolved before successful updates: PyKeOps required its exact platform/GPU cache subdirectory; one resume invocation had an incorrect SOURCE path and exited before updating; and the first Senseiver setup exposed a Flow-only EMA option in the shared template. The flag is now confined to the Flow entrypoint, with inheritance regression coverage. Review also found raw deterministic inference differed from the declared observation-consistent endpoint; external fixed evaluation and its existing rollout check now use the shared reconstruction under `no_grad`, while historical paths keep their previous semantics. Early test attempts during implementation also caught missing `full_pass` sampling in a fixture and a transient in-progress helper import; final focused suites pass.

## Changed files

The patch is limited to the following 18 files; original SOURCEs and generated evidence remain outside Git. The small shared clipping correction preserves imaginary gradient components when applying its existing double-precision coefficient:

```text
src/phycoflow_reconstruction/training/fidelity_budget.py
src/phycoflow_reconstruction/training/post_training.py
src/phycoflow_reconstruction/training/checkpointing.py
src/phycoflow_reconstruction/training/rollout_contract.py
src/phycoflow_reconstruction/training/gradients.py
src/phycoflow_reconstruction/config/validate.py
pyproject.toml
cases/turbulent_combustion/configs/posttrain/fidelity_budget/_common.yaml
cases/turbulent_combustion/configs/posttrain/fidelity_budget/lagrangian.yaml
cases/turbulent_combustion/configs/posttrain/fidelity_budget/rectify.yaml
cases/turbulent_combustion/configs/posttrain/fidelity_budget/extended_log_barrier.yaml
cases/turbulent_combustion/configs/posttrain/fidelity_budget/smoke_flow.yaml
cases/turbulent_combustion/configs/posttrain/fidelity_budget/smoke_senseiver.yaml
tests/test_fidelity_budget.py
tests/test_fidelity_budget_config.py
tests/test_fidelity_budget_integration.py
docs/Fidelity_Budget_Posttraining.md
docs/Multitask_Prep.md
```

## Interpretation and user-owned validation

The budgets describe stochastic expected TRAIN risks. A batch can violate a SOURCE mean even at initialization; a positive batch violation is not by itself paired forgetting. The tiny calibration does not accurately estimate population risk. Rectification and barriers add sampling-dependent approximations: the expected barrier of batch risk differs from the barrier of expected risk. The fixed validation SOURCE baseline is evaluated separately and stored with validation ratios; TRAIN thresholds are never relabeled as validation SOURCE measurements.

Zero initial multipliers and AdamW memory allow transient violations. Rectification may oscillate and need not monotonically repair actual risk. The barrier exerts pressure inside the feasible region and does not certify feasibility. The scientific budget, normalization floors, dual learning rate and barrier schedule remain explicit design choices; these policies are not hyperparameter-free. Endpoint MSE does not establish preservation of RF bridge loss, diffusion objectives, conditional diversity, likelihood-related behavior, or all physical structure. Scalar magnitudes, multiplier size, or one-step changes cannot rank policies.

Subsequent substantive experiments belong to the user. Keep one coherence definition, common per-field budgets and declared reference protocol across both model types; match exposure and cost, keep TRAIN/validation/TEST roles separate, evaluate fidelity per field and each coherence family independently, and distinguish last from selected checkpoint identities. Assess reference-estimator uncertainty and repeated stochastic draws before interpreting budget compliance. Use a longer horizon only by explicitly configuring a separate user-owned experiment. No efficacy comparison or subsequent experiment is launched by this preparation.

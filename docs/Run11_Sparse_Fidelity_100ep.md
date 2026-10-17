# Run 11: retained LR warm-up and one fidelity penalty every 380 updates

This is one authorized exploratory candidate, configured for **100 reporting epochs / 3,800 AdamW updates**, on physical GPU 1. It is prepared on the ongoing branch `codex/fidelity-budget-posttraining`, inheriting **`1ba2e9357b59669116b842deea683f8d7740b51a`**. The completed original Run 11 and warm-up candidate remain separate, preserved artifacts. Run 12 and Run 13 remain paused; other GPU jobs are outside this candidate.

## Feasibility and limits

Retaining the AdamW warm-up is feasible: update 1 uses `5e-6`, update 76 uses `5e-5`, and later updates retain `5e-5`. Its position comes from the restored global update count. The earlier [20-epoch test](Run11_LR_Warmup_20ep.md) reduced the first-20-epoch TRAIN mean-field endpoint-MSE peak by 10.83%, retained coherence improvement, and did not eliminate overshoot or improve fixed validation consistently. Zero initial multipliers still mean zero fidelity correction on the first update. Pre-update SOURCE coherence is a separate baseline, not an optimization-induced error spike.

The user selected **one correction update every 380 updates**, rather than an entire correction epoch. For one-based successful update n, define s(n)=1 when n mod 380=0, otherwise 0. Internal ConFIG supplies d_C(n) from the three coherence family gradients; the composed model direction is d_C(n)/C0 + s(n) sum_f lambda_f(n) grad h_f(n). AdamW applies one globally clipped model step, then Cooper updates each multiplier to max(0, lambda_f(n)+0.01 h_f(n)), using detached current pre-update violations **on every batch**. No 380-fold rescaling is applied. Thus corrections are scheduled at 380, 760, …, 3,800; the first 379 updates have no primal fidelity correction, and a scheduled traversal is omitted when all multipliers are zero. This changes the feedback dynamics and can produce drift or large accumulated multipliers. It is not equivalent to the original dense Lagrangian method and offers no +5% guarantee.

Sparse penalties omit the extra fidelity-gradient backward on ordinary unscheduled updates. They retain the shared reconstruction, endpoint risk arithmetic, all three coherence-gradient traversals, and the existing FP64 ConFIG matrix calculation on every update. The FP64 work belongs to coherence aggregation, so this change does not address that cost directly. Replacing it with a small Gram solve would need a separate numerical review because forming a Gram matrix squares conditioning; no such change is included here.

Existing history confirms Run 11's lower memory and higher early latency relative to Run 03: over the first 2,280 updates their batch-weighted mean update times were 2.7595 and 1.1885 seconds, respectively; recorded peak allocations were approximately 15.33 and 20.18 GB. This is observational evidence only: GPU placement, workstation load, objectives, aggregation and run segments differ. It does not isolate FP64 cost or predict this candidate's speedup. No comparator was rerun.

## Exact scientific configuration

Profile: `cases/turbulent_combustion/configs/posttrain/fidelity_budget/run11s_lagrangian_lr_warmup_sparse380_100ep.yaml`. It inherits Run 11 through the existing warm-up profile and changes only the horizon, sparse penalty interval and output identity relative to that profile:

```yaml
defaults: [run11w_config_lagrangian_lr_warmup_20ep.yaml]
optimization:
  epochs: 100
  multitask:
    external_policy:
      lagrangian:
        penalty_interval_updates: 380
output: {experiment_name: _MultiTask/11s_lagrangian_lr_warmup_sparse380_100ep}
```

Inherited settings include base LR `5e-5`, `lr_warmup: {enabled: true, start_lr: 5e-6, updates: 76}`, dual LR `0.01`, zero initial multipliers, +5% per-field SOURCE-relative budgets, full-model AdamW, weight decay `1e-6`, clipping `1`, batch 32, 4,096 queries, two Euler steps, internal ordered A/B/T ConFIG, `family_scalarization.method: none`, unit family scales, and full CO/T 32×128 H0/H1 self/mutual topology. The original seeds, fixed validation panel and 8-batch ×2-draw TRAIN calibration remain unchanged. The 8,000-sample TRAIN set gives 38 reporting updates per epoch with the inherited fraction; 100 reporting epochs represent 15.2 nominal dataset passes, not 100 complete passes.

Fresh model weights come from the original Flow SOURCE `cases/turbulent_combustion/runs/tc_gl_rbf_cq_cached_kv_5000ep/20260826T212231Z_7b68c461/checkpoints/last.pt`; SOURCE AdamW and controller state are not adopted. The existing EMA behavior and SOURCE EMA counter are retained. Dataset `/home/wanglz/Desktop/src/PhyCoFlow/0_demo_TurbulentCombustion/Dataset/Merged_CH4COTU1P.h5` and original normalization statistics are reused directly. No physical SOURCE or dataset copy is made.

## Launch and validation record

The code/configuration is committed and pushed before launch. The first invocation is bounded to 38 updates to inspect its ordinary checkpoint; the same child then resumes for at most 3,762 additional updates. Both invocations retain the configured 100-epoch horizon and stop at update 3,800. There is no second candidate, sweep or extension. The commands below use the case CLI; the saved launch scripts additionally use the existing mechanical instrumentation wrapper for actual model-call/backward/step counts.

```bash
export CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export PHYCOFLOW_TOPOLOGY_WORKERS=4
export KEOPS_CACHE_FOLDER=/home/wanglz/.cache/phycoflow_formal_11_13/keops_gpu1
/home/wanglz/miniconda3/envs/phycoflow_env/bin/python cases/turbulent_combustion/run.py post-train \
  --config configs/posttrain/fidelity_budget/run11s_lagrangian_lr_warmup_sparse380_100ep.yaml --max-steps 38
# Set RUN11S to the exact child emitted by the first invocation.
/home/wanglz/miniconda3/envs/phycoflow_env/bin/python cases/turbulent_combustion/run.py post-train \
  --config configs/posttrain/fidelity_budget/run11s_lagrangian_lr_warmup_sparse380_100ep.yaml \
  --resume "$RUN11S" --max-steps 3762
```

Prelaunch verification passed **98 distinct CPU tests**: 18 sparse-penalty tests, 15 policy-math tests, 30 configuration regressions, 14 internal-ConFIG direction tests, 17 warm-up tests and four genuine reconstruction/topology integration cases. Coverage includes exact 379/380/381 and 759/760/761 boundaries, scalar and ConFIG risk-gradient omission, every-batch dual steps after model steps, default interval-one equivalence and bit-identical toy model/AdamW/controller resume across both sides of update 380. Integration cases use one-epoch fixture SOURCE training and one/two-epoch child horizons, one update per reporting epoch, including a sparse child split across checkpoint/resume. No long training job was used to test the cadence. Targeted Ruff and whitespace checks passed; resolved candidate equality against the inherited warm-up profile was checked for exactly the declared changes.

Live evidence will be recorded below. Standard histories average scalar risks/coherence/norms/times over each reporting epoch; nested field/controller mappings retain the last batch. `fidelity_penalty_active_fraction` records the fraction of scheduled corrections per epoch, and mean `constraint_gradient_evaluations` records actual traversals per update. Warm-up telemetry preserves all first 76 updates; the local instrumentation retains corrections and their neighboring updates. Endpoint and coherence values describe the pre-update prediction. Budget eligibility is assessed separately on the unchanged fixed validation panel and cannot be inferred from an averaged TRAIN loss.

The user owns subsequent efficacy assessment. Inspect validation eligibility for total and each field at common exact steps, actual correction-gradient norms/clipping, multiplier accumulation between impulses, epoch-weighted timings, and SOURCE-reference integrity. A faster launch or lower TRAIN coherence is not proof of sustained fidelity or scientific superiority.

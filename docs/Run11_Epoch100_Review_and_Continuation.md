# Run 11f: epoch-100 review and continuation

On October 17, 2026, Run 11f completed **100 reporting epochs / 3,800 AdamW updates**. The result is **mixed**: similar coherence reduction and lower recorded latency, but the final CO error slightly exceeded its budget. Following the user's explicit request, the same child resumed toward **1,000 epochs / 38,000 updates**. Completion of that continuation is not yet verified.

## First-100-epoch comparison

Old run: `cases/turbulent_combustion/runs/_MultiTask/11_config_lagrangian_1000ep/20261016T010804Z_e5999942`. New child: `cases/turbulent_combustion/runs/_MultiTask/11f_lagrangian_lr_warmup_dense_fast_100ep/20261017T170131Z_615167aa`. Its birth directory name and identity remain unchanged. Only epochs 1–100 and the 20 exact common validation steps 190–3,800 are compared. Each metric uses its own `finite_counts` weighting, including the old first epoch's 1+37 partial rows; nested field-risk entries are last-batch snapshots, not epoch means.

| Measurement | Original Run 11 | New Run 11f |
|---|---:|---:|
| Mean seconds/update, epochs 1–100 | 2.7493 | 1.6793 |
| Mean seconds per 38-update epoch | 104.47 | 63.81 |
| Epoch-100 mean TRAIN coherence | 0.00778305 | 0.00760985 |
| Coherence reduction from shared SOURCE C0 | 65.58% | 66.34% |
| Epoch-100 mean TRAIN endpoint data loss | 0.02815920 | 0.02771704 |
| Epoch-100 validation total normalized MSE | 0.4664 | 0.4755 |
| Validation total MSE change from SOURCE | −4.76% | −2.90% |
| Fully eligible common validation checks | 11/20 | 11/20 |
| Epoch-100 full fidelity guard | Pass | Fail: CO |

Epoch costs sum ordinary update timers, excluding validation, checkpointing, plotting, initialization and segment gaps. The observed 38.92% lower historical timer / 1.637× ratio cannot be attributed entirely to the patch: GPU placement, workstation load, warm-up and segmentation differ. The earlier same-GPU three-update implementation control measured only 0.75% lower mean latency, too few samples to establish a stable gain.

Coherence is the epoch-mean raw A+B+T sum. Old/new fidelity reference artifacts match exactly: C0=`0.022609325824305415`, mean field SOURCE risk=`0.027931112316212182`, from the declared 8-TRAIN-batch × 2-draw calibration. At epochs 1/5/20/50/100, coherence reduction is respectively 32.8/53.1/59.6/62.1/65.6% for old Run 11 and 29.0/52.0/59.6/62.1/66.3% for Run 11f. The high pre-update SOURCE coherence is separate from an optimization-induced error spike. TRAIN means do not establish per-field validation fidelity.

| Epoch-100 validation MSE change from SOURCE | Original | New |
|---|---:|---:|
| CH4 | −3.88% | +0.04% |
| CO | +0.92% | **+5.16%** |
| T | +0.44% | +3.79% |
| U_1 | −5.56% | −4.18% |
| p | −7.59% | −6.88% |

Total-MSE eligibility occurred on 14/20 old and 15/20 new checks. CH4/CO/T/U_1/p passed on 20/15/18/15/13 old checks and 20/15/19/17/15 new checks. Full eligibility requires total and every field within +5% of SOURCE; improved total error does not rescue CO at epoch 100. New eligible selections at update 950 (epoch 25, topology) and 3,610 (epoch 95, fidelity) were preserved alongside last at 3,800. The old full-1,000-epoch selected checkpoint is excluded. Repeated checks reuse one fixed panel and are not independent samples; this historical, single-seed contrast does not establish superiority or future stability.

## State-preserving continuation

Branch: `codex/fidelity-budget-posttraining`; inherited SHA: `5fbfe733633810effb6e4145521191357c1377f5`; pushed launch/code SHA: `ad36f521702d8fa31de652be66f50154ced8a253`, containing numerical optimization `4a8fc2d1fe0a990fa44b6ab4c3ed382e5fb34bc2`. The explicit `RunStore.extend_epochs(child, 1000)` metadata operation changes **only** `optimization.epochs: 100 → 1000`, updates the existing manifest digest and records the previous checkpoint binding. Ordinary resume's configuration check remains unchanged. Pre-extension metadata and hard links to last/best/best_fidelity checkpoints are preserved in the child's `artifacts/epoch_extensions/100_to_1000/`; atomic rolling-checkpoint replacement leaves these snapshots intact.

The epoch-100 restricted checkpoint audit found all 145 AdamW counters/controller at 3,800, EMA at 318,800, model LR `5e-5`, dual LR `0.01`, penalty interval 1, saved RNG/index-generator/family/normalization state, exact SOURCE reference and successful SOURCE immutability verification. SOURCE weights and physical dataset are reused without copying or modifying them. Optimizer, controller, EMA, reference and warm-up are restored, not reset.

The profile remains `configs/posttrain/fidelity_budget/run11f_lagrangian_lr_warmup_dense_fast_100ep.yaml`, relative to `cases/turbulent_combustion`. Full-model training, batch 32, 4,096 queries, two Euler steps, internal FP64 A/B/T ConFIG, family scalarization `none`, unit family scales, full CO/T 32×128 H0/H1 self/mutual topology, seeds/calibration/panel and +5% per-field SOURCE-relative budgets remain unchanged. Fidelity acts every update; AdamW remains at its post-warm-up LR `5e-5`.

```bash
export CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export PHYCOFLOW_TOPOLOGY_WORKERS=4
export KEOPS_CACHE_FOLDER=/home/wanglz/.cache/phycoflow_formal_11_13/keops_gpu1
# After the one-time explicit RunStore.extend_epochs(CHILD, 1000) operation:
/home/wanglz/miniconda3/envs/phycoflow_env/bin/python cases/turbulent_combustion/run.py post-train \
  --config configs/posttrain/fidelity_budget/run11f_lagrangian_lr_warmup_dense_fast_100ep.yaml \
  --override optimization.epochs=1000 \
  --resume cases/turbulent_combustion/runs/_MultiTask/11f_lagrangian_lr_warmup_dense_fast_100ep/20261017T170131Z_615167aa \
  --max-steps 34200
```

The actual command uses the existing mechanical instrumentation wrapper around this CLI, in tmux `phycoflow_run11f_gpu1`, PID **734186**, launched **2026-10-17 19:10:25 UTC / 3:10:25 p.m. EDT**. Physical GPU 1 (`GPU-3ceda40c-fd5c-4b88-6c47-b3301711571e`) was free immediately beforehand. No other job was displaced. Exactly 34,200 additional updates are allowed, ending at update 38,000 / epoch 1,000.

Actual update 3,801 shows all AdamW counters 3,800→3,801, controller 3,800→3,801, exact saved multiplier/reference values before the update and LR `5e-5`. The first three continued updates each used one AdamW step, one shared differentiable reconstruction, two ordinary Flow velocity calls, three family-gradient traversals plus one fidelity-gradient traversal, with zero native-loss replay or teacher calls. The unchanged projected dual update follows the model step at LR `0.01`.

## Checks and local evidence

**14 focused tests passed** in `tests/test_epoch_extension.py` and `tests/test_run_contracts.py`: archived checkpoint bytes/state, unchanged ordinary resume rejection, explicit horizon-only migration, snapshot survival after atomic checkpoint replacement, invalid targets/config bindings and injected metadata-write rollback. Targeted Ruff/format and whitespace checks passed. These CPU temporary-fixture tests launch no training campaigns. The sole research execution is the expressly authorized continuation to 1,000; no additional model evaluation or functional training run was launched for this review.

Local records under `cases/turbulent_combustion/runs/_MultiTask/11f_operations/` include `run11f_vs_old_epochs_1_100.json` / `.md`, the history-only extractor `compare_run11f_1_100.py`, `epoch100_pre_extension_audit.json`, `continuation_1000_record.json`, and actual `resume_1000_20261017T191012Z.sh` / `.log`. Live mechanical counters are in `/tmp/phycoflow_run11f_resume1000_20261017T191012Z.json`; durable startup evidence is also saved in the child's artifacts. Generated diagnostics, checkpoints and data remain outside Git. Later validation should track every field's budget and eligible selections; dense Lagrangian feedback does not guarantee fidelity at every update or on unseen data.

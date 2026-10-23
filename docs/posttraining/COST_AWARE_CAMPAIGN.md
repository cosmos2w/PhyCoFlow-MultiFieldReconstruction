# Cost-aware campaign runner

The foreground controller in `scripts/posttraining/cost_aware_campaign.py` pins a frozen SOURCE/config lineage, measures the registered execution alternatives, and admits bounded children one at a time. It does not read TEST data or invoke configuration validators. Dataset identity receipts are registered before a trainer or evaluator is imported. Children use the configured physical GPU (GPU 2 by default), run without a queue, and are limited to 100 epochs. Campaign-wide accounting enforces at most 800 science epochs, 50 auxiliary epochs, 850 combined exposures, 16 hours of process time with a one-hour closeout reserve, and 90 minutes of profiling/calibration.

Run commands from the repository in `phycoflow_env`. The current campaign root is the default; set it explicitly when working with another prepared root:

```bash
conda activate phycoflow_env
export CAMPAIGN_ROOT=/data2/wanglz/CostAwareLPC/CostAwareLPC_20261023T185427Z_01a16a15
RUNNER=scripts/posttraining/cost_aware_campaign.py
python "$RUNNER" --root "$CAMPAIGN_ROOT" inspect G
python "$RUNNER" --root "$CAMPAIGN_ROOT" profile G --gpu 2 --max-seconds 900
python "$RUNNER" --root "$CAMPAIGN_ROOT" plan G
```

`inspect` writes immutable config, capability, and pending-profile request records without opening dataset files. `profile` prints its proposed command and requires `--execute` to start the bounded SOURCE-only calibration/profile child. `plan` consumes the resulting measured TRAIN-only profile. A measured profile and selected plan are required before launching the strategy arm.

The launch, resume, and evaluation commands also print a dry-run command unless `--execute` is supplied. A training child can run for no more than 100 epochs and remains in the foreground:

```bash
python "$RUNNER" --root "$CAMPAIGN_ROOT" launch G dense --gpu 2 --max-seconds 10000 --execute
python "$RUNNER" --root "$CAMPAIGN_ROOT" resume G dense --run "$CAMPAIGN_ROOT/runs/G/dense/RUN_ID" --gpu 2 --max-seconds 7000 --execute
python "$RUNNER" --root "$CAMPAIGN_ROOT" evaluate G dense --run "$CAMPAIGN_ROOT/runs/G/dense/RUN_ID" --checkpoint best --max-samples 128 --gpu 2 --execute
python "$RUNNER" --root "$CAMPAIGN_ROOT" evaluate G dense --run "$CAMPAIGN_ROOT/runs/G/dense/RUN_ID" --checkpoint last --panel disjoint --max-samples 128 --gpu 2 --execute
python "$RUNNER" --root "$CAMPAIGN_ROOT" status
python "$RUNNER" --root "$CAMPAIGN_ROOT" stop
```

Replace `G` with a task in the frozen inventory, `dense` with `heuristic` or `strategy` where appropriate, and `RUN_ID` with the exact campaign child directory. Evaluation accepts only the validation split and sample counts divisible into exact groups of 32. `stop` signals only the runner-owned foreground child. The controller records reservations and actual observed charges in `manifests/campaign_runner_ledger.jsonl`; a stopped run resumes from its same frozen lineage.

## Execution contract

The reusable layer is `training/execution_strategy`; the campaign controller supplies experiment ownership and resource accounting. It is off unless `optimization.execution_strategy.enabled: true` is set with a bound task, measured profile, selected plan and independent execution seed. Profiles are specific to the SOURCE, full scientific configuration, hardware and numerical implementation; copy neither a selected probability nor a SOURCE risk denominator to another task. Version one accepts fixed family weights with `coherence.family_balance.mode: none`, scalar additive Lagrangian optimization, and dense endpoint/dual feedback. Existing ConFIG configurations retain their explicit method; periodic coefficient reuse remains quarantined.

Each sampled additive contribution includes its original coefficient and inverse inclusion probability. Conditional scalar/gradient unbiasedness holds before clipping and AdamW, and does not assert identical trajectories or held-out feasibility. Correlated filtration choices use their exact finite support rather than an independent-Bernoulli variance formula. Outer family probability times conditional filtration probability bounds the combined importance multiplier at four. Output-gradient norms are proxies; selection uses bounded parameter-gradient probes and an extra-variance ceiling equal to measured intrinsic TRAIN variance.

The planner compares at most six measured whole plans, including full execution, using conservative cost envelopes, acquisition and scheduled monitoring costs, a 5% savings floor, and the remaining horizon. Reference-cache states are matched before current-batch oracle preparation; production update timings exclude diagnostic accounting scans. Four batches do not establish a steady-state cache or latency certificate. Full execution remains a valid outcome. The runtime uses only TRAIN observations, revisits at ten-epoch intervals, and performs parameter-gradient audits at twenty-epoch intervals. Unsafe noise, sustained field-risk violations or excessive monitoring overhead trigger conservative full execution; this first version does not automatically reduce coverage again.

Resume uses the same frozen configuration and accepted checkpoint cursor. It restores model, optimizer, dense duals, Torch RNG, data-index generator and execution state, and checks the numerical implementation and next execution decision. The current adapters use Torch-backed or explicitly seeded randomness; global Python/NumPy RNG consumers in future adapters require additional recovery support. An interrupted checkpoint interval is charged conservatively as auxiliary exposure. Stop and wait for the owned active receipt to clear before resuming; no continuation beyond the original 100 epochs is admitted.

## Mature quality history

`phycoflow_reconstruction.evaluation.quality_history` assesses already-computed validation evidence; it does not load data or run validators. Provide each check's explicit `split`, epoch and completion time, raw numerators/counts for coherence, A, T, endpoint aggregate, and every endpoint field, plus four independently computed B groups of exactly 32 snapshots. Synthetic checks belong only in tests and cannot serve as campaign evidence.

The assessor evaluates complete trailing windows `(e-40, e]` at checks ending at epochs 60, 70, 80, 90, and 100. It pools raw numerators and counts before forming ratios. The coherence-sum target is at most 0.90; each A, B, and T family ratio must be at most 1.05; endpoint aggregate and every endpoint field must also be at most 1.05. Confirmation requires two adjacent complete passing windows, and its completion time is the completion time of the second check. The report then tracks later retention, losses, excursions and their magnitude/duration, and right-censors retention when later evidence is unavailable.

Build the campaign history from retained metadata and existing validation artifacts:

```bash
python scripts/posttraining/analyze_cost_aware_campaign.py --root "$CAMPAIGN_ROOT"
```

This parser reads retained artifacts and writes `analysis/quality_history.json`, the one-row-per-check `analysis/quality_history.csv`, and the long-form paired-component `analysis/quality_components.csv`; it does not open datasets, checkpoints, or validators. Primary `uniform_evaluations` are the scheduled native rows in each run's `metrics/topology_validation.jsonl`, paired to that run's `evaluation/before.json` SOURCE baseline. Portable final uniform/disjoint evaluations remain separate endpoint audits and do not replace scheduled history. It only accepts source-matched full validation checks with 128 snapshots, four exact 32-snapshot groups, 4096 query points, and six topology filtrations. Missing runs and checks remain pending, and ambiguous elapsed-time evidence is not used for attainment.

Native checks preserve completion UTC, controller `command_elapsed_seconds`, active process time, wall time from the first training command, and both `task_profile_acquisition_seconds` and `allocated_profile_acquisition_seconds`. Each task's measured profile acquisition is shown on each run for traceability and allocated to the strategy arm's cold time only; do not sum the repeated task-level value across arms. Dense and fixed-heuristic arms do not require the execution-plan profiles. The campaign ledger still charges all profile attempts to the global resource budget. The current evaluator stores paired group means rather than raw per-entry accumulators, so reported numerators are explicitly reconstructed from those means and frozen counts. Component diagnostics retain their original metric paths and are not summed into family totals.

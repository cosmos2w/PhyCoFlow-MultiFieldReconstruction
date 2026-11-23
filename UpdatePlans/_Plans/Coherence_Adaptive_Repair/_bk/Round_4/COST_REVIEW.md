> **Historical record:** Round 4 is closed. This document is preserved for reference and is not a current launch instruction.

# Measured teacher cost review

This operational review follows the demonstrated epoch-1 persistence failure and cold-epoch timing discrepancy. It does not change the empirical-law solver, observations, support, normalizers, objective, update stream or initial/maximum teacher horizon of 100 reporting epochs. Completed Round 3 and TEST remain closed.

Use the scheduled closed-lease observation. Preserve the current complete checkpoint by hard link and the matching history by copy in a named review directory before binding them; future checkpoint replacement and history append must not invalidate the review evidence. The saved scientific configuration remains unchanged.

Take the last five completed reporting epochs from the resumed lease. Match their recorded update wall times to the ordered checkpoint-write durations in that lease receipt. Let r = 100 minus the completed epoch age. Let T and C be all closed teacher lease GPU-wall and process-CPU seconds, including the failed first lease. Use the original safety factor 1.2 and the already declared startup/closeout allowances:

- Projected total teacher GPU work = T + 60 + 1.2 * r * (mean warm epoch update wall + mean matched checkpoint-write wall) + 1599 seconds.
- For CPU persistence/recording overhead, subtract the sum of that lease's recorded epoch CPU times from its complete per-lease CPU timer, then divide the nonnegative remainder by its completed epochs. This includes startup, checkpointing, plotting and teardown overhead conservatively; do not add nested job timers again.
- Projected total teacher CPU work = C + 60 + 1.2 * r * (mean warm epoch process CPU + measured overhead per epoch) + 1698 worker-seconds.

These are bounded-work planning forecasts, not lower bounds on every possible future runtime. Retain the measured window and component values so the projection is inspectable. No efficacy inference follows from this cost window.

After Luna reports the durable pause and closed lease, the reviewed one-shot helper is `scripts/posttraining/review_car_round4_teacher_cost.py`. It refuses active leases before reading teacher artifacts, preserves the checkpoint by hard link and history/scientific configuration by copy, and leaves both original forecasts unchanged. Use the existing attempt `SSS_Empirical_Endpoint_Teacher_Initial`, resumed token `3c4fb9ddec1647298aa9aef16467fd29`, a new descriptive review name, zero reserve committed elsewhere unless another allocation is documented, and `configs/execution_code_teacher_resource.json`. It writes `assessment/teacher_cost_reviews/<review-name>/warm_forecast.json`; it makes no stopping, admission or adoption decision. Four focused tests cover the formulas, closed-phase snapshot path, original preservation, active-lease refusal and compatibility with the shared admission/stop validators. Review its actual output before writing a decision.

The teacher's base proposed GPU allowance is 14400 seconds. WORKPLAN section 7 additionally allocates 18000 GPU-seconds for variation, failures and durable closeout; this permits no extra candidate, recipe or epoch extension. Let R be reserve already committed elsewhere, and let O_candidate and O_whole be all other candidate/comparison GPU work, including measured preparation/qualification, failed-start bounds, startup allowance, and the still-required native students and assessment forecasts. The effective maximum total teacher GPU budget is:

`min(14400 + 18000 - R, 35317 - O_candidate, 86400 - O_whole)`.

The CPU ceiling remains 21600 worker-seconds. If the measured age-100 projection fits both ceilings, an immutable resource admission may allocate the rounded-up projected teacher GPU total, up to the effective maximum; record the reserve draw above 14400 explicitly. Bind the measured forecast, manifest, checkpoint, unchanged scientific configuration, ledger and exact execution capsules before launch. A new capsule may amend only operational resource enforcement, missing-directory persistence and curve presentation; retain all original scientific source and teacher update/normalizer/replay bodies, and preserve the saved code-root lineage while recording actual execution code per lease.

If the remaining age-100 forecast cannot fit the effective GPU envelope or CPU limit, close for cost feasibility using a hash-bound budget-stop decision and partial-teacher cost/report paths. Do not claim that an incomplete teacher failed its TRAIN800 utility gate. Students remain unqualified. A completed teacher still needs the original TRAIN800 replay/native-feasibility gate before any student launch; target I/O and inclusive costs must then be refreshed for the native comparison. No additional arm or epoch extension follows either result.

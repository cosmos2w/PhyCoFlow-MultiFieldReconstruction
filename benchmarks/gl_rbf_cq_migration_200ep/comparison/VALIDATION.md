# Validation report

## Overall assessment: Share with caveats

The three-arm comparison is methodologically aligned with the requested
downstream decision and its headline calculations are verified. The saved
report is suitable for review on the validation branch. The required caveats
are the single training seed, the authorized B128-to-B40 protocol adjustment,
and structural-only (not browser-interaction) HTML verification.

## Methodology review

- Population and protocol: the same chronological dataset split, B40/Q4096,
  T-only 192--384-sensor protocol, seed 42, 200 epochs, and 200 steps per epoch
  are used by all formal arms. B128 failed during Arm-A backward allocation;
  B40 was authorized and frozen before any comparable formal arm completed.
- Comparison basis: all quality claims use the same 20-sample validation
  manifest, 4,096 query indices, 32 generation steps, normalization digest,
  and the seven prespecified milestone checkpoints.
- Weight selector: Arm A uses its configured legacy weights. Arms B and C use
  their prespecified configured EMA weights; live-weight evaluations remain
  diagnostic and are not mixed into the primary comparison.
- Effect definitions: B minus A is the migration effect; C minus B is the
  execution effect; C minus A is the total latest-model effect. Later minus
  earlier is used throughout, and all three quality metrics are lower-is-better.
- Causal boundary: the same-state, same-batch, same-RF-schedule 50-step probe is
  the causal C-versus-B execution evidence. Differences between independently
  trained B/C epoch-200 checkpoints are accumulated trajectory drift and are
  not presented as an execution-quality effect.

## Issues found and resolved

1. **Medium — convergence threshold mismatch.** The interrupted draft reported
   mean relative L2 at 0.80 rather than the requested 0.75 checkpoint. The
   generator now reports 0.75, 0.70, and 0.65 and was rerun twice with
   byte-identical outputs.
2. **Medium — stale guide-gap narrative.** The draft described semantic
   checkpoint identity, vendored-snapshot integrity, and lifecycle guidance as
   open after they had been fixed. The generated report now distinguishes the
   three fixes in commit `47de065a5b80a871297620e9703fb0bff528dff4`
   from four optional evidence-schema follow-ups.
3. **Low — generator static QA.** Import ordering, redundant f-string prefixes,
   executable shebang metadata, and one exception type were corrected. Ruff now
   passes both report generators.

No unresolved calculation, protocol, source-selection, or comparison blocker
was found.

## Calculation spot-checks

- Epoch-200 fixed-manifest deltas were independently recomputed from the frozen
  A/B/C source JSON and match `final_summary.json` to less than `1e-12`:
  B-A mean relative L2 `+0.009784340858459473`, C-B
  `+0.01798391342163086`, and C-A `+0.027768254280090332`.
- Migration resource effects match the raw formal telemetry: B-A peak allocated
  memory `-83.45285764946966%`, reserved memory `-84.5363440314601%`,
  steady mean epoch time `-54.005279547556654%`, and steady mean step time
  `-54.65629185306423%`.
- The controlled execution artifact verifies 50/50 measured steps with exactly
  four K/V projections for B and one for C. Its C-B median step change is
  `-3.157231093910251%`; maximum absolute loss and gradient-norm differences
  are `2.384185791015625e-07` and `2.9173436599805314e-07`.
- Threshold timing uses cumulative formal epoch-wall telemetry through the first
  evaluated checkpoint meeting each threshold. At mean relative L2 <= 0.75,
  A first reaches the threshold at epoch 100 (`5972.0 s`), while B and C reach
  it at epoch 60 (`1650.0 s` and `1607.3 s`).
- All three arms resolve to the same dataset, normalizer, fixed sensor-manifest,
  and query-index hashes. B/C initialization identity contains 148 equal state
  keys, and all formal histories contain 40,000 finite steps with no backward
  retry or non-unit scale row.

## Visualization and artifact review

The report uses a grouped bar chart rather than a line because only seven
discrete checkpoints were evaluated. The chart snapshot contains exactly 21
rows at arm-by-milestone grain, uses a common lower-is-better scale, and is
adjacent to the crossover interpretation and limitation. Exact values remain
available in two tables and `milestones.csv`.

The official portable builder reports validation `passed`, package `passed`,
and verification `structural_only`. Exact payload equality and the semantic
fallback passed. Chromium was not installed, so responsive browser interaction,
source-dialog behavior, and visual geometry were not verified; no browser was
downloaded for the report task.

## Required caveats

- This is one seed on one dataset/split and one GPU class; it does not estimate
  across-seed variability.
- A versus B/C formal resource deltas include architecture and lifecycle changes.
  Only the controlled B-versus-C probe isolates cached-K/V execution.
- Configured EMA is the primary B/C selector. C's live epoch-200 diagnostic is
  better than its EMA checkpoint, but changing selectors after observing the
  result would invalidate the prespecified comparison.
- The portable HTML is self-contained and structurally verified, but browser
  interaction QA remains unverified because Chromium is unavailable.

## Reproducible checks

- `generate_comparison.py` was run to the benchmark directory and a temporary
  directory; `milestones.csv`, `final_summary.json`, and `RESULTS.md` compared
  byte-for-byte.
- The portable generator was rerun and packaged with Node 20 through the official
  delivery script.
- Ruff passed both generators.
- Focused checkpoint, lifecycle, portable-gate, and training-preview tests passed
  20/20 with physical GPU 0 exposed.
- `git diff --check` passed, and the user's untracked benchmark Markdown remained
  untouched.

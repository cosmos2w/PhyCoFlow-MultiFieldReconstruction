# Test_1002 evidence index

**Updated retention recommendation:** see the [checkpoint retention review](./CHECKPOINT_RETENTION_REVIEW.md). The first-pass boundaries below describe what the initial fixture cleanup protected; they do not establish that every retained checkpoint is worth keeping. The review recommends 32 model files (2.61 GiB) and identifies 352 removal candidates (26.52 GiB), with no checkpoint deletion performed yet.

This directory retains the R1–R5 experiment lineage, including unsuccessful fidelity checks, controls, confirmations, profiling runs and bounded sibling recoveries. Start with the [R5 report](../../../../MODEL_UPGRADE_1002_R5.md) and [R4 report](../../../../MODEL_UPGRADE_1002_R4.md). These reports distinguish completed bounded experiments from formal-run preparation; cleanup does not change their scientific conclusions.

## First-pass retention boundaries

| Location | Purpose and retention decision |
| --- | --- |
| `T00_*` through `T52_*` | Original controls, individual A/B/C arms, optimizer comparisons and repaired R1 lineages. Keep their checkpoints, resolved configs, calibration, histories, figures and arrays. |
| `R2_00_*` through `R2_90_*` | Native-risk controls, optimizer arms, second seed, learning-rate comparison and sibling smokes. Keep mature and intermediate checkpoint identities, recovery/interruption evidence and failed fidelity measurements. |
| `R3_10_*`, `R3_20_*`, `R3_30_*`, `R3_90_*` | Corridor control, finite-primary candidate, confirmation and bounded sibling. Keep all scientific checkpoints and evidence. |
| `R4_00_*`, `R4_10_*`, `R4_20_*`, `R4_30_*` | Mathematical validation, physical runtime profiles, reference/adaptive trajectories and confirmation. Profiling outputs are measured evidence, not disposable test fixtures. |
| `R5_N_native_only`, `R5_F_scalar_reference`, `R5_S_source_regularized`, `R5_confirmation` | Native-only control, matched scalar comparison, continuation and fresh-seed confirmation. Keep their full scientific lineage. |
| `R5_profile_*`, `R5_formal_source_sibling` | Actual performance measurements and SOURCE-initialized bounded recovery evidence. Keep checkpoint/config/runtime identities; the sibling does not establish 5,000-epoch stability. |
| `_reports`, `R3_analysis`, `R3_final_figures`, `R4_reports`, `R5_reports` | Numerical tables, raw sums/denominators, evaluation contracts, contour arrays, PDF masters, diagnostic decisions and final reviews. Keep original numerical artifacts and failed diagnostics as well as final summaries. |
| `_audit`, `R5_campaign` | Provenance, frozen source snapshots, stage ledgers, process/budget reviews, command logs and campaign receipts. Keep these; an audit-folder name does not imply disposability. |
| `_configs`, `R3_configs`, `R4_configs`, `R5_configs` | Exact historical experiment settings. Keep to preserve experiment identity and replay contracts. |
| `R2_native_tests`, `R3_tests`, `R4_tests`, `R5_tests`, `R5_tools` | Source helpers, test sources, JUnit/log/JSON receipts and retained reference implementations. Keep source and pass/failure receipts; large synthetic fixture trees have been archived. |
| `R4_runtime_cache`, `R5_runtime_cache` | Mixed runtime caches and validation evidence. Bytecode was removed, but receipts, source-containing trees and other unclassified files remain. Do not delete these directories wholesale. |
| `_cleanup/20261006` | Exact pre/post inventories, cleanup plan, protected hashes, live-use audit, verified fixture archive and completion receipt. Keep this recovery bundle locally; it is not a Git payload. |

Scientific run directories remain at their original paths because report links, configs, checkpoint provenance and recovery records refer to them. SOURCE checkpoints outside this directory and datasets were outside the cleanup scope. Existing worktree report deletions and the modified R3 formal config were left untouched.

## Cleanup performed on October 6, 2026

The recursive audit recorded 19,372 real directories. The completed cleanup leaves 1,952, including the new receipt directory hierarchy: 17,420 fewer directories, approximately a 90% reduction. The top-level campaign layout was preserved rather than relocated.

Exactly 2,147 numbered pytest fixture roots, containing 29,332 regular files, were packed into a verified archive before removing their unpacked trees. The bundle includes 5,737 synthetic fixture `.pt` files and 1,869 external pytest `current` pointers; scientific campaign checkpoints remain unpacked and unchanged. Fixture trees containing Python/shell/notebook source or tracked files were excluded from archival. Every archived regular file was SHA-256 checked against its original bytes; symlink targets were checked without following them, and no regular fixture file had multiple hardlinks.

Exactly 21,318 disposable `.pyc`/`.pyo` files were deleted. Empty ancestors of removed fixtures and bytecode were pruned. All 9,562 retained non-directory entries passed identity/metadata checks, and 3,493 retained checkpoints, sources, numerical artifacts and PDFs additionally passed SHA-256 checks. The final accessible process command/cwd/open-file scan found no owners of cleanup targets. Some system-owned or otherwise inaccessible processes could not be fully inspected; the denied PID list is recorded in the [live-use audit](./_cleanup/20261006/live_use.json). No training or evaluation was launched for this cleanup.

The fixture archive is 656,991,697 bytes, replacing 1,005,517,078 bytes of unpacked regular fixture data. Deleted bytecode accounts for another 280,452,499 bytes. Net payload savings are 628,977,880 bytes (about 600 MiB), before receipt overhead and filesystem block accounting. This pass primarily reduces directory clutter; most retained storage is substantive scientific evidence.

- [Completion receipt](./_cleanup/20261006/result.json)
- [Exact cleanup plan and archived member inventory](./_cleanup/20261006/plan.json)
- [Before inventory](./_cleanup/20261006/before_inventory.json) and [after inventory](./_cleanup/20261006/after_inventory.json)
- [Retained scientific/source SHA-256 manifest](./_cleanup/20261006/protected_hashes.json)
- [Archived fixture SHA-256 manifest](./_cleanup/20261006/fixture_hashes.json)
- [Recoverable fixture archive](./_cleanup/20261006/synthetic_test_fixtures.tar.gz)

## Restoring archived test fixtures

Ordinary tests regenerate their own temporary fixtures. To inspect an old fixture at its recorded path, verify the archive checksum and extract only the relevant member into this directory. Historical references to archived fixture paths are intentional and recoverable; scientific paths and retained report receipts still resolve normally. Extracting an old pytest `current` pointer restores its original target, so restore that target too if needed.

Run the following from `Test_1002`, replacing the example member with a fixture root listed in `plan.json`:

```bash
sha256sum _cleanup/20261006/synthetic_test_fixtures.tar.gz
# Expected: 8fe3121fe3ed22e60c8f34c4f9120992eb4b3f908b0bc1303cee7d62b6a869d9
tar -tzf _cleanup/20261006/synthetic_test_fixtures.tar.gz
tar -xzf _cleanup/20261006/synthetic_test_fixtures.tar.gz -- R4_00_exact_math/pytest_tmp/test_v4_accumulator_reports_po0
```

Do not remove additional scientific checkpoints, original arrays, PDF masters, code or calibration/provenance based on names such as `tmp`, `cache`, `profile` or `audit`. Further campaign-level reduction requires a separate decision about which replay and comparison evidence to retain.

# Project organization and storage migration

The project now separates maintained code, dataset catalogs, bulk run/evidence storage, plan reading copies and portable report publications. The HOME checkout fell from 57,679,835,136 bytes (53.72 GiB) to approximately 0.25 GiB after publication and Git staging. About 53.47 GiB of physical checkout storage was released. Measurements use `du -x` without following symlinks; they exclude the external dataset owners reached through catalog links. Final exact accounting is recorded in the DATA2 migration operations receipt.

Start from the [research index](../../README.md), [plan library](../../_Plans/README.md), [report library](../README.md), [physical storage map](../../../docs/Storage_Map.md) and [management rules](../../../docs/Experiment_Artifacts.md). The [retained cleanup list](CLEANUP_REVIEW.md) collects items requiring a later owner/dependency decision. This was an engineering organization task; no model training, inference, metric assessment or new experiment was run.

## Storage and archives

All five case `runs` roots now resolve directly into DATA2 `cases/<case>/runs/`. The existing turbulent-combustion Adaptive Repair subtree was preserved rather than copied over. Checkpoints, complete histories, ten-epoch curves, raw assessments, operations and failed attempts remain together in their case-owned storage. Existing path-bound historical runs retain their literal directories and are classified in `runs/_bk/README.md`; their immutable identities were not renamed for cosmetic uniformity.

Two large research collections, root-level legacy report/figure exports, historical coherence source and the one-off ExampleVisual working directory also moved to clearly named DATA2 archive locations. The [storage map](../../../docs/Storage_Map.md) lists each logical/physical mapping. Existing DATA2 compute campaigns and fitted-artifact collections retain their frozen paths and are indexed as legacy material.

The old flat workplan and report collections are grouped under each library's `_bk/Legacy_Local_*` directory. Before grouping and presentation-link rebasing, exact SHA-256-inventoried original snapshots were saved on DATA2 as `_bk/research_collections/Legacy_Plans` and `Legacy_Reports`. Sixteen source-identity compatibility aliases and eight transitive support aliases remain because historical dependencies require them. The snapshots preserve original bytes; reading copies and their source maps are explicitly distinguished.

The small original Adaptive Repair contract collection remains at its old HOME location. The reviewed plan library contains ordinary-file reading copies, with Round 5 current and Rounds 1–4 under `_bk`. Round 3/4 closure, SOURCE/C8 and TEST restrictions are unchanged. Canonical report publications now live in `UpdatePlans/_Reports/Coherence_Adaptive_Repair/`, with Round 5 current and prior rounds under its `_bk`; the four old root review aliases are retained only for compatibility.

Existing dataset owners were preserved: four catalog entries resolve into `/data/wanglz/PhyCoFlow/datagen/processed/`, while two combustion files still belong to the sibling demo project. Those external HOME files are not part of the checkout's measured 53.72 GiB and require a separate owner/dependency review before migration. New project-owned datasets belong in DATA2 `datasets/<case>/<version>/`.

## Verification and deletion boundary

The eleven bulk groups passed complete source/target metadata and ownership inventories, content-checksum comparison using rsync xxh128, hardlink accounting, link checks and mutation checks before alias switching. Verified source rollback copies remained available until post-switch checks passed. Cleanup removed only manifest-listed duplicate originals: 24,507 regular-file paths representing 57,386,275,182 logical bytes. Logical path totals count hardlink entries and differ from physical allocation savings. No unique scientific file was deleted. The two externally shared hardlink boundaries were recorded; their outside members were untouched.

The old Round 2 file-event watcher, PID 2009362 with the exact expected script identity, was stopped by SIGTERM after the user explicitly confirmed it was unnecessary. The process exited. Its script, event log and stop receipt were retained; no other process was stopped.

The local-only operations root is `/data2/wanglz/PhyCoFlow-MultiFieldReconstruction/operations/migrations/20261123_Project_Organization/`. It contains the preflight allowlist, full inventories, prepared/switched/completed receipts, migration script, watcher disposition, archive receipts, run navigation inventory and final verification/Git receipts. These machine-specific payloads are intentionally excluded from GitHub.

## Markdown and publication acceptance

The new Markdown checker parses actual links and normalizes URL paths before filesystem resolution. It detects the demonstrated failure where a filesystem traversal through a report symlink resolves `../../assessment`, but a Markdown viewer normalizes the URL to a different location. Portable mode rejects absolute local URLs, external checkout paths and symlink-dependent publication links. Five focused tests cover that failure, reference links/images/fragments, local asset aliases, absolute local URLs and symlinked or out-of-root documents without links; all passed. Ruff passed for the new helper and its tests.

The five Adaptive Repair publications contain 60 files totaling 20,763,886 bytes. All 22 PNG/PDF figure pairs match their original source bytes. The 11 Markdown documents passed 121 local-link checks; all 33 main-report PDF pages were visually inspected. A relocated copy preserved all 60 hashes, all 55 rendered PDF pages including figure masters were pixel-identical, and all 88 PDF link annotations resolved within the bundle. Rounds 3 and 4 had no original standalone main PDF, so clearly identified publication PDFs were generated from accepted Markdown. Round 1's historical weighting terminology discrepancy is preserved and documented in its evidence index, without recomputing scientific results.

A separate browser preview was unavailable because the installed Firefox could not run in this environment. Markdown links were checked, images decoded and the PDF render inspected. All 210 previously resolving relative Markdown targets in the old flat collections still resolve after grouping; 245 pre-existing missing links or anchors are recorded in the archive indexes. Those uncurated archives are not claimed as portable publications. A clean Git-index export passed 133 local-link checks across 15 published report/index documents, decoded all 22 PNG previews and preserved the 60-file scientific publication bundle independently of local DATA2 aliases. No machine-specific symlink or raw model payload was staged.

## Ongoing management

The upgraded artifact policy and contributor guide define code boundaries, HOME/DATA2 ownership, `_bk` retention, source/publication identity, migration checks and GitHub inclusion. `.gitignore` now treats both directory and symlink forms correctly, excludes raw historical collections and makes reviewed plan/report libraries trackable. GitHub CI checks portable publication links in a clean checkout; scientific payloads and machine-specific aliases remain local. There is no automatic deletion by age or archive status, no silent HOME storage fallback and no automatic experiment extension.

A later [navigation follow-up](NAVIGATION_FOLLOWUP.md) corrects the initially over-retained compatibility entrances: 31 redundant aliases are now archived, and current reading links are updated. The earlier statements above about aliases describe the initial migration state; the follow-up records the current state.

The storage reorganization is adopted. The retained cleanup items remain in place pending the user's later review; they do not block the new reading/navigation structure.

# Project organization and storage migration

Outcome: provide one clear research reading entrance, archive historical plans/reports/runs under `_bk/`, move inactive generated HOME payloads to DATA2 with verified compatibility paths, and make selected reports readable from a fresh Git checkout. This is an engineering/storage task; it authorizes no training, inference, model evaluation or scientific protocol changes.

The source baseline is commit `fc77dcdbcbecb2d340767ce76df2799974b40273`. Initial physical checkout use is 53.72 GiB, including 47.12 GiB under cases and 6.43 GiB under UpdatePlans. Existing DATA2 Adaptive Repair campaigns and external dataset owners are protected.

## Scope and allocation

- Main agent owns migration, shared policy, Git rules, navigation, link checking and final verification. GPT-6.1 Terra max agents own the ordinary-file Adaptive Repair plan collection and the portable final report collection.
- HOME retains maintained source/configurations, original tracked historical contracts, compact reading copies and approved publication bundles. DATA2 retains datasets, complete run payloads, generated versions, raw evidence and immutable legacy research collections.
- The migration allowlist contains physical `cases/*/runs` directories, the two large legacy Compute Enhancement/Manuscript Foundation collections, legacy root reports/figures/code collection and the ignored ExampleVisual work snapshot. Existing DATA2 content is never overwritten. The already-DATA2-owned Adaptive Repair subtree is preserved during the case-run merge.
- A separate bounded archive pass groups legacy `_Plans` and `_Reports` material under `_bk/`; uncertain or path-bound compatibility entries remain recorded rather than deleted.
- Use existing rsync transfer checksums and complete metadata/ownership/hardlink inventories. Preserve originals until copy, link rebasing and post-alias verification succeed. Remove only manifest-listed verified HOME duplicate copies, never unique scientific content.
- No GPU allocation is needed. Do not poll experiment progress. The obsolete Round 2 file-event watcher was explicitly declared unnecessary by the user and may be stopped after verifying its exact identity; retain its script, event log and stop receipt.

## Storage and verification

Migration receipts and the one-off operation script belong under `/data2/wanglz/PhyCoFlow-MultiFieldReconstruction/operations/migrations/20261123_Project_Organization/`. Record source/target mapping, source state, transfer-checksum verification, explicit relative-link rebases, compatibility aliases and measured disk use. Preserve hardlinks within each copied tree; record unavoidable external hardlink boundary separation without touching external originals.

Canonical plans and reports are physical files under `UpdatePlans/_Plans/` and `_Reports/`. Current Round 5 reading material remains visible; Round 1–4 material belongs to the corresponding `_bk/`. Original scientific contracts and report versions remain unchanged; reading/publication copies record source identities and make presentation-only link changes.

Validate Markdown destinations after URL path normalization, not by following a symlink before resolving `..`. Portable publications contain their selected PNG previews and PDF masters, use a compact evidence index for local-only raw artifacts, and have no mandatory DATA2 link dependency. Inspect the generated publication PDFs, their embedded links and previews. Test the reusable link checker against the demonstrated symlink-parent regression and validate the portable collection from a clean Git export.

## Endpoint

Complete migration and archive navigation, publish approved readable collections, update the existing artifact policy/agent/contribution/ignore rules, and deliver an English operational report with before/after usage and a consolidated pending-deletion list. Preserve active work, frozen identities, SOURCE/C8, TEST locks and all unique scientific content. Commit durable changes after reviewing the outgoing patch; generated migration payloads remain ignored on DATA2. Do not extend this task into experiments, cleanup of other projects, deletion of uncertain material or a new storage framework.

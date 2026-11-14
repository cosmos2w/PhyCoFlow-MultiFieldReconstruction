# Experiment artifacts and storage

Keep reusable code, reviewed protocols and compact documentation in the repository. On this host, place generated campaign payloads in the existing project storage root `/data2/wanglz/PhyCoFlow-MultiFieldReconstruction/`; keep a stable case-owned logical path under `cases/<case>/runs/` as an alias to the physical campaign location. Create new campaigns directly on that storage and establish their alias before launch. Check that the intended DATA2 filesystem is mounted, writable and has capacity; if it is unavailable, stop and resolve the storage issue. Never silently fall back to home storage or change an existing alias's target.

For example, the logical alias `cases/turbulent_combustion/runs/coherence_adaptive_repair/round_2/` can point to the mirrored physical directory `/data2/wanglz/PhyCoFlow-MultiFieldReconstruction/cases/turbulent_combustion/runs/coherence_adaptive_repair/round_2/`. This is an example for a new round, not a claim that an existing round has been moved.

Each campaign has one clear `README.md` index at its logical root, linking to its plan, frozen configuration and source identities, attempts, evidence, reports and applicable restrictions. Group new campaign contents by purpose:

```text
<campaign>/
  README.md       navigation and campaign identity
  configs/        resolved configurations and manifests
  operations/     launch, completion, resource and cost receipts
  sources/        immutable SOURCE snapshots and identities
  runs/           formal/, qualification/ and pilot/ groups, each with named arms and immutable attempt IDs
  artifacts/      fitted banks and reusable generated artifacts
  assessment/     split-gated panels, metrics and evidence
  reports/        readable reports and figure masters/previews
  archive/        retained superseded or closed material
  cache/          explicitly disposable, reproducible caches
```

Keep reusable tools in tracked source; frozen execution copies and their provenance remain campaign artifacts. Use unique named attempt directories and preserve failed, paused and completed attempts, their checkpoints and logs. Record archive status in the index or metadata without renaming hash-bound attempt paths. Never overwrite a previous attempt or present a restart as the original execution. Record the physical and logical locations, source/config/code hashes, split and population, seeds/coupling, and artifact provenance. Capture complete per-job wall-time and GPU lease boundaries for cost attribution; distinguish independent CPU work from CPU time inside its own GPU lease and retain failed-attempt costs. Forecasts and incomplete timers are not measured zeros.

Relocate an existing campaign only after it is terminal and ownership, open handles, links, available capacity and applicable user authorization have been verified. Make a verified copy on the target filesystem, compare an inventory and content hashes including required metadata, then establish and verify the stable alias and all referenced paths. Preserve originals until the copy and references have been independently verified and cleanup is authorized within the existing task scope. Never move, replace, or relink an active run. A legacy layout stays as it is until an authorized migration; its navigation index must describe the actual locations rather than imply that the payloads were reorganized.

Use `tools/storage/verify_campaign_relocation.py` for read-only `inventory`, `compare` and post-alias `links` checks. Write its sealed receipts outside the inventoried trees. It compares content hashes, mode, ownership, modification time, symlink text and hardlink relationships; physical inode numbers, allocation and directory storage size may differ across filesystems. It does not establish that writers have stopped, move files, or compare ACLs and extended attributes. Preserve and verify those separately when present. Re-inventory the final physical location after renaming a staging directory so post-alias checks refer to its actual path. Check path-resolution contracts before migration: a logical symlink preserves access but may not preserve string comparisons against immutable resolved-path receipts.

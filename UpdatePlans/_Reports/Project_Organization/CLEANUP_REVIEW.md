# Retained items for cleanup review

No item in this table was deleted. The recommendation is to retain these items for now; deletion requires a specific owner/dependency decision after the organization outcome is reviewed. This is not an instruction to restart an old experiment.

| Item | Why retained | Suggested disposition |
| --- | --- | --- |
| Four root `Coherence_Adaptive_Repair_*_Review` symlinks | Frozen original plans and decision history still reference these paths | Retain as ignored compatibility aliases; remove only after those incoming dependencies are retired. Use the ordinary-file report library for reading. |
| Sixteen old flat plan/report aliases and eight support aliases | Historical incoming links and a saved SHA-256 source manifest require the exact original bytes | Retain ignored aliases to the DATA2 original snapshots; they are provenance entries, not current Markdown reading locations. |
| Two turbulent-combustion HDF5 files under sibling `0_demo_TurbulentCombustion/Dataset/` | Each is 8,060,571,792 logical bytes (about 7.51 GiB); both are outside this checkout and may serve sibling workflows | Review that owner's dependencies before a separate DATA2 migration. Existing dataset catalog links remain valid. |
| Old flat plan/report originals and reading archives | Exact DATA2 snapshots preserve source identity; grouped reading copies preserve local historical access | Keep both roles until a reviewed publication or explicit cleanup supersedes them. Old broken links are inventoried rather than rewritten into unsupported claims. |
| Legacy source collections, diagnostics and raw reports on DATA2 | These can contain unique scientific evidence, code provenance or failed attempts | Keep; being ignored or under `_bk` does not establish uselessness. |
| Build/test caches and `dist/` | Reproducible local development output, small compared with the migrated payloads | Optional later cleanup; no storage benefit justifies expanding this task. |
| Retired Round 2 watcher script and event log | Process was stopped on the user's explicit instruction; files document its historical behavior | Keep the durable operations snapshot; do not restart. |

Two hardlinked source entries crossed this repository's boundary before migration: turbulent-combustion `normalization.pt` and `epoch_2000.pt` inside `tc_gl_rbf_cq_cached_kv_5000ep/20260826T212231Z_7b68c461/`. The external hardlink members were not removed or modified. DATA2 copies preserve bytes but cannot retain a physical hardlink across filesystems; inventories record that boundary split.

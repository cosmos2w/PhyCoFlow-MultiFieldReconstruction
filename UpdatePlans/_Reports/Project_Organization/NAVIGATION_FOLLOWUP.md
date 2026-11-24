# Navigation follow-up and file lifecycle rules

The first organization pass preserved historical path mentions too broadly as current compatibility entrances. This left duplicate review aliases at the repository root and 24 old flat aliases alongside the new plan/report libraries. They are now archived: four review aliases under `UpdatePlans/_Reports/_bk/Retired_Local_Aliases/`, 24 flat aliases under each library's `_bk/Legacy_Aliases/`, and three legacy source/export aliases under root `_bk/legacy_sources/` or `_bk/legacy_exports/`. All 31 DATA2 target device/inode identities are unchanged. No original regular file, scientific payload or frozen contract was moved or deleted.

Use the [report library](../README.md) for reading and [the artifact policy](../../../docs/Experiment_Artifacts.md) for all new work. Maintained plan reading links now use ordinary historical reading copies. Frozen source documents and hash manifests preserve literal historical paths; the local operations receipt records how those paths resolve to retained originals. A historical path mention alone does not require a duplicate current entrance.

The local-only audit root is `/data2/wanglz/PhyCoFlow-MultiFieldReconstruction/operations/migrations/20261123_Navigation_Retirement/`. It preserves the pre-existing Round 6 edits, the exact alias plan, old/new symlink text, unchanged target identities, presentation-only link changes and final checks. Round 6 planning content and scientific definitions are preserved; this follow-up does not authorize an experiment.

New reports must be ordinary-file publication bundles under `UpdatePlans/_Reports/<campaign>/`, with selected Markdown/PDF, figure PNG/PDF pairs and compact evidence documentation. Raw report versions, render HTML, page previews and receipts remain with DATA2 campaign storage. Current library levels contain only README, ordinary campaign directories and `_bk`; a read-only layout check and clean-checkout CI enforce that boundary.

Small disposable scratch belongs in a unique task-owned `/tmp/phycoflow-<task>-<unique>/` directory. Large/resumable scratch belongs in DATA2 cache storage. Promote unique results and required job/report verification evidence before cleaning confirmed task-owned reproducible files. Preserve uncertain or pre-existing material for one review list. Never follow a symlink when removing an alias; `_bk` has no automatic deletion deadline. The authoritative [temporary-file lifecycle](../../../docs/Experiment_Artifacts.md#temporary-files-and-cleanup) covers these cases.

## Remaining source and build moves awaiting authorization

The broader source/build migration was rejected by automatic approval review because moving the original source path and broad legacy entries, together with deleting verified local build copies, could break frozen/path-bound contracts and exceeded the confirmed-duplicate boundary. The accepted narrower operation changes symlink navigation only. The following concrete optional next step would archive by rename, preserve every file, update current reading references, preserve frozen bytes and leave no compatibility aliases at the old visible locations:

| Existing path | Proposed retained archive/source location | Required handling |
| --- | --- | --- |
| `UpdatePlans/Coherence_Adaptive_Repair/` | `UpdatePlans/_Plans/_bk/Original_Sources/Coherence_Adaptive_Repair/` | Preserve all original bytes and tracked source identities; rebase the source index and current readers only |
| `UpdatePlans/Coherence_Manuscript_Foundation` | `UpdatePlans/_Sources/Coherence_Manuscript_Foundation` | Move only its alias; update new Round 6 source citations without changing source bytes |
| `UpdatePlans/CoherenceComputeEnhancement_20261106` | `UpdatePlans/_Plans/_bk/CoherenceComputeEnhancement_20261106` | Move only its alias; preserve the physical DATA2 collection |
| `UpdatePlans/Project_Retrospective_20261120/` | `UpdatePlans/_Reports/_bk/Legacy_Local_Reports/Project_Retrospective_20261120/` | Preserve the original report, figures and helper; archive presentation dependencies together |
| `build/` | `_bk/build_outputs/build/` | Rename only after a fresh no-writer check; preserve every file |
| `dist/` | `_bk/build_outputs/dist/` | Rename only; preserve the wheel |

These six paths remain in place. Existing checks found no open handles under the original source/build directories, and `phycoflow_env`'s editable installation points directly to `src/`; those facts reduce but do not override the approval-review block. No broad source or build migration was performed.

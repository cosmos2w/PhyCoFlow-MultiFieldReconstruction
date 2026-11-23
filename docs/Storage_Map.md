# Project storage map

This map describes the organization established on 23 November 2026. [Experiment artifacts](Experiment_Artifacts.md) defines the ongoing rules; the [organization report](../UpdatePlans/_Reports/Project_Organization/report.md) records measurements, verification and retained cleanup candidates.

The HOME checkout is `/home/wanglz/Desktop/src/PhyCoFlow/Proj_MultiFieldReconstruction`. The physical bulk-storage owner is `/data2/wanglz/PhyCoFlow-MultiFieldReconstruction`. Git contains reusable source, configurations, contracts, catalog documentation, reviewed plans and compact publication bundles; it does not distribute the local storage aliases.

| HOME path | Ownership and actual location |
| --- | --- |
| `src/`, `configs/`, `scripts/`, `tests/`, `tools/`, `.github/` | Ordinary tracked code and development tools; runtime outputs belong in DATA2 |
| `cases/<case>/` | Thin case code, configuration and diagnostics source |
| `cases/<case>/runs` | Ignored symlink to DATA2 `cases/<case>/runs`; includes checkpoints, complete histories, ten-epoch curves, raw assessment, operations and failed attempts |
| `datasets/` | Tracked schemas/catalogs and ignored local payload links; datasets retain their existing physical owners below |
| `UpdatePlans/_Plans/` | Ordinary reviewed plan reading copies; closed plans under collection-owned `_bk/` |
| `UpdatePlans/_Reports/` | Ordinary reviewed publication bundles with colocated figures/evidence summaries; closed reports under collection-owned `_bk/` |
| `docs/` and `ModelExplain.md` | Maintained engineering and method reference documentation |
| `UpdatePlans/Coherence_Adaptive_Repair/` | Small retained original scientific contracts and local campaign aliases; reading copies live in the plan library |
| `UpdatePlans/CoherenceComputeEnhancement_20261106` | Ignored alias to DATA2 `_bk/research_collections/CoherenceComputeEnhancement_20261106` |
| `UpdatePlans/Coherence_Manuscript_Foundation` | Ignored alias to DATA2 `_bk/research_collections/Coherence_Manuscript_Foundation` |
| `LegacyCoherence` | Ignored historical-source alias to DATA2 `_bk/research_collections/Legacy_LegacyCoherence` |
| `reports`, `figures` | Ignored compatibility aliases to DATA2 `reports/_bk/Legacy_reports` and `reports/_bk/Legacy_figures` |
| `cases/turbulent_combustion/diagnostics/ExampleVisual/.work` | Ignored alias to DATA2 `_bk/diagnostics/ExampleVisual_Work` |
| Four root `Coherence_Adaptive_Repair_*_Review` aliases | Retained local compatibility for frozen references; the report library is the reading entry |

The intended DATA2 structure is:

```text
PhyCoFlow-MultiFieldReconstruction/
  README.md                         physical storage navigation
  cases/<case>/runs/
    README.md                       current and historical run index
    <campaign>/                     immutable attempts and case-owned evidence
      configs/ operations/ sources/ runs/ artifacts/ assessment/ reports/
      _bk/ cache/                   closed material and explicitly disposable caches
    _bk/README.md                   historical inventory; path-bound originals retain their locations
  datasets/<case>/<version>/        new project-owned dataset versions
  operations/migrations/<task>/     inventories, checksums and migration receipts
  reports/
    <standalone-collection>/         non-campaign generated reports
    _bk/                            retained historical exports
  _bk/
    research_collections/            exact historical plans, reports and source snapshots
    diagnostics/                     retained one-off diagnostic work
```

Existing DATA2 `artifacts/coherence_adaptive_repair/`, `experiments/LPCAccel_20261022_01a16674/`, and the dated `coherence-compute-enhancement-round*/` collections remain at their recorded locations. Their path-bound originals are catalogued as legacy material; moving them again would break provenance for no additional HOME-space saving. Existing `archive/` folders likewise retain their literal names. `_bk` is the name for new archive organization, not an instruction to rename frozen receipts.

All five case `runs` roots now reside on DATA2. The turbulent-combustion root retains the existing Adaptive Repair campaign; its Round 1–5 source, assessment, complete history and checkpoint locations are unchanged. Its current index points to Round 5 and to the historical `_bk` inventory. Older model-source and historical run directories retain their path identities even when classified as archived. This is an explicit compatibility exception, not an active-run list.

Dataset ownership is separate from the run migration:

| Catalog entries | Existing physical owner | Disposition |
| --- | --- | --- |
| Brusselator, Kolmogorov, KS, mass-transport demo | `/data/wanglz/PhyCoFlow/datagen/processed/` | Existing external data owner retained; catalog links verified |
| Two turbulent-combustion HDF5 files | Sibling project `../0_demo_TurbulentCombustion/Dataset/` on HOME | Outside this checkout and potentially shared; retained for owner/dependency review rather than silently moved |
| New project-owned datasets | DATA2 `datasets/<case>/<version>/` | Create directly on DATA2 and expose ignored catalog links; never copy payloads into Git |

The old flat plan and report collections are grouped under their library `_bk/Legacy_Local_*` directories. Exact original snapshots on DATA2 preserve original source bytes before presentation-link rebasing. They are local historical context, not new public or scientific authority.

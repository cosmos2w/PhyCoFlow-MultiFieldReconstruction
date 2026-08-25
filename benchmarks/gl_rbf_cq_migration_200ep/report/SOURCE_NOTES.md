# Source and packaging notes

## Report contract

- Surface: portable technical HTML report.
- Primary quality selector: configured evaluation weights. Arms B and C therefore use their configured EMA lifecycle; Arm A has no EMA lifecycle.
- Snapshot grain: arm × fixed milestone checkpoint.
- Fixed comparison set: epochs 1, 20, 40, 60, 100, 150, and 200 for A, B, and C (21 rows).
- Lower-is-better metrics: normalized MSE, mean relative L2, and worst-field relative L2.
- Generator: Python standard library only. `generate_report.py` validates protocol/hash invariants before it writes `artifact.json`.
- Determinism: two independent generator invocations produced byte-identical `artifact.json` files (`cmp` passed).

## Chart map

| Contract field | Implementation |
| --- | --- |
| Question | How does reconstruction quality compare across matched milestone checkpoints? |
| Takeaway | B/C lead from epochs 20 through 150, while A is best at epoch 200; A also leads at epoch 1. |
| Analytical family / intent | Comparison |
| Native chart type | Grouped vertical bar |
| Dataset | `milestone_quality`, exactly 21 rows |
| X encoding | `epoch_label`, ordinal; the seven checkpoints are discrete |
| Y encoding | `mean_relative_l2`, quantitative; axis starts from the native bar baseline and lower is better |
| Color encoding | `arm_label`, nominal, categorical restrained palette named `restrained-neutral-blue-orange` |
| Non-color distinction | Every series is named in the legend and repeated in a stable A/B/C grouped position at each labeled epoch |
| Final surface | Self-contained portable HTML |

The explanatory markdown immediately before and after the chart states the crossover conclusion and warns against interpreting the bars as a continuous learning curve.

## Omission rationale

- No line chart: only seven discrete checkpoint epochs were evaluated, so connecting them would imply unsupported interpolation.
- No additional metric charts: normalized MSE, worst-field relative L2, and exact source precision are retained in the milestone table; plotting all three would add scale and metric-family clutter without changing the decision.
- No per-field chart: all five field values remain in the frozen source evidence; the report emphasizes the prespecified aggregate and worst-field endpoints.
- No live-weight chart: configured EMA is the prespecified primary B/C evaluation selector. Mixing live and configured weights in the main comparison would blur the decision contract.
- No A-versus-B/C formal-runtime chart: formal A and CQ timing also reflect architecture and lifecycle differences. Those exact values are reported with an explicit confounding caveat, while the controlled B-versus-C probe supplies the causal execution evidence.
- No execution chart: one paired B/C execution comparison is more legible as exact prose alongside the quality chart, and avoids promoting a secondary engineering measurement over the primary reconstruction-quality result.
- No B128 chart: B128 failed Arm-A backward allocation and is protocol-selection evidence, not a completed comparable arm. The authorized common B40 replacement is documented under limitations.

## Report structure mapping

| Required role | Visible block(s) |
| --- | --- |
| Title | `title` |
| Technical summary | `technical_summary` |
| Key findings | `key_findings_heading`, `chart_takeaway`, `milestone_chart`, `chart_interpretation`, `final_deltas`, `exact_milestones` |
| Scope / metric definitions | `scope_metrics` |
| Experimental design / migration validation | `experimental_design`, `migration_effect`, `execution_effect`, `total_effect` |
| Limitations / robustness | `limitations` |
| Recommended next steps | `recommendations` |
| Further questions | `further_questions` |

## Source inventory

All packaged source paths are repository-relative and contain no parent traversal or machine-local absolute path.

| Source ID | Repository-relative source | Use |
| --- | --- | --- |
| `arm_a` | `Proj_MultiFieldReconstruction/benchmarks/gl_rbf_cq_migration_200ep/baseline/A_performance.json` | Frozen corrected legacy baseline, protocol, telemetry, and fixed milestones |
| `arm_b` | `Proj_MultiFieldReconstruction/benchmarks/gl_rbf_cq_migration_200ep/runs_summary/B_summary.json` | Portable CQ + legacy MHA formal result and configured-EMA milestones |
| `arm_c` | `Proj_MultiFieldReconstruction/benchmarks/gl_rbf_cq_migration_200ep/runs_summary/C_summary.json` | Portable CQ + cached K/V formal result and configured-EMA milestones |
| `execution` | `Proj_MultiFieldReconstruction/benchmarks/gl_rbf_cq_migration_200ep/execution/B_vs_C_execution.json` | Same-state, same-batch, same-RNG B-versus-C controlled execution probe |
| `derived_report` | `Proj_MultiFieldReconstruction/benchmarks/gl_rbf_cq_migration_200ep/report/generate_report.py` | Deterministic validation/join/delta transform for quantitative narrative |
| `milestone_query` | same generator | Exact SQL surface query over the generator-materialized 21-row milestone snapshot |
| `delta_query` | same generator | Exact SQL surface query over the generator-materialized three-row delta snapshot |

The generator reads the four frozen evidence JSON files directly; it does not read or depend on any `phase1_audit` product.

## Package receipt and QA

Successful delivery used:

- Runtime: `/home/wanglz/miniconda3/envs/ModularDT/bin/node`, Node `v20.20.1`.
- Builder: `/home/wanglz/.codex/plugins/cache/openai-curated-remote/data-analytics/0.2.8-13ceeea1f599/skills/build-report/scripts/deliver_portable_artifact.mjs`.
- Input/output: `artifact.json` → `report.html`.
- Receipt: `ok=true`; validation `passed`; package `passed`; verification `structural_only`.
- Structural counts: 16 blocks, 1 chart, 2 tables, 0 metric cards, 0 custom-HTML blocks.
- Source dialog / interaction: `not_verified` because no installed Chromium headless-shell executable was available. The packager did not download a browser.
- Browser warning code: `browser_unavailable`.
- Structural verification time: 8.2 ms; total successful delivery time: 101.4 ms.

Pre-package and post-package QA:

- All A/B/C formal statuses are completed.
- All three arms match B40/Q4096/seed42 and the same normalizer, sensor-manifest, and query-index digests.
- B/C primary milestone evaluations are configured EMA.
- Execution probe asserts the same batch, initial state, and rectified-flow seed schedule; observed K/V projections are 4 for B and 1 for C on every measured step.
- Snapshot row bounds are 21 milestone rows and 3 final-delta rows.
- Chart type is `bar`, X type is `ordinal`, and the dataset contains exactly the 21 required arm/milestone rows.
- Quantitative chart/table blocks resolve through `sourceId` to explicit SQL surface queries; quantitative markdown resolves to its direct evidence or deterministic transform source.
- Exact table values are stored as text to prevent the portable renderer's presentation formatter from rounding source precision.

Packaged file checksums:

| File | SHA-256 | Bytes |
| --- | --- | ---: |
| `generate_report.py` | `afeb018cf11bc4785263a7eb121d930534ed220920e01c4aad542b1700b2207a` | 28,140 |
| `artifact.json` | `37b175de0f80c35310cfc446fb79378f4b57940934be22778c36cd635d34fe1a` | 36,639 |
| `report.html` | `9b7404492f5f13ac299e87f34b484b603fb9bb94e280f5327ebafcfdabeb99eb` | 495,665 |

The first delivery attempt used the system Node `v12.22.9` and stopped at JavaScript parse time before reading the artifact. A second attempt with the package-local Node binary stopped at dynamic-library loading before reading the artifact. Two subsequent validation attempts identified and resolved the builder's SQL-source provenance requirement. The receipt above is the final successful package and structural verification result.

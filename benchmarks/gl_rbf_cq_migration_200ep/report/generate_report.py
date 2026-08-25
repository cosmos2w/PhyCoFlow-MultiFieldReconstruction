#!/usr/bin/env python3
"""Build the deterministic portable-report artifact from frozen benchmark evidence."""

from __future__ import annotations

import argparse
import json
from decimal import Decimal
from pathlib import Path
from typing import Any


REPORT_DIR = Path(__file__).resolve().parent
BENCHMARK_DIR = REPORT_DIR.parent
REPO_ROOT = REPORT_DIR.parents[3]

SOURCE_PATHS = {
    "arm_a": BENCHMARK_DIR / "baseline" / "A_performance.json",
    "arm_b": BENCHMARK_DIR / "runs_summary" / "B_summary.json",
    "arm_c": BENCHMARK_DIR / "runs_summary" / "C_summary.json",
    "execution": BENCHMARK_DIR / "execution" / "B_vs_C_execution.json",
}
EXPECTED_EPOCHS = [1, 20, 40, 60, 100, 150, 200]
ARM_LABELS = {
    "A": "A — legacy GL_rbf_ENH",
    "B": "B — CQ legacy_mha",
    "C": "C — CQ cached_kv",
}


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def repo_relative(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT).as_posix()


def exact(value: Any) -> str:
    """Render a JSON number without report-layer rounding."""
    return str(value)


def signed_decimal_delta(left: Any, right: Any) -> str:
    delta = Decimal(str(left)) - Decimal(str(right))
    return f"{delta:+f}"


def percent_change(left: float, right: float) -> float:
    return (left / right - 1.0) * 100.0


def convergence(source: dict[str, Any], arm: str) -> list[dict[str, Any]]:
    if arm == "A":
        rows = source["formal_run"]["fixed_manifest_convergence"]
    else:
        rows = source["fixed_manifest_convergence"]
    require(
        [row["epoch"] for row in rows] == EXPECTED_EPOCHS,
        f"Arm {arm} does not contain the required milestone epochs",
    )
    return rows


def validate_sources(
    arm_a: dict[str, Any],
    arm_b: dict[str, Any],
    arm_c: dict[str, Any],
    execution: dict[str, Any],
) -> None:
    require(arm_a["formal_run"]["status"] == "completed", "Arm A is not complete")
    require(arm_b["completion"]["status"] == "completed", "Arm B is not complete")
    require(arm_c["completion"]["status"] == "completed", "Arm C is not complete")
    require(arm_b["ema"]["evaluated_with_ema"] is True, "Arm B primary evaluation is not EMA")
    require(arm_c["ema"]["evaluated_with_ema"] is True, "Arm C primary evaluation is not EMA")

    for arm, source in (("A", arm_a), ("B", arm_b), ("C", arm_c)):
        convergence(source, arm)

    manifest_hashes = {
        arm_a["formal_run"]["fixed_validation_manifest_sha256"],
        arm_b["fixed_evaluation"]["sensor_manifest_sha256"],
        arm_c["fixed_evaluation"]["sensor_manifest_sha256"],
    }
    query_hashes = {
        arm_a["formal_run"]["fixed_query_indices_sha256"],
        arm_b["fixed_evaluation"]["query_indices_sha256"],
        arm_c["fixed_evaluation"]["query_indices_sha256"],
    }
    normalizer_hashes = {
        arm_a["formal_run"]["normalizer_digest"],
        arm_b["data_and_normalization"]["normalizer_digest"],
        arm_c["data_and_normalization"]["normalizer_digest"],
    }
    require(len(manifest_hashes) == 1, "Fixed sensor-manifest hashes differ across arms")
    require(len(query_hashes) == 1, "Fixed query-index hashes differ across arms")
    require(len(normalizer_hashes) == 1, "Normalizer digests differ across arms")

    protocol_triplets = [
        (
            arm_a["protocol"]["batch_size"],
            arm_a["protocol"]["query_points"],
            arm_a["protocol"]["seed"],
        ),
        (
            arm_b["configuration"]["batch_size"],
            arm_b["configuration"]["query_points"],
            arm_b["configuration"]["seed"],
        ),
        (
            arm_c["configuration"]["batch_size"],
            arm_c["configuration"]["query_points"],
            arm_c["configuration"]["seed"],
        ),
    ]
    require(len(set(protocol_triplets)) == 1, "B/Q/seed protocol differs across arms")
    require(protocol_triplets[0] == (40, 4096, 42), "Unexpected formal B/Q/seed protocol")
    require(execution["protocol"]["batch_size"] == 40, "Execution probe batch differs")
    require(execution["protocol"]["query_points"] == 4096, "Execution probe Q differs")
    require(execution["protocol"]["same_batch"] is True, "Execution probe batch is not matched")
    require(
        execution["protocol"]["same_initial_state"] is True,
        "Execution probe initialization is not matched",
    )
    require(
        execution["protocol"]["same_rf_seed_schedule"] is True,
        "Execution probe RF schedule is not matched",
    )
    require(
        set(execution["arms"]["B_legacy_mha_full"]["observed_kv_projection_calls_per_step"]) == {4},
        "Arm B did not sustain four K/V projections per measured step",
    )
    require(
        set(execution["arms"]["C_cached_kv_full"]["observed_kv_projection_calls_per_step"]) == {1},
        "Arm C did not sustain one K/V projection per measured step",
    )


def build_artifact() -> dict[str, Any]:
    arm_a = load_json(SOURCE_PATHS["arm_a"])
    arm_b = load_json(SOURCE_PATHS["arm_b"])
    arm_c = load_json(SOURCE_PATHS["arm_c"])
    execution = load_json(SOURCE_PATHS["execution"])
    validate_sources(arm_a, arm_b, arm_c, execution)

    source_rows = {
        "A": convergence(arm_a, "A"),
        "B": convergence(arm_b, "B"),
        "C": convergence(arm_c, "C"),
    }
    milestone_rows: list[dict[str, Any]] = []
    for epoch in EXPECTED_EPOCHS:
        for arm in ("A", "B", "C"):
            row = next(item for item in source_rows[arm] if item["epoch"] == epoch)
            milestone_rows.append(
                {
                    "epoch": epoch,
                    "epoch_label": str(epoch),
                    "arm": arm,
                    "arm_label": ARM_LABELS[arm],
                    "mse_normalized": row["mse_normalized"],
                    "mse_normalized_exact": exact(row["mse_normalized"]),
                    "mean_relative_l2": row["mean_relative_l2"],
                    "mean_relative_l2_exact": exact(row["mean_relative_l2"]),
                    "worst_field_relative_l2": row["worst_field_relative_l2"],
                    "worst_field_relative_l2_exact": exact(row["worst_field_relative_l2"]),
                    "checkpoint_report_sha256": row["report_sha256"],
                }
            )
    require(len(milestone_rows) == 21, "Grouped-bar dataset must contain exactly 21 rows")

    final = {
        arm: next(row for row in source_rows[arm] if row["epoch"] == 200)
        for arm in ("A", "B", "C")
    }
    delta_rows = []
    for comparison, left, right, interpretation in (
        ("B − A", "B", "A", "Migration effect"),
        ("C − B", "C", "B", "Execution-mode effect"),
        ("C − A", "C", "A", "Total latest-model effect"),
    ):
        delta_rows.append(
            {
                "comparison": comparison,
                "interpretation": interpretation,
                "mse_normalized_delta_exact": signed_decimal_delta(
                    final[left]["mse_normalized"], final[right]["mse_normalized"]
                ),
                "mean_relative_l2_delta_exact": signed_decimal_delta(
                    final[left]["mean_relative_l2"], final[right]["mean_relative_l2"]
                ),
                "worst_field_relative_l2_delta_exact": signed_decimal_delta(
                    final[left]["worst_field_relative_l2"],
                    final[right]["worst_field_relative_l2"],
                ),
            }
        )

    a_epoch_median = arm_a["formal_run"]["steady_state_epoch_wall_seconds_median"]
    b_epoch_median = arm_b["timing_and_memory"]["steady_epoch_wall_time_s"]["median"]
    c_epoch_median = arm_c["timing_and_memory"]["steady_epoch_wall_time_s"]["median"]
    a_peak_alloc = arm_a["formal_run"]["peak_cuda_allocated_bytes"]
    b_peak_alloc = arm_b["timing_and_memory"]["formal_peak_cuda_allocated_bytes"]
    c_peak_alloc = arm_c["timing_and_memory"]["formal_peak_cuda_allocated_bytes"]
    b_minus_a_quality = signed_decimal_delta(final["B"]["mean_relative_l2"], final["A"]["mean_relative_l2"])
    c_minus_b_quality = signed_decimal_delta(final["C"]["mean_relative_l2"], final["B"]["mean_relative_l2"])
    c_minus_a_quality = signed_decimal_delta(final["C"]["mean_relative_l2"], final["A"]["mean_relative_l2"])
    step_effect = execution["execution_effect_C_vs_B"]["whole_step_ms"]["median_percent_change"]
    alloc_effect = execution["memory_effect_C_vs_B"]["peak_cuda_allocated_bytes"]["percent_change"]

    completed = max(
        arm_b["completion"]["completed_utc"], arm_c["completion"]["completed_utc"]
    )
    generated_at = completed.replace("+00:00", "Z")
    source_paths = {key: repo_relative(path) for key, path in SOURCE_PATHS.items()}
    generator_path = repo_relative(Path(__file__))
    metric_definitions = [
        "mean_relative_l2: arithmetic mean of per-field relative L2 over CH4, CO, T, U_1, and p; lower is better",
        "mse_normalized: mean squared error in training-normalized field space; lower is better",
        "worst_field_relative_l2: maximum per-field relative L2; lower is better",
    ]
    sources = [
        {"id": "arm_a", "label": "Frozen Arm A performance", "path": source_paths["arm_a"]},
        {"id": "arm_b", "label": "Frozen Arm B summary", "path": source_paths["arm_b"]},
        {"id": "arm_c", "label": "Frozen Arm C summary", "path": source_paths["arm_c"]},
        {
            "id": "execution",
            "label": "Controlled B-versus-C execution benchmark",
            "path": source_paths["execution"],
        },
        {
            "id": "derived_report",
            "label": "Deterministic benchmark report transform",
            "path": generator_path,
            "query": {
                "engine": "stdlib-python",
                "language": "python",
                "query": f"python3 {generator_path} --output {repo_relative(REPORT_DIR / 'artifact.json')}",
                "description": "Validate frozen protocols and join the three configured-weight convergence series by milestone epoch; compute epoch-200 signed deltas directly from source decimals.",
                "executed_at": generated_at,
                "tables_used": list(source_paths.values()),
                "metric_definitions": metric_definitions,
            },
        },
        {
            "id": "milestone_query",
            "label": "Milestone-quality surface query",
            "path": generator_path,
            "query": {
                "engine": "snapshot-sql",
                "language": "sql",
                "sql": "SELECT epoch, epoch_label, arm, arm_label, mse_normalized, mse_normalized_exact, mean_relative_l2, mean_relative_l2_exact, worst_field_relative_l2, worst_field_relative_l2_exact, checkpoint_report_sha256 FROM milestone_quality ORDER BY epoch, CASE arm WHEN 'A' THEN 1 WHEN 'B' THEN 2 WHEN 'C' THEN 3 END",
                "description": "Select the complete 21-row milestone snapshot materialized and validated by the deterministic Python generator.",
                "executed_at": generated_at,
                "tables_used": ["milestone_quality"],
                "metric_definitions": metric_definitions,
            },
        },
        {
            "id": "delta_query",
            "label": "Epoch-200 delta surface query",
            "path": generator_path,
            "query": {
                "engine": "snapshot-sql",
                "language": "sql",
                "sql": "SELECT comparison, interpretation, mse_normalized_delta_exact, mean_relative_l2_delta_exact, worst_field_relative_l2_delta_exact FROM final_deltas ORDER BY CASE comparison WHEN 'B − A' THEN 1 WHEN 'C − B' THEN 2 WHEN 'C − A' THEN 3 END",
                "description": "Select the three signed epoch-200 effect rows computed from the unrounded source decimals.",
                "executed_at": generated_at,
                "tables_used": ["final_deltas"],
                "metric_definitions": metric_definitions,
            },
        },
    ]

    chart = {
        "id": "milestone_quality_chart",
        "title": "Fixed-manifest reconstruction quality at matched milestones",
        "subtitle": "Configured evaluation weights; B/C use EMA. Lower mean relative L2 is better.",
        "showDescription": True,
        "intent": "comparison",
        "question": "How does reconstruction quality compare across matched milestone checkpoints?",
        "rationale": "Seven epochs are discrete checkpoint milestones, so grouped bars expose arm-by-arm comparisons without implying interpolation between checkpoints.",
        "comparisonContext": {
            "baseline": "Arm A corrected legacy GL_rbf_ENH",
            "grain": "arm × milestone checkpoint",
            "normalization": "fixed 20-sample validation manifest; configured weights",
            "semanticFamily": "comparison",
            "unit": "mean relative L2",
        },
        "type": "bar",
        "dataset": "milestone_quality",
        "sourceId": "milestone_query",
        "encodings": {
            "x": {"field": "epoch_label", "type": "ordinal", "label": "Epoch milestone"},
            "y": {
                "field": "mean_relative_l2",
                "type": "quantitative",
                "aggregate": "none",
                "label": "Mean relative L2",
            },
            "color": {"field": "arm_label", "type": "nominal", "label": "Arm"},
            "tooltip": [
                {"field": "epoch", "type": "ordinal", "label": "Epoch"},
                {"field": "arm_label", "type": "nominal", "label": "Arm"},
                {
                    "field": "mean_relative_l2",
                    "type": "quantitative",
                    "label": "Mean relative L2",
                },
            ],
        },
        "xAxisTitle": "Milestone epoch",
        "yAxisTitle": "Mean relative L2 (lower is better)",
        "layout": "full",
        "maxRows": 21,
        "palette": {"kind": "categorical", "name": "restrained-neutral-blue-orange"},
        "legend": {"position": "bottom", "sort": "spec", "title": "Model arm"},
        "labels": {"values": "none"},
        "settings": {"groupMode": "grouped", "orientation": "vertical", "sort": "custom"},
        "surface": {"surface": "card", "interactiveLegend": True, "viewMode": "visualization"},
    }

    tables = [
        {
            "id": "final_delta_table",
            "title": "Epoch-200 effects at configured evaluation weights",
            "subtitle": "Signed row-arm minus comparator deltas; positive values are worse because all three metrics are lower-is-better.",
            "showDescription": True,
            "dataset": "final_deltas",
            "sourceId": "delta_query",
            "layout": "full",
            "density": "spacious",
            "columns": [
                {"field": "comparison", "label": "Comparison", "type": "text"},
                {"field": "interpretation", "label": "Effect", "type": "text"},
                {"field": "mse_normalized_delta_exact", "label": "Δ normalized MSE (exact)", "type": "text"},
                {"field": "mean_relative_l2_delta_exact", "label": "Δ mean rel L2 (exact)", "type": "text"},
                {"field": "worst_field_relative_l2_delta_exact", "label": "Δ worst rel L2 (exact)", "type": "text"},
            ],
        },
        {
            "id": "milestone_exact_table",
            "title": "Exact fixed-manifest milestone results",
            "subtitle": "All seven matched checkpoints for each arm; source precision is preserved as text.",
            "showDescription": True,
            "dataset": "milestone_quality",
            "sourceId": "milestone_query",
            "layout": "full",
            "density": "dense",
            "defaultSort": {"field": "epoch", "direction": "asc"},
            "columns": [
                {"field": "epoch", "label": "Epoch", "type": "number"},
                {"field": "arm", "label": "Arm", "type": "text"},
                {"field": "mse_normalized_exact", "label": "Normalized MSE (exact)", "type": "text"},
                {"field": "mean_relative_l2_exact", "label": "Mean rel L2 (exact)", "type": "text"},
                {"field": "worst_field_relative_l2_exact", "label": "Worst rel L2 (exact)", "type": "text"},
            ],
        },
    ]

    blocks = [
        {
            "id": "title",
            "type": "markdown",
            "layout": "full",
            "body": "# GL-RBF/CQ 200-epoch migration benchmark\n\nA fixed-manifest technical validation of the corrected downstream legacy baseline, portable CQ migration, and cached-K/V execution mode.",
        },
        {
            "id": "technical_summary",
            "type": "markdown",
            "layout": "full",
            "sourceId": "derived_report",
            "body": (
                "## Technical summary\n\n"
                f"At epoch 200, configured-weight mean relative L2 was **{exact(final['A']['mean_relative_l2'])}** for A, "
                f"**{exact(final['B']['mean_relative_l2'])}** for B, and **{exact(final['C']['mean_relative_l2'])}** for C. "
                f"The migration effect B − A was **{b_minus_a_quality}**, the execution-mode effect C − B was **{c_minus_b_quality}**, "
                f"and the total latest-model effect C − A was **{c_minus_a_quality}**. All are positive—and therefore worse—on this lower-is-better metric. "
                "The portable CQ arms led at the intermediate checkpoints from epochs 20 through 150, but the corrected legacy arm finished best at epoch 200."
            ),
        },
        {
            "id": "key_findings_heading",
            "type": "markdown",
            "layout": "full",
            "body": "## Key findings",
        },
        {
            "id": "chart_takeaway",
            "type": "markdown",
            "layout": "full",
            "sourceId": "derived_report",
            "body": (
                "**Takeaway:** B and C lead A at every matched milestone from epoch 20 through epoch 150; A reverses that ordering at epoch 200. "
                "Epoch 1 is also an A lead. The grouped design treats the seven checkpoints as discrete comparisons, not a continuous time series."
            ),
        },
        {"id": "milestone_chart", "type": "chart", "layout": "full", "chartId": "milestone_quality_chart"},
        {
            "id": "chart_interpretation",
            "type": "markdown",
            "layout": "full",
            "sourceId": "derived_report",
            "body": (
                f"The final quality gaps are small in absolute mean-relative-L2 terms: B − A = **{b_minus_a_quality}**, "
                f"C − B = **{c_minus_b_quality}**, and C − A = **{c_minus_a_quality}**. "
                "The milestone ordering therefore supports a crossover conclusion, not a claim that one arm dominates throughout training."
            ),
        },
        {"id": "final_deltas", "type": "table", "layout": "full", "tableId": "final_delta_table"},
        {"id": "exact_milestones", "type": "table", "layout": "full", "tableId": "milestone_exact_table"},
        {
            "id": "scope_metrics",
            "type": "markdown",
            "layout": "full",
            "sourceId": "derived_report",
            "body": (
                "## Scope and metric definitions\n\n"
                "All arms use the authorized common **B40/Q4096**, T-only **192–384 sensor**, seed-42, 200-epoch/40,000-step protocol. "
                "Fixed validation uses 20 identical samples, 4,096 identical query indices, and 32 generation steps. "
                "**Mean relative L2** is the arithmetic mean of per-field relative L2 over CH4, CO, T, U_1, and p. "
                "**Normalized MSE** is MSE in the shared training-normalized field space. **Worst-field relative L2** is the maximum field result. "
                "All three are lower-is-better. Configured EMA weights are primary for B and C; A has no EMA lifecycle."
            ),
        },
        {
            "id": "experimental_design",
            "type": "markdown",
            "layout": "full",
            "sourceId": "derived_report",
            "body": (
                "## Experimental design and migration validation\n\n"
                "**A** is the corrected large-scale downstream `GL_rbf_ENH` baseline. **B** replaces the model with portable `GL_rbf_CQ` while retaining `legacy_mha + full`. "
                "**C** changes only the CQ condition-attention execution to `cached_kv + full`. The downstream dataset contract, chronological split, field normalization, trainer, optimizer family, "
                "fixed sensor manifest, fixed query indices, and milestone evaluator are matched. All three formal runs completed 40,000 steps with finite recorded loss and gradient norms and no backward retries. "
                f"The immutable sensor-manifest digest is `{arm_a['formal_run']['fixed_validation_manifest_sha256']}` and the query-index digest is `{arm_a['formal_run']['fixed_query_indices_sha256']}`."
            ),
        },
        {
            "id": "migration_effect",
            "type": "markdown",
            "layout": "full",
            "sourceId": "derived_report",
            "body": (
                "### Migration effect: B versus A\n\n"
                f"B ends **{b_minus_a_quality}** higher in mean relative L2. Its median steady epoch wall time is **{exact(b_epoch_median)} s** versus "
                f"**{exact(a_epoch_median)} s** for A ({percent_change(b_epoch_median, a_epoch_median):+.12f}%), and formal peak allocated memory is "
                f"**{b_peak_alloc} bytes** versus **{a_peak_alloc} bytes** ({percent_change(b_peak_alloc, a_peak_alloc):+.12f}%). "
                "Those resource differences are end-to-end arm differences—not an isolated kernel causal estimate—because the architectures and B/C query-microbatched lifecycle differ from A."
            ),
        },
        {
            "id": "execution_effect",
            "type": "markdown",
            "layout": "full",
            "sourceId": "execution",
            "body": (
                "### Execution effect: C versus B\n\n"
                f"In the controlled same-state, same-batch GPU 0 probe (5 warmups, 50 measured steps), cached K/V changes median whole-step time by "
                f"**{exact(step_effect)}%** and peak allocated memory by **{exact(alloc_effect)}%**. Observed K/V projections are exactly **4 per step** for B and **1 per step** for C. "
                f"Maximum absolute loss and gradient-norm differences are **{exact(execution['numerical_equivalence']['max_absolute_loss_difference'])}** and "
                f"**{exact(execution['numerical_equivalence']['max_absolute_gradient_norm_difference'])}**, respectively."
            ),
        },
        {
            "id": "total_effect",
            "type": "markdown",
            "layout": "full",
            "sourceId": "derived_report",
            "body": (
                "### Total latest-model effect: C versus A\n\n"
                f"C ends **{c_minus_a_quality}** higher in mean relative L2. Its median steady epoch wall time is **{exact(c_epoch_median)} s** versus "
                f"**{exact(a_epoch_median)} s** for A ({percent_change(c_epoch_median, a_epoch_median):+.12f}%), and formal peak allocated memory is "
                f"**{c_peak_alloc} bytes** versus **{a_peak_alloc} bytes** ({percent_change(c_peak_alloc, a_peak_alloc):+.12f}%). "
                "This is the net validated arm difference, not a decomposition of every architectural contribution."
            ),
        },
        {
            "id": "limitations",
            "type": "markdown",
            "layout": "full",
            "sourceId": "derived_report",
            "body": (
                "## Limitations and robustness\n\n"
                "This benchmark has one training seed, one dataset/split, one hardware class, and one sensor/query protocol; it does not estimate across-seed uncertainty. "
                "Only seven fixed checkpoints were evaluated, so the chart must not be read as a continuous learning curve. B/C configured EMA is the prespecified primary selector, while A has no EMA, "
                "which is operationally faithful but not a live-weight-only ablation. The controlled C-versus-B probe isolates execution behavior; formal-run timing across A and B/C also reflects architectural and lifecycle differences. "
                "The original B128 protocol OOMed during Arm-A backward even after an allocator-identical retry; the user-authorized B40 setting was then frozen for all arms without reducing Q4096 or the 192–384 sensor range."
            ),
        },
        {
            "id": "recommendations",
            "type": "markdown",
            "layout": "full",
            "sourceId": "derived_report",
            "body": (
                "## Recommended next steps\n\n"
                "1. Keep A as the quality reference for this exact 200-epoch configured-weight protocol; it has the best final fixed-manifest metrics.\n"
                "2. If portability is required, prefer B for the stronger epoch-200 configured-EMA quality among the CQ arms.\n"
                "3. If CQ execution cost is the priority, use C's cached-K/V path and retain the controlled probe as its causal performance evidence; do not describe its small final quality gap as an execution-speed result.\n"
                "4. Before changing the default downstream model, repeat the matched protocol across additional seeds and predeclare whether configured EMA or live weights is the deployment selector."
            ),
        },
        {
            "id": "further_questions",
            "type": "markdown",
            "layout": "full",
            "body": (
                "## Further questions\n\n"
                "- Does the epoch-150-to-200 crossover reproduce across seeds or later checkpoints?\n"
                "- How sensitive are the CQ conclusions to EMA decay and to a live-weight deployment policy?\n"
                "- Do field-specific downstream objectives change the ranking, especially for U_1 and pressure?\n"
                "- Can cached K/V preserve its step-time and memory advantage at other query sizes, sensor counts, or GPUs?"
            ),
        },
    ]

    manifest = {
        "version": 1,
        "surface": "report",
        "title": "GL-RBF/CQ 200-epoch migration benchmark",
        "description": "Matched fixed-manifest comparison of corrected legacy GL_rbf_ENH, portable GL_rbf_CQ, and cached-K/V execution.",
        "generatedAt": generated_at,
        "charts": [chart],
        "tables": tables,
        "sources": sources,
        "blocks": blocks,
    }
    return {
        "surface": "report",
        "manifest": manifest,
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "ready",
            "datasets": {
                "milestone_quality": milestone_rows,
                "final_deltas": delta_rows,
            },
        },
        "sources": sources,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPORT_DIR / "artifact.json",
        help="Artifact output path (default: report/artifact.json)",
    )
    args = parser.parse_args()
    artifact = build_artifact()
    rendered = json.dumps(artifact, indent=2, ensure_ascii=False) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(f"Wrote {args.output} ({len(rendered.encode('utf-8'))} bytes)")


if __name__ == "__main__":
    main()

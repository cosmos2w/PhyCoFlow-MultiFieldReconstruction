# Post-training results and protocol

This compact record summarizes the completed Run 00, Run 06, Run 11 and Run 11f comparison. Its values come from the completed assessment retained with the originating checkout's ignored local report archive; that archive is optional evidence and is not included in a fresh clone.

## Four-run comparison

All four children completed 1,000 reporting epochs / 38,000 AdamW updates from the same pretrained Flow SOURCE. The broader validation assessment covers 128 snapshots and reuses matched SOURCE and child checkpoint identities. Percent changes below compare each child with that validation SOURCE; positive endpoint-MSE changes are regressions. Selected checkpoints have unequal exposure and do not form an equal-step method comparison.

The original selection panel contains 32 validation snapshots; the broader assessment uses four 32-snapshot validation ensembles. Common assessment uses live weights, two Euler steps, fixed shared queries and seed 2027. The table's T3 denotes the common CO/T persistence reassessment with three mutual filtration lines; the training and original checkpoint-selection protocol used one mutual line. Coherence is evaluated per ensemble and then averaged, rather than treating all 128 snapshots as one ensemble. Keep the query geometry, family definitions, ensemble size and SOURCE reference fixed when reproducing these values.

| Run | Update rule | Selected checkpoint on broader panel | Broader selected T3 reduction | Final checkpoint on broader panel | Broader final T3 reduction | Broader final CO MSE change |
| --- | --- | --- | ---: | --- | ---: | ---: |
| 00 · Plain Sum | Native RF plus raw A+B+T sum | Epoch 255; guard passes | 50.95% | Epoch 1,000; fails CO | 51.94% | +5.77% |
| 06 · Joint ConFIG | Joint RF/A/B/T ConFIG | Epoch 980; fails CO | 67.09% | Epoch 1,000; fails CO | 65.41% | +6.90% |
| 11 · Lagrangian | Internal A/B/T ConFIG plus dense SOURCE-relative fidelity feedback | Epoch 895; guard passes | 66.38% | Epoch 1,000; passes | 67.00% | +3.44% |
| 11f · Lagrangian with LR warm-up | Internal A/B/T ConFIG plus dense SOURCE-relative fidelity feedback; model LR warm-up | Epoch 595; guard passes | 66.29% | Epoch 1,000; fails CO | 68.85% | +5.43% |

Run 11f is the documented example because the run completed the specified warm-up recipe and its selected checkpoint passed both the original fixed selection panel and the broader assessment. At epoch 595 / update 22,610, `best.pt` has a 66.29% broader T3 reduction and a worst broader per-field MSE change of +2.18% for p. At epoch 1,000 / update 38,000, `last.pt` has a 68.85% broader T3 reduction but CO MSE increases +5.43% on the broader panel and +5.019% on the original 32-snapshot selection panel, exceeding the +5% field allowance.

The selected checkpoint is the eligible candidate produced under the original fixed-panel selection rule. The subsequent 128-snapshot reassessment corroborates that saved checkpoint; it did not retroactively change which checkpoint was selected. `best_fidelity.pt` tracks the minimum aggregate fixed-panel MSE and is not an eligibility certificate for every field. Under topology-with-fidelity selection, `best.pt` may remain a SOURCE fallback if no child checkpoint qualifies, so inspect the saved selection metadata and update cursor.

The single-seed evidence does not establish scientific superiority, a universal fidelity guarantee, physical causality, or a speedup. Run 11 had the largest raw A+B+T reduction among final checkpoints (19.48% on the broader panel); Run 11f had the largest final T3 reduction. Lagrangian campaign timers were slower than the joint/native-RF routes in this recorded cohort, and the `dense_fast` historical run name is not measured complete-campaign latency evidence.

No TEST data was evaluated for this report. Repeated checkpoints share one training seed and validation source; repeated panel checks and related snapshots do not provide independent training replicates or confidence intervals.

## Durable workflow and local evidence

Use [Post-training strategies](Posttraining_Strategies.md) for the maintained portable recipe, dependency extras, source binding, smoke, fresh run, resume and explicit-checkpoint validation evaluation. Older Run 11 operational notes are optional historical evidence preserved with local run artifacts and may not be present in a fresh clone.

The completed presentation report, figures, summary tables, restricted checkpoint evidence, manifests, and saved assessments are ignored generated artifacts retained only in the originating checkout. The earlier 00/06 assessments were reused only after exact checkpoint-identity matching; the completed 11/11f assessment performed read-only forward evaluation of SOURCE, last and selected checkpoints on the declared panels. It performed zero optimizer updates and zero TEST evaluations.

From a checkout that has access to the four local run directories, the maintained CPU-only summary utility reads weighted training histories, completion cursors and provenance without loading checkpoints, computing validation ratios, running inference or using a GPU:

```bash
RUN00=cases/turbulent_combustion/runs/_MultiTask/00_plain_sum_1000ep/RUN_ID
RUN06=cases/turbulent_combustion/runs/_MultiTask/06_joint_config_1000ep/RUN_ID
RUN11=cases/turbulent_combustion/runs/_MultiTask/11_config_lagrangian_1000ep/RUN_ID
RUN11F=cases/turbulent_combustion/runs/_MultiTask/11f_lagrangian_lr_warmup_dense_fast_100ep/RUN_ID
python scripts/posttraining/summarize_runs.py \
  --run "00=$RUN00" \
  --run "06=$RUN06" \
  --run "11=$RUN11" \
  --run "11f=$RUN11F" \
  --output /path/to/local-summary
```

The utility writes `summary.md`, `summary.json`, `summary.csv`, and `epoch_history.csv` from training telemetry and saved completion/provenance records; it does not load a checkpoint, compute final validation ratios, or regenerate the selected-checkpoint comparison table above. Use the local saved report evidence to review selected checkpoints. Reassessment scripts under the ignored report and operations directories are dated, depend on saved local assessments and case data, and are archive-only rather than portable fresh-clone evaluation commands.

Optional history figures use the `plot` extra and the maintained `render-history` entry point; this command reads the named run's saved telemetry:

```bash
python -m pip install -e '.[plot]'
python cases/turbulent_combustion/run.py render-history --run "$CHILD_RUN"
```

## Maintained references

- [SOURCE-relative fidelity policies and internal ConFIG](Fidelity_Budget_Posttraining.md)
- [Post-training strategies and portable recipe](Posttraining_Strategies.md)
- [Post-training implementation and maintenance guide](Posttraining_Development.md)
- [Earlier scalarization and joint-aggregation quick start](Multitask_Prep.md)
- [Configuration composition and validation rules](configuration.md)

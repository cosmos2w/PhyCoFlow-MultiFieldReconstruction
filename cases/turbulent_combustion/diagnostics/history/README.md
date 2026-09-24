# Training history figure suite

These figures separate four questions:

| Figure | Role and contents |
| --- | --- |
| `loss_history.png` | Training objective trajectories: total, data, coherence, physics when present, and fixed-sample validation. Each objective has an independent scale; panel heights are not additive and do not indicate gradient or update influence. For post-training, the labels identify pre-update data loss and the family-weighted coherence sum. |
| `coherence_history.png` | Total coherence objective and weighted contributions from each configured family, followed by raw component trajectories. Component strips report the last raw and effective weighted values; thin traces retain epoch values and bold traces show a centered 2% window median. |
| `optimization_diagnostics.png` | Epoch-mean data, coherence, and combined gradient norms; data/coherence gradient cosine; and update conflict frequency. These are gradient diagnostics, separate from loss magnitudes. |
| `checkpoint_fidelity.png` | Fixed-panel topology selection score and source-relative normalized MSE change for total and each field. Circles mark eligible candidates, crosses mark rejected candidates or field-limit violations, and the star marks the selected checkpoint in the frozen window. |

Family colors are stable across figures: global distribution is coral, cross spectrum is blue, and topology is ochre. Line styles repeat those identities without relying on color alone. Loss and raw component traces are not interchangeable with effective weighted contributions. Gradient norms describe update direction/magnitude diagnostics; they are not loss values.

## Frozen active-run example

The example below uses the active `20260924T030724Z_4208231c` run through epoch 800. The script reads its JSONL/config and writes only under this diagnostics directory. It stores a compact JSONL projection containing only plotted fields, the resolved configuration, row counts, cutoff, and SHA-256 hashes of the full source files. The nested checkpoint evaluation payloads are omitted from the example snapshot.

Rebuild all four images from the repository root:

```bash
rtk proxy python cases/turbulent_combustion/diagnostics/history/render_history_examples.py \
  --run-dir cases/turbulent_combustion/runs/coherence_fix_ABC_sliced_persistence_formal_5000ep_gpu1/20260924T030724Z_4208231c \
  --through-epoch 800 \
  --output-dir cases/turbulent_combustion/diagnostics/history/active_run_epoch_0800
```

The fixed-panel score is an eligibility-constrained checkpoint-selection signal. A rejected candidate's score remains visible for diagnosis but does not become the selected checkpoint; the star is computed from eligible candidates through the frozen epoch. Relative MSE changes use the run's source checkpoint and configured total/per-field limits. These are fixed-panel selection diagnostics, not held-out generalization estimates or topology-quality guarantees.

Example outputs:

- [Loss objectives](active_run_epoch_0800/loss_history.png)
- [Coherence family and component history](active_run_epoch_0800/coherence_history.png)
- [Optimization diagnostics](active_run_epoch_0800/optimization_diagnostics.png)
- [Checkpoint fidelity](active_run_epoch_0800/checkpoint_fidelity.png)
- [Snapshot manifest](active_run_epoch_0800/manifest.json)

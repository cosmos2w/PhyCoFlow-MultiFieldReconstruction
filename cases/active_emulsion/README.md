# Active-emulsion topology post-training

The [case recipe](configs/posttrain/topology.yaml) uses efficient sliced
persistence on periodic 128×128 physical fields. Self components compare phi,
vx, and vy; the mutual component compares phi with signed vorticity. The
[family configuration](configs/coherence/topology.yaml) declares the grid,
periods, filtrations, and four mutual slices.

From the repository root, with a compatible completed native source run:

```bash
python cases/active_emulsion/run.py post-train \
  --config cases/active_emulsion/configs/posttrain/topology.yaml \
  --override source_run=/path/to/source/run
```

The source supplies the dataset path, model, and sensor configuration. The example
[block-mean layout](configs/sensors/block_means.yaml) pools physical observations;
its nonlinear field transform belongs inside the physical model adapter.
[Dataset documentation](../../datasets/active_emulsion/README.md) describes the
required payload and metadata.

The case selects 30% of training snapshots, stratified by `regime` and `m`, with
at least four temporal bins per original simulation (`file`). Its budget is 3,000
reporting epochs of nine updates. The sampler continues across those reporting
boundaries and retains short final batches. Adjust the schedule and subset for
your dataset and compute budget.

| Run artifact | Meaning |
|---|---|
| `progress.json` | Updates, reporting epochs, and sample exposures |
| `artifacts/training_subset.json` | Selected rows and metadata mapping |
| `metrics/constraint_updates.jsonl` | Optimization and fidelity diagnostics |
| `metrics/topology_validation.jsonl` | Fixed-panel validation results |
| `evaluation/checkpoint_status.json` | Checkpoint eligibility |
| `checkpoints/last.pt` | Model, optimizer, and recovery state |

The [shared guide](../../TOPOLOGY_POSTTRAINING.md) covers the objective,
installation, execution controls, and segmented recovery. Cluster profiles,
checkpoint import adapters, and historical campaign reports are local artifacts.

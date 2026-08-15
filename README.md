# Multi-Field Reconstruction

This workspace benchmarks reconstruction of complete physical fields from
partial, sparse observations. It separates reusable data/model/training code
from case-specific physics, configurations, visualizations, and run artifacts.

The first supported cases are turbulent combustion, Brusselator,
two-dimensional Kolmogorov flow, Kuramoto–Sivashinsky (KS), and a
mass-transport–fluid future case scaffold. Kolmogorov is a formal initial
benchmark; the current mass-transport demo remains integration-only until an
official dataset exists. The initial model matrix covers deterministic point
models, neural operators, diffusion/latent generative models, and
PointCloudFFM.

## Setup

```bash
cd Proj_MultiFieldReconstruction
conda env create -f environment.yml
conda activate phycoflow_reconstruction
python -m pip install -e .
python scripts/validate_dataset.py --all
```

If payloads live elsewhere, create a safe relative link without copying data:

```bash
python scripts/link_dataset.py --case brusselator --source /path/to/brusselator.h5
```

Existing environments may instead install the package editable with
`python -m pip install -e . --no-deps` when compatible dependencies are already
available.

## Working rule

Launch work from a case directory. The thin case launcher delegates to the
general package and writes only under that case's ignored `runs/` directory.

```bash
cd Cases/brusselator
python run.py validate --config configs/dataset.yaml
python run.py train-base --config configs/base/pointcloud_ffm.yaml --max-steps 1
```

`--max-steps` is intended only for focused integration checks. Formal runs must
use reviewed configs without this override.

The Kolmogorov case uses the same entry point and reconstructs `u,v,p` from
sparse velocity measurements:

```bash
cd Cases/kolmogorov
python run.py validate --config configs/dataset.yaml
python run.py train-base --config configs/base/pointcloud_ffm.yaml --max-steps 1
```

The historical combustion checkpoint is validated without modifying it:

```bash
cd Cases/turbulent_combustion
python import_demo50.py --config configs/compatibility/demo50.yaml
```

Phase-5 post-training always writes a child run. For a native Brusselator
PointCloudFFM source, provide the immutable base-run directory explicitly:

```bash
cd Cases/brusselator
python run.py post-train \
  --config configs/posttrain/global_distribution_reference.yaml \
  --override source_run=runs/<base-experiment>/<run-id>
```

The Demo50 compatibility configuration reproduces the paired-supervised
global-distribution semantics and explicit stale-field mapping:

```bash
cd Cases/turbulent_combustion
python run.py post-train \
  --config configs/posttrain/demo50_global_distribution.yaml
```

`training_reference` and `paired_supervised` are intentionally separate
configurations. The former strips paired targets from the coherence rollout
and uses only a serialized training-split reference bank. Do not interpret a
`--max-steps` integration run as a post-training result.

Phase 6 provides matched child configs for every compatible Phase-4
Brusselator checkpoint under `configs/posttrain/phase6_*.yaml`. After the child
runs exist, validate and regenerate the comparison artifact with:

```bash
python -m phycoflow_reconstruction.evaluation.posttrain_comparison \
  --matrix Cases/brusselator/comparisons/phase6_gpu1_matrix.yaml \
  --json Cases/brusselator/comparisons/phase6_gpu1_summary.json \
  --markdown Cases/brusselator/comparisons/phase6_gpu1_summary.md
```

The checked-in Phase-6 summary is a one-step GPU integration comparison. It
validates adapter transfer and matched protocols; it is not a tuned benchmark.

Phase 7 adds one common checkpoint evaluator with case-owned physical
diagnostics. It records the resolved config, checkpoint, portable sensor
manifest, query indices, dataset fingerprint, timings, and plotting payload:

```bash
cd Cases/brusselator
python run.py evaluate-run \
  --run runs/<experiment>/<run-id> \
  --sensor-config configs/sensors/u_only_random.yaml \
  --split validation --max-samples 1
```

Verified Brusselator PDE paths are deliberately separate:

```bash
python run.py train-direct --config configs/direct_physics/pinn.yaml
python run.py post-train \
  --config configs/posttrain/physics_periodic.yaml \
  --override source_run=runs/<base-experiment>/<run-id>
```

The frozen Phase-8 integration release is in `benchmarks/v0_integration/`.
From a fresh editable install with a linked Brusselator dataset, reproduce its
documented one-update workflow using:

```bash
bash scripts/reproduce_brusselator_integration.sh
```

See `REPRODUCIBILITY.md` for the artifact trace and its strict non-scientific
scope.

Latent flow uses two explicit base runs. Stage 1 learns the autoencoder; Stage
2 must set `model.stage1_checkpoint` to the immutable Stage-1 `last.pt` or
`best.pt`. The loader checks model identity and stage before loading only the
autoencoder weights.

## Layout

- `src/phycoflow_reconstruction/`: reusable contracts, data, models,
  hierarchical coherence families, training, evaluation, and utilities.
- `Dataset/`: tracked explanations plus local-only payload links.
- `Cases/`: case definitions, configs, launchers, visualizations, and runs.
- `configs/schema/`: the three mutually exclusive training-stage schemas.
- `tests/`: focused contract tests rather than per-model smoke scripts.
- `benchmarks/`: reviewed release specifications, frozen sensors, aggregate
  results, and audit records; local checkpoints remain outside Git.
- `figures/`: reproducible visualization scripts and generated figure contracts.

See `ProjectCreate.md` for the staged build plan and `ModelExplain.md` for the
scientific/model conventions. `UPSTREAM.md` records dependency and clean-room
reference decisions.

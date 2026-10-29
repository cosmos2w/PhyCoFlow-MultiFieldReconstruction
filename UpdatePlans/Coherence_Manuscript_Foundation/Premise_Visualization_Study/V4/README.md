# V4 evidence library: reproduction and extension

This library compares frozen `last.pt` checkpoints on the existing 100 TEST frames, 256 T-only observations per frame and the native 100 × 403 grid. Predictions are training-normalized, in `(CH4, CO, T, U_1, p)` order. The cohort is a paired illustrative study, not an independent final test. See [the source cohort manifest](../metadata/study_manifest.json), [checkpoint snapshot](metadata/checkpoints.json), [model registry](scripts/registry.json), and [report](REPORT.md).

## Retained inputs and provenance

- `../data/gt_test100.npz`: normalized target fields, coordinate grids, indices, sensor inputs and normalizer. `../metadata/study_manifest.json` and `../metadata/observation_manifest.json` document the original protocol.
- `data/*_test100.npz`: six `[100,100,403,5]` prediction arrays; three relative links reuse byte-verified earlier arrays. New DeepONet, GeoFNO and DiffusionPDE outputs are saved here. No checkpoint copies are retained.
- `metadata/checkpoints.json` freezes the actual checkpoint and resolved-config SHA-256 values. Each `metadata/<model>.json` binds a prediction array hash, settings and source. `data/metrics/` holds cohort metrics and definitions; `data/events_*.csv`, `data/events_cases.npz` and `metadata/events_summary.json` hold coupled-region evidence; `data/controls.npz` and `metadata/controls.json` hold controlled examples. PNGs are previews and PDFs are masters in `figures/`.
- Current DiffusionPDE uses this repository's conditional EDM formulation and Heun sampler. It is not evidence of equivalence to the upstream DiffusionPDE system. The checkpoint is epoch 1400; logged training continued through epoch 1403 before the authorized interruption. Validation in this plot is a sampled validation loss. Source histories: [training](../../../../cases/turbulent_combustion/runs/tc_diffusion_pde_5000ep/20261027T222333Z_187a0271/metrics/history.jsonl), [validation](../../../../cases/turbulent_combustion/runs/tc_diffusion_pde_5000ep/20261027T222333Z_187a0271/metrics/validation_history.jsonl).

![Paused DiffusionPDE training and sampled validation loss histories](figures/diffusion_paused_loss.png)

Training history extends to epoch 1403; the frozen last checkpoint is epoch 1400. [PDF master](figures/diffusion_paused_loss.pdf).

## Extraction and adding models

Use the `phycoflow_env` Python from the repository root. `snapshot` must run before opening TEST data for a fresh study; the existing snapshot is immutable and deliberately refuses replacement. For a new model add one `name: run-relative-path` entry to `scripts/registry.json`, then run `inference.py extend` to freeze the new `last.pt` and configuration without changing existing frozen identities. The model must use the project `build_model` and `reconstruct` interface, the same dataset, normalizer, and sensor protocol. Run `extract --model NAME --device cpu` or `--device cuda:0`; for CUDA set `CUDA_VISIBLE_DEVICES` to the intended physical UUID before launch and verify its occupancy. Each completed file has a matching metadata file. Per-frame partials allow resume with the same checkpoint, protocol, device and visibility; no training is performed. Native generation settings and configured EMA are retained, and target fields are removed from new inference inputs. Previously saved results are reused only after matching hashes and protocol.

```bash
PY=/home/wanglz/miniconda3/envs/phycoflow_env/bin/python
V4=UpdatePlans/Coherence_Manuscript_Foundation/Premise_Visualization_Study/V4
# For an added registry entry:
$PY "$V4/scripts/inference.py" extend
$PY "$V4/scripts/inference.py" extract --model NEW_NAME --device cpu
```

## Metrics and figure redraw

From the repository root, the registry drives the complete CPU workflow. The full recomputation can take substantial time. The definitions and exact inputs are recorded in `data/metrics/definitions.json`.

```bash
PY=/home/wanglz/miniconda3/envs/phycoflow_env/bin/python
V4=UpdatePlans/Coherence_Manuscript_Foundation/Premise_Visualization_Study/V4
$PY "$V4/scripts/run.py" --all       # analyze, redraw, rebuild report, verify and render
$PY "$V4/scripts/run.py" --analyze   # recompute complete registry metrics and event catalog
$PY "$V4/scripts/run.py" --redraw    # redraw from saved metrics, cases and controls
$PY "$V4/scripts/run.py" --verify    # verify and render the current report
```

Inference remains a separate explicit action. Added model entries automatically appear in the metrics and event catalogs and figures. A new model receives an explicitly pending source-review entry in the audit; inspect its source and observation treatment and review the rebuilt report's claims before publication. Existing model audits are retained. The verifier checks frozen source identities, all prediction hashes, full cohort and observations, expected registry-sized catalog counts, PNG decoding, PDF master links and local Markdown links; it writes a rendered HTML preview and a PDF using PyMuPDF in the `fig` environment. It requires access to the original dataset, runs and checkpoint paths.

## Paused training

Training was interrupted and is not restarted by these scripts. To resume only when explicitly authorized, verify GPU ownership, checkpoint identity and existing run history first. The case CLI accepts `train-base --resume RUN_DIRECTORY` with the original config and overrides, and its resume code checks the resolved config hash. The interrupted job used `configs/base/diffusion_pde_edm_5000ep.yaml` with `runtime.device=cuda:0`; use the intended physical GPU through `CUDA_VISIBLE_DEVICES`. Resume starts from `last.pt` epoch 1400, so history entries after that checkpoint represent pre-interruption progress, not checkpointed model state; inspect how duplicate epochs are handled before continuation. Do not infer completion from the `5000ep` run name or stale status file.

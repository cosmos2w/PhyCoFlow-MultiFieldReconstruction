# Scalable Anchored Coherence

Use this profile to refine your trained SOURCE with three complementary views of a multi-field state. **State** measures physical occupation of field values, **Scale** measures signed organization across spatial bands, and **Shape** measures excursion-set topology. The profile shares sampled field/pair work, fits representations on TRAIN, and freezes internal gradient calibration after the first 10 epochs. Your case supplies the scientific settings; the shared package owns the computation and lifecycle.

The recommended fragment is [`scalable_anchored_coherence.yaml`](../configs/defaults/scalable_anchored_coherence.yaml). It selects the C8-derived scalable recipe. Existing configurations continue to select their original objectives. The mathematical definitions and legacy formulas are in [`ModelExplain.md`](../ModelExplain.md#7-data-driven-coherence-model).

## What to supply for a new case

| Case choice | What it controls |
|---|---|
| Completed SOURCE and checkpoint identity | Immutable starting model, its normalizer, solver and observation contract |
| Ordered fields and units | Every marginal, pair index, field floor and fitted tensor |
| Physical masses and coordinates | Occupation probability, spectral integration and field-risk units |
| Geometry, masks and boundaries | Connected pieces, retained modes, Shape raster and persistence domain |
| TRAIN fitting sample IDs and seed | Reproducible scales, direction selection and pooled Shape calibration |
| Positive task-unit floors and priorities | Weak-field behavior and relative scientific priorities |
| Fidelity allowance and selection rule | Acceptable per-field reconstruction change and checkpoint choice |
| Training horizon and draw budget | Optimizer exposure and number of independent conditional reconstructions |
| Anchoring qualification | Fixed SOURCE/child output coefficient and measured inference cost |

Use approximately 10% of samples for matched evaluation by default; keep sample IDs and reconstruction draws identical across the compared models. Training horizons are stated in epochs. A short startup check verifies execution; it does not qualify a learned improvement.

## Compose the profile

From `cases/<case>/configs/posttrain/`, compose your ordinary case profile and the shared fragment:

```yaml
defaults:
  - _common.yaml
  - ../../../../configs/defaults/scalable_anchored_coherence.yaml
source_run: null
# For a full nonperiodic rectilinear snapshot grid:
dataset: {physical_measure: native_rectilinear_area}
```

Supply the completed SOURCE at launch using the standard [post-training commands](../README.md#63-validate-launch-and-resume-on-your-data). A new profile inherits the SOURCE model and dataset contract; choose its physical coordinate scaling, spectral bands, Shape raster and fidelity allowance in the case file. The shared settings use 32 State candidates, two Scale bands, joint weight 0.25, unit self/rho priorities, a `1e-4` model-unit Scale floor, and a `32 × 32` Shape raster with 32 diagram projections. These are exposed starting settings; fit/qualify the case's units and resolution before a substantive run. The anchoring coefficient is deliberately absent from this fragment.

## The recommended computation

| View | Full reference | Scalable profile |
|---|---|---|
| State | Fixed larger mixed-direction bank and every field marginal | Freeze up to eight hard/diverse directions from TRAIN/SOURCE; retain every marginal |
| Scale | Complete additive field/pair population | Importance-correct sampled signed and rho leaves; retain complete joint, piece means and log-power-self anchors |
| Shape | Complete fixed-transform query observer | Shared sampled field/pair IDs, selected signs/lines only, separately versioned pooled TRAIN calibration |
| Internal coefficients | Unit coefficients | Parameter-gradient probes at epochs 1, 3, 6, 8 and 10; freeze after epoch 10 |

The full reference is useful for qualification. Changing from per-query to pooled Shape scales changes the finite observer, so its raw score is not numerically interchangeable with the earlier complete observer. Assess both models through the same frozen observer when comparing outcomes.

### State

The occupation law is the pushforward of normalized physical masses through the field values. Weighted one-dimensional transport uses the exact merged empirical CDF breakpoints. The common State distance combines all marginal channels and fixed mixed projections in one Hilbert norm; conditional scoring uses the same distance for attraction to the reference and separation between independent members.

Direction fitting removes sign-equivalent duplicates, selects four angularly diverse directions and then up to four hard directions with a diversity penalty. Hardness is measured on fixed matched TRAIN/SOURCE reconstructions. Selection, normalization and weights freeze before learning. For one field, the mixed channel is absent. The exact endpoint embedding is reused across pairs; no quantile-grid approximation is introduced by this cache.

### Scale

The spatial representation contains signed band fluctuations and mass-weighted means on every connected piece. The complete joint score includes both in one vector. Full log-power-self and pairwise signed rho add complementary information. Their improvement cannot replace signed/joint preservation.

Only additive signed projections and rho terms are sampled. For an implicit population of size `P`, `K` draws, proposal probability `p_i` and coalesced multiplicity `n_i`, each observed leaf has population-mean coefficient `n_i / (K * P * p_i)`. The sampler uses a positive uniform mixture, so every pair retains support. A uniform-only mode provides a reference. Full anchors and model inference still depend on field count and spatial resolution.

### Shape

The field transform, raster and normalization freeze on TRAIN. Unary signs are positive/negative; mutual queries use four sign combinations and three fixed positive lines. H0/H1 finite diagrams and essential births remain separate. Pooled scales distinguish self/mutual, sign, line, homology dimension and finite/essential categories. Positive floors support near-zero discrepancies without using prediction-dependent denominators.

Scale and Shape share the sampled outer field/pair IDs. Shape constructs only selected signs/lines, with their declared inner probabilities. A finite query budget provides all-pair support rather than exhaustive visitation. Physical cell-overlap, interpolation, smoothing, masks and periodicity belong to the observer's definition. A reduced raster measures topology on that raster.

## Measures and sampling

[`coherence/measures.py`](../src/phycoflow_reconstruction/coherence/measures.py) supplies explicit physical measures and exact fingerprints. [`data/physical_measure.py`](../src/phycoflow_reconstruction/data/physical_measure.py) carries them as auxiliary metadata through sample/query selection, independently of model inputs. For a rectilinear grid, choose the declared native nodal measure; for other domains provide explicit masses and their geometry/units receipt.

Occupation distances normalize their endpoint masses into probability measures. A sparse self-normalized occupation measure is a ratio approximation to the native occupation law. Linear field-risk integration has a different sampling contract: use the native normalized masses and the actual inclusion/proposal correction. Renormalizing the randomly selected mass changes that estimator. Do not infer a physical measure from point density or a mask without a case declaration.

## Fitting, updates and resume

The current ready-to-run profile uses fixed, valid snapshot queries from SOURCE models that accept selected query sets. Its prefit rejects structured-grid SOURCE models and invalid selected nodes. Grouped and native-topology evaluation use their existing separate workflows; the new profile's common assessment is the frozen observer described below. The numerical measure and scoring primitives can also be used through explicit case-owned support contracts.

Fit geometry, field units, direction selection and pooled Shape scales only from the declared TRAIN data and completed SOURCE. Artifacts bind sample IDs, field order, normalizer, physical/query support, SOURCE checkpoint, configuration and definition versions. Loading a stale artifact fails before training. Fitting uses isolated random streams and does not advance the ordinary training batches or optimizer.

Each update shares observation-conditioned reconstructions across all three families. Generative models supply independent latent/prior realizations; deterministic models use one real prediction and an explicitly paired score. Dense targets enter the objective after reconstruction. The starting profile sets a 10% SOURCE-relative allowance for each field's TRAIN physical-measure MSE; choose the allowance deliberately for your task. All accepted updates pass through the existing shared optimizer lifecycle.

The Lagrangian policy applies online TRAIN fidelity feedback. Per-field risk can exceed the allowance on individual updates or held-out panels; adoption therefore uses a separate matched evaluation guard for the raw or TRAIN-qualified anchored output.

During the first 10 epochs, parameter-gradient probes use the already computed reconstruction graph of the first ordinary training batch at epochs 1, 3, 6, 8 and 10. The journal records its sample IDs and query identity. This keeps the ordinary training stream intact; the fixed campaign calibration batch stays in the research record. The semantic blocks are State marginal/mixed, Scale signed/joint/log-power-self/rho, and Shape self/mutual. Each fair block keeps attraction and member separation together. Within a family, EMA-smoothed norms produce median-relative positive coefficients clipped to `[0.5, 2]`; unresolved near-zero blocks remain at one. State coefficients belong inside its common square root for every pair. These are internal coefficients; whole-family weights do not have the same effect under ConFIG.

Sampler observations, controller state and calibration advance only with successful optimizer updates. A checkpoint saves their cursor and random states with model/optimizer/EMA state. An unchanged resume recovers those values; a changed definition or fitted identity starts a new child run.

Run-owned `logs/` contains complete histories and curve PNG/PDF files. Curves refresh at first available history, every 10 completed epochs, validation, pause and completion. Readers accept legacy `metrics/` histories. Recovery preserves an immutable digest-named copy before removing records beyond the checkpoint. ConFIG family values, sampled TRAIN diagnostics, held-out metrics and actual optimizer-step diagnostics are labeled separately.

History figures give each family and raw component its own scale. Recorded values remain visible beneath an explicitly labeled centered-median trend; modest changes use padded linear axes, while genuinely wide positive ranges use logarithmic axes. The epoch-10 calibration freeze is marked where applicable. These views describe the recorded training batches; they do not replace a fixed held-out comparison.

## SOURCE-output anchoring

[`coherence/physical_risk.py`](../src/phycoflow_reconstruction/coherence/physical_risk.py) supplies linear physical-field MSE and the exact blend-risk quadratic on a declared matched support. It separates full quadrature, without-replacement inclusion correction and with-replacement proposal correction.

The public interface is [`blend_matched_outputs`](../src/phycoflow_reconstruction/evaluation/source_anchor.py). It composes already computed outputs as

```text
anchored = (1 - alpha) * source + alpha * child
```

Your case qualifies `alpha` on TRAIN, freezes it, and assesses the fixed choice on matched held-out samples. Supply both checkpoint hashes and matched identities for ordered samples, observations, field order, latent/prior state, solver, normalizer, query support and clamping. The qualification record includes the measured cost of SOURCE and child inference. Evaluating both models is part of the method's inference cost; the interface does not hide either execution inside a base model.

A convex blend preserves hard sensor values when both endpoints obey the same hard clamp. It does not guarantee arbitrary nonlinear physical constraints. Evaluate per-field fidelity and all three coherence views on the blended output. Retain SOURCE if a case's qualified policy provides no worthwhile benefit.

## Matched evaluation

Use `FrozenCoherenceObserver` from `phycoflow_reconstruction.evaluation.frozen_coherence` for a common comparison on saved outputs. Its constructor fixes the TRAIN artifact identity, fitted families, reference tensor, ordered sample IDs, coordinates, endpoint masses and a uniform shared query pack. Use one common positive coefficient set across every compared model; the default is unit coefficients. `.score(members)` accepts `[J, B, N, C]` tensors, with one deterministic member or two independent conditional members, and returns per-sample scores plus definition and observer-receipt identities. It performs no inference, fitting or training-state updates. This Python interface lets a case own its evaluation coverage and coupling.

For the independent scalable scores, `evaluation.batch_size` limits inference and scoring memory without reducing the selected sample panel. Keep its value and sample order identical across compared models: the receipt records each batch's draw seed and the fixed common observer-query seed. Scores and per-condition errors are pooled by sample count; observed/unobserved entry errors use their actual entry counts, including a shorter final batch. This setting is separate from `evaluation.group_size`, which defines ensembles for the existing covariance workflows.

## Measured research example

The frozen research tag `coherence-research-20261112` retains the complete Round 8 campaign. Two matched 1,000-epoch arms compared the full computation with the scalable candidate on one turbulent-combustion dataset. Each used two independent draws and shared a counted 10-epoch warmup. The complete campaign used 23.51 occupied GPU-hours.

The scalable candidate used 3.49% less amortized GPU time than the full arm. Both TRAIN and assessment quality-preservation checks passed; the predeclared assessment requirement for a material additional raw quality gain did not pass. This supports adopting the computation as the preferred scalable recipe while keeping the original scientific decision in the research record.

For the TRAIN-qualified `alpha=0.5` blend, the measured assessment changes relative to SOURCE were:

| Arm | State improvement | Scale improvement | Shape improvement | Largest field-MSE increase |
|---|---:|---:|---:|---:|
| Full computation | 1.24% | 1.01% | 6.95% | 0.89% |
| Scalable computation | 1.78% | 1.27% | 7.23% | 0.79% |

The assessment used matched approximately 10% condition samples and a fixed common observer; Shape used a physical-overlap `32 × 128` raster. The comparison is one seed on one dataset. The example's coefficient, grid and tolerances remain case choices. Measured two-model blend latency was about 0.30 seconds per matched request versus 0.15 seconds for SOURCE alone; use measured costs on your own model and hardware.

## Implementation map

| Contract | Shared module |
|---|---|
| State geometry and conditional scoring | `coherence/families/global_distribution/`, `coherence/state_compute.py` |
| Signed Scale and connected-piece means | `coherence/families/multiscale_spectrum/`, `coherence/scale_compute.py` |
| Formal and pooled Shape | `coherence/families/topology/`, `coherence/shape_compute.py` |
| Shared implicit populations | `coherence/query_sampling.py`, `coherence/query_runtime.py` |
| Parameter-gradient calibration | `training/internal_calibration.py` |
| TRAIN fitting and identity validation | `training/scalable_fitting.py`, `training/scalable_prefit.py` |
| Frozen matched assessment | `evaluation/frozen_coherence.py` |
| Ordinary post-training lifecycle | `training/post_training.py` |
| Recoverable recording and actual-step diagnostics | `training/journals.py`, `training/update_diagnostics.py` |
| Output anchoring | `evaluation/source_anchor.py` |

Baseline model architectures and parameter keys are unchanged. Load existing model checkpoints with their original resolved configuration, normalizer and EMA contract. Legacy experimental optimizer resumes that bind implementation hashes require their recorded implementation. Training continuation additionally needs the corresponding optimizer, controller, fitted-family and random state; a legacy run does not acquire the new scalable definition by resuming it with a new fragment.

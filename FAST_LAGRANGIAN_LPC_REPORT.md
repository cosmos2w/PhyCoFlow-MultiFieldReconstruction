# Fast Lagrangian refinement and LPC ordering: bounded execution report

Campaign date: 2026-10-22 (America/New_York). Branch: `cosmos2w/fast-lagrangian-lpc`. This report retains the completed short-run evidence and is being extended for the 2026-10-23 continuation. The user now authorizes three SOURCE-initialized 100-epoch Goal-1 arms on physical GPUs 1/2/0, followed by the gated Goal-2 diagnostic and at most one supported 100-epoch ordered arm on GPU 0. Long-run results remain pending until measured; the earlier no-long-launch restriction is superseded for this bounded continuation.

## Authorized continuation — 2026-10-23

The user now authorizes G1-C/G1-S/G1-R for 100 fresh SOURCE-initialized epochs on physical GPUs 1/2/0 respectively, followed by Goal 2 only after all three finish and are assessed. Physical GPU 0 becomes the main working device afterward. All three devices were verified idle. The scientific Python source and prepared configuration hashes remain unchanged. Per-device lifetime locks, short shared admission locking, cross-device arm/orphan checks, conservative interrupted-exposure accounting, and combined live GPU-time accounting support the concurrent launch. The continuation campaign/evaluator suite passed all 39 focused tests.

Readiness is applied from `manifests/goal1_readiness.authorized_20261023.json`, linked to the explicit user authorization receipt. Per-arm maximum runtime allowances are twice the matched ten-epoch projections, with two GPU hours reserved for Goal 2, at least one for evaluation/diagnostic work, and bounded shutdown margins inside the 16-hour combined cap. The 100-epoch study uses the existing first-32 full-descriptor validation every ten epochs; the final uniform-128 audit stays distinct. Per-check visibility timing is monitored externally with observation intervals. Concurrent long-run time-to-quality is observational across physical devices and shared CPU/storage; the prior single-GPU timings remain the matched idle-hardware latency evidence. Long-run outcomes are pending until measured; the short-run results below are retained as stage-one evidence.

## Answer and current evidence boundary

The existing Run 11f A/B/T definitions, endpoint operator, model, normalizer, five dense endpoint constraints, and epoch exposure are preserved. The selected acceleration is P3: one shared randomly sampled polarity, with both unary fields and the existing joint line computed together, including H0/H1. The scalar randomized-family variant additionally samples the entire topology family with probability one half and applies its importance factor once. Full existing A/B/T remains the evaluation target.

Four fresh ten-epoch profiles completed on physical GPU 1. P3 with dense ConFIG saved 18.3% inclusive user-command wall time against full-bank dense ConFIG. Moving from P3 ConFIG to P3 scalar sum saved 34.8%; sampling the whole topology family on that scalar route saved 16.4% against its matched dense-family scalar control. T3’s total saving against T0 is 55.5%, but that comparison also changes scalarization and stochastic optimization behavior. These are measured short-run costs, not retained-quality speedups.

```text
implementation_verified: true
latency_effect: positive for supported short-run comparisons
quality_retention_100ep: unsupported
model_agnostic_interface: real_adapter_smoke_verified
lpc_ordering: not_executed
long_run_stability: untested
```

Periodic ConFIG coefficient extraction reconstructs the pinned full-width FP64 TorchJD oracle, retaining coefficient signs. Reusing those coefficients on fresh current gradients failed the small P3 staleness probe at both tested cadences. T4/G1-P are therefore excluded. No stale parameter gradient or direction is used by the implementation. This mechanism is unsupported for the selected estimator in this campaign.

The matched 100-epoch Goal-1 learning histories were unexecuted at the end of the short study. The user explicitly authorized their execution on 2026-10-23, followed by Goal 2. Until the three runs finish and their complete evidence is assessed, 100-epoch quality retention, the mature 60–100 epoch window, and matched time-to-quality remain pending. The real-model residual-gradient diagnostic and any ordered-refinement training stay gated until then; the residual helper currently has numerical unit coverage only.

**TEST exception:** one readiness attempt mistakenly called the existing dataset validator. Its structural probe read one final HDF5 scalar in the TEST frame range. No TEST model evaluation, metric, selection, or estimator tuning was performed. The scalar value was not retained. The incident is recorded in `manifests/test_access_exception.json`, and subsequent validation is config-only. This campaign cannot be described as having zero TEST access.

## Verified baseline and immutable inputs

Remote `main` was verified with `git ls-remote` at `716f48397dbd40301ecd9d72c09a24a405936ca7`. The starting working HEAD is `a1c02200c06e48c15e528d7c0f10636bba976f54`; its common ancestor with the reviewed merge is `04b38070516947514dbdbd732900740b8abc8a49`. The material tree difference is the newer Run 11S summed-coherence configuration, its external-policy validator correction, and its regression test. That work was retained. The older local `main` reference was not used to reset the working tree.

The protocol anchor is `cases/turbulent_combustion/runs/_MultiTask/11f_lagrangian_lr_warmup_dense_fast_100ep/20261017T170131Z_615167aa/resolved_config.yaml`. It resolves to 1,000 historical reporting epochs, despite its directory label. The historical optimizer cursor is 38,000 and the epoch exposure is 38 minibatches of 32 samples, or 1,216 samples per reporting epoch. The TRAIN split contains 8,000 snapshots. A reporting epoch is not a complete dataset pass.

Every independent scientific arm starts from `/home/wanglz/Desktop/src/PhyCoFlow/Proj_MultiFieldReconstruction/cases/turbulent_combustion/runs/tc_gl_rbf_cq_cached_kv_5000ep/20260826T212231Z_7b68c461/checkpoints/last.pt`, SHA256 `03f2f45b966a00850cc45c011b4cf1cd644b9408de355324fec8dc434a84a810`. It is the original pretrained SOURCE, not Run 11f's post-trained state. The resolved anchor SHA256 is `7709a76f3a265c37d3522c36f9b14ff3fa8d9f3826745e95f6b32572ca348771`.

| Contract | Verified value |
|---|---|
| Fields and normalization | CH4, CO, T, U_1, p; original TRAIN mean/std artifact |
| Model and reconstruction | GL_rbf_ENH_CQ / `gl_rbf_cq`, cached condition K/V, top-k 32, two Euler steps, same endpoint-smooth correction and final clamp |
| Training exposure | Batch 32; 4,096 fixed shared queries; 38 minibatches/reporting epoch |
| Topology | CO and T; 32×128; H0/H1; sublevel and superlevel; one CO+T joint line; 32 distance projections; original essential handling and standardization |
| Model optimizer | AdamW, LR 5e-5, weight decay 1e-6, clip 1; original warm-up from 5e-6 across 76 minibatches/two epochs |
| Dense fidelity | Every batch measures all five endpoint MSEs and applies active primal fidelity contributions; successful model action precedes projected dual action; dual LR 0.01; +5% training bounds |
| Calibration | Full descriptors; eight TRAIN batches; two generative draws; seed 700103; fixed C0 and field baselines |
| Timing validation | Original fixed 32-snapshot validation panel, full descriptors; same sensor/query/noise contract across arms |
| TEST | No evaluation/selection; one incidental validator scalar read, disclosed above |

The fixed C0 is `0.022609325824305415`. T0–T3 reproduce the saved historical Run 11f reference object exactly, including all field denominators, floors and bounds; `manifests/protocol_identity.json` retains that comparison and the numerical source-file hashes. The trials executed this study working tree; the delivered branch commit records those files.

The installed environment is `phycoflow_env`: Torch 2.5.1+cu121, TorchJD 0.17.1, cooper-optim 1.0.1, GUDHI 3.13.0, PyKeOps 2.3, NumPy 2.0.1, and h5py 3.14.0. Physical GPU 1 is NVIDIA RTX 6000 Ada, UUID `GPU-3ceda40c-fd5c-4b88-6c47-b3301711571e`. It was idle before reservation; unrelated jobs on GPUs 0 and 2 were left running. CPU threads and persistence workers are fixed at four. The GPU is exclusive to each trial; the host CPU/storage are shared with those existing jobs.

## What is exact, estimated, approximate, or changed

A is the existing raw-state distribution discrepancy. B retains magnitude-squared/squared-band-energy calculations, including a fourth-order cross-band contribution in the original fields. T is the existing persistence-v3 shape discrepancy. These are state, scale, and shape **proxies**; they are not a pure Local/Pair/Collective partition or a decomposition of field-law KL. `T` is not synonymous with the theoretical `C_LPC` sector.

For complete filtration q, write `T = sum_q w_q t_q`. Each `t_q` keeps all homology dimensions and finite/essential operations. The four unary units each carry the original self-category mean weight; each of the two joint units additionally carries its fixed positive-line minimum-direction factor. They are not six equally weighted terms.

P3 samples one polarity with probability 1/2, shared by the unary and joint categories. Its Horvitz–Thompson coefficients reproduce the original two-sign category means; there is no second factor of two applied to an already reduced sampled mean. S2 samples one of four unary units and one of two joint units, applying each unit's original weight divided by its inclusion probability. Generated and reference sides always use identical sampled identities. Generated pairings are recomputed; target pairing reuse uses the existing content-keyed primitive. The historical hash-bound topology implementation is untouched.

Paired dense TRAIN references are required; omitting a native model loss does not make this method data-free.

The exact finite-support expectation of loss and input gradient is checked against the original full objective. On the real SOURCE TRAIN endpoint, P3 mean loss relative error is 5.43e-8 and gradient relative error 8.67e-6; S2 gives 2.71e-8 and 7.81e-6. These FP32 reductions are numerical checks on a fixed pairing stratum, not bitwise identities or guarantees at nonsmooth pairing changes. The separate real-endpoint profile measures full/unary/joint forward-plus-backward scopes at 0.589/0.375/0.200 seconds, with six/four/two complete filtrations. In the full case, pairing/index-transfer host wall is 0.346 seconds, distance host dispatch is 0.080 seconds, and synchronized backward is 0.157 seconds; these overlapping/dispatch scopes must not be summed as independent complete costs. All 192 reference pairing calls hit the already warm fixed-endpoint cache, while 192 generated pairings are fresh. This is not an observed production cache hit rate; `analysis/work_unit_profile.json` records the scopes and limitations. P3 relative gradient variance is 0.998, versus 1.180 for S2. Their individual direction cosines vary substantially. P3 was chosen conservatively before learning outcomes were examined.

The scalar family estimator is `(A + B + I_T*T_hat/p_T)/C0`, with `p_T=1/2`, plus the unchanged dense fidelity term. When T is omitted, its family callable, raster, filtrations, and GUDHI pairing are not executed. A skipped T is logged as missing, with its execution identity/probability retained. Both the inner `T_estimate` and the outer importance contribution are named explicitly. Calibration and evaluation never receive the training execution plan. Independent counter-seeded family/filtration streams depend on the accepted cursor and do not consume data, sensor, or model-noise RNG state.

Unbiasedness applies to the scalar gradient estimator before clipping and AdamW. It does not make the nonlinear ConFIG map of sampled family gradients unbiased, and it does not equate stochastic, clipped, or AdamW trajectories. All sampled-topology arms share that approximation for their matched comparisons.

ConFIG coefficient extraction uses a reduced QR/SVD on normalized full-width FP64 gradients and the full-width pseudoinverse rank threshold. It avoids normal-equation rank recovery. The new oracle rank diagnostic is authoritative for this check; the older aggregator's Gram-based rank flag remains diagnostic only and does not drive the update. Real SOURCE reconstruction errors are below 7e-16; signed-coefficient fixtures and an AdamW displacement comparison also pass. Actual SOURCE probe coefficients happened to be positive; that does not justify forcing other coefficients positive.

In the fixed-SOURCE P3 replay, holding the batch-0 coefficients gives direction cosines 1.000, 0.961, 0.971, and 0.891 across four fresh batches. Reducing the refresh interval to two still produces a 0.800 cosine on the second refresh/reuse pair. These probes include minibatch, reconstruction-noise, and filtration changes, but no optimizer updates; they do not establish moving-model staleness bounds. Both cadences fail the engineering audit threshold 0.9 for all declared reuse pairs, so the periodic arm is excluded. The default-off implementation has sparse current-gradient audits and requests a fresh oracle refresh when severe staleness is detected; those safeguards do not turn the failed pretest into a supported acceleration.

## Short timing and quality evidence

| Arm / calculation | Command wall, min | Coordinated process, min | Trainer, min | Epoch 1 / 2 wall, s | Epochs 3–10 wall median [range], s | Peak allocated, GiB | T-active batches |
|---|---:|---:|---:|---:|---:|---:|---:|
| T0: Full T + dense ConFIG | 17.55 | 17.39 | 16.78 | 103 / 97 | 98.0 [97–122] | 14.28 | 380/380 |
| T1: P3 + dense ConFIG | 14.33 | 14.25 | 13.71 | 83 / 79 | 80.0 [79–101] | 14.25 | 380/380 |
| T2: P3 + scalar sum | 9.34 | 9.26 | 8.71 | 53 / 49 | 50.5 [50–68] | 12.91 | 380/380 |
| T3: P3 + scalar sum + family p=1/2 | 7.81 | 7.68 | 7.13 | 44 / 39 | 42.5 [37–57] | 12.91 | 192/380 |

Command wall includes launcher preflight. T0/T1 command receipts resolve to whole seconds; T2/T3 additionally have external monotonic timing. T0 includes its failed bootstrap attempt (five command seconds, 0.179 coordinated seconds, zero training exposure). Coordinated process time begins after campaign admission and includes child initialization/calibration, training, routine full validation, and finalization; the separate 64-snapshot command audit is excluded from timing comparisons and included in the resource ledger. Trainer time starts later and is not the inclusive cost. Epoch wall times include reporting/checkpoint/validation work; the final epoch includes finalization. The retained JSON also contains the corresponding training-only epoch times.

The first two epochs include the unchanged warm-up. Epoch wall times come from the trainer's completed-epoch lines, rounded to seconds, including its checkpoint/validation/reporting scope. The 10-epoch inclusive process timer additionally covers SOURCE loading/calibration, initial evaluation, and finalization. Mean update timers are retained separately and are not called complete training cost. Initialization, cache warm-up, refresh/skip work, and failures are charged rather than removed from the budget.

All timing children are fresh SOURCE children. Target pairing caches use the same capacity (4,096 entries) and worker policy. Each process starts its own in-memory pairing cache; compiled KeOps kernels are shared through the owned campaign cache, warmed by the recovery smoke before T0. All profiles use the same current code and scientific descriptor primitives. Timing is a single sequential pass per profile on one physical GPU, with a shared host and no repeated-arm uncertainty estimate. Differences below the practical 15% inclusive saving target remain inconclusive for deployment screening.

| Arm | Epoch | Full A / SOURCE | Full B / SOURCE | Full T / SOURCE | Aggregate endpoint / SOURCE | Worst field ratio | Strict / engineering corridor | Useful target at check |
|---|---:|---:|---:|---:|---:|---:|---|---|
| T0 | 5 | 0.9145 | 0.8683 | 0.4774 | 0.9562 | 1.0092 | pass / pass | yes |
| T0 | 10 | 0.6698 | 1.1120 | 0.4345 | 0.9005 | 1.0454 | pass / pass | no |
| T1 | 5 | 1.2415 | 0.9688 | 0.4910 | 1.1250 | 1.2293 | fail / fail | no |
| T1 | 10 | 0.9186 | 0.9073 | 0.4997 | 1.0008 | 1.0448 | pass / pass | yes |
| T2 | 5 | 1.1055 | 0.9166 | 0.4220 | 1.0796 | 1.1289 | fail / fail | no |
| T2 | 10 | 0.7950 | 0.9297 | 0.4101 | 0.9406 | 1.0346 | pass / pass | yes |
| T3 | 5 | 0.9144 | 0.9115 | 0.5031 | 0.9670 | 1.0509 | fail / pass | no |
| T3 | 10 | 0.8372 | 0.9177 | 0.4305 | 0.9477 | 1.0425 | pass / pass | yes |

All ratios above use the same fixed first-32 validation SOURCE baseline; epoch zero is 1.0000 in every column. Duplicate identical final validation rows are reduced by accepted epoch. The useful-target flag is only a check-level observation, requiring strict feasibility, at least 10% T reduction, and A/B within +5%; it does not establish mature-window retention.

| Arm | Epoch | CH4 | CO | T | U_1 | p |
|---|---:|---:|---:|---:|---:|---:|
| T0 | 5 | 0.9592 | 1.0092 | 0.9831 | 0.9401 | 0.9470 |
| T0 | 10 | 0.9930 | 1.0454 | 1.0249 | 0.9203 | 0.7568 |
| T1 | 5 | 0.9857 | 1.0465 | 1.0063 | 1.1124 | 1.2293 |
| T1 | 10 | 0.9795 | 1.0448 | 1.0098 | 1.0071 | 0.9698 |
| T2 | 5 | 0.9728 | 1.0275 | 0.9950 | 1.0869 | 1.1289 |
| T2 | 10 | 0.9655 | 1.0346 | 1.0037 | 0.9530 | 0.8568 |
| T3 | 5 | 0.9841 | 1.0509 | 1.0029 | 0.9369 | 0.9582 |
| T3 | 10 | 0.9871 | 1.0425 | 1.0174 | 0.9365 | 0.8956 |

Endpoint ratios are model-unit field MSE, child/SOURCE. T0 selects epoch 10; T1 selects epoch 10; T2 selects epoch 10; T3 selects epoch 10. Full selected and final metrics are retained separately. T1’s epoch-5 U_1/p excursion is visible; its separate best-fidelity checkpoint remains at SOURCE epoch zero, whereas its topology-selected and final states are epoch 10. No ten-epoch checkpoint is promoted into a 100-epoch result.

| Arm | Exact child path below payload root |
|---|---|
| T0 | `runs/T0/20261023T021811Z_f7a4ad72` |
| T1 | `runs/T1/20261023T024006Z_433b002f` |
| T2 | `runs/T2/20261023T030742Z_9cadd70e` |
| T3 | `runs/T3/20261023T031859Z_ffea7252` |

No arm meets the useful target at both saved nonzero checks. T3’s epoch-5 CO ratio is 1.0509: a strict miss that remains within the separate engineering corridor. These ten-epoch outcomes check gross instability and provide full-metric context for timing. They do not select a scientific winner. Training estimates are not substituted for full validation metrics, and a small fidelity miss is kept visible. Strict +5% per-field/aggregate feasibility and the separate +5% aggregate/+10% per-field engineering corridor retain their distinct meanings.

The predeclared common useful target for the future 100-epoch study is strict endpoint feasibility, at least 10% full-T reduction, and A/B no more than 5% above SOURCE, sustained at two routine checks. Its mature-window time-to-quality comparison is pending. The JSON audit uses uniform validation snapshots in complete groups of 32. B is the mean of B(32) ensemble statistics, not B computed once on the entire larger panel. Dense endpoint and per-field MSE pool exactly with fully valid equal-sized groups; worst-field relative L2 is recomputed from pooled field scores. Observed/unobserved-entry MSE stays in individual group records because sensor counts vary. This panel is distinct from the historical first-32 selection panel and must not be merged into a historical leaderboard. The T0 final checkpoint completed a real 64-snapshot CLI audit in 11.20 coordinated process seconds; its A/B/T ratios are 0.903/1.042/0.472 and endpoint field ratios are 0.976/1.010/0.990/1.023/0.982. That single-arm command check is not an accelerated-arm generalization comparison. Its immutable JSON is `analysis/T0_last_epoch10_n64_8427446ffd19_20261023T030504200236Z_validation.json`.

## Compatibility, recovery, and resource accounting

The real deterministic compatibility SOURCE is `cases/turbulent_combustion/runs/tc_deeponet_5000ep/20260829T125329Z_2ebc837b/checkpoints/last.pt`. Its existing model, observation contract, dataset, and normalizer were inherited. The one-epoch smoke uses the same endpoint policy without RF hooks, retains nonzero trainable gradients, and verifies SOURCE immutability. It establishes adapter compatibility, not deterministic-model efficacy or superiority.

The separate flow recovery smoke has a two-epoch horizon. It stopped at epoch one with an atomic checkpoint, then resumed the same unchanged arm through epoch two. Calibration, multipliers, RNG/cursors, implementation bindings, normalizer, and source references were recovered; the fast per-batch log contains no duplicate accepted cursor after history recovery. It is separate from the fresh timing children and is not extended automatically. Focused fixtures also check periodic coefficient-state recovery and a repository-link RunStore round trip.

| Resource | Actual exposure / charge | Limit |
|---|---:|---:|
| Timing profiles | 4 × 10 = 40 epochs | ≤5 × 10 = 50 |
| Adapter + recovery smokes | 1 + 2 = 3 epochs | 6 |
| Actual total training | 43 epochs | 556 |
| Conservative epoch accounting, including reserved smokes | 46 epochs | 556 |
| Goal-1 / Goal-2 100-epoch arms | 0 / 0 | ≤4 / ≤1 |
| Extra full64 validation command | 11.20 s | Within evaluation reserve |
| Offline GPU diagnostic reservation | 180 s, conservative upper charge | 2,700 s |
| Smoke process reservation | 900 s, conservative upper charge | Included below |
| Combined GPU-use charge | 1.1128 h | 16 h |

The campaign runner refuses occupied GPU 1, different arm/source/config bindings, more than five timing profiles, more than 50 timing epochs, a lineage horizon above 100, or insufficient 556-epoch/16-hour capacity after reserving evaluation and smoke work. Its watchdog charges the complete owned process, with a safe-stop margin. Interrupted attempts are charged conservatively when their final accepted exposure is unknown; resume never silently erases discarded work. Public status distinguishes observed accepted epochs from conservative charges/reservations. Training and evaluation share one persistent `flock` and a locked GPU-idle recheck. Stop waits for the verified child and coordinator to release the reservation. No process can automatically start another arm.

All numerical payloads, processed outputs, compiler/Python/Matplotlib caches, and test temporaries reside at `/data2/wanglz/PhyCoFlow-MultiFieldReconstruction/experiments/LPCAccel_20261022_01a16674`, verified on mounted ext4 `/dev/nvme0n1p1`. The run link is `cases/turbulent_combustion/runs/_LPCAccel/LPCAccel_20261022_01a16674`; the report link is `reports/LPCAccel_20261022_01a16674`. Only short multiprocessing Unix-socket directories use `/tmp`, because Linux rejects longer AF_UNIX paths. Historical SOURCE and run payloads were not migrated, overwritten, or cleaned.

## Code mappings and validation

| Boundary | Implementation |
|---|---|
| Complete filtration enumeration, P3/S2, exact category weights, raster adapter | `src/phycoflow_reconstruction/coherence/families/topology/filtration_sampling.py` |
| Independent execution streams, strict opt-in combinations, implementation binding | `src/phycoflow_reconstruction/training/fast_lagrangian.py` |
| One shared endpoint, genuine T omission, full calibration/evaluation, truthful estimate history | `src/phycoflow_reconstruction/training/post_training.py` |
| Existing dense fidelity/dual composition; oracle refresh capture and periodic recovery | `src/phycoflow_reconstruction/training/fidelity_budget.py` |
| Stable signed coefficient recovery, fresh-loss reuse, default-off residual helper | `src/phycoflow_reconstruction/training/periodic_config.py` |
| Strict estimator/config/route validation | `src/phycoflow_reconstruction/config/validate.py` |
| Truncate speculative per-batch estimator logs at recovery checkpoint | `src/phycoflow_reconstruction/training/run_store.py` |
| Prepare/status/explicit foreground launch/stop/resume/evaluation and campaign bounds | `scripts/posttraining/fast_lagrangian_campaign.py` |
| Real SOURCE TRAIN estimator/coefficient probe | `scripts/posttraining/fast_lagrangian_pretest.py` |
| One-epoch deterministic and two-epoch flow recovery smokes | `scripts/posttraining/fast_lagrangian_smoke.py` |
| Compact full-descriptor validation, exact 32-snapshot groups, no plot exports | `scripts/posttraining/fast_lagrangian_evaluate.py` |

The complete accelerated-history audit checks all 380 accepted minibatches in each of T1–T3: every batch retains five finite endpoint risks, active dense fidelity, continuous multipliers, and the original projected dual formula (maximum absolute error below 3e-17). Sampling decisions replay exactly from the accepted cursor. T3 executes T on 192/380 batches; all 188 omissions retain missing topology values. `analysis/accelerated_dense_update_audit.json` stores these checks. T0 uses the unchanged original path and has no new per-batch audit log.

The final relevant CPU regression suite passed 251 tests with one CUDA-only test skipped in the CPU run. That warm-up/primal-dual oracle test was subsequently run alone on idle GPU 1 and passed. The final launcher/evaluator suite separately passed all 24 targeted tests after the admission, locking, stop, output-identity and pooling fixes. The retained validation receipts record these distinct checks. Coverage includes all-options-off policy behavior, original topology value/gradient oracle, exact P3/S2 expectations, signed/rank-degenerate coefficient examples, current-gradient reuse, interval-one oracle/AdamW equivalence, dense dual updates, true T omission, deterministic endpoint dispatch, strict recovery bindings, safe links, atomic checkpoint recovery, and launch/split/cap exclusions. Ruff and diff checks are also recorded. Numerical tests and real-SOURCE smokes are distinct from scientific learning evidence. The campaign manifest pins all 142 numerical Python files, including the update integration, original descriptors, model and normalizer code; campaign launch/resume/evaluation rejects a changed source tree. Native checkpoints separately bind the three new helper files only, so direct native resumes require the frozen integration version. P3/S2 support the verified cubical Run-11 topology profile, not arbitrary replacement topology definitions.

## User-operated commands

Run these from the repository root with `phycoflow_env`. The commands print an action by default; `--execute` runs exactly one foreground action. They do not queue a successor. The wrapper invokes the existing public `cases/turbulent_combustion/run.py post-train --config CONFIG` entry point; training is not implemented in a separate campaign trainer. The prepared G1 configurations remain fixed at 100 epochs and original SOURCE initialization; the excluded periodic arm remains blocked.

```bash
export LPC_CAMPAIGN=/data2/wanglz/PhyCoFlow-MultiFieldReconstruction/experiments/LPCAccel_20261022_01a16674
conda run --no-capture-output -n phycoflow_env python scripts/posttraining/fast_lagrangian_campaign.py --root "$LPC_CAMPAIGN" status
```

The next user-operated study is the matched G1-C/G1-S comparison; G1-R additionally isolates randomized-family scalar training if its measured cost is useful. G1-P remains excluded. No recipe has 100-epoch quality support yet. The timing receipt is recorded; the separate Goal-1 readiness proposal does not enable or launch anything until the user applies it. No long launch was executed here.

```bash
# Review the proposed arms, measured cost projections and reserve first.
cat "$LPC_CAMPAIGN/manifests/goal1_readiness.proposed.json"
conda run --no-capture-output -n phycoflow_env python scripts/posttraining/fast_lagrangian_campaign.py --root "$LPC_CAMPAIGN" authorize goal1 --evidence "$LPC_CAMPAIGN/manifests/goal1_readiness.proposed.json"

# One explicit foreground arm; choose G1-C, G1-S or eligible G1-R.
conda run --no-capture-output -n phycoflow_env python scripts/posttraining/fast_lagrangian_campaign.py --root "$LPC_CAMPAIGN" launch G1-S --execute
```

The resolved candidates use the existing standard trainer. Its portable syntax is `python cases/turbulent_combustion/run.py post-train --config CONFIG [--resume OWN_RUN]`; the tested campaign wrapper supplies physical-GPU admission, mounted-cache paths, the short socket bootstrap, environment settings and budget accounting around that call. The two-epoch recovery smoke exercised this same trainer's own-state resume. Direct calls do not enforce the campaign's resource ledger, so the bounded-study commands above use the wrapper.


To interrupt the currently verified campaign-owned training process, use the following command. It does not signal unrelated jobs. Recovery uses the last atomic rolling checkpoint and replays only the same configured arm; discarded exposure remains charged. A checkpoint older than the interruption is not an exact continuation from the final attempted batch.

```bash
conda run --no-capture-output -n phycoflow_env python scripts/posttraining/fast_lagrangian_campaign.py --root "$LPC_CAMPAIGN" stop
conda run --no-capture-output -n phycoflow_env python scripts/posttraining/fast_lagrangian_campaign.py --root "$LPC_CAMPAIGN" resume G1-S --run "$LPC_CAMPAIGN/runs/G1-S/REPLACE_WITH_OWN_RUN" --execute
```

Full-descriptor validation uses the original dense evaluator and the matched SOURCE on the same uniform panel. It writes compact JSON under the campaign analysis directory and cannot select TEST. Evaluate both final and existing selected checkpoints; retain complete histories for the mature-window comparison.

```bash
conda run --no-capture-output -n phycoflow_env python scripts/posttraining/fast_lagrangian_campaign.py --root "$LPC_CAMPAIGN" evaluate G1-S --run "$LPC_CAMPAIGN/runs/G1-S/REPLACE_WITH_OWN_RUN" --checkpoint last --max-samples 64 --execute
conda run --no-capture-output -n phycoflow_env python scripts/posttraining/fast_lagrangian_campaign.py --root "$LPC_CAMPAIGN" evaluate G1-S --run "$LPC_CAMPAIGN/runs/G1-S/REPLACE_WITH_OWN_RUN" --checkpoint best --max-samples 64 --execute
```

## Figures, retained evidence, and remaining work

[Filtration estimators and signed coefficient staleness](reports/LPCAccel_20261022_01a16674/filtration_and_coefficient_pretests.pdf)

<object data="reports/LPCAccel_20261022_01a16674/filtration_and_coefficient_pretests.pdf" type="application/pdf" width="100%" height="520" aria-label="Filtration estimators and signed coefficient staleness"><a href="reports/LPCAccel_20261022_01a16674/filtration_and_coefficient_pretests.pdf">Open PDF master: Filtration estimators and signed coefficient staleness</a></object>

One real SOURCE TRAIN endpoint: complete-work-unit cost and exact finite-support gradient variance; fresh-batch P3 reuse comparisons on a fixed model. The 0.9 engineering line exposes the failed interval-two reuse pair. These are descriptor/replay scopes, not complete training speedups.

[Bounded timing and fieldwise validation](reports/LPCAccel_20261022_01a16674/bounded_timing_and_validation.pdf)

<object data="reports/LPCAccel_20261022_01a16674/bounded_timing_and_validation.pdf" type="application/pdf" width="100%" height="520" aria-label="Bounded timing and fieldwise validation"><a href="reports/LPCAccel_20261022_01a16674/bounded_timing_and_validation.pdf">Open PDF master: Bounded timing and fieldwise validation</a></object>

Four complete fresh ten-epoch user commands with unchanged epoch exposure. Complete epoch wall times include validation/checkpoint/finalization work. Full final A/B/T and each endpoint field are shown against the matched SOURCE; they provide short-run context, not 100-epoch quality retention.

[Complete short validation histories](reports/LPCAccel_20261022_01a16674/short_validation_histories.pdf)

<object data="reports/LPCAccel_20261022_01a16674/short_validation_histories.pdf" type="application/pdf" width="100%" height="520" aria-label="Complete short validation histories"><a href="reports/LPCAccel_20261022_01a16674/short_validation_histories.pdf">Open PDF master: Complete short validation histories</a></object>

Full first-32 validation at epochs 0, 5 and 10 for all four timing arms. The worst field ratio retains temporary fidelity excursions, including T1 at epoch 5. Duplicate final records are reduced by accepted epoch. The mature window and comparative time-to-quality remain pending.


Numerical sources and exact run paths are retained in `analysis/timing_summary.json`, `analysis/pretest_full.json`, `analysis/pretest_p3.json`, `manifests/preflight.json`, `manifests/campaign.json`, `manifests/launch_ledger.jsonl`, `manifests/smoke_receipt.json`, and the per-run configuration/history/evaluation/checkpoint files. The figure regeneration script is `analysis/build_report_figures.py`. The planned 100-epoch field/error maps and LPC diagnostic master remain deferred with their unexecuted scientific stages; the short-run PDFs do not stand in for them. Within the current campaign all user-operated lineages remain capped at 100 epochs. A larger user-selected horizon belongs to a separate explicit future study, after the quality decision; it is neither prepared as an automatic continuation nor recommended by these short results. Only owned disposable fixtures, redundant generated presentation images, and QA rasters are eligible for cleanup. Original numerical evidence and recovery state are retained. Cleanup and final delivery receipts are `manifests/cleanup_result.json` and `manifests/delivery.json`; the latter records the pushed branch/commit and protected-input verification.

Before Goal 2, complete the eligible Goal-1 arms and assess full histories, strict fieldwise fidelity, the mature window, and common-target time. Only then perform the small real-model residual-gradient diagnostic. Any optional ordered arm must minimize full-defined T/T0 with dense A/B no-regression and endpoint constraints, fixed full SOURCE calibration, and at most 100 fresh epochs. It changes objective preferences and is not a conditional KL estimator. No conditional estimators, descriptor regressors, new descriptor definitions, new optimizers, flow-only methods, or unrecorded continuation are included here.

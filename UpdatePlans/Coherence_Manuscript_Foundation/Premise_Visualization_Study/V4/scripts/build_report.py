#!/usr/bin/env python3
"""Assemble the English evidence library from reviewed, saved measurements."""
from pathlib import Path
import json
import subprocess
import numpy as np
import pandas as pd

V4 = Path(__file__).resolve().parents[1]
STUDY = V4.parent
NAMES = {'deeponet': 'DON (DeepONet)', 'diffusion_pde': 'DiffusionPDE (EDM)',
         'geofno': 'GeoFNO', 'mimonet': 'MIMONet', 'senseiver': 'Senseiver',
         'gl_rbf_cq': 'FFM (GL-RBF-CQ)'}
FIELDS = ('CH4', 'CO', 'T', 'U_1', 'p')
SHORT = {'deeponet': 'DON', 'diffusion_pde': 'EDM', 'geofno': 'GeoFNO',
         'mimonet': 'MIMONet', 'senseiver': 'Senseiver', 'gl_rbf_cq': 'FFM'}
EVENT_LABELS = {'lowCH4_highCO': 'Low CH4 / high CO',
                'transitionCH4_highCO': 'CH4 transition / high CO',
                'highCO_hotT': 'High CO / hot T', 'lowCH4_hotT': 'Low CH4 / hot T'}


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |',
                      '| ' + ' | '.join(['---'] * len(headers)) + ' |',
                      *['| ' + ' | '.join(map(str, row)) + ' |' for row in rows]])


def main():
    metrics = V4 / 'data/metrics'
    c = pd.read_csv(metrics / 'candidates.csv')
    events = json.loads((V4 / 'metadata/events_summary.json').read_text())
    selected = json.loads((metrics / 'selected_examples.json').read_text())
    inventory = json.loads((V4 / 'figures/inventory.json').read_text())
    definitions = json.loads((metrics / 'definitions.json').read_text())
    models = definitions['models']
    for name in models:
        NAMES.setdefault(name, name.replace('_', ' '))
        SHORT.setdefault(name, name.replace('_', ' '))
    with np.load(STUDY / 'data/gt_test100.npz') as f:
        gt, scale, offset = f['fields'], f['scale'], f['offset']
    identities, fidelity = [], []
    for name in models:
        m = json.loads((V4 / f'metadata/{name}.json').read_text())
        identities.append([SHORT[name], f"{m['epoch']:g}", m['global_step'],
                           m['evaluation_weights'], 'verified prior arrays' if 'previous' in m['source'] else 'new extraction',
                           f"[identity](metadata/{name}.json)"])
        with np.load(V4 / f'data/{name}_test100.npz') as f:
            z = f['fields'].astype(float)
        l2 = np.linalg.norm((z-gt)*scale, axis=(1, 2)) / np.linalg.norm(gt*scale+offset, axis=(1, 2))
        fidelity.append([NAMES[name], *[f'{100*v:.2f}%' for v in np.median(l2, axis=0)]])
    matched = c[c.matched_20 & c.state_support]
    material = matched.state_gap_gt_y_sd.abs() >= .05
    stable = material & matched.state_stable
    copula = (matched.copula_tv_a - matched.copula_tv_b).abs() >= .1
    summaries = []
    for gate in ('all', 'matched_40', 'matched_20', 'matched_10'):
        q = c if gate == 'all' else c[c[gate]]
        q = q[q.state_support]
        hit = q.state_gap_gt_y_sd.abs() >= .05
        dom = q.l2_dominance != 0
        inv = hit & dom & ((q.state_gap*q.l2_dominance) < 0)
        summaries.append([gate, len(q), int(hit.sum()), int((hit & q.state_stable).sum()),
                          int(((q.copula_tv_a-q.copula_tv_b).abs() >= .1).sum()),
                          f'{int(inv.sum())}/{int(dom.sum())}'])
    event_rows = []
    for r in events['selection']:
        a, b = r['records']
        event_rows.append([r['frame'], EVENT_LABELS.get(r['event'], r['event']),
                           f"{SHORT[r['model_a']]} / {SHORT[r['model_b']]}",
                           f"{100*a['l2_a']:.1f}/{100*a['l2_b']:.1f}%",
                           f"{100*b['l2_a']:.1f}/{100*b['l2_b']:.1f}%",
                           f"{a['joint_iou']:.3f}/{b['joint_iou']:.3f}"])
    parts = []
    figure_index = []

    def add(text):
        parts.append(text.strip() + '\n')

    def figure(stem, caption, alt=None):
        index = len(figure_index) + 1
        figure_index.append((index, stem, caption))
        add(f'![{alt or caption}](figures/{stem}.png)\n\n**Figure {index}.** {caption} [PDF master](figures/{stem}.pdf).')

    add('# Coupled-field relationships beyond reconstruction error: a combustion evidence library\n\n29 October 2026 · V4 · Paper-premise materials, complete candidate banks and reusable figure sources')
    add(f'''## What this library establishes

Multi-field reconstruction has two distinct evaluation targets: individual fields should approach their references, and the reconstructed fields should retain the observed organization of their joint state. A vector of fieldwise L2 errors does not determine that organization. This library combines verified constructive controls, native-coordinate coupled-region views, and a complete {len(models)}-model comparison census to make that distinction inspectable.

The strongest new field-first examples are frames **9544** and **9111**: the model with higher error in both relevant fields nevertheless preserves the selected joint region better. In frame 9544, FFM has CO/T native-area-weighted L2 of 47.8%/15.5%, versus DiffusionPDE's 42.8%/14.0%, but high-CO/hot-T joint IoU is 0.199 versus 0.077. The unary CO and T region IoUs remain close. The point is a concrete disagreement between fieldwise accuracy and a coupled-state query, rather than a claim that either reconstruction is nearly exact.

Across the complete 100-frame cohort, {int(material.sum()):,}/{len(matched):,} comparisons matching both fieldwise Euclidean L2 values within 20% have a conditional-relationship error contrast of at least 0.05 reference Y-field standard deviations; {int(stable.sum()):,} also retain the model ordering across all five bin resolutions. A separate rank-based diagnostic finds {int(copula.sum()):,}/{len(matched):,} with at least 0.10 difference in rank-copula histogram TV error. These descriptive comparisons share frames, field pairs and models. Their counts characterize this fixed cohort, not independent trials or universal architecture failure rates.

The deliverable is deliberately broader than a final manuscript figure: five controlled galleries, seven coupled-region comparisons, eight detailed State comparisons, full-cohort diagnostics, and estimator/attribution audits. Every displayed preview has a PDF master. Exact arrays and selection decisions remain separate from plotting, so the evidence can be re-ranked and redrawn.''')
    add('## 1. The paper argument: LPC motivates observable questions, and SSS makes them testable')
    add('''The target is a conditional joint physical state inferred from sparse partial observations. Under the analyticity and existence assumptions in the theory document, LPC partitions mixed cumulants by order, distinct sites and distinct fields: Local collects one-site mixed terms; Pair collects two-field/two-site second-order terms; Collective collects the remaining mixed terms spanning multiple sites. This is an exact organization under its assumptions. The operational State–Scale–Shape closure selects finite observations of that richer law; it is not an invertible replacement for the law.

State asks which values occur together. Scale asks how amplitudes and signed alignment organize in spatial modes. Shape asks how field-defined regions intersect, connect and persist. A physically useful query should therefore point back to a recognizable field configuration: the CO-rich band relative to a CH4 transition, CO-rich material inside a hot region, or the continuity of a joint excursion. The present study assesses static snapshot organization. It does not identify reaction rates, causal coupling or satisfaction of governing equations.

A simple mechanism makes the premise more than a metric preference. If two unresolved fields satisfy V = U², their squared-error-optimal conditional means satisfy E[V|Y] − E[U|Y]² = Var(U|Y). With nonzero conditional variance, the pair of mean estimates leaves that nonlinear relation. This establishes a possible failure under explicit conditions; a generative model's single draw is not presumed to be a conditional mean. The strict controls below independently establish matched-error counterexamples without requiring that training-mechanism assumption.

Three distinctions prevent the diagnostics from becoming labels without content. A raw joint-distribution error includes marginal changes; the rank-copula audit separately probes rank association. A unary spectrum does not determine cross-field phase. A joint mask defect can inherit unary defects, so unary masks and their errors must remain visible. Perfect reconstruction would preserve suitable continuous coherence queries; the premise concerns finite and often appreciable errors, not an inconsistency at zero error.

Sources: [LPC theory](../../01_Theory/LPC_THEORY.md), [SSS computation](../../02_Computation/SSS_COMPUTATION.md), [post-training interface](../../03_Posttraining/POSTTRAINING_STRATEGY.md).''')
    add('## 2. Constructive evidence: hold fieldwise error fixed, then inspect relationships')
    add('''The strict controls progressively add preserved quantities. The nonlinear State construction matches each field MSE and shares the same reconstructed first field; one candidate satisfies v = u² while the other does not. The periodic Scale construction additionally preserves unary Fourier power exactly while relative phase changes. The Shape construction retains unary Betti numbers across fifteen thresholds while joint connectivity is one versus two. State and Shape do not claim identical marginals. Exact periodic Fourier identities do not transfer to the nonuniform combustion grid.

The real-field permutation controls preserve both unary empirical distributions and equal per-field Euclidean errors algebraically, while common and opposed operations change same-site pairing differently. They intentionally include cyclic index seams and unequal physical displacements on this nonuniform grid. The four calibrated noise/convolution/phase/warp families are output sensitivity tests. Their selective and sometimes weak responses are retained rather than treated as universal failures.''')
    control_caps = {
        'controls_strict_toys': 'Strict synthetic controls hold each field MSE fixed while nonlinear State, signed cross-spectrum, and joint connectivity differ. The Scale toy preserves unary power; the Shape toy preserves unary Betti numbers. These are separate contracts.',
        'controls_real_state': 'A common versus opposed cyclic rearrangement of frame 9575 preserves both unary empirical distributions and identical field errors while changing CH4–CO pairing. The wrap seam and substantial CO error are explicit constructed features, not physical transport.',
        'controls_permutation_sweep': 'Ten uniformly sampled saved frames, three field pairs and five cyclic shift sizes show how identical field errors and unchanged unary distributions permit different rank-binned joint occupation and signed correlation. Faint curves are frames and bold curves medians; shaded MSE ranges are not confidence intervals.',
        'controls_strict_sweep': 'A phase-amplitude sweep retains unary Fourier power and shared/opposed field-error equality while cross-products diverge. The independent Shape construction retains unary Betti numbers but different joint connectivity over fifteen thresholds. Large sweep amplitudes have large field errors.',
        'controls_real_families': 'Four calibrated synthetic intervention families on frame 9575 match all five normalized MSE values and restore observed temperature exactly. Their correlation responses differ and can be small. A visibly different error texture alone does not prove a large coupled-state defect.'}
    for stem, caption in control_caps.items():
        figure(stem, caption)
    add('Control source data and checked contracts: [controls.npz](data/controls.npz), [numerical ledger](metadata/controls.json). The sweep retains chart-ready measurements rather than every shifted field.')
    add('## 3. Frozen model and data contracts')
    add('''All candidate weights are the requested **last** checkpoint, frozen before the new extraction. A nominal `5000ep` directory does not establish completed training. DON, MIMONet and Senseiver are at epoch 5000; the requested GeoFNO is at epoch 2580 and the requested DiffusionPDE at epoch 1400. FFM is retained as the original reference endpoint. Identical previous predictions are reused only after matching checkpoint, configuration and prediction hashes. The three new extractions are DON, GeoFNO and DiffusionPDE.

The common cohort is the original 100 uniformly spaced frames in chronological TEST global IDs 9000–9999, at native nonuniform 100×403 sites and field order (CH4, CO, T, U_1, p). Each frame uses the same 256 saved temperature observations, original TRAIN normalization, full ordered native queries and original deterministic seed protocol. MIMONet retains ascending point-index sensor packing. Generative endpoints use one fixed draw, so this is not a conditional-diversity assessment.''')
    add(table(['Model', 'Checkpoint epoch', 'Updates', 'Weights', 'Endpoint source', 'Record'], identities))
    add('''The model names describe the actual local adapters. DON uses branch/trunk basis reconstruction. MIMONet uses multiplicative branch/trunk merging and its T256 point-index interface. Senseiver uses latent cross-attention. The local GeoFNO adapter uses a rasterized neuraloperator FNO; it does not introduce a learned geometry warp. This requested DiffusionPDE run is explicitly **EDM**, with conditional U-Net, EMA, bf16 and 32-step Heun inference; it is different from the older DDPM/DDIM run discussed in earlier audits. The FFM endpoint uses EMA and four Euler steps. The source audit does not assert equivalence between a local model label and every part of an external original method.

Native observation treatment remains part of the contract: FFM retains endpoint consistency plus its final T clamp; EDM conditions and clamps observed T during denoising/final output; deterministic regressors receive no added clamp. Equal measurement sets therefore do not imply identical hard observation enforcement. Dense target fields are absent from new model inference inputs. [Source/code audit](metadata/model_audit.json), [frozen checkpoint identities](metadata/checkpoints.json).''')
    add('### Actual fieldwise fidelity of the cohort')
    add(table(['Model', 'CH4 median L2', 'CO median L2', 'T median L2', 'U_1 median L2', 'p median L2'], fidelity))
    add('These are unweighted native-node decoded Euclidean relative L2 values. CO and velocity errors remain substantial. Pressure has a large mean, so its small relative L2 is not evidence of accurate pressure fluctuations. No headline example is selected on pressure L2 alone. Dataset field/coordinate units have not been independently established; numeric temperature levels are dataset values.')
    add('## 4. Field-first atlas: where coupled states are retained, lost or invented')
    add('''The event atlas starts from four empirical reference questions: CO-rich material in low CH4; CO-rich material inside the CH4 transition; CO-rich material in a hot region; and hot material in low CH4. CH4 cuts use 25% and 75% of the GT value range, avoiding tied-floor quantile artifacts. CO-rich is above the area-weighted GT 90th percentile and hot is above the area-weighted GT 75th percentile. Every state uses these same GT values; predicted quantiles are not used to force equal occupied area. Actual support is recorded.

Read each page from continuous fields to partner contours and then the joint footprint. Solid white contours show that column's partner region at the common GT cut; dashed gray contours retain the reference partner boundary. Green retains reference joint area, orange identifies missing joint area, and purple identifies extra predicted joint area. These are coupled-region geometry queries, not topology or a complete chemical constraint.

This atlas deliberately reports native-area-weighted decoded L2 alongside native-area region metrics. The full cohort and State atlas use unweighted decoded Euclidean L2. Both conventions are valid declared queries, but their values and match gates must not be substituted for each other.''')
    add(table(['Frame', 'Coupled query', 'Models A / B', 'A field L2', 'B field L2', 'Joint IoU A/B'], event_rows))
    event_order = [5, 6, 0, 1, 2, 3, 4]
    for idx in event_order:
        r = events['selection'][idx]
        a, b = r['records']
        stem = f"events_{idx+1:02d}_{r['event']}"
        caption = (f"Frame {r['frame']}, {r['event']}, {NAMES[r['model_a']]} versus {NAMES[r['model_b']]}. "
                   f"Native-area field L2 is {100*a['l2_a']:.1f}%/{100*a['l2_b']:.1f}% versus {100*b['l2_a']:.1f}%/{100*b['l2_b']:.1f}%; joint IoU {a['joint_iou']:.3f} versus {b['joint_iou']:.3f}. "
                   f"Unary region IoUs are {a['unary_a_iou']:.3f}/{a['unary_b_iou']:.3f} versus {b['unary_a_iou']:.3f}/{b['unary_b_iou']:.3f}; GT joint support is {100*a['joint_gt_area']:.2f}% of native area. "
                   'All displayed model pairs meet the 20% per-field area-L2 matching gate. Solid white is the current partner boundary; dashed gray is its GT reference.')
        if idx in (5, 6):
            caption += ' The higher-L2 model has higher joint IoU in both-field dominance, while unary IoUs are close; they are not exactly controlled.'
        if idx == 0:
            caption += ' MIMONet nearly loses the CO-rich unary region, so this dramatic joint failure has a clear unary contribution.'
        if idx == 1:
            caption += ' The reference transition intersection is relatively small and should not be presented as a domain-wide event.'
        figure(stem, caption)
    add(f'''Frame 9544 is a useful opening because the two CO and T region IoUs are similar yet their intersection overlaps the reference differently. Frame 9111 supplies a second model pairing and reverses the model role: DiffusionPDE is worse in both field errors but better for the low-CH4/high-CO footprint. Together with the preserved-marginal controls and rank-copula audit, these examples support a relational premise beyond comparing field error magnitudes. They still do not isolate every unary contribution.

Other atlas pages are retained for different purposes. Frame 9393 shows why a spectacular joint-mask loss can partly arise from unary amplitude failure. Frame 9898 makes the narrow transition query visible. The lower-CH4/hot-T pages supply lower-error alternatives and model coverage. The selections are exploratory, explicitly indexed, and accompanied by all {400*len(models):,} event-state rows and {400*len(models)*(len(models)-1)//2:,} event/model-pair rows. Across the four declared events there are {events['inversion_candidates_match20']} support-qualified, 20%-matched comparisons where a model worse in both area-L2 values has joint IoU at least 0.10 better. These comparisons overlap and are not independent repetitions. [Event ledger](metadata/events_summary.json), [all event states](data/events_metrics.csv), [all event comparisons](data/events_pairs.csv).''')
    add('## 5. State atlas: connect visible field changes to conditional and joint occupation')
    add('''The State pages place the continuous fields and their common-cut partner contours before the conditional curves. Their missing/excess joint maps use a different declared 70th-percentile/range-midpoint slice from the event atlas, so the two masks should not be compared as if they were identical. The conditional query is E[Y | X in bin], with each state assigned by its own X values, fixed GT bin edges, common minimum 64-node bin support, and GT occupancy weights. Unsupported bins are omitted. Full bin counts remain in retained arrays. Five resolutions (8, 10, 12, 16, 20) assess ordering stability.

The raw 2D histogram and its marginal histograms respond to absolute states. The empirical-rank copula uses each state's own marginal ranks; it adds a dependence audit, while tied values and finite bins remain part of the estimator. The model-minus-GT joint occupation panels retain a common scale within each comparison. Neither a raw conditional difference nor a joint histogram alone isolates dependence.''')
    inv_by_stem = {r['stem']: r for r in inventory}
    for s in selected:
        q = c[(c.global_index == s['global_index']) & (c.pair == s['pair']) &
              (c.model_a == s['model_a']) & (c.model_b == s['model_b'])].iloc[0]
        caption = (f"Frame {s['global_index']}, {s['pair']}, {NAMES[s['model_a']]} versus {NAMES[s['model_b']]}. "
                   f"Unweighted Euclidean field L2 is {100*q.l2_a_x:.1f}%/{100*q.l2_a_y:.1f}% versus {100*q.l2_b_x:.1f}%/{100*q.l2_b_y:.1f}%. "
                   f"Conditional discrepancy is {q.state_a:.5f} versus {q.state_b:.5f} normalized Y units, common GT support {100*q.state_coverage:.2f}%; the ordering is stable over all five tested bin counts. "
                   f"Rank-copula TV-to-GT is {q.copula_tv_a:.3f} versus {q.copula_tv_b:.3f}. "
                   'Clouds use fixed per-field shared limits and white current-partner contours; footprint and occupation differences identify the affected coupled configuration. Selection is exploratory.')
        figure(s['stem'], caption)
    add('## 6. Cohort evidence: how often does the accuracy ordering fail for these queries?')
    add('''Every unordered field pair, frame and model pair is retained: 10 × 100 × 15 = 15,000 comparisons. The new screen removes the old requirement that both models lie below their own cohort medians. Instead it reports nested 10%, 20% and 40% per-field L2 match strata, preserving poorer-fidelity cases with their literal errors. A match gap is |eA−eB|/max(eA,eB) for each field. A model is L2-dominant only if its error is higher in both fields; an ordering reversal occurs when its descriptor error is lower.

The table below separates any-order reversals from material contrasts. A conditional contrast of 0.05 means five percent of the GT Y-field spatial standard deviation; copula contrast 0.10 is a difference in bounded rank-histogram TV errors. These thresholds organize an exploratory evidence bank and are not significance tests. The normalized conditional effect is the difference of two errors-to-GT, rather than the direct distance between their conditional curves.''')
    add(table(['L2 gate', 'Eligible n', 'State gap >=0.05 SD', 'Also bin-stable', 'Copula gap >=0.10', 'Material inversions / dominant n'], summaries))
    for stem in ('inversion_heatmaps', 'fidelity_strata', 'cohort_state', 'cohort_joint_tv', 'cohort_copula_tv'):
        figure(stem, inv_by_stem[stem]['caption'] + ' The inversion heatmap counts any strict ordering reversal; material-effect counts are reported separately above.')
    add('''The breadth claim is specific: accuracy–relationship disagreement is repeatedly observed across this declared model/frame/query collection, rather than appearing only in the original 9867 example. The data do not establish that all architectures fail, that every plausible reconstruction is physically inconsistent, or that the numerical rate transfers to other datasets. Chronologically related snapshots, repeated field pairs and repeated model comparisons preclude interpreting the row count as independent sample size. No checkpoint was selected using these relationship scores.

Complete outputs: [candidate CSV](data/metrics/candidates.csv), [frame descriptors](data/metrics/frame_metrics.csv), [material contrasts by pair](data/metrics/material_prevalence.csv), [cohort table](data/metrics/cohort.csv), [all definitions and hashes](data/metrics/definitions.json), [exploratory candidate index](data/metrics/candidate_index.json).''')
    add('## 7. Scale and Shape: complementary probes with explicit attribution')
    add('''The visualization Scale estimator is a native-coordinate, trapezoid-area NUDFT with mean removal, Hann taper and fixed physical octave bands. Signed complex cross-products are normalized by unary powers, and bands require at least 1% power in each reference and compared field. Unary spectra remain alongside cross-products. The selected comparison is independently recomputed with a rectangular window and common two-window support. The visible ordering changes by band/window: it supplies a sensitivity audit, not a robust model ranking or an isolated mutual-Scale failure.

This estimator differs from the training code's graph-basis, ensemble same-mode coherence and cross-band energy statistics. In particular, the latter magnitude-squared statistics can lose sign, and cross-band energy covariance is not the full second-order LPC Pair block. The stronger signed diagnostic is useful here because the premise study is testing information that a descriptor may omit; it is not silently replacing the historical training objective.''')
    figure('example_scale', inv_by_stem['example_scale']['caption'] + ' Model ordering is window/band dependent; unary spectra also differ. This is an estimator-sensitivity illustration.')
    figure('cohort_scale', inv_by_stem['cohort_scale']['caption'] + ' The cohort census uses Hann only; two-window robustness is assessed for the displayed selected case, not every row.')
    add('''Shape retains geometric overlap separately from topology. The cohort's topology summary is the absolute discrepancy in four-connected joint excursion component counts, over three GT level settings and three native-area floors (0.003, 0.006, 0.010 of domain area). It is not persistent homology. CH4–CO and CH4–T use CH4-low and partner-high slices; other pairs use high–high. Actual thresholds, GT support and predicted occupancy are saved; supported cohort summaries require nontrivial GT joint area across slices.

The real-field pages establish coupled-region organization and its relation to fieldwise accuracy. Component-count differences may arise from unary amplitude, unary connectivity, thresholding or joint arrangement. Their attribution controls are shown rather than asserting a pure mutual-topology positive. The strict Shape toy remains the independently controlled proof that equal unary topology can coexist with different joint topology. V3's unary-inheritance negative findings remain valid and are not overwritten.''')
    for stem in ('example_shape', 'cohort_shape_geometry_mean', 'cohort_shape_component_abs_mean'):
        figure(stem, inv_by_stem[stem]['caption'])
    add('## 8. What this evidence motivates for post-training')
    add('''A common differentiable reconstructed endpoint permits the same relational queries to be applied to different native predictors. Paired TRAIN references supply relationship targets, and SOURCE-relative per-field risks express the allowed fidelity change. This separates the scientific query from architecture-specific pretraining objectives. Scalarization and ConFIG are solver choices inside that interface, rather than the scientific premise itself.

The present V4 work performs inference and diagnostics only. It supplies candidate queries and failure examples that can motivate subsequent refinement and its independent evaluation. It does not add a new post-training efficacy result. Existing Premise_Add endpoint results, complete routine histories and execution-cost probes retain their own observation/solver/selection contracts; they must not be concatenated with this T256 TEST cohort as a before/after experiment. Paired target use, selected versus final checkpoints, full history and total cost remain separate reporting questions.

**Paper-facing premise wording:** “Across pretrained multi-field reconstruction models, similar fieldwise error can coexist with different thermochemical associations and coupled-region organization. Controlled examples preserve fieldwise errors and selected unary statistics while changing cross-field relationships; a fixed combustion cohort exhibits repeated accuracy–relationship ordering disagreements. These observations motivate complementary data-derived State, Scale and Shape queries for fidelity-constrained refinement.”

“Physical coherence” here denotes agreement with the declared empirical coupled-state observations. A governing-equation, causal or cross-dataset claim requires additional evidence.''')
    add('## 9. Execution, preservation and reuse')
    add('''The original GPU2 EDM training was interrupted with SIGINT after process ownership, command, working directory and physical UUID were verified. Its last checkpoint remains epoch 1400; the completed train history extends to epoch 1403, and available validation/preview history to epoch 1400. The following curve labels these distinct histories. No training was resumed, and model extraction used only physical GPU2. Historical run files and all V1/V2/V3/Add evidence remain in place.''')
    figure('diffusion_paused_loss', 'Paused DiffusionPDE EDM history: completed sampled TRAIN loss through epoch 1403, saved validation checks through epoch 1400, and last checkpoint at epoch 1400. Native denoising losses and reconstruction validation metrics remain separately labelled; a low denoising loss is not a reconstruction-quality certificate.')
    add('''[Pause record](metadata/training_pause.json), [loss history record](metadata/paused_loss_curve.json), [verification](metadata/verification.json), [model source audit](metadata/model_audit.json).

The retained model arrays are normalized full-grid endpoints, not checkpoint copies. Matching old archives use relative links; new models store one compressed endpoint archive. No completed per-frame inference cache remains. Small selected maps, conditional curves, joint occupation, event masks/thresholds and spectral band summaries support redraw. Complete candidate tables intentionally preserve both accepted and rejected possibilities. They are evidence catalogs, not discarded optimization intermediates.

For a new model, add its run to the registry, freeze its last checkpoint with `inference.py extend`, and extract only that model with the same cohort and protocol. Then recompute the model-pair census and selection, redraw figures, rebuild this report and verify the rendered result. Existing frozen identities must not be replaced. Changing an existing model's last checkpoint requires a new study/version identity. Detailed commands and the inference environment are in [README](README.md).''')
    add('## Figure index')
    add('[Complete figure atlas: all PDF masters at their original page sizes](FIGURE_ATLAS.pdf). The atlas preserves vector text and axes and is convenient for inspecting dense panels or selecting future manuscript figures.')
    add(table(['Figure', 'Retained preview and PDF master'], [[i, f'[{stem}](figures/{stem}.png) · [PDF](figures/{stem}.pdf)'] for i, stem, _ in figure_index]))
    add('## Reading and artifact map')
    add('''[Original V1](../REPORT.md) · [V2](../REPORT_V2.md) · [V3 story](../V3_PRESENTATION_20MIN.md) · [V3 evidence](../V3_EVIDENCE_LEDGER.md) · [post-training addition](../Premise_Add/SPEAKER_NOTES.md) · [latest presentation review](../Premise_Add/Presentation_Review_20261029/SPEAKER_SCRIPT_AND_REVIEW.md) · [reproduction and extension](README.md).

This report is generated from saved evidence by `scripts/build_report.py`. Its source paragraphs and captions remain unwrapped. Figure files are independent Python-generated PDF masters with relative PNG previews.''')
    (V4 / 'REPORT.md').write_text('\n'.join(parts))
    (V4 / 'metadata/report_figures.json').write_text(json.dumps([{'number': i, 'stem': s, 'caption': cap} for i, s, cap in figure_index], indent=2) + '\n')
    subprocess.run(['/home/wanglz/miniconda3/envs/fig/bin/python', str(V4 / 'scripts/build_atlas.py')], check=True)
    print(f'Wrote REPORT.md with {len(figure_index)} figures')


if __name__ == '__main__':
    main()

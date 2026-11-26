"""Render and validate terminal executor preparation evidence for the Round 7 report."""
import math
from scripts.analysis.car_round7_preparation_readout import action_product_lines

FAMILIES = ('global_distribution', 'multiscale_spectrum', 'topology')
OBJECTIVES = ('composite', 'joint_anchor')
SETTINGS = ('separate_repeat_2', 'factored_static_features')


def validate_preparation(data, reference_sha256, kappa, bind_all, document):
    if data.get('schema') != 'car_round7_terminal_preparation_readout_v1' or data.get('status') != 'completed':
        raise ValueError('completed terminal preparation readout required')
    list(bind_all(data))
    for state in ('fresh_source', 'mature_checkpoint'):
        screen = data[state]
        if screen['state'] != state or screen['reference_sha256'] != reference_sha256 or screen['field_row_active'] != (state == 'mature_checkpoint'):
            raise ValueError('preparation screen state or training reference differs')
        combinations = {(item['objective'], item['setting']) for item in screen['attempts']}
        if len(screen['attempts']) != 6 or combinations != {(o, s) for o in OBJECTIVES for s in ('separate_repeat_1', *SETTINGS)}:
            raise ValueError('both objective screens need two baselines and the cached candidate')
        for item in screen['attempts']:
            if set(item['per_family_row_comparison_to_first_separate']) != set(FAMILIES):
                raise ValueError('all three family row comparisons are required')
    if {r['objective'] for r in data['production_resume'] if r['qualified']} != set(OBJECTIVES) or len(data['production_resume']) != 2:
        raise ValueError('both production resume qualifications required')
    strict = data['strict_resume_comparison']
    if strict['results'] != document(strict['source']) or len(strict['results']) != 2:
        raise ValueError('strict resume comparison differs from bound evidence')
    required = ('strict_scientific_checkpoint_differences', 'strict_recording_definition_differences',
                'strict_scientific_history_differences', 'logs/epoch_calibrated_pi_updates.jsonl',
                'metrics/training_coupling.jsonl', 'logs/scalable_updates.jsonl')
    if any(row.get('comparison_atol') != 0 or row.get('comparison_rtol') != 0 or
           any(row.get(key) != [] for key in required) for row in strict['results']):
        raise ValueError('strict resume comparisons must have zero tolerance and no differences')
    pilot = data['joint_pilot']
    if (pilot['reporting_epochs'] != 10 or pilot['contiguous_actions'] != 120 or
        pilot['reference_sha256'] != reference_sha256 or not math.isclose(pilot['kappa'], kappa, rel_tol=1e-14) or
        len(pilot['epoch_observations']) != 10 or len(pilot['epoch_timing']) != 10 or len(pilot['curves_decoded']) != 4):
        raise ValueError('terminal pilot recording, training reference or kappa differs')
    return data


def preparation_lines(data):
    def number(x):
        if x is None: return 'n/a'
        if not isinstance(x, (int, float)) or not math.isfinite(x): raise ValueError('finite parity value required')
        return f'{x:.3g}'
    def pair(x): return f"{number(x['max_abs'])} / {number(x['relative_l2'])}"
    lines = ['## Executor numerical and recording qualification', '',
             'Each cell is maximum absolute error / relative L2 error against the first separate-backward run. The repeated separate run provides same-state numerical variability; the cached factored run is the candidate. These float32 comparisons do not establish trajectory equivalence.', '',
             '| State | Objective | Compared run | State row | Scale row | Shape row | Weighted field row | Direction | Actual displacement |',
             '| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for state in ('fresh_source', 'mature_checkpoint'):
        for objective in OBJECTIVES:
            for setting in SETTINGS:
                item = next(x for x in data[state]['attempts'] if x['objective'] == objective and x['setting'] == setting)
                families = item['per_family_row_comparison_to_first_separate']
                metrics = item['comparison_to_first_separate']
                cells = [pair(families[name]) for name in FAMILIES] + [pair(metrics[name]) for name in ('field', 'direction', 'displacement')]
                lines.append(f"| {state} | {objective} | {setting} ({item['parameter_row_dtype']}) | " + ' | '.join(cells) + ' |')
    lines += ['', 'The fresh SOURCE field row is inactive; relative errors for zero-norm references are unavailable. Family and field norms, absolute L2 errors, and value/risk comparisons are retained in the bound preparation readout.', '',
              'Both objective-specific fresh-process save/reload/resume checks passed. Independent zero-tolerance scientific checkpoint, definitions, history, PI, coupling and scalable-update comparisons found no differences after the explicitly recorded timing/accounting exclusions.', '',
              '| Pilot epoch | Learning rate | κ | Raw joint | Weighted joint | Shape self | Shape mutual | Active field fraction |',
              '| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for item in (data['joint_pilot']['epoch_observations'][0], data['joint_pilot']['epoch_observations'][-1]):
        keys = ('epoch', 'optimizer_lr', 'round7_scale_kappa', 'coherence_component/multiscale_spectrum/joint/raw', 'coherence_component/multiscale_spectrum/joint/weighted_contribution', 'topology.self', 'topology.mutual', 'active_field_fraction')
        lines.append('| ' + ' | '.join(number(item[key]) for key in keys) + ' |')
    final = data['joint_pilot']['epoch_observations'][-1]
    family_values = '; '.join(f"{name}: {number(final[f'coherence_family/{name}/raw'])} / {number(final[f'coherence_family/{name}/weighted_contribution'])}" for name in FAMILIES)
    calibration = data['joint_pilot']['scalable_calibration']
    lines += ['', f"Epoch 10 sampled raw / weighted family scalars are {family_values}. Prefit coefficients constant across actions = {calibration['coefficients_constant']}; calibration-frozen flag constant = {calibration['calibration_frozen_constant']} (first {calibration['calibration_frozen_first']}, last {calibration['calibration_frozen_last']}). Query exposure and member/query identities for all eight groups are retained in the bound readout.", '', 'Scores are pre-action epoch averages and LR is averaged per accepted action; PI multipliers refer to before-action state and displacement to the actual post-action Adam update.']
    lines.extend(['', 'First-order products across the 120 accepted pilot actions:', ''])
    lines.extend(action_product_lines(data['joint_pilot']['actual_displacement_products']))
    timing = data['joint_pilot']['epoch_timing']
    total_wall = sum(float(row['wall_seconds']) for row in timing)
    total_train = sum(float(row['train_seconds']) for row in timing)
    total_callbacks = sum(float(row['callback_and_reporting_seconds']) for row in timing)
    lines += ['', f'The pilot recorded 10 complete reporting epochs and 120 contiguous accepted actions: {total_wall:.3f} s total epoch span, including {total_train:.3f} s train and {total_callbacks:.3f} s callback/reporting time. Its PI records and both decoded loss/coherence curve formats are bound in the preparation readout. These sampled training observations qualify execution and recording; they are not model efficacy evidence.', '']
    return lines

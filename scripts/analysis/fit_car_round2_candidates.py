#!/usr/bin/env python3
"""Fit bounded TRAIN candidate geometry; task/parameter qualification follows.

This CPU endpoint discovery never declares a final selected bank or an allocator.
It reuses the prepared source-specific C8 core, physical support and J1/J2 view.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path

CODE_ROOT = Path(os.environ.get('CAR_CODE_ROOT', Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(CODE_ROOT / 'src'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared-run', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--max-wall-seconds', type=int, default=1800)
    args = parser.parse_args()
    start = time.time()
    out, run = args.output_dir.resolve(), args.prepared_run.resolve()
    if out.exists():
        raise FileExistsError(out)
    if args.max_wall_seconds <= 0:
        raise ValueError('positive bounded CPU job duration required')
    out.mkdir(parents=True)
    receipt = {'schema': 'car_round2_candidates_job_v1', 'start_unix': start,
               'independent_cpu': True, 'gpu_lease': None, 'status': 'running'}
    def timeout(_signum, _frame):
        raise TimeoutError('bounded candidate discovery CPU allowance exceeded')
    previous = signal.signal(signal.SIGALRM, timeout)
    signal.alarm(args.max_wall_seconds)
    try:
        import torch
        import yaml

        from phycoflow_reconstruction.coherence import build_enabled_families
        from phycoflow_reconstruction.data.factory import open_field_dataset
        from phycoflow_reconstruction.training.run_store import file_sha256
        from phycoflow_reconstruction.training.scalable_fitting import fit_scalable_artifacts
        torch.set_num_threads(args.threads)
        config = yaml.safe_load((run / 'resolved_config.yaml').read_text())
        if config['dataset'].get('split', 'train') != 'train':
            raise ValueError('TRAIN-only candidate discovery required')
        dataset = open_field_dataset(config['dataset'], split='train')
        try:
            families = build_enabled_families(config['coherence'], dataset.data_spec, dataset.normalizer)
            manifest = json.loads((run / 'run_manifest.json').read_text())
            for name, family in families.items():
                path = run / 'artifacts' / f'{name}_family.pt'
                if file_sha256(path) != manifest['coherence_family_state_sha256s'][name]:
                    raise ValueError(f'prepared family hash changed: {name}')
                family.load_state_artifact(torch.load(path, map_location='cpu', weights_only=True))
            panel_path, core_path = [run / 'artifacts' / f'{name}.pt'
                                     for name in ('source_panel', 'scalable_coherence')]
            panel = torch.load(panel_path, map_location='cpu', weights_only=True)
            core = torch.load(core_path, map_location='cpu', weights_only=True)
            if panel['provenance'] != core['provenance'] or len(panel['sample_ids']) != max(1, __import__('math').ceil(.1 * len(dataset))):
                raise ValueError('exact declared approximately 10% TRAIN SOURCE panel required')
            expected_members = 4 if config['model']['name'] == 'gl_rbf_cq' else 1
            if panel['members_count'] != expected_members:
                raise ValueError('prepared SOURCE panel must be genuine J4 law or J1 point')
            # The input binding uses the complete saved SOURCE member panel;
            # the discovery score uses all its native members, with J1 paired.
            artifact = fit_scalable_artifacts(
                families=families, source=panel['members'], reference=panel['reference'],
                coordinates=panel['coordinates'], provenance=panel['provenance'],
                source_weights=panel['masses'], reference_weights=panel['masses'],
                seed=int(config['runtime'].get('seed', 42)) + 810001,
                reuse_core=core, adaptive_settings={'panel_batch_size': 8, 'floor': .01,
                    'sketch_size': 128, 'redundancy_limit': .95, 'units': 'model'})
            bank = artifact['adaptive_bank']
            (out / 'provisional_bank.json').write_text(json.dumps(bank, indent=2, allow_nan=False) + '\n')
            (out / 'candidate_atlas.json').write_text(json.dumps({
                'schema': 'car_round2_candidate_atlas_v1', 'status': 'awaiting_actual_parameter_response',
                'source_panel_sha256': file_sha256(panel_path), 'core_sha256': file_sha256(core_path),
                'sample_ids': panel['sample_ids'], 'scoring_mode': bank['scoring_mode'],
                'candidate_count': bank['candidate_generation']['candidate_count'],
                'reserved_task_used': False, 'rows': [
                    {'id': row['id'], 'candidate': row['candidate'],
                     'source_discrepancy': bank['measurements'][row['id']]['discrepancy'],
                     'repeatability': bank['measurements'][row['id']]['repeatability'],
                     'endpoint_gradient_heuristic': bank['measurements'][row['id']]['sensitivity'],
                     'measured_endpoint_cost_seconds': bank['measurements'][row['id']]['cost'],
                     'actual_parameter_response': None, 'development_relevance': None,
                     'final_selection': None} for row in bank['selection_audit']]
            }, indent=2, allow_nan=False) + '\n')
            receipt.update(status='completed', sample_count=len(panel['sample_ids']),
                           candidate_count=bank['candidate_generation']['candidate_count'],
                           provisional_bank_sha256=file_sha256(out / 'provisional_bank.json'))
        finally:
            dataset.close()
    except BaseException as exc:
        receipt.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)
        receipt.update(end_unix=time.time(), wall_seconds=time.time() - start)
        (out / 'job_receipt.json').write_text(json.dumps(receipt, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
